from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import requests
import torch


@dataclass
class EncoderConfig:
    # backend 决定文本编码来源；其余字段是三种 backend 的统一配置层。
    backend: str = "sbert"  # sbert | llm_hf | llm_api
    model_name: str = "all-MiniLM-L6-v2"
    max_length: int = 512
    batch_size: int = 32
    instruction_prefix: str = ""
    api_base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    trust_remote_code: bool = False
    normalize: bool = True


class BaseTextEncoder:
    def __init__(self, config: EncoderConfig, device: str | torch.device | None = None) -> None:
        self.config = config
        if device is None or str(device).lower() == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif isinstance(device, torch.device):
            self.device = str(device)
        else:
            self.device = str(device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    @property
    def signature(self) -> str:
        # signature 用于缓存命名；只要编码配置变化，缓存 key 就会变化。
        raw = f"{self.config.backend}|{self.config.model_name}|{self.config.max_length}|{self.config.instruction_prefix}|{self.config.normalize}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        if not self.config.normalize:
            return arr.astype(np.float32)
        # 统一做 L2 normalize，方便后续余弦相似度和图构建。
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (arr / norms).astype(np.float32)


class SentenceTransformerEncoder(BaseTextEncoder):
    def __init__(self, config: EncoderConfig, device: str | None = None) -> None:
        super().__init__(config, device)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(config.model_name, device=self.device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            # all-MiniLM-L6-v2 常见维度是 384；这里只是空输入时的占位约定。
            return np.zeros((0, 384), dtype=np.float32)
        arr = self.model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return self._normalize(arr)


class HFMeanPoolingEncoder(BaseTextEncoder):
    def __init__(self, config: EncoderConfig, device: str | None = None) -> None:
        super().__init__(config, device)
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            hidden_size = int(getattr(self.model.config, "hidden_size", 768))
            return np.zeros((0, hidden_size), dtype=np.float32)
        prepared = [
            # 某些 embedding 模型会吃 instruction prefix，这里可选拼接。
            f"{self.config.instruction_prefix}\n{text}" if self.config.instruction_prefix else text
            for text in texts
        ]
        outs: list[np.ndarray] = []
        for i in range(0, len(prepared), self.config.batch_size):
            batch = prepared[i:i+self.config.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self.device)
            output = self.model(**encoded)
            token_embeddings = output[0]
            # mean pooling 仅对 attention mask 覆盖的 token 求平均。
            attn = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled = (token_embeddings * attn).sum(dim=1) / torch.clamp(attn.sum(dim=1), min=1e-9)
            outs.append(pooled.detach().cpu().numpy())
        arr = np.concatenate(outs, axis=0)
        return self._normalize(arr)


class OpenAICompatibleEmbeddingEncoder(BaseTextEncoder):
    def __init__(self, config: EncoderConfig, device: str | None = None) -> None:
        super().__init__(config, device)
        self.base_url = (config.api_base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.environ.get(config.api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"Environment variable {config.api_key_env} is required for llm_api backend."
            )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            # text-embedding-3-large 常见维度是 3072；这里同样是空输入占位。
            return np.zeros((0, 3072), dtype=np.float32)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        outs: list[np.ndarray] = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = list(texts[i:i+self.config.batch_size])
            payload = {
                "model": self.config.model_name,
                "input": batch,
            }
            resp = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            # 服务端返回可能不按输入顺序排列，因此按 index 排序后再拼。
            data = sorted(data, key=lambda x: x["index"])
            arr = np.array([row["embedding"] for row in data], dtype=np.float32)
            outs.append(arr)
        arr = np.concatenate(outs, axis=0)
        return self._normalize(arr)


def build_text_encoder(config: EncoderConfig, device: str | None = None) -> BaseTextEncoder:
    backend = config.backend.lower()
    if backend == "sbert":
        return SentenceTransformerEncoder(config, device=device)
    if backend == "llm_hf":
        return HFMeanPoolingEncoder(config, device=device)
    if backend == "llm_api":
        return OpenAICompatibleEmbeddingEncoder(config, device=device)
    raise ValueError(f"Unsupported encoder backend: {config.backend}")
