from __future__ import annotations

import os
import time
from dataclasses import dataclass

import pandas as pd
import requests
import torch


@dataclass
class ChatAPIConfig:
    base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str = ""
    temperature: float = 0.2
    max_retries: int = 3
    timeout: int = 120


class OpenAICompatibleChatGenerator:
    def __init__(self, config: ChatAPIConfig) -> None:
        self.config = config
        self.api_key = config.api_key or os.environ.get(config.api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"Missing API key. Pass api_key directly or set env: {config.api_key_env}"
            )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
        }
        last_err = None
        for _ in range(self.config.max_retries):
            try:
                resp = requests.post(
                    f"{self.config.base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout,
                )
                if not resp.ok:
                    body = resp.text[:2000]
                    raise RuntimeError(
                        f"HTTP {resp.status_code} from chat API. Response body: {body}"
                    )
                return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                last_err = e
                time.sleep(1.0)
        raise RuntimeError(f"Chat generation failed: {last_err}") from last_err


@dataclass
class LocalHFChatConfig:
    model_name: str
    hf_token_env: str = "HF_TOKEN"
    max_new_tokens: int = 96
    temperature: float = 0.2
    top_p: float = 0.9
    device_map: str = "auto"
    trust_remote_code: bool = False
    torch_dtype: str = "auto"


class LocalHFChatGenerator:
    def __init__(self, config: LocalHFChatConfig) -> None:
        self.config = config
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = os.environ.get(config.hf_token_env)
        dtype = self._resolve_dtype(config.torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            token=token,
            trust_remote_code=config.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            token=token,
            trust_remote_code=config.trust_remote_code,
            device_map=config.device_map,
            torch_dtype=dtype,
        )
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def _resolve_dtype(name: str):
        key = (name or "auto").lower()
        if key == "float16":
            return torch.float16
        if key == "bfloat16":
            return torch.bfloat16
        if key == "float32":
            return torch.float32
        return "auto"

    @torch.no_grad()
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = self.config.temperature > 0
        outputs = self.model.generate(
            **encoded,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=do_sample,
            temperature=self.config.temperature if do_sample else None,
            top_p=self.config.top_p if do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = outputs[0][encoded["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text


def save_rows(rows: list[dict], output_csv: str) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
