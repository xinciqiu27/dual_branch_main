from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset


class PairwiseTrainDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        negative_pool_map: dict[int, list[int]],
        query_text_emb_map: dict[str, np.ndarray],
        num_apis: int,
        seed: int = 42,
    ) -> None:
        self.samples = samples
        self.negative_pool_map = negative_pool_map
        self.query_text_emb_map = query_text_emb_map
        self.num_apis = num_apis
        self.rng = random.Random(seed)
        self.query_text_dim = 0
        if query_text_emb_map:
            self.query_text_dim = next(iter(query_text_emb_map.values())).shape[0]

    def __len__(self) -> int:
        return len(self.samples)

    def _get_query_text(self, key: str) -> np.ndarray:
        if key in self.query_text_emb_map:
            return self.query_text_emb_map[key]
        return np.zeros((self.query_text_dim,), dtype=np.float32)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        neg = self.rng.choice(self.negative_pool_map[idx])
        return {
            "mashup_id": sample["mashup_id"],
            "selected_api_ids": sample["selected_api_ids"],
            "positive_api_id": sample["positive_api_id"],
            "negative_api_id": neg,
            "query_text_emb": self._get_query_text(sample["query_key"]),
            "query_key": sample["query_key"],
        }


def collate_pairwise(batch: list[dict]) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_len = max(len(item["selected_api_ids"]) for item in batch)

    mashup_id = torch.tensor([item["mashup_id"] for item in batch], dtype=torch.long)
    pos_api = torch.tensor([item["positive_api_id"] for item in batch], dtype=torch.long)
    neg_api = torch.tensor([item["negative_api_id"] for item in batch], dtype=torch.long)

    selected = torch.zeros((batch_size, max_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
    for i, item in enumerate(batch):
        ids = item["selected_api_ids"]
        selected[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[i, :len(ids)] = 1.0

    query_text = torch.tensor(np.stack([item["query_text_emb"] for item in batch]), dtype=torch.float32)
    return {
        "mashup_id": mashup_id,
        "selected_api_ids": selected,
        "selected_mask": mask,
        "positive_api_id": pos_api,
        "negative_api_id": neg_api,
        "query_text_emb": query_text,
    }
