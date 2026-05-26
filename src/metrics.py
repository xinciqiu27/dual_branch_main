from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    # 只看前 k 个位置；后续位置不参与折损累计收益。
    rel = np.asarray(list(relevances)[:k], dtype=np.float32)
    if rel.size == 0:
        return 0.0
    # 第 1 个位置折损为 1，后续位置按 log2 递减。
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    return float(np.sum(rel * discounts))


def ndcg_at_k(binary_hits: Sequence[int], num_pos: int, k: int) -> float:
    # 实际排序的 DCG。
    dcg = dcg_at_k(binary_hits, k)
    # 理想排序等价于把所有正样本都尽量排到最前面。
    ideal = [1] * min(num_pos, k)
    idcg = dcg_at_k(ideal, k)
    return 0.0 if idcg == 0 else dcg / idcg


def first_relevant_rank(sorted_items: Sequence[int], positives: set[int]) -> int:
    # 返回第一个命中正样本的位置；用于计算 MRR。
    for i, item in enumerate(sorted_items, start=1):
        if item in positives:
            return i
    # 若完全没有命中，则返回越界位置，调用方据此给 0 分。
    return len(sorted_items) + 1


def evaluate_ranked_list(sorted_items: Sequence[int], positives: set[int], ks: Sequence[int]) -> dict[str, float]:
    # 正样本集合通常是“当前 query 剩余可补全的 API”。
    pos = set(positives)
    out: dict[str, float] = {}
    rank = first_relevant_rank(sorted_items, pos)
    out["MRR"] = 0.0 if rank > len(sorted_items) else 1.0 / rank
    for k in ks:
        # 截取前 k 个推荐结果。
        topk = list(sorted_items[:k])
        hit_count = sum(1 for x in topk if x in pos)
        recall = hit_count / max(1, len(pos))
        precision = hit_count / max(1, k)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        # NDCG 使用二值相关性：命中正样本记 1，否则记 0。
        binary = [1 if x in pos else 0 for x in topk]
        out[f"Recall@{k}"] = recall
        out[f"Precision@{k}"] = precision
        out[f"F1@{k}"] = f1
        out[f"NDCG@{k}"] = ndcg_at_k(binary, len(pos), k)
    return out


def average_metric_dict(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    # 默认所有 metric dict 的 key 完全一致。
    keys = metrics[0].keys()
    return {k: float(np.mean([m[k] for m in metrics])) for k in keys}
