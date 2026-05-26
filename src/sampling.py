from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from typing import Iterable

import numpy as np


def query_key(mashup_id: int, selected_api_ids: list[int]) -> str:
    # query_key 唯一标识“同一个 mashup + 同一个 selected set”。
    selected_api_ids = sorted(selected_api_ids)
    return f"m{mashup_id}|s{'-'.join(map(str, selected_api_ids))}"


def build_train_samples(
    invocations: dict[int, list[int]],
    mashup_ids: list[int],
    max_selected: int = 3,
    max_subsets_per_positive: int = 3,
    cold_apis: set[int] | None = None,
) -> list[dict]:
    samples: list[dict] = []
    cold_apis = cold_apis or set()
    for mid in mashup_ids:
        # 同一 mashup 的历史 API 集合被看作一个功能组合。
        apis = sorted(set(invocations[mid]))
        for pos in apis:
            if pos in cold_apis:
                continue
            # 训练时 selected 集合显式排除正样本 pos。
            remaining = [a for a in apis if a != pos and a not in cold_apis]
            if not remaining:
                continue
            count = 0
            for r in range(1, min(max_selected, len(remaining)) + 1):
                for subset in itertools.combinations(remaining, r):
                    # 一个样本表示：给定 mashup 和已选 API 子集，补全 pos 这个 API。
                    samples.append({
                        "mashup_id": mid,
                        "selected_api_ids": list(subset),
                        "positive_api_id": pos,
                        "all_api_ids": apis,
                        "query_key": query_key(mid, list(subset)),
                    })
                    count += 1
                    if count >= max_subsets_per_positive * max_selected:
                        break
                if count >= max_subsets_per_positive * max_selected:
                    break
    return samples


def build_eval_queries(
    invocations: dict[int, list[int]],
    mashup_ids: list[int],
    max_selected: int = 3,
    max_queries_per_mashup: int = 6,
    cold_apis: set[int] | None = None,
) -> list[dict]:
    queries: list[dict] = []
    cold_apis = cold_apis or set()
    for mid in mashup_ids:
        apis = sorted(set(invocations[mid]))
        built = 0
        for r in range(1, min(max_selected, len(apis) - 1) + 1):
            for subset in itertools.combinations(apis, r):
                selected = list(subset)
                # 评估时的正样本是“该 mashup 中剩余、且未被 selected 覆盖的 API”。
                targets = sorted([a for a in apis if a not in selected and (not cold_apis or a in cold_apis)])
                if not targets:
                    continue
                queries.append({
                    "mashup_id": mid,
                    "selected_api_ids": selected,
                    "target_api_ids": targets,
                    "all_api_ids": apis,
                    "query_key": query_key(mid, selected),
                })
                built += 1
                if built >= max_queries_per_mashup:
                    break
            if built >= max_queries_per_mashup:
                break
    return queries


def build_cold_api_eval_queries(
    invocations: dict[int, list[int]],
    mashup_ids: list[int],
    cold_apis: set[int] | None = None,
    max_selected: int = 3,
    max_queries_per_mashup: int = 6,
) -> list[dict]:
    queries: list[dict] = []
    cold_apis = cold_apis or set()
    for mid in mashup_ids:
        apis = sorted(set(invocations[mid]))
        warm_apis = [a for a in apis if a not in cold_apis]
        cold_targets = [a for a in apis if a in cold_apis]
        if not warm_apis or not cold_targets:
            continue
        built = 0
        for r in range(1, min(max_selected, len(warm_apis)) + 1):
            for subset in itertools.combinations(warm_apis, r):
                selected = list(subset)
                queries.append({
                    "mashup_id": mid,
                    "selected_api_ids": selected,
                    "target_api_ids": cold_targets,
                    "all_api_ids": apis,
                    "query_key": query_key(mid, selected),
                })
                built += 1
                if built >= max_queries_per_mashup:
                    break
            if built >= max_queries_per_mashup:
                break
    return queries


def build_holdout_eval_queries(
    observed_invocations: dict[int, list[int]],
    target_invocations: dict[int, list[int]],
    mashup_ids: list[int],
) -> list[dict]:
    queries: list[dict] = []
    for mid in mashup_ids:
        selected = sorted(set(observed_invocations.get(mid, [])))
        targets = sorted(set(target_invocations.get(mid, [])))
        if not selected or not targets:
            continue
        queries.append({
            "mashup_id": mid,
            "selected_api_ids": selected,
            "target_api_ids": targets,
            "all_api_ids": sorted(set(selected) | set(targets)),
            "query_key": query_key(mid, selected),
        })
    return queries


def build_negative_pool_map(
    train_samples: list[dict],
    train_invocations: dict[int, list[int]],
    all_api_ids: list[int],
    mode: str = "strict_global",
    api_sim_matrix: np.ndarray | None = None,
) -> dict[int, list[int]]:
    # 记录某个 selected set 在训练集中曾与哪些 API 共现过。
    # strict_global 的核心思想就是：这些“见过的可补全项”不要当负例。
    selected_to_union: dict[tuple[int, ...], set[int]] = defaultdict(set)
    selected_to_pos: dict[tuple[int, ...], set[int]] = defaultdict(set)
    for s in train_samples:
        key = tuple(sorted(s["selected_api_ids"]))
        selected_to_union[key].update(s["all_api_ids"])
        selected_to_pos[key].add(s["positive_api_id"])

    out: dict[int, list[int]] = {}
    universe = set(all_api_ids)
    for i, s in enumerate(train_samples):
        selected = tuple(sorted(s["selected_api_ids"]))
        current_all = set(s["all_api_ids"])
        if mode == "random":
            # 最简单随机负采样：只避开当前 mashup 已真实调用过的 API。
            pool = sorted(list(universe - current_all))
        else:
            forbidden = selected_to_union[selected]
            # strict_global / hard 都先从更严格的全局禁止集合中排除候选。
            pool = sorted(list(universe - forbidden))
        if not pool:
            # 若过滤后为空，退回简单随机池，避免样本失效。
            pool = sorted(list(universe - current_all))
        if mode == "hard" and api_sim_matrix is not None:
            pos = s["positive_api_id"]
            # hard negative 不是在线挖掘模型误判，而是基于文本相似度选更像正例的候选。
            sims = [(cand, float(api_sim_matrix[pos, cand])) for cand in pool]
            sims.sort(key=lambda x: x[1], reverse=True)
            topn = max(1, min(50, len(sims)))
            pool = [cand for cand, _ in sims[:topn]]
        out[i] = pool
    return out


def choose_negative(pool: list[int], rng: random.Random) -> int:
    return rng.choice(pool)
