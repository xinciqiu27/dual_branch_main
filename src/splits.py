from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class SplitResult:
    # train/val/test 中存的是 mashup 的内部索引，不是原始 ID。
    train_mashups: list[int]
    val_mashups: list[int]
    test_mashups: list[int]
    cold_apis: list[int] | None = None


def standard_split_from_provided(
    train_invocations: dict[int, list[int]],
    test_invocations: dict[int, list[int]],
    val_ratio: float = 0.125,
    seed: int = 42,
    exclude_val_from_test: bool = True,
) -> SplitResult:
    rng = np.random.default_rng(seed)
    train_mashups = sorted(train_invocations.keys())
    rng.shuffle(train_mashups)
    # Warm-start standard protocol:
    # 1) train supervision comes from train.txt on train mashups
    # 2) validation holds out one observed train API on val mashups
    # 3) test uses train.txt as observed context and test.txt as held-out targets
    # 4) val/test mashups must be disjoint to avoid using the same mashup for
    #    early stopping and final reporting
    n_val = max(1, int(len(train_mashups) * val_ratio))
    val = sorted(train_mashups[:n_val])
    train = sorted(train_mashups[n_val:])
    if exclude_val_from_test:
        test = sorted(set(test_invocations.keys()) - set(val))
    else:
        test = sorted(test_invocations.keys())
    return SplitResult(train_mashups=train, val_mashups=val, test_mashups=test)


def cold_mashup_split(all_invocations: dict[int, list[int]], train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42) -> SplitResult:
    mids = np.array(sorted(all_invocations.keys()))
    rng = np.random.default_rng(seed)
    rng.shuffle(mids)
    n = len(mids)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    train = sorted(mids[:n_train].tolist())
    val = sorted(mids[n_train:n_train+n_val].tolist())
    test = sorted(mids[n_train+n_val:].tolist())
    if not test:
        # 数据很小时保证 test 非空。
        test = val[-1:]
        val = val[:-1]
    return SplitResult(train_mashups=train, val_mashups=val, test_mashups=test)


def cold_api_split(all_invocations: dict[int, list[int]], api_freq: dict[int, int], cold_api_ratio: float = 0.2, seed: int = 42) -> SplitResult:
    # choose cold apis from lower-frequency half to avoid removing all popular anchors
    apis = np.array(sorted(api_freq.keys()))
    freqs = np.array([api_freq[a] for a in apis])
    order = np.argsort(freqs)
    candidate = apis[order[: max(1, len(apis)//2)]]
    rng = np.random.default_rng(seed)
    rng.shuffle(candidate)
    n_cold = max(1, int(len(apis) * cold_api_ratio))
    cold = set(candidate[:n_cold].tolist())

    train, val, test = [], [], []
    for m, apis_used in sorted(all_invocations.items()):
        used = set(apis_used)
        has_cold = len(used & cold) > 0
        has_warm = len(used - cold) > 0
        if has_cold and has_warm:
            # can create context with warm selected apis and predict cold
            test.append(m)
        elif has_cold and not has_warm:
            # not usable for complementary context; keep in val for inspection
            val.append(m)
        else:
            train.append(m)
    if not val and train:
        val = train[-max(1, len(train)//10):]
        train = train[:-len(val)]
    return SplitResult(train_mashups=sorted(train), val_mashups=sorted(val), test_mashups=sorted(test), cold_apis=sorted(cold))
