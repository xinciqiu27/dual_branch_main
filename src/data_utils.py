from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .encoders import BaseTextEncoder
from .graphs import build_threshold_graph, build_topk_graph
from .sampling import query_key


def load_tssgcf_style_data(data_dir: str | Path) -> dict:
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    mashup_df = pd.read_csv(raw_dir / "Mashup_desc.csv")
    api_df = pd.read_csv(raw_dir / "API_desc.csv")

    required_m_cols = {"mashup_id", "description", "mashup_name"}
    required_a_cols = {"api_id", "description", "api_name"}
    if not required_m_cols.issubset(mashup_df.columns):
        raise ValueError(f"Mashup_desc.csv must contain columns: {required_m_cols}")
    if not required_a_cols.issubset(api_df.columns):
        raise ValueError(f"API_desc.csv must contain columns: {required_a_cols}")

    train_inv = read_invocation_txt(raw_dir / "train.txt")
    test_inv = read_invocation_txt(raw_dir / "test.txt")
    # all_inv 主要用于 cold split 和图构建。
    all_inv = merge_invocations(train_inv, test_inv)

    valid_mashups = sorted(all_inv.keys())
    valid_apis = sorted({a for apis in all_inv.values() for a in apis})

    mashup_df = mashup_df[mashup_df["mashup_id"].isin(valid_mashups)].copy()
    api_df = api_df[api_df["api_id"].isin(valid_apis)].copy()

    mashup_df.sort_values("mashup_id", inplace=True)
    api_df.sort_values("api_id", inplace=True)

    mid2idx = {mid: i for i, mid in enumerate(mashup_df["mashup_id"].tolist())}
    aid2idx = {aid: i for i, aid in enumerate(api_df["api_id"].tolist())}
    idx2aid = {v: k for k, v in aid2idx.items()}
    idx2mid = {v: k for k, v in mid2idx.items()}

    mapped_all_inv = {mid2idx[mid]: sorted([aid2idx[a] for a in apis if a in aid2idx]) for mid, apis in all_inv.items() if mid in mid2idx}
    mapped_train_inv = {mid2idx[mid]: sorted([aid2idx[a] for a in apis if a in aid2idx]) for mid, apis in train_inv.items() if mid in mid2idx}
    mapped_test_inv = {mid2idx[mid]: sorted([aid2idx[a] for a in apis if a in aid2idx]) for mid, apis in test_inv.items() if mid in mid2idx}

    api_freq = Counter(a for apis in mapped_all_inv.values() for a in apis)

    return {
        "mashup_df": mashup_df.reset_index(drop=True),
        "api_df": api_df.reset_index(drop=True),
        "mid2idx": mid2idx,
        "aid2idx": aid2idx,
        "idx2mid": idx2mid,
        "idx2aid": idx2aid,
        "all_invocations": mapped_all_inv,
        "train_invocations": mapped_train_inv,
        "test_invocations": mapped_test_inv,
        "api_freq": dict(api_freq),
    }


def read_invocation_txt(path: str | Path) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # 一行格式默认是：mashup_id api_id api_id ...
            items = [int(x) for x in line.strip().split() if x.strip()]
            if not items:
                continue
            out[items[0]] = items[1:]
    return out


def merge_invocations(*parts: dict[int, list[int]]) -> dict[int, list[int]]:
    out: dict[int, set[int]] = {}
    for part in parts:
        for m, apis in part.items():
            out.setdefault(m, set()).update(apis)
    return {m: sorted(list(s)) for m, s in out.items()}


def load_optional_llm_csvs(
    data_dir: str | Path,
    mid2idx: dict[int, int],
    aid2idx: dict[int, int],
    mashup_expanded_csv: str | Path | None = None,
    api_expanded_csv: str | Path | None = None,
    residual_text_csv: str | Path | None = None,
) -> dict:
    data_dir = Path(data_dir)
    llm_dir = data_dir / "llm"
    out = {
        "mashup_expanded": {},
        "api_expanded": {},
        "residual_text": {},
    }
    mashup_path = Path(mashup_expanded_csv) if mashup_expanded_csv else llm_dir / "mashup_expanded.csv"
    api_path = Path(api_expanded_csv) if api_expanded_csv else llm_dir / "api_expanded.csv"
    residual_path = Path(residual_text_csv) if residual_text_csv else llm_dir / "residual_text.csv"

    if mashup_path.exists():
        # 存在就覆盖原始 description；不存在则保持原始文本。
        df = pd.read_csv(mashup_path)
        id_col = "mashup_id" if "mashup_id" in df.columns else df.columns[0]
        text_col = "expanded_text" if "expanded_text" in df.columns else df.columns[-1]
        for _, row in df.iterrows():
            mid = int(row[id_col])
            if mid in mid2idx:
                out["mashup_expanded"][mid2idx[mid]] = str(row[text_col])
    if api_path.exists():
        df = pd.read_csv(api_path)
        id_col = "api_id" if "api_id" in df.columns else df.columns[0]
        text_col = "expanded_text" if "expanded_text" in df.columns else df.columns[-1]
        for _, row in df.iterrows():
            aid = int(row[id_col])
            if aid in aid2idx:
                out["api_expanded"][aid2idx[aid]] = str(row[text_col])
    if residual_path.exists():
        df = pd.read_csv(residual_path)
        key_col = "query_key" if "query_key" in df.columns else df.columns[0]
        text_col = "residual_text" if "residual_text" in df.columns else df.columns[-1]
        for _, row in df.iterrows():
            out["residual_text"][str(row[key_col])] = str(row[text_col])
    return out


def build_entity_texts(bundle: dict, llm_texts: dict) -> dict:
    mashup_df = bundle["mashup_df"]
    api_df = bundle["api_df"]
    mashup_texts = []
    api_texts = []
    for i, row in mashup_df.iterrows():
        mashup_texts.append(llm_texts["mashup_expanded"].get(i, str(row["description"])))
    for i, row in api_df.iterrows():
        api_texts.append(llm_texts["api_expanded"].get(i, str(row["description"])))
    return {"mashup_texts": mashup_texts, "api_texts": api_texts}


def generate_template_residual_text(
    mashup_name: str,
    mashup_desc: str,
    selected_api_names: list[str],
    selected_api_descs: list[str],
) -> str:
    # 模板 residual 文本并不直接写出目标 API，而是描述“还缺什么能力”。
    parts = [
        f"Mashup name: {mashup_name}",
        f"Mashup requirement: {mashup_desc}",
        f"Selected APIs: {', '.join(selected_api_names) if selected_api_names else 'None'}",
        "Selected API capabilities:",
    ]
    for name, desc in zip(selected_api_names, selected_api_descs):
        parts.append(f"- {name}: {desc}")
    parts.append("Remaining required capability:")
    return "\n".join(parts)


def encode_texts_with_cache(
    encoder: BaseTextEncoder,
    texts: list[str],
    cache_dir: str | Path,
    prefix: str,
) -> np.ndarray:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5()
    digest.update(f"{len(texts)}|{encoder.signature}|{prefix}".encode("utf-8"))
    for text in texts:
        digest.update(b"\n")
        digest.update(str(text).encode("utf-8"))
    key = digest.hexdigest()[:12]
    path = cache_dir / f"{prefix}_{encoder.signature}_{key}.npy"
    if path.exists():
        return np.load(path)
    # 首次不存在缓存时才真正编码。
    arr = encoder.encode(texts)
    np.save(path, arr)
    return arr


def build_complete_query_text(
    mashup_desc: str,
    template_residual_text: str,
    llm_residual_text: str | None = None,
) -> str:
    parts = [f"Structured residual requirement:\n{template_residual_text}"]
    if llm_residual_text and llm_residual_text.strip():
        parts.append(f"LLM-inferred missing capability:\n{llm_residual_text.strip()}")
    return "\n\n".join(parts)


def restrict_residual_texts_to_queries(
    queries: list[dict],
    residual_text_dict: dict[str, str],
) -> tuple[dict[str, str], dict[str, float]]:
    split_residual_text: dict[str, str] = {}
    for q in queries:
        key = q["query_key"]
        text = str(residual_text_dict.get(key, "")).strip()
        if text:
            split_residual_text[key] = text
    total = len(queries)
    covered = len(split_residual_text)
    return split_residual_text, {
        "covered_queries": covered,
        "total_queries": total,
        "coverage": float(covered / total) if total else 0.0,
    }


def build_query_text_embedding_map(
    queries: list[dict],
    bundle: dict,
    llm_texts: dict,
    encoder: BaseTextEncoder,
    cache_dir: str | Path,
) -> dict[str, np.ndarray]:
    mashup_df = bundle["mashup_df"]
    api_df = bundle["api_df"]
    residual_text_dict = llm_texts["residual_text"]

    texts = []
    keys = []
    for q in queries:
        key = q["query_key"]
        mid = q["mashup_id"]
        selected = q["selected_api_ids"]
        mashup_name = str(mashup_df.iloc[mid]["mashup_name"])
        mashup_desc = llm_texts["mashup_expanded"].get(mid, str(mashup_df.iloc[mid]["description"]))
        names = [str(api_df.iloc[a]["api_name"]) for a in selected]
        descs = [llm_texts["api_expanded"].get(a, str(api_df.iloc[a]["description"])) for a in selected]
        template_text = generate_template_residual_text(mashup_name, mashup_desc, names, descs)
        llm_residual_text = residual_text_dict.get(key, "")
        texts.append(
            build_complete_query_text(
                mashup_desc=mashup_desc,
                template_residual_text=template_text,
                llm_residual_text=llm_residual_text,
            )
        )
        keys.append(key)
    arr = encode_texts_with_cache(encoder, texts, cache_dir, "query_text")
    return {k: arr[i] for i, k in enumerate(keys)}


def build_residual_embedding_map(
    queries: list[dict],
    bundle: dict,
    llm_texts: dict,
    encoder: BaseTextEncoder,
    cache_dir: str | Path,
) -> dict[str, np.ndarray]:
    mashup_df = bundle["mashup_df"]
    api_df = bundle["api_df"]
    residual_text_dict = llm_texts["residual_text"]

    texts = []
    keys = []
    for q in queries:
        key = q["query_key"]
        if key in residual_text_dict:
            text = residual_text_dict[key]
        else:
            # 如果没有外部 residual 文本，就按模板把 mashup 需求和 selected API 能力拼起来。
            mid = q["mashup_id"]
            selected = q["selected_api_ids"]
            mashup_name = str(mashup_df.iloc[mid]["mashup_name"])
            mashup_desc = str(mashup_df.iloc[mid]["description"])
            names = [str(api_df.iloc[a]["api_name"]) for a in selected]
            descs = [str(api_df.iloc[a]["description"]) for a in selected]
            text = generate_template_residual_text(mashup_name, mashup_desc, names, descs)
        texts.append(text)
        keys.append(key)
    arr = encode_texts_with_cache(encoder, texts, cache_dir, "residual")
    return {k: arr[i] for i, k in enumerate(keys)}


def build_graphs_and_text_embeddings(
    encoder: BaseTextEncoder,
    bundle: dict,
    llm_texts: dict,
    cache_dir: str | Path,
    graph_mode: str = "threshold",
    mashup_topk: int = 20,
    api_topk: int = 20,
    mashup_threshold: float = 0.34,
    api_threshold: float = 0.34,
    add_self_loop: bool = True,
) -> dict:
    texts = build_entity_texts(bundle, llm_texts)
    mashup_emb = encode_texts_with_cache(encoder, texts["mashup_texts"], cache_dir, "mashup")
    api_emb = encode_texts_with_cache(encoder, texts["api_texts"], cache_dir, "api")
    if graph_mode == "topk":
        mashup_adj, mashup_full_sim = build_topk_graph(
            mashup_emb,
            topk=mashup_topk,
            threshold=mashup_threshold,
            add_self_loop=add_self_loop,
        )
        api_adj, api_full_sim = build_topk_graph(
            api_emb,
            topk=api_topk,
            threshold=api_threshold,
            add_self_loop=add_self_loop,
        )
    else:
        mashup_adj, mashup_full_sim = build_threshold_graph(
            mashup_emb,
            threshold=mashup_threshold,
            add_self_loop=add_self_loop,
        )
        api_adj, api_full_sim = build_threshold_graph(
            api_emb,
            threshold=api_threshold,
            add_self_loop=add_self_loop,
        )
    return {
        "mashup_text_emb": mashup_emb,
        "api_text_emb": api_emb,
        "mashup_adj": mashup_adj,
        "api_adj": api_adj,
        "mashup_full_sim": mashup_full_sim,
        "api_full_sim": api_full_sim,
    }
