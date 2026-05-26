from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data_utils import (
    build_graphs_and_text_embeddings,
    build_query_text_embedding_map,
    load_optional_llm_csvs,
    load_tssgcf_style_data,
    restrict_residual_texts_to_queries,
)
from src.dataset import PairwiseTrainDataset, collate_pairwise
from src.encoders import EncoderConfig, build_text_encoder
from src.graphs import mask_graph_nodes, sparse_to_torch, symmetric_normalize
from src.losses import TextualSimilarityLoss
from src.model import BranchConfig, DualBranchAPIRec
from src.sampling import (
    build_cold_api_eval_queries,
    build_eval_queries,
    build_holdout_eval_queries,
    build_negative_pool_map,
    build_train_samples,
)
from src.splits import cold_api_split, cold_mashup_split, standard_split_from_provided
from src.train_utils import EarlyStopping, evaluate_queries, save_json, train_one_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # 数据与设备配置。
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--non-deterministic", dest="deterministic", action="store_false")
    p.set_defaults(deterministic=True)
    p.add_argument("--encoder-backend", type=str, default="sbert", choices=["sbert", "llm_hf", "llm_api"])
    p.add_argument("--encoder-model", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--instruction-prefix", type=str, default="")
    p.add_argument("--api-base-url", type=str, default="https://api.openai.com/v1")
    p.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--mashup-expanded-csv", type=str, default="")
    p.add_argument("--api-expanded-csv", type=str, default="")
    p.add_argument("--residual-text-csv", type=str, default="data/llm/residual_text_deepseek.csv")
    p.add_argument("--eval-ks", type=str, default="2,4,5,6,8,10,20")
    p.add_argument("--selection-metric", type=str, default="NDCG@5")
    # 512
    p.add_argument("--max-length", type=int, default=512)
    # 32
    p.add_argument("--encoder-batch-size", type=int, default=64)

    # 数据划分协议。
    p.add_argument("--split-mode", type=str, default="standard", choices=["standard", "cold_mashup", "cold_api"])
    p.add_argument("--seed", type=int, default=42)
    # 0.2
    p.add_argument("--cold-api-ratio", type=float, default=0.2)

    # 优化器与模型超参。
    # 100
    p.add_argument("--epochs", type=int, default=500)
    # 256
    p.add_argument("--batch-size", type=int, default=128)
    # 0.0005
    p.add_argument("--lr", type=float, default=0.0004)
    p.add_argument("--patience", type=int, default=15)
    # 256
    p.add_argument("--emb-dim", type=int, default=256)
    # 2
    p.add_argument("--n-layers", type=int, default=2)
    # 0.1
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--lambda-exp", type=float, default=0.0)
    p.add_argument("--lambda-imp", type=float, default=0.0)
    # 0.25
    p.add_argument("--lambda-tss", type=float, default=0.25)
    p.add_argument("--lambda-l2", type=float, default=1e-5)

    # 文本相似图构建参数。
    p.add_argument("--mashup-topk", type=int, default=20)
    p.add_argument("--api-topk", type=int, default=20)
    p.add_argument("--mashup-threshold", type=float, default=0.34)
    p.add_argument("--api-threshold", type=float, default=0.34)
    p.add_argument("--graph-mode", type=str, default="threshold", choices=["threshold", "topk"])
    p.add_argument("--no-self-loop", action="store_true")

    # query / 样本构造参数。
    p.add_argument("--max-selected", type=int, default=3)
    p.add_argument("--max-subsets-per-positive", type=int, default=3)
    p.add_argument("--max-queries-per-mashup", type=int, default=6)

    # 训练开关与融合方式。
    p.add_argument("--negative-mode", type=str, default="strict_global", choices=["random", "strict_global", "hard"])
    p.add_argument("--w-explicit", type=int, default=1)
    p.add_argument("--w-implicit", type=int, default=1)
    p.add_argument("--w-residual", type=int, default=1)
    p.add_argument("--w-tss", type=int, default=1)
    p.add_argument("--fusion-mode", type=str, default="mlp", choices=["avg", "learnable", "mlp"])
    p.add_argument("--query-fusion-mode", type=str, default="fixed_avg", choices=["fixed_avg", "residual_gate"])
    p.add_argument("--api-fusion-mode", type=str, default="fixed_avg", choices=["fixed_avg", "dynamic_gate"])
    p.add_argument("--joint-loss-variant", type=str, default="full", choices=["full", "no_imp_aux", "total_only"])
    p.add_argument("--context-loss-type", type=str, default="bpr", choices=["bpr", "bce"])

    # staged training 参数。
    p.add_argument("--training-mode", type=str, default="joint", choices=["staged", "no_stage2", "joint"])
    p.add_argument("--save-dir", type=str, default="outputs/default_joint_run")
    p.add_argument("--stage1-epochs", type=int, default=20)
    p.add_argument("--stage2-epochs", type=int, default=20)
    p.add_argument("--stage3-epochs", type=int, default=40)
    p.add_argument("--stage3-lr", type=float, default=0.0010)
    return p.parse_args()


def parse_eval_ks(text: str) -> tuple[int, ...]:
    parts = [int(x.strip()) for x in (text or "").split(",") if x.strip()]
    if not parts:
        return (2, 4, 5, 6, 8, 10)
    return tuple(sorted(set(parts)))


def seed_everything(seed: int) -> None:
    # 统一固定 Python / NumPy / PyTorch 的随机种子。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    requested = (device_arg or "auto").lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_runtime(device: torch.device, deterministic: bool = True) -> None:
    if device.type != "cuda":
        return
    if getattr(torch.backends, "cudnn", None) is not None:
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True


def build_training_node_masks(
    bundle: dict,
    split,
    split_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    mashup_mask = np.ones(len(bundle["mashup_df"]), dtype=bool)
    api_mask = np.ones(len(bundle["api_df"]), dtype=bool)
    if split_mode == "standard":
        return mashup_mask, api_mask
    train_mashups = sorted(split.train_mashups)
    mashup_mask[:] = False
    mashup_mask[train_mashups] = True
    active_apis: set[int] = set()
    for mid in train_mashups:
        source_invocations = bundle["train_invocations"] if split_mode == "standard" else bundle["all_invocations"]
        active_apis.update(source_invocations.get(mid, []))
    api_mask[:] = False
    if active_apis:
        api_mask[sorted(active_apis)] = True
    return mashup_mask, api_mask


def _length_stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
    }


def build_protocol_report(
    args,
    split,
    train_samples: list[dict],
    val_queries: list[dict],
    test_queries: list[dict],
    query_feature_report: dict,
) -> dict:
    train_set = set(split.train_mashups)
    val_set = set(split.val_mashups)
    test_set = set(split.test_mashups)
    report = {
        "split_mode": args.split_mode,
        "standard_protocol": (
            {
                "name": "tssgcf_compatible_warm_start",
                "train_supervision": "train.txt on train_mashups only",
                "validation": "one observed train API held out on val_mashups for model selection only",
                "test": "train.txt as observed context, test.txt as held-out targets",
                "train_test_mashup_overlap": len(train_set & test_set),
                "val_test_mashup_overlap": len(val_set & test_set),
                "node_training_scope": "all standard mashups/apis",
            }
            if args.split_mode == "standard"
            else None
        ),
        "train_mashup_count": len(split.train_mashups),
        "val_mashup_count": len(split.val_mashups),
        "test_mashup_count": len(split.test_mashups),
        "cold_api_count": len(split.cold_apis or []),
        "train_selected_stats": _length_stats([len(s["selected_api_ids"]) for s in train_samples]),
        "val_selected_stats": _length_stats([len(q["selected_api_ids"]) for q in val_queries]),
        "test_selected_stats": _length_stats([len(q["selected_api_ids"]) for q in test_queries]),
        "val_target_stats": _length_stats([len(q["target_api_ids"]) for q in val_queries]),
        "test_target_stats": _length_stats([len(q["target_api_ids"]) for q in test_queries]),
        "graph_config": {
            "graph_mode": args.graph_mode,
            "mashup_topk": args.mashup_topk,
            "api_topk": args.api_topk,
            "mashup_threshold": args.mashup_threshold,
            "api_threshold": args.api_threshold,
            "self_loop": not args.no_self_loop,
        },
        "query_feature_report": query_feature_report,
    }
    return report


def validate_graph_bundle(graph_bundle: dict) -> None:
    for name in ("mashup_text_emb", "api_text_emb", "mashup_full_sim", "api_full_sim"):
        arr = np.asarray(graph_bundle[name])
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN/Inf.")
    for name in ("mashup_adj", "api_adj"):
        mat = graph_bundle[name].tocoo()
        if mat.shape[0] != mat.shape[1]:
            raise ValueError(f"{name} is not square: {mat.shape}")
        if not np.isfinite(mat.data).all():
            raise ValueError(f"{name} contains NaN/Inf edge weights.")
        asym = (mat - mat.T).tocoo()
        if asym.nnz > 0 and not np.allclose(asym.data, 0.0, atol=1e-6):
            raise ValueError(f"{name} is not symmetric after construction.")
        diag = mat.diagonal()
        if diag.size and np.any(diag <= 0):
            raise ValueError(f"{name} has non-positive diagonal entries.")


def validate_protocol(
    args,
    bundle: dict,
    split,
    train_samples: list[dict],
    val_queries: list[dict],
    test_queries: list[dict],
    negative_pool_map: dict[int, list[int]],
    cold_apis: set[int],
) -> None:
    train_set = set(split.train_mashups)
    val_set = set(split.val_mashups)
    test_set = set(split.test_mashups)
    if args.split_mode == "standard":
        if train_set & val_set:
            raise ValueError("Standard split should keep validation mashups out of the training mashup set.")
    elif train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("Split mashup sets overlap.")

    for i, sample in enumerate(train_samples):
        selected = set(sample["selected_api_ids"])
        all_apis = set(sample["all_api_ids"])
        pos = sample["positive_api_id"]
        if not selected:
            raise ValueError(f"Train sample {i} has empty selected_api_ids.")
        if pos in selected:
            raise ValueError(f"Train sample {i} includes its positive API in selected_api_ids.")
        if pos not in all_apis:
            raise ValueError(f"Train sample {i} positive API is missing from all_api_ids.")
        pool = negative_pool_map.get(i, [])
        if not pool:
            raise ValueError(f"Train sample {i} has an empty negative pool.")
        if any(neg in all_apis for neg in pool):
            raise ValueError(f"Train sample {i} negative pool contains observed positives.")
        if args.split_mode == "cold_api":
            if pos in cold_apis or selected & cold_apis:
                raise ValueError(f"Train sample {i} leaks cold APIs into training.")
            if any(neg in cold_apis for neg in pool):
                raise ValueError(f"Train sample {i} negative pool contains cold APIs.")

    for name, queries in (("val", val_queries), ("test", test_queries)):
        for i, query in enumerate(queries):
            selected = set(query["selected_api_ids"])
            targets = set(query["target_api_ids"])
            if not selected:
                raise ValueError(f"{name} query {i} has empty selected_api_ids.")
            if not targets:
                raise ValueError(f"{name} query {i} has empty target_api_ids.")
            if selected & targets:
                raise ValueError(f"{name} query {i} has overlapping selected and target APIs.")
            if args.split_mode == "standard":
                observed = set(bundle["train_invocations"].get(query["mashup_id"], []))
                if selected != observed and name == "test":
                    raise ValueError(f"{name} query {i} does not match standard hold-out observed context.")
            elif args.split_mode == "cold_api":
                if selected & cold_apis:
                    raise ValueError(f"{name} query {i} leaks cold APIs into selected context.")
                if not targets.issubset(cold_apis):
                    raise ValueError(f"{name} query {i} contains non-cold targets in cold_api mode.")


def prepare_common(args):
    # 这一步把训练和评估所需的公共对象全部准备好。
    seed_everything(args.seed)
    device = resolve_device(args.device)
    configure_runtime(device, deterministic=args.deterministic)
    print(f"Using device: {device}")
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_tssgcf_style_data(data_dir)
    llm_texts = load_optional_llm_csvs(
        data_dir,
        bundle["mid2idx"],
        bundle["aid2idx"],
        mashup_expanded_csv=args.mashup_expanded_csv or None,
        api_expanded_csv=args.api_expanded_csv or None,
        residual_text_csv=args.residual_text_csv or None,
    )

    if args.split_mode == "standard":
        split = standard_split_from_provided(
            bundle["train_invocations"],
            bundle["test_invocations"],
            val_ratio=0.125,
            seed=args.seed,
            exclude_val_from_test=False,
        )
    elif args.split_mode == "cold_mashup":
        split = cold_mashup_split(bundle["all_invocations"], seed=args.seed)
    else:
        split = cold_api_split(bundle["all_invocations"], bundle["api_freq"], cold_api_ratio=args.cold_api_ratio, seed=args.seed)

    encoder_cfg = EncoderConfig(
        backend=args.encoder_backend,
        model_name=args.encoder_model,
        max_length=args.max_length,
        batch_size=args.encoder_batch_size,
        instruction_prefix=args.instruction_prefix,
        api_base_url=args.api_base_url,
        api_key_env=args.api_key_env,
        trust_remote_code=args.trust_remote_code,
        normalize=True,
    )
    encoder = build_text_encoder(encoder_cfg, device=device)

    # 构图与节点文本向量编码。
    graph_bundle = build_graphs_and_text_embeddings(
        encoder=encoder,
        bundle=bundle,
        llm_texts=llm_texts,
        cache_dir=data_dir / "cache",
        graph_mode=args.graph_mode,
        mashup_topk=args.mashup_topk,
        api_topk=args.api_topk,
        mashup_threshold=args.mashup_threshold,
        api_threshold=args.api_threshold,
        add_self_loop=not args.no_self_loop,
    )
    mashup_train_mask_np, api_train_mask_np = build_training_node_masks(
        bundle,
        split,
        args.split_mode,
    )
    if args.split_mode != "standard":
        graph_bundle["mashup_adj"] = mask_graph_nodes(
            graph_bundle["mashup_adj"],
            np.where(mashup_train_mask_np)[0],
            keep_self_loop=True,
        )
        graph_bundle["api_adj"] = mask_graph_nodes(
            graph_bundle["api_adj"],
            np.where(api_train_mask_np)[0],
            keep_self_loop=True,
        )
    graph_bundle["mashup_adj"] = symmetric_normalize(graph_bundle["mashup_adj"])
    graph_bundle["api_adj"] = symmetric_normalize(graph_bundle["api_adj"])
    validate_graph_bundle(graph_bundle)

    cold_apis = set(split.cold_apis or [])
    # 训练集是 pairwise 三元组，验证/测试集是 query -> target 集合。
    train_samples = build_train_samples(
        invocations=bundle["all_invocations"] if args.split_mode != "standard" else bundle["train_invocations"],
        mashup_ids=split.train_mashups,
        max_selected=args.max_selected,
        max_subsets_per_positive=args.max_subsets_per_positive,
        cold_apis=cold_apis if args.split_mode == "cold_api" else None,
    )
    if args.split_mode == "standard":
        # Standard protocol: evaluate with observed train APIs as context and
        # held-out APIs as targets. Do not enumerate subsets from train∪test.
        val_rng = np.random.default_rng(args.seed)
        val_train_inv: dict[int, list[int]] = {}
        val_target_inv: dict[int, list[int]] = {}
        for mid in split.val_mashups:
            apis = sorted(set(bundle["train_invocations"].get(mid, [])))
            if len(apis) < 2:
                continue
            target = int(val_rng.choice(np.asarray(apis, dtype=np.int64)))
            val_train_inv[mid] = [a for a in apis if a != target]
            val_target_inv[mid] = [target]
        val_queries = build_holdout_eval_queries(
            observed_invocations=val_train_inv,
            target_invocations=val_target_inv,
            mashup_ids=split.val_mashups,
        )
        test_queries = build_holdout_eval_queries(
            observed_invocations=bundle["train_invocations"],
            target_invocations=bundle["test_invocations"],
            mashup_ids=split.test_mashups,
        )
    else:
        if args.split_mode == "cold_api":
            val_queries = build_cold_api_eval_queries(
                bundle["all_invocations"],
                split.val_mashups,
                cold_apis=cold_apis,
                max_selected=args.max_selected,
                max_queries_per_mashup=args.max_queries_per_mashup,
            )
            test_queries = build_cold_api_eval_queries(
                bundle["all_invocations"],
                split.test_mashups,
                cold_apis=cold_apis,
                max_selected=args.max_selected,
                max_queries_per_mashup=args.max_queries_per_mashup,
            )
        else:
            val_queries = build_eval_queries(
                bundle["all_invocations"],
                split.val_mashups,
                max_selected=args.max_selected,
                max_queries_per_mashup=args.max_queries_per_mashup,
            )
            test_queries = build_eval_queries(
                bundle["all_invocations"],
                split.test_mashups,
                max_selected=args.max_selected,
                max_queries_per_mashup=args.max_queries_per_mashup,
            )

    all_queries = train_samples + val_queries + test_queries
    split_residual_text, total_query_coverage = restrict_residual_texts_to_queries(
        all_queries,
        llm_texts.get("residual_text", {}),
    )
    _, val_query_coverage = restrict_residual_texts_to_queries(
        val_queries,
        llm_texts.get("residual_text", {}),
    )
    _, test_query_coverage = restrict_residual_texts_to_queries(
        test_queries,
        llm_texts.get("residual_text", {}),
    )
    _, train_query_coverage = restrict_residual_texts_to_queries(
        train_samples,
        llm_texts.get("residual_text", {}),
    )
    split_llm_texts = dict(llm_texts)
    split_llm_texts["residual_text"] = split_residual_text
    query_text_map = build_query_text_embedding_map(
        all_queries,
        bundle=bundle,
        llm_texts=split_llm_texts,
        encoder=encoder,
        cache_dir=data_dir / "cache",
    )

    negative_pool_map = build_negative_pool_map(
        train_samples=train_samples,
        train_invocations=bundle["all_invocations"],
        all_api_ids=[
            a for a in range(len(bundle["api_df"]))
            if not (args.split_mode == "cold_api" and a in cold_apis)
        ],
        mode=args.negative_mode,
        api_sim_matrix=graph_bundle["api_full_sim"],
    )
    validate_protocol(
        args=args,
        bundle=bundle,
        split=split,
        train_samples=train_samples,
        val_queries=val_queries,
        test_queries=test_queries,
        negative_pool_map=negative_pool_map,
        cold_apis=cold_apis,
    )
    protocol_report = build_protocol_report(
        args=args,
        split=split,
        train_samples=train_samples,
        val_queries=val_queries,
        test_queries=test_queries,
        query_feature_report={
            "residual_text_scope": "current_split_queries_only",
            "train_queries": train_query_coverage,
            "val_queries": val_query_coverage,
            "test_queries": test_query_coverage,
            "all_queries": total_query_coverage,
            "external_residual_query_count": len(split_residual_text),
        },
    )
    print("Protocol report:", json.dumps(protocol_report, ensure_ascii=False))
    dataset = PairwiseTrainDataset(
        samples=train_samples,
        negative_pool_map=negative_pool_map,
        query_text_emb_map=query_text_map,
        num_apis=len(bundle["api_df"]),
        seed=args.seed,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_pairwise,
        pin_memory=device.type == "cuda",
    )

    mashup_adj = sparse_to_torch(graph_bundle["mashup_adj"], device=device)
    api_adj = sparse_to_torch(graph_bundle["api_adj"], device=device)
    # 文本相似矩阵作为 TSS 正则的静态参照。
    mashup_full_sim = torch.tensor(graph_bundle["mashup_full_sim"], dtype=torch.float32, device=device)
    api_full_sim = torch.tensor(graph_bundle["api_full_sim"], dtype=torch.float32, device=device)
    mashup_text_emb = torch.tensor(graph_bundle["mashup_text_emb"], dtype=torch.float32, device=device)
    api_text_emb = torch.tensor(graph_bundle["api_text_emb"], dtype=torch.float32, device=device)
    input_text_dim = int(graph_bundle["api_text_emb"].shape[1])

    branch_cfg = BranchConfig(
        use_explicit=bool(args.w_explicit),
        use_implicit=bool(args.w_implicit),
        use_residual=bool(args.w_residual),
        use_tss=bool(args.w_tss),
        fusion_mode=args.fusion_mode,
        query_fusion_mode=args.query_fusion_mode,
        api_fusion_mode=args.api_fusion_mode,
    )
    model = DualBranchAPIRec(
        num_mashups=len(bundle["mashup_df"]),
        num_apis=len(bundle["api_df"]),
        input_text_dim=input_text_dim,
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        branch_config=branch_cfg,
    ).to(device)
    model.set_train_masks(
        mashup_mask=torch.tensor(mashup_train_mask_np, dtype=torch.bool, device=device),
        api_mask=torch.tensor(api_train_mask_np, dtype=torch.bool, device=device),
    )

    tss_loss_fn = TextualSimilarityLoss(
        mashup_full_sim=mashup_full_sim,
        api_full_sim=api_full_sim,
        alpha_m=args.mashup_threshold,
        alpha_a=args.api_threshold,
        beta=1.0,
        weight=args.lambda_tss,
        mashup_mask=torch.tensor(mashup_train_mask_np, dtype=torch.bool, device=device),
        api_mask=torch.tensor(api_train_mask_np, dtype=torch.bool, device=device),
    )

    meta = {
        # 保存一次运行的关键元信息，便于后续复现实验。
        "args": vars(args),
        "split": {
            "train_mashups": split.train_mashups,
            "val_mashups": split.val_mashups,
            "test_mashups": split.test_mashups,
            "cold_apis": split.cold_apis,
        },
        "encoder_signature": encoder.signature,
        "device": str(device),
        "num_train_samples": len(train_samples),
        "num_val_queries": len(val_queries),
        "num_test_queries": len(test_queries),
        "protocol_report": protocol_report,
    }
    save_json(meta, save_dir / "run_meta.json")

    return {
        "device": device,
        "bundle": bundle,
        "split": split,
        "encoder": encoder,
        "graph_bundle": graph_bundle,
        "train_loader": train_loader,
        "val_queries": val_queries,
        "test_queries": test_queries,
        "query_text_map": query_text_map,
        "mashup_adj": mashup_adj,
        "api_adj": api_adj,
        "mashup_text_emb": mashup_text_emb,
        "api_text_emb": api_text_emb,
        "tss_loss_fn": tss_loss_fn,
        "model": model,
        "save_dir": save_dir,
        "cold_apis": cold_apis,
    }


def run_joint_training(args):
    # joint 模式一步到位训练全部模块。
    ctx = prepare_common(args)
    model = ctx["model"]
    device = ctx["device"]
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=max(2, args.patience // 3))
    stopper = EarlyStopping(patience=args.patience, mode="max")
    best_metric = -1.0
    best_epoch = 0
    best_val_metrics = {}
    best_path = ctx["save_dir"] / "best_model.pt"
    eval_ks = parse_eval_ks(args.eval_ks)

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model, ctx["train_loader"], optimizer, device,
            ctx["mashup_adj"], ctx["api_adj"], ctx["mashup_text_emb"], ctx["api_text_emb"], ctx["tss_loss_fn"],
            lambda_exp=args.lambda_exp, lambda_imp=args.lambda_imp, lambda_l2=args.lambda_l2,
            train_mode="joint", joint_loss_variant=args.joint_loss_variant, context_loss_type=args.context_loss_type,
        )
        val_metrics = evaluate_queries(
            model, ctx["val_queries"], ctx["query_text_map"], device,
            ctx["mashup_adj"], ctx["api_adj"], ctx["mashup_text_emb"], ctx["api_text_emb"],
            num_apis=len(ctx["bundle"]["api_df"]),
            ks=eval_ks,
            only_targets_from=ctx["cold_apis"] if args.split_mode == "cold_api" else None,
        )
        main_metric = val_metrics.get(args.selection_metric, 0.0)
        scheduler.step(main_metric)
        print(f"[Epoch {epoch}] train={train_stats} val={val_metrics}")
        if main_metric > best_metric:
            best_metric = main_metric
            best_epoch = epoch
            best_val_metrics = copy.deepcopy(val_metrics)
            torch.save(model.state_dict(), best_path)
        if stopper.step(main_metric):
            print("Early stopping triggered.")
            break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate_queries(
        model, ctx["test_queries"], ctx["query_text_map"], device,
        ctx["mashup_adj"], ctx["api_adj"], ctx["mashup_text_emb"], ctx["api_text_emb"],
        num_apis=len(ctx["bundle"]["api_df"]),
        ks=eval_ks,
        only_targets_from=ctx["cold_apis"] if args.split_mode == "cold_api" else None,
    )
    save_json(test_metrics, ctx["save_dir"] / "test_metrics.json")
    summary = {
        "training_mode": "joint",
        "selection_metric": args.selection_metric,
        "best_val_metric": best_metric,
        "best_val_recall10": best_val_metrics.get("Recall@10", 0.0),
        "best_val_metrics": best_val_metrics,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
    }
    save_json(summary, ctx["save_dir"] / "summary.json")
    print("Test metrics:", test_metrics)


def freeze_module(module, requires_grad: bool) -> None:
    # staged training 用这个工具切换模块是否参与梯度更新。
    for p in module.parameters():
        p.requires_grad = requires_grad


def configure_stage(model, stage: str, args) -> str:
    # stage1: structure-only; stage2: context/residual; stage3: joint
    if stage == "stage1":
        freeze_module(model.explicit_model.mashup_id_emb, True)
        freeze_module(model.explicit_model.api_id_emb, True)
        freeze_module(model.explicit_model.text_proj, True)
        freeze_module(model.explicit_model.query_text_proj, False)
        freeze_module(model.implicit_model, False)
        freeze_module(model.fusion_mlp, False)
        model.branch_logits.requires_grad = False
        # Stage1 only trains the TSSGCF explicit backbone.
        model.branch_config.use_implicit = False
        model.branch_config.use_residual = False
        return "structure"
    if stage == "stage2":
        freeze_module(model.explicit_model.mashup_id_emb, False)
        freeze_module(model.explicit_model.api_id_emb, False)
        freeze_module(model.explicit_model.text_proj, False)
        freeze_module(model.explicit_model.query_text_proj, True)
        freeze_module(model.implicit_model, True)
        freeze_module(model.fusion_mlp, False)
        model.branch_logits.requires_grad = False
        model.branch_config.use_implicit = True
        model.branch_config.use_residual = True
        return "context"
    # stage3
    # 最后联合微调所有参数。
    for p in model.parameters():
        p.requires_grad = True
    model.branch_config.use_implicit = bool(args.w_implicit)
    model.branch_config.use_residual = bool(args.w_residual)
    return "joint"


def run_staged_training(args):
    # 默认入口走 staged training：先结构、再上下文、最后联合。
    ctx = prepare_common(args)
    model = ctx["model"]
    device = ctx["device"]
    save_dir = ctx["save_dir"]
    best_metric = -1.0
    best_epoch = 0
    best_stage = ""
    best_val_metrics = {}
    best_path = save_dir / "best_model.pt"
    eval_ks = parse_eval_ks(args.eval_ks)

    if args.training_mode == "no_stage2":
        plan = [
            ("stage1", args.stage1_epochs, args.lr),
            ("stage3", args.stage3_epochs, args.stage3_lr),
        ]
    else:
        plan = [
            ("stage1", args.stage1_epochs, args.lr),
            ("stage2", args.stage2_epochs, args.lr),
            ("stage3", args.stage3_epochs, args.stage3_lr),
        ]
    epoch_counter = 0
    for stage_name, epochs, lr in plan:
        mode = configure_stage(model, stage_name, args)
        # 每个 stage 只优化当前 requires_grad=True 的参数。
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = Adam(params, lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=max(2, args.patience // 4))
        stopper = EarlyStopping(patience=max(3, args.patience // 2), mode="max")
        print(f"=== {stage_name} / mode={mode} / epochs={epochs} / lr={lr} ===")
        for _ in range(epochs):
            epoch_counter += 1
            train_stats = train_one_epoch(
                model, ctx["train_loader"], optimizer, device,
                ctx["mashup_adj"], ctx["api_adj"], ctx["mashup_text_emb"], ctx["api_text_emb"], ctx["tss_loss_fn"],
                lambda_exp=args.lambda_exp, lambda_imp=args.lambda_imp, lambda_l2=args.lambda_l2,
                train_mode=mode, joint_loss_variant=args.joint_loss_variant, context_loss_type=args.context_loss_type,
            )
            val_metrics = evaluate_queries(
                model, ctx["val_queries"], ctx["query_text_map"], device,
                ctx["mashup_adj"], ctx["api_adj"], ctx["mashup_text_emb"], ctx["api_text_emb"],
                num_apis=len(ctx["bundle"]["api_df"]),
                ks=eval_ks,
                only_targets_from=ctx["cold_apis"] if args.split_mode == "cold_api" else None,
            )
            main_metric = val_metrics.get(args.selection_metric, 0.0)
            scheduler.step(main_metric)
            print(f"[Epoch {epoch_counter}] stage={stage_name} train={train_stats} val={val_metrics}")
            if main_metric > best_metric:
                best_metric = main_metric
                best_epoch = epoch_counter
                best_stage = stage_name
                best_val_metrics = copy.deepcopy(val_metrics)
                torch.save(model.state_dict(), best_path)
            if stopper.step(main_metric):
                print(f"Early stop in {stage_name}.")
                break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate_queries(
        model, ctx["test_queries"], ctx["query_text_map"], device,
        ctx["mashup_adj"], ctx["api_adj"], ctx["mashup_text_emb"], ctx["api_text_emb"],
        num_apis=len(ctx["bundle"]["api_df"]),
        ks=eval_ks,
        only_targets_from=ctx["cold_apis"] if args.split_mode == "cold_api" else None,
    )
    save_json(test_metrics, save_dir / "test_metrics.json")
    summary = {
        "training_mode": args.training_mode,
        "selection_metric": args.selection_metric,
        "best_val_metric": best_metric,
        "best_val_recall10": best_val_metrics.get("Recall@10", 0.0),
        "best_val_metrics": best_val_metrics,
        "best_epoch": best_epoch,
        "best_stage": best_stage,
        "test_metrics": test_metrics,
    }
    save_json(summary, save_dir / "summary.json")
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    args = parse_args()
    if args.training_mode == "joint":
        run_joint_training(args)
    else:
        run_staged_training(args)
