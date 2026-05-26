from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import bpr_loss, pointwise_bce_loss
from .metrics import average_metric_dict, evaluate_ranked_list


@dataclass
class EarlyStopping:
    patience: int = 10
    mode: str = "max"

    def __post_init__(self) -> None:
        self.best = None
        self.bad_count = 0

    def step(self, value: float) -> bool:
        improved = False
        if self.best is None:
            improved = True
        elif self.mode == "max" and value > self.best:
            improved = True
        elif self.mode == "min" and value < self.best:
            improved = True
        if improved:
            self.best = value
            self.bad_count = 0
            return False
        self.bad_count += 1
        return self.bad_count >= self.patience


def move_batch(batch: dict[str, torch.Tensor], device: str | torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def train_one_epoch(
    model,
    train_loader: DataLoader,
    optimizer,
    device,
    mashup_adj,
    api_adj,
    mashup_text_emb,
    api_text_emb,
    tss_loss_fn,
    lambda_exp: float = 0.5,
    lambda_imp: float = 0.5,
    lambda_l2: float = 1e-5,
    train_mode: str = "joint",
    joint_loss_variant: str = "full",
    context_loss_type: str = "bpr",
) -> dict[str, float]:
    model.train()
    running = {"loss": 0.0, "rank": 0.0, "exp": 0.0, "imp": 0.0, "tss": 0.0}
    steps = 0
    for batch in tqdm(train_loader, desc="Train", leave=False):
        batch = move_batch(batch, device)
        optimizer.zero_grad()

        out_pos = model.score_batch(
            batch["mashup_id"],
            batch["selected_api_ids"],
            batch["selected_mask"],
            batch["positive_api_id"],
            batch["query_text_emb"],
            mashup_adj,
            api_adj,
            mashup_text_emb,
            api_text_emb,
        )
        out_neg = model.score_batch(
            batch["mashup_id"],
            batch["selected_api_ids"],
            batch["selected_mask"],
            batch["negative_api_id"],
            batch["query_text_emb"],
            mashup_adj,
            api_adj,
            mashup_text_emb,
            api_text_emb,
        )

        if train_mode == "structure":
            pos_total = out_pos["explicit"]
            neg_total = out_neg["explicit"]
            loss_rank = bpr_loss(pos_total, neg_total)
            loss_exp = out_pos["total"].new_tensor(0.0)
            loss_imp = out_pos["total"].new_tensor(0.0)
        elif train_mode == "context":
            if (
                model.branch_config.use_implicit
                and model.branch_config.use_explicit
                and model.branch_config.use_residual
            ):
                # Keep the context-stage objective while allowing gradients to
                # reach the residual-aware explicit query path.
                pos_total = out_pos["total"]
                neg_total = out_neg["total"]
            else:
                pos_total = out_pos["implicit"] if model.branch_config.use_implicit else out_pos["explicit"]
                neg_total = out_neg["implicit"] if model.branch_config.use_implicit else out_neg["explicit"]
            if context_loss_type == "bce":
                loss_rank = pointwise_bce_loss(pos_total, neg_total)
            else:
                loss_rank = bpr_loss(pos_total, neg_total)
            loss_exp = out_pos["total"].new_tensor(0.0)
            loss_imp = out_pos["total"].new_tensor(0.0)
        else:
            loss_rank = bpr_loss(out_pos["total"], out_neg["total"])
            if joint_loss_variant == "total_only":
                loss_exp = out_pos["total"].new_tensor(0.0)
                loss_imp = out_pos["total"].new_tensor(0.0)
            elif joint_loss_variant == "no_imp_aux":
                loss_exp = (
                    bpr_loss(out_pos["explicit"], out_neg["explicit"])
                    if model.branch_config.use_explicit
                    else out_pos["total"].new_tensor(0.0)
                )
                loss_imp = out_pos["total"].new_tensor(0.0)
            else:
                loss_exp = (
                    bpr_loss(out_pos["explicit"], out_neg["explicit"])
                    if model.branch_config.use_explicit
                    else out_pos["total"].new_tensor(0.0)
                )
                loss_imp = (
                    bpr_loss(out_pos["implicit"], out_neg["implicit"])
                    if model.branch_config.use_implicit
                    else out_pos["total"].new_tensor(0.0)
                )

        loss_tss = (
            tss_loss_fn(out_pos["final_mashup"], out_pos["final_api"])
            if model.branch_config.use_tss
            else out_pos["total"].new_tensor(0.0)
        )
        l2 = sum(p.pow(2).sum() for p in model.parameters()) * lambda_l2
        loss = loss_rank + lambda_exp * loss_exp + lambda_imp * loss_imp + loss_tss + l2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running["loss"] += float(loss.item())
        running["rank"] += float(loss_rank.item())
        running["exp"] += float(loss_exp.item())
        running["imp"] += float(loss_imp.item())
        running["tss"] += float(loss_tss.item())
        steps += 1
    return {k: v / max(1, steps) for k, v in running.items()}


@torch.no_grad()
def evaluate_queries(
    model,
    queries: list[dict],
    query_text_emb_map: dict[str, np.ndarray],
    device,
    mashup_adj,
    api_adj,
    mashup_text_emb,
    api_text_emb,
    num_apis: int,
    ks: Sequence[int] = (2, 4, 6, 8, 10),
    only_targets_from: set[int] | None = None,
) -> dict[str, float]:
    model.eval()
    metrics = []
    g_mashup, g_api, final_mashup, final_api = model.encode_nodes(mashup_adj, api_adj, mashup_text_emb, api_text_emb)
    for q in tqdm(queries, desc="Eval", leave=False):
        selected = q["selected_api_ids"]
        selected_tensor = torch.tensor(selected, dtype=torch.long, device=device).unsqueeze(0)
        mashup_id = torch.tensor([q["mashup_id"]], dtype=torch.long, device=device)
        query_text_emb = torch.tensor(query_text_emb_map[q["query_key"]], dtype=torch.float32, device=device).unsqueeze(0)

        explicit_m = model.build_explicit_query_rep(
            final_mashup=final_mashup,
            mashup_idx=mashup_id,
            query_text_emb=query_text_emb,
            use_query_text=model.branch_config.use_residual,
        )
        explicit_scores = model.explicit_model.score(explicit_m.expand(num_apis, -1), final_api)

        if model.branch_config.use_implicit:
            selected_expand = final_api[selected_tensor].expand(num_apis, -1, -1)
            selected_mask = torch.ones((num_apis, len(selected)), dtype=torch.float32, device=device)
            context = model.attentive_context(selected_expand, final_api, selected_mask)
            implicit_scores = model.implicit_model.score(context, final_api)
        else:
            implicit_scores = torch.zeros_like(explicit_scores)

        if model.branch_config.use_explicit and model.branch_config.use_implicit:
            total_scores = model.fuse_scores(explicit_scores, implicit_scores)
        elif model.branch_config.use_explicit:
            total_scores = explicit_scores
        else:
            total_scores = implicit_scores

        total_scores = total_scores.detach().cpu().numpy()
        total_scores[selected] = -1e9
        if only_targets_from is not None:
            positives = set([a for a in q["target_api_ids"] if a in only_targets_from])
            if not positives:
                continue
        else:
            positives = set(q["target_api_ids"])
        ranked = list(np.argsort(-total_scores))
        metrics.append(evaluate_ranked_list(ranked, positives, ks))
    return average_metric_dict(metrics)


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
