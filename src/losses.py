from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12).mean()


def pointwise_bce_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)
    pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels)
    neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels)
    return 0.5 * (pos_loss + neg_loss)


class TextualSimilarityLoss(nn.Module):
    def __init__(
        self,
        mashup_full_sim: torch.Tensor,
        api_full_sim: torch.Tensor,
        alpha_m: float = 0.35,
        alpha_a: float = 0.35,
        beta: float = 1.0,
        weight: float = 0.25,
        mashup_mask: torch.Tensor | None = None,
        api_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("mashup_full_sim", mashup_full_sim)
        self.register_buffer("api_full_sim", api_full_sim)
        self.register_buffer(
            "mashup_mask",
            mashup_mask.bool() if mashup_mask is not None else torch.ones(mashup_full_sim.size(0), dtype=torch.bool),
        )
        self.register_buffer(
            "api_mask",
            api_mask.bool() if api_mask is not None else torch.ones(api_full_sim.size(0), dtype=torch.bool),
        )
        self.alpha_m = alpha_m
        self.alpha_a = alpha_a
        self.beta = beta
        self.weight = weight

    def _side_loss(
        self,
        reps: torch.Tensor,
        full_sim: torch.Tensor,
        alpha: float,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        active = torch.where(mask)[0]
        if active.numel() <= 1:
            return reps.new_tensor(0.0)
        reps = reps.index_select(0, active)
        full_sim = full_sim.index_select(0, active).index_select(1, active)
        reps = F.normalize(reps, p=2, dim=-1)
        sim = reps @ reps.T
        penalty = F.relu(alpha - full_sim)
        reg = penalty.pow(self.beta) * F.relu(sim).pow(self.beta + 1.0)
        reg = reg - torch.diag(torch.diag(reg))
        denom = max(1, reg.numel() - reg.size(0))
        return reg.sum() / denom

    def forward(self, mashup_reps: torch.Tensor, api_reps: torch.Tensor) -> torch.Tensor:
        if self.weight <= 0:
            return mashup_reps.new_tensor(0.0)
        lm = self._side_loss(mashup_reps, self.mashup_full_sim, self.alpha_m, self.mashup_mask)
        la = self._side_loss(api_reps, self.api_full_sim, self.alpha_a, self.api_mask)
        return self.weight * (lm + la)
