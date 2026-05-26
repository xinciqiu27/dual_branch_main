from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCNEncoder(nn.Module):
    def __init__(self, n_layers: int = 2) -> None:
        super().__init__()
        self.n_layers = n_layers

    def forward(self, adj: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # 保存每一层传播结果，最后按 LightGCN 的常见做法做层平均。
        layers = [emb]
        h = emb
        for _ in range(self.n_layers):
            h = torch.sparse.mm(adj, h)
            layers.append(h)
        out = torch.stack(layers, dim=0).mean(dim=0)
        return F.normalize(out, p=2, dim=-1)


class PairwiseMLPScorer(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = hidden_dim or emb_dim
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        # left/right 一般分别是 query 表征和 candidate API 表征。
        return self.net(torch.cat([left, right], dim=-1)).squeeze(-1)


class FusionMLP(nn.Module):
    def __init__(self, hidden_dim: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, explicit_score: torch.Tensor, implicit_score: torch.Tensor) -> torch.Tensor:
        # 融合层只接两个标量分支分数，而不是更复杂的向量交互。
        scores = torch.stack([explicit_score, implicit_score], dim=-1)
        return self.net(scores).squeeze(-1)


class TextProjectionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        # 两层 MLP 做文本投影。
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScalarGateMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


@dataclass
class BranchConfig:
    use_explicit: bool = True
    use_implicit: bool = True
    use_residual: bool = True
    use_tss: bool = True
    fusion_mode: str = "mlp"  # mlp | avg | learnable
    query_fusion_mode: str = "fixed_avg"  # fixed_avg | residual_gate
    api_fusion_mode: str = "fixed_avg"  # fixed_avg | dynamic_gate


class ExplicitModel(nn.Module):
    def __init__(
        self,
        num_mashups: int,
        num_apis: int,
        input_text_dim: int,
        emb_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        score_mode: str = "dot",
    ) -> None:
        super().__init__()
        self.score_mode = score_mode
        self.mashup_id_emb = nn.Embedding(num_mashups, emb_dim)
        self.api_id_emb = nn.Embedding(num_apis, emb_dim)
        self.mashup_encoder = LightGCNEncoder(n_layers=n_layers)
        self.api_encoder = LightGCNEncoder(n_layers=n_layers)
        self.text_proj = TextProjectionMLP(input_text_dim, emb_dim, emb_dim)
        self.query_text_proj = TextProjectionMLP(input_text_dim, emb_dim, emb_dim)
        self.api_fusion_gate = ScalarGateMLP(emb_dim * 2, hidden_dim=emb_dim, dropout=dropout)
        self.query_residual_gate = ScalarGateMLP(emb_dim * 2, hidden_dim=emb_dim, dropout=dropout)
        self.explicit_scorer = PairwiseMLPScorer(emb_dim, dropout=dropout) if score_mode == "mlp" else None
        self.register_buffer("mashup_train_mask", torch.ones(num_mashups, dtype=torch.bool), persistent=False)
        self.register_buffer("api_train_mask", torch.ones(num_apis, dtype=torch.bool), persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.mashup_id_emb.weight)
        nn.init.xavier_uniform_(self.api_id_emb.weight)
        for module in self.text_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.query_text_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for gate in (self.api_fusion_gate, self.query_residual_gate):
            for module in gate.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def set_train_masks(
        self,
        mashup_mask: torch.Tensor | None = None,
        api_mask: torch.Tensor | None = None,
    ) -> None:
        if mashup_mask is not None:
            self.mashup_train_mask = mashup_mask.bool().to(self.mashup_train_mask.device)
        if api_mask is not None:
            self.api_train_mask = api_mask.bool().to(self.api_train_mask.device)

    def encode_nodes(
        self,
        mashup_adj: torch.Tensor,
        api_adj: torch.Tensor,
        mashup_text_emb: torch.Tensor,
        api_text_emb: torch.Tensor,
        api_fusion_mode: str = "fixed_avg",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # g_* 是图结构表征；*_text 是文本投影表征；final_* 是二者平均后的最终节点表示。
        g_mashup = self.mashup_encoder(mashup_adj, self.mashup_id_emb.weight)
        g_api = self.api_encoder(api_adj, self.api_id_emb.weight)
        mashup_text = F.normalize(self.text_proj(mashup_text_emb), p=2, dim=-1)
        api_text = F.normalize(self.text_proj(api_text_emb), p=2, dim=-1)
        final_mashup = 0.5 * (g_mashup + mashup_text)
        if api_fusion_mode == "dynamic_gate":
            api_gate = self.api_fusion_gate(torch.cat([g_api, api_text], dim=-1))
            final_api = F.normalize(api_gate * g_api + (1.0 - api_gate) * api_text, p=2, dim=-1)
        else:
            final_api = 0.5 * (g_api + api_text)
        inactive_m = ~self.mashup_train_mask
        if torch.any(inactive_m):
            g_mashup = g_mashup.clone()
            final_mashup = final_mashup.clone()
            g_mashup[inactive_m] = 0.0
            final_mashup[inactive_m] = mashup_text[inactive_m]
        inactive_a = ~self.api_train_mask
        if torch.any(inactive_a):
            g_api = g_api.clone()
            final_api = final_api.clone()
            g_api[inactive_a] = 0.0
            final_api[inactive_a] = api_text[inactive_a]
        return g_mashup, g_api, final_mashup, final_api

    def encode_query_text(self, query_text_emb: torch.Tensor) -> torch.Tensor:
        if query_text_emb.numel() == 0:
            return query_text_emb
        return F.normalize(self.query_text_proj(query_text_emb), p=2, dim=-1)
        # residual 是 query 级需求向量，不是实体级节点向量。

    def build_explicit_query_rep(
        self,
        final_mashup: torch.Tensor,
        mashup_idx: torch.Tensor,
        query_text_emb: torch.Tensor,
        use_query_text: bool = True,
        query_fusion_mode: str = "fixed_avg",
    ) -> torch.Tensor:
        mashup_query = final_mashup[mashup_idx]
        if not use_query_text:
            return mashup_query
        query_text_proj = self.encode_query_text(query_text_emb)
        query_text_norm = query_text_proj.norm(p=2, dim=-1, keepdim=True)
        if query_fusion_mode == "residual_gate":
            query_gate = self.query_residual_gate(torch.cat([mashup_query, query_text_proj], dim=-1))
            fused_query = F.normalize(mashup_query + query_gate * query_text_proj, p=2, dim=-1)
            return torch.where(query_text_norm > 1e-8, fused_query, mashup_query)
        return torch.where(
            query_text_norm > 1e-8,
            F.normalize(0.5 * (mashup_query + query_text_proj), p=2, dim=-1),
            mashup_query,
        )

    def score(self, mashup_rep: torch.Tensor, api_rep: torch.Tensor) -> torch.Tensor:
        if self.explicit_scorer is None:
            return torch.sum(mashup_rep * api_rep, dim=-1)
        return self.explicit_scorer(mashup_rep, api_rep)


class ImplicitModel(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        dropout: float = 0.1,
        score_mode: str = "dot",
    ) -> None:
        super().__init__()
        self.score_mode = score_mode
        self.context_mlp = nn.Sequential(
            # 输入是 [selected_api_emb ; candidate_api_emb] 的拼接。
            nn.Linear(emb_dim * 2, emb_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.attn_linear = nn.Linear(emb_dim, 1)
        self.implicit_scorer = PairwiseMLPScorer(emb_dim, dropout=dropout) if score_mode == "mlp" else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.context_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.attn_linear.weight)
        nn.init.zeros_(self.attn_linear.bias)

    def attentive_context(
        self,
        selected_api_emb: torch.Tensor,
        candidate_api_emb: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        # MoCoACM-style candidate-aware attention: each selected API is scored
        # after interacting with the candidate API representation.
        cand_expand = candidate_api_emb.unsqueeze(1).expand_as(selected_api_emb)
        pair_features = torch.cat([selected_api_emb, cand_expand], dim=-1)
        hidden = self.context_mlp(pair_features)
        attn_scores = self.attn_linear(hidden).squeeze(-1)
        # padding 位置强制置为极小值，避免被 softmax 分到权重。
        attn_scores = attn_scores.masked_fill(selected_mask <= 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)
        # 注意这里加权求和的是 selected_api_emb 本身，不是 hidden。
        context = torch.sum(selected_api_emb * attn_weights.unsqueeze(-1), dim=1)
        return F.normalize(context, p=2, dim=-1)

    def score(self, context_rep: torch.Tensor, api_rep: torch.Tensor) -> torch.Tensor:
        if self.implicit_scorer is None:
            return torch.sum(context_rep * api_rep, dim=-1)
        return self.implicit_scorer(context_rep, api_rep)


class DualBranchAPIRec(nn.Module):
    def __init__(
        self,
        num_mashups: int,
        num_apis: int,
        input_text_dim: int,
        emb_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        branch_config: BranchConfig | None = None,
    ) -> None:
        super().__init__()
        self.branch_config = branch_config or BranchConfig()
        self.emb_dim = emb_dim

        self.explicit_model = ExplicitModel(
            num_mashups=num_mashups,
            num_apis=num_apis,
            input_text_dim=input_text_dim,
            emb_dim=emb_dim,
            n_layers=n_layers,
            dropout=dropout,
            score_mode="dot",
        )
        self.implicit_model = ImplicitModel(
            emb_dim=emb_dim,
            dropout=dropout,
            score_mode="dot",
        )
        self.fusion_mlp = FusionMLP(hidden_dim=max(16, emb_dim // 4), dropout=dropout)
        self.branch_logits = nn.Parameter(torch.zeros(2))

        # Compatibility handles for the existing training script.
        self.mashup_id_emb = self.explicit_model.mashup_id_emb
        self.api_id_emb = self.explicit_model.api_id_emb
        self.text_proj = self.explicit_model.text_proj
        self.query_text_proj = self.explicit_model.query_text_proj

    def set_train_masks(
        self,
        mashup_mask: torch.Tensor | None = None,
        api_mask: torch.Tensor | None = None,
    ) -> None:
        self.explicit_model.set_train_masks(mashup_mask=mashup_mask, api_mask=api_mask)

    def encode_nodes(
        self,
        mashup_adj: torch.Tensor,
        api_adj: torch.Tensor,
        mashup_text_emb: torch.Tensor,
        api_text_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.explicit_model.encode_nodes(
            mashup_adj,
            api_adj,
            mashup_text_emb,
            api_text_emb,
            api_fusion_mode=self.branch_config.api_fusion_mode,
        )

    def encode_query_text(self, query_text_emb: torch.Tensor) -> torch.Tensor:
        return self.explicit_model.encode_query_text(query_text_emb)

    def build_explicit_query_rep(
        self,
        final_mashup: torch.Tensor,
        mashup_idx: torch.Tensor,
        query_text_emb: torch.Tensor,
        use_query_text: bool = True,
    ) -> torch.Tensor:
        return self.explicit_model.build_explicit_query_rep(
            final_mashup=final_mashup,
            mashup_idx=mashup_idx,
            query_text_emb=query_text_emb,
            use_query_text=use_query_text,
            query_fusion_mode=self.branch_config.query_fusion_mode,
        )

    def attentive_context(
        self,
        selected_api_emb: torch.Tensor,
        candidate_api_emb: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.implicit_model.attentive_context(selected_api_emb, candidate_api_emb, selected_mask)

    def fuse_scores(self, explicit_score: torch.Tensor, implicit_score: torch.Tensor) -> torch.Tensor:
        if self.branch_config.fusion_mode == "avg":
            return 0.5 * explicit_score + 0.5 * implicit_score
        if self.branch_config.fusion_mode == "learnable":
            # learnable 模式学习两个全局标量权重，而不是 query-aware 权重。
            w = torch.softmax(self.branch_logits, dim=0)
            return w[0] * explicit_score + w[1] * implicit_score
        return self.fusion_mlp(explicit_score, implicit_score)

    def score_batch(
        self,
        mashup_idx: torch.Tensor,
        selected_api_idx: torch.Tensor,
        selected_mask: torch.Tensor,
        candidate_api_idx: torch.Tensor,
        query_text_emb: torch.Tensor,
        mashup_adj: torch.Tensor,
        api_adj: torch.Tensor,
        mashup_text_emb: torch.Tensor,
        api_text_emb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # 每次 score_batch 都会重新编码全部节点；这让实现简单，但训练时开销较大。
        g_mashup, g_api, final_mashup, final_api = self.encode_nodes(
            mashup_adj, api_adj, mashup_text_emb, api_text_emb
        )
        candidate_emb = final_api[candidate_api_idx]
        selected_emb = final_api[selected_api_idx]
        explicit_m = self.build_explicit_query_rep(
            final_mashup=final_mashup,
            mashup_idx=mashup_idx,
            query_text_emb=query_text_emb,
            use_query_text=self.branch_config.use_residual,
        )

        explicit_score = self.explicit_model.score(explicit_m, candidate_emb)
        context = self.attentive_context(selected_emb, candidate_emb, selected_mask)
        implicit_score = self.implicit_model.score(context, candidate_emb)

        if self.branch_config.use_explicit and self.branch_config.use_implicit:
            total = self.fuse_scores(explicit_score, implicit_score)
        elif self.branch_config.use_explicit:
            total = explicit_score
        elif self.branch_config.use_implicit:
            total = implicit_score
        else:
            total = explicit_score.new_zeros(explicit_score.shape)

        return {
            "total": total,
            "explicit": explicit_score,
            "implicit": implicit_score,
            "g_mashup": g_mashup,
            "g_api": g_api,
            "final_mashup": final_mashup,
            "final_api": final_api,
        }


class HybridRecSys(DualBranchAPIRec):
    pass


LightGCN = LightGCNEncoder
