from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


def cosine_similarity_matrix(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    x = emb / norms
    sim = x @ x.T
    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float32)


def build_topk_graph(
    emb: np.ndarray,
    topk: int = 20,
    threshold: float = 0.0,
    add_self_loop: bool = True,
) -> tuple[sp.coo_matrix, np.ndarray]:
    sim = cosine_similarity_matrix(emb)
    n = sim.shape[0]
    rows, cols, data = [], [], []
    for i in range(n):
        row = sim[i]
        if topk >= n:
            idx = np.argsort(-row)
        else:
            idx = np.argpartition(-row, kth=min(topk, n - 1))[: topk + 1]
            idx = idx[np.argsort(-row[idx])]
        for j in idx:
            if i == j and not add_self_loop:
                continue
            if row[j] >= threshold:
                rows.append(i)
                cols.append(j)
                data.append(float(row[j]))
    mat = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    mat = mat.maximum(mat.T).tocoo()
    if add_self_loop:
        mat = (mat + sp.eye(n, dtype=np.float32, format="coo")).tocoo()
    norm = symmetric_normalize(mat)
    return norm, sim


def build_threshold_graph(
    emb: np.ndarray,
    threshold: float = 0.0,
    add_self_loop: bool = True,
) -> tuple[sp.coo_matrix, np.ndarray]:
    sim = cosine_similarity_matrix(emb)
    mask = sim >= threshold
    if add_self_loop:
        np.fill_diagonal(mask, True)
        np.fill_diagonal(sim, 1.0)
    rows, cols = np.where(mask)
    data = sim[rows, cols].astype(np.float32, copy=False)
    mat = sp.coo_matrix((data, (rows, cols)), shape=sim.shape, dtype=np.float32)
    mat = mat.maximum(mat.T).tocoo()
    if add_self_loop:
        diag = sp.eye(mat.shape[0], dtype=np.float32, format="coo")
        mat = mat.tolil()
        mat.setdiag(0.0)
        mat = (mat.tocoo() + diag).tocoo()
    return mat, sim


def mask_graph_nodes(
    mat: sp.coo_matrix,
    active_nodes: np.ndarray | list[int],
    keep_self_loop: bool = True,
) -> sp.coo_matrix:
    mat = mat.tocoo()
    n = mat.shape[0]
    active = np.zeros(n, dtype=bool)
    active[np.asarray(active_nodes, dtype=np.int64)] = True
    keep = active[mat.row] & active[mat.col]
    if keep_self_loop:
        keep &= mat.row != mat.col
    out = sp.coo_matrix((mat.data[keep], (mat.row[keep], mat.col[keep])), shape=mat.shape, dtype=np.float32)
    if keep_self_loop:
        out = (out + sp.eye(n, dtype=np.float32, format="coo")).tocoo()
    return out


def symmetric_normalize(mat: sp.coo_matrix) -> sp.coo_matrix:
    mat = mat.tocoo()
    deg = np.array(mat.sum(axis=1)).squeeze()
    deg = np.where(deg == 0, 1.0, deg)
    inv_sqrt = np.power(deg, -0.5)
    d = sp.diags(inv_sqrt)
    return (d @ mat @ d).tocoo()


def sparse_to_torch(mat: sp.coo_matrix, device: str | torch.device = "cpu") -> torch.Tensor:
    mat = mat.tocoo()
    idx = torch.tensor(np.vstack([mat.row, mat.col]), dtype=torch.long)
    val = torch.tensor(mat.data, dtype=torch.float32)
    shape = torch.Size(mat.shape)
    return torch.sparse_coo_tensor(idx, val, shape, device=device).coalesce()
