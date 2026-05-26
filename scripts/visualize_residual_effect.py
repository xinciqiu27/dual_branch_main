from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True, help="Training output directory containing run_meta.json and best_model.pt")
    p.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint path. Defaults to <run-dir>/best_model.pt")
    p.add_argument("--query-split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--num-query-plots", type=int, default=5)
    p.add_argument("--candidate-sample", type=int, default=40, help="Number of non-target, non-selected APIs to sample for local plots")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_saved_args(run_dir: Path) -> argparse.Namespace:
    run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    with patch.object(sys, "argv", ["train.py"]):
        defaults = train.parse_args()
    merged = vars(defaults).copy()
    merged.update(run_meta["args"])
    merged["save_dir"] = str(run_dir)
    return SimpleNamespace(**merged)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mean_or_zero(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.zeros((arr.shape[-1],), dtype=np.float32)
    return arr.mean(axis=0)


def pca_2d(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) <= 1:
        return np.zeros((len(vectors), 2), dtype=np.float32)
    return PCA(n_components=2, random_state=42).fit_transform(vectors).astype(np.float32)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def project_points(points: np.ndarray, width: int = 900, height: int = 700, pad: int = 60) -> np.ndarray:
    xs = points[:, 0]
    ys = points[:, 1]
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    if math.isclose(min_x, max_x):
        min_x -= 1.0
        max_x += 1.0
    if math.isclose(min_y, max_y):
        min_y -= 1.0
        max_y += 1.0
    sx = (width - 2 * pad) / (max_x - min_x)
    sy = (height - 2 * pad) / (max_y - min_y)
    scale = min(sx, sy)
    out = []
    for x, y in points:
        px = pad + (x - min_x) * scale
        py = height - pad - (y - min_y) * scale
        out.append((float(px), float(py)))
    return np.asarray(out, dtype=np.float32)


def save_svg(path: Path, body_lines: list[str], width: int = 900, height: int = 700) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#666"/></marker></defs>',
        *body_lines,
        "</svg>",
    ]
    path.write_text("\n".join(svg), encoding="utf-8")


def make_global_shift_svg(path: Path, coords_no: np.ndarray, coords_yes: np.ndarray, gains: np.ndarray) -> None:
    all_points = np.vstack([coords_no, coords_yes])
    screen = project_points(all_points)
    no_screen = screen[: len(coords_no)]
    yes_screen = screen[len(coords_no) :]
    body = [
        '<text x="30" y="35" font-size="24" font-family="Arial">Residual demand shifts query representations</text>',
        '<text x="30" y="62" font-size="14" font-family="Arial" fill="#555">gray: no residual, red: with residual, arrow: shift direction</text>',
    ]
    for i, (p0, p1) in enumerate(zip(no_screen, yes_screen)):
        gain = gains[i]
        width = 1.0 if gain <= 0 else min(4.0, 1.0 + gain * 20.0)
        body.append(
            f'<line x1="{p0[0]:.2f}" y1="{p0[1]:.2f}" x2="{p1[0]:.2f}" y2="{p1[1]:.2f}" '
            f'stroke="#999" stroke-width="{width:.2f}" marker-end="url(#arrow)" opacity="0.65"/>'
        )
        body.append(f'<circle cx="{p0[0]:.2f}" cy="{p0[1]:.2f}" r="2.8" fill="#888" opacity="0.8"/>')
        body.append(f'<circle cx="{p1[0]:.2f}" cy="{p1[1]:.2f}" r="3.2" fill="#d62728" opacity="0.85"/>')
    save_svg(path, body)


def make_two_group_svg(
    path: Path,
    api_coords: np.ndarray,
    mashup_coords: np.ndarray,
    mashup_label: str,
    mashup_color: str,
) -> None:
    all_points = np.vstack([api_coords, mashup_coords])
    screen = project_points(all_points)
    api_screen = screen[: len(api_coords)]
    mashup_screen = screen[len(api_coords) :]
    body = [
        f'<text x="30" y="35" font-size="24" font-family="Arial">{mashup_label}</text>',
        '<text x="30" y="62" font-size="14" font-family="Arial" fill="#555">blue: static API embeddings, colored: mashup/query embeddings</text>',
    ]
    for x, y in api_screen:
        body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.4" fill="#1f77b4" opacity="0.24"/>')
    for x, y in mashup_screen:
        body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.5" fill="{mashup_color}" opacity="0.42"/>')
    legend_y = 95
    for name, color in [
        ("api_static", "#1f77b4"),
        (mashup_label, mashup_color),
    ]:
        body.append(f'<circle cx="35" cy="{legend_y}" r="4" fill="{color}"/>')
        body.append(f'<text x="48" y="{legend_y + 5}" font-size="13" font-family="Arial">{name}</text>')
        legend_y += 22
    save_svg(path, body)


def make_local_query_svg(path: Path, points: np.ndarray, labels: list[str], title: str) -> None:
    color_map = {
        "query_no_residual": "#7f7f7f",
        "query_with_residual": "#d62728",
        "selected_api": "#1f77b4",
        "target_api": "#2ca02c",
        "other_api": "#c7c7c7",
    }
    screen = project_points(points)
    body = [
        f'<text x="30" y="35" font-size="24" font-family="Arial">{title}</text>',
        '<text x="30" y="62" font-size="14" font-family="Arial" fill="#555">red query should ideally move toward green target APIs</text>',
    ]
    for idx, (xy, label) in enumerate(zip(screen, labels)):
        x, y = xy
        if idx == 0:
            q0 = xy
        if idx == 1:
            q1 = xy
        color = color_map[label]
        radius = 5 if "query" in label else 3.2
        body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{color}" opacity="0.9"/>')
    body.append(
        f'<line x1="{q0[0]:.2f}" y1="{q0[1]:.2f}" x2="{q1[0]:.2f}" y2="{q1[1]:.2f}" '
        f'stroke="#555" stroke-width="2" marker-end="url(#arrow)" opacity="0.85"/>'
    )
    legend_y = 95
    for name, color in [
        ("query_no_residual", "#7f7f7f"),
        ("query_with_residual", "#d62728"),
        ("selected_api", "#1f77b4"),
        ("target_api", "#2ca02c"),
        ("other_api", "#c7c7c7"),
    ]:
        body.append(f'<circle cx="35" cy="{legend_y}" r="4" fill="{color}"/>')
        body.append(f'<text x="48" y="{legend_y + 5}" font-size="13" font-family="Arial">{name}</text>')
        legend_y += 22
    save_svg(path, body)


def make_local_overlay_svg(path: Path, points: np.ndarray, labels: list[str], title: str) -> None:
    color_map = {
        "query_no_residual": "#2ca02c",
        "query_with_residual": "#d62728",
        "selected_api": "#1f77b4",
        "target_api": "#ff7f0e",
        "other_api": "#c7c7c7",
    }
    radius_map = {
        "query_no_residual": 5.0,
        "query_with_residual": 5.0,
        "selected_api": 3.4,
        "target_api": 4.2,
        "other_api": 2.7,
    }
    opacity_map = {
        "query_no_residual": 0.95,
        "query_with_residual": 0.95,
        "selected_api": 0.85,
        "target_api": 0.92,
        "other_api": 0.38,
    }
    screen = project_points(points)
    body = [
        f'<text x="30" y="35" font-size="24" font-family="Arial">{title}</text>',
        '<text x="30" y="62" font-size="14" font-family="Arial" fill="#555">green: no residual, red: with residual, orange: target APIs, blue: selected APIs</text>',
    ]
    for (x, y), label in zip(screen, labels):
        body.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius_map[label]:.2f}" fill="{color_map[label]}" opacity="{opacity_map[label]:.2f}"/>'
        )
    legend_y = 95
    for name, color in [
        ("query_no_residual", "#2ca02c"),
        ("query_with_residual", "#d62728"),
        ("target_api", "#ff7f0e"),
        ("selected_api", "#1f77b4"),
        ("other_api", "#c7c7c7"),
    ]:
        body.append(f'<circle cx="35" cy="{legend_y}" r="4" fill="{color}"/>')
        body.append(f'<text x="48" y="{legend_y + 5}" font-size="13" font-family="Arial">{name}</text>')
        legend_y += 22
    save_svg(path, body)


def main() -> None:
    cli_args = parse_args()
    run_dir = Path(cli_args.run_dir)
    checkpoint = Path(cli_args.checkpoint) if cli_args.checkpoint else run_dir / "best_model.pt"
    args = load_saved_args(run_dir)
    args.seed = cli_args.seed

    ctx = train.prepare_common(args)
    model = ctx["model"]
    device = ctx["device"]
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    query_list = ctx["test_queries"] if cli_args.query_split == "test" else ctx["val_queries"]
    query_text_map = ctx["query_text_map"]
    api_df = ctx["bundle"]["api_df"]
    mashup_df = ctx["bundle"]["mashup_df"]

    with torch.no_grad():
        _, _, final_mashup_t, final_api_t = model.encode_nodes(
            ctx["mashup_adj"],
            ctx["api_adj"],
            ctx["mashup_text_emb"],
            ctx["api_text_emb"],
        )
    final_mashup = final_mashup_t.detach().cpu().numpy()
    final_api = final_api_t.detach().cpu().numpy()

    summary_rows: list[dict] = []
    query_no_list = []
    query_yes_list = []
    target_gain_list = []

    rng = np.random.default_rng(cli_args.seed)

    for q in query_list:
        mid = q["mashup_id"]
        selected = q["selected_api_ids"]
        targets = q["target_api_ids"]
        if not targets:
            continue
        query_text = torch.tensor(query_text_map[q["query_key"]], dtype=torch.float32, device=device).unsqueeze(0)
        mashup_idx = torch.tensor([mid], dtype=torch.long, device=device)
        with torch.no_grad():
            q_no = model.build_explicit_query_rep(
                final_mashup=final_mashup_t,
                mashup_idx=mashup_idx,
                query_text_emb=query_text,
                use_query_text=False,
            )[0].detach().cpu().numpy()
            q_yes = model.build_explicit_query_rep(
                final_mashup=final_mashup_t,
                mashup_idx=mashup_idx,
                query_text_emb=query_text,
                use_query_text=True,
            )[0].detach().cpu().numpy()
        selected_centroid = mean_or_zero(final_api[np.asarray(selected, dtype=np.int64)])
        target_centroid = mean_or_zero(final_api[np.asarray(targets, dtype=np.int64)])
        row = {
            "query_key": q["query_key"],
            "mashup_id": mid,
            "mashup_name": str(mashup_df.iloc[mid]["mashup_name"]),
            "selected_count": len(selected),
            "target_count": len(targets),
            "cos_no_to_target": cosine(q_no, target_centroid),
            "cos_yes_to_target": cosine(q_yes, target_centroid),
            "target_gain": cosine(q_yes, target_centroid) - cosine(q_no, target_centroid),
            "cos_no_to_selected": cosine(q_no, selected_centroid),
            "cos_yes_to_selected": cosine(q_yes, selected_centroid),
            "selected_gain": cosine(q_yes, selected_centroid) - cosine(q_no, selected_centroid),
            "query_shift_l2": float(np.linalg.norm(q_yes - q_no)),
        }
        summary_rows.append(row)
        query_no_list.append(q_no)
        query_yes_list.append(q_yes)
        target_gain_list.append(row["target_gain"])

    out_dir = run_dir / "residual_viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / f"{cli_args.query_split}_query_summary.csv",
        summary_rows,
        [
            "query_key",
            "mashup_id",
            "mashup_name",
            "selected_count",
            "target_count",
            "cos_no_to_target",
            "cos_yes_to_target",
            "target_gain",
            "cos_no_to_selected",
            "cos_yes_to_selected",
            "selected_gain",
            "query_shift_l2",
        ],
    )

    query_no_arr = np.asarray(query_no_list, dtype=np.float32)
    query_yes_arr = np.asarray(query_yes_list, dtype=np.float32)
    target_gain_arr = np.asarray(target_gain_list, dtype=np.float32)
    global_2d = pca_2d(np.vstack([query_no_arr, query_yes_arr]))
    n = len(query_no_arr)
    global_rows = []
    for i, row in enumerate(summary_rows):
        global_rows.append(
            {
                "query_key": row["query_key"],
                "mashup_id": row["mashup_id"],
                "mashup_name": row["mashup_name"],
                "x_no": float(global_2d[i, 0]),
                "y_no": float(global_2d[i, 1]),
                "x_yes": float(global_2d[n + i, 0]),
                "y_yes": float(global_2d[n + i, 1]),
                "target_gain": row["target_gain"],
                "query_shift_l2": row["query_shift_l2"],
            }
        )
    write_csv(
        out_dir / f"{cli_args.query_split}_global_query_shift.csv",
        global_rows,
        ["query_key", "mashup_id", "mashup_name", "x_no", "y_no", "x_yes", "y_yes", "target_gain", "query_shift_l2"],
    )
    make_global_shift_svg(
        out_dir / f"{cli_args.query_split}_global_query_shift.svg",
        global_2d[:n],
        global_2d[n:],
        target_gain_arr,
    )

    api_and_query = np.vstack([final_api, query_no_arr, query_yes_arr])
    api_query_2d = pca_2d(api_and_query)
    api_coords = api_query_2d[: len(final_api)]
    query_no_coords = api_query_2d[len(final_api) : len(final_api) + n]
    query_yes_coords = api_query_2d[len(final_api) + n :]
    api_query_rows = []
    for api_idx, xy in enumerate(api_coords):
        api_query_rows.append(
            {
                "id": f"api_{api_idx}",
                "type": "api_static",
                "x": float(xy[0]),
                "y": float(xy[1]),
            }
        )
    for i, row in enumerate(summary_rows):
        api_query_rows.append(
            {
                "id": row["query_key"],
                "type": "mashup_query_no_residual",
                "x": float(query_no_coords[i, 0]),
                "y": float(query_no_coords[i, 1]),
            }
        )
        api_query_rows.append(
            {
                "id": row["query_key"],
                "type": "mashup_query_with_residual",
                "x": float(query_yes_coords[i, 0]),
                "y": float(query_yes_coords[i, 1]),
            }
        )
    write_csv(
        out_dir / f"{cli_args.query_split}_api_query_overlay.csv",
        api_query_rows,
        ["id", "type", "x", "y"],
    )
    make_two_group_svg(
        out_dir / f"{cli_args.query_split}_api_query_no_residual.svg",
        api_coords,
        query_no_coords,
        "mashup_query_no_residual",
        "#2ca02c",
    )
    make_two_group_svg(
        out_dir / f"{cli_args.query_split}_api_query_with_residual.svg",
        api_coords,
        query_yes_coords,
        "mashup_query_with_residual",
        "#d62728",
    )

    top_rows = sorted(summary_rows, key=lambda x: x["target_gain"], reverse=True)[: cli_args.num_query_plots]
    all_api_ids = np.arange(len(api_df), dtype=np.int64)
    for rank, row in enumerate(top_rows, start=1):
        q = next(item for item in query_list if item["query_key"] == row["query_key"])
        selected = np.asarray(q["selected_api_ids"], dtype=np.int64)
        targets = np.asarray(q["target_api_ids"], dtype=np.int64)
        forbidden = set(selected.tolist()) | set(targets.tolist())
        candidates = np.asarray([a for a in all_api_ids if a not in forbidden], dtype=np.int64)
        if len(candidates) > cli_args.candidate_sample:
            candidates = rng.choice(candidates, size=cli_args.candidate_sample, replace=False)

        query_text = torch.tensor(query_text_map[q["query_key"]], dtype=torch.float32, device=device).unsqueeze(0)
        mashup_idx = torch.tensor([q["mashup_id"]], dtype=torch.long, device=device)
        with torch.no_grad():
            q_no = model.build_explicit_query_rep(final_mashup_t, mashup_idx, query_text, use_query_text=False)[0].detach().cpu().numpy()
            q_yes = model.build_explicit_query_rep(final_mashup_t, mashup_idx, query_text, use_query_text=True)[0].detach().cpu().numpy()

        vectors = [q_no, q_yes]
        labels = ["query_no_residual", "query_with_residual"]
        ids = ["query_no_residual", "query_with_residual"]
        for aid in selected:
            vectors.append(final_api[aid])
            labels.append("selected_api")
            ids.append(f"selected_api_{aid}")
        for aid in targets:
            vectors.append(final_api[aid])
            labels.append("target_api")
            ids.append(f"target_api_{aid}")
        for aid in candidates:
            vectors.append(final_api[aid])
            labels.append("other_api")
            ids.append(f"other_api_{aid}")

        vectors_arr = np.asarray(vectors, dtype=np.float32)
        coords = pca_2d(vectors_arr)
        detail_rows = []
        for ident, label, xy in zip(ids, labels, coords):
            detail_rows.append(
                {
                    "id": ident,
                    "label": label,
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                }
            )
        base = out_dir / f"{cli_args.query_split}_detail_rank{rank}_m{q['mashup_id']}"
        write_csv(base.with_suffix(".csv"), detail_rows, ["id", "label", "x", "y"])
        title = f"Query {rank}: mashup={q['mashup_id']} gain={row['target_gain']:.4f}"
        make_local_query_svg(base.with_suffix(".svg"), coords, labels, title)
        make_local_overlay_svg(base.with_name(base.name + "_overlay").with_suffix(".svg"), coords, labels, title)

    aggregate = {
        "query_count": len(summary_rows),
        "mean_target_gain": float(np.mean(target_gain_arr)) if len(target_gain_arr) else 0.0,
        "median_target_gain": float(np.median(target_gain_arr)) if len(target_gain_arr) else 0.0,
        "positive_gain_ratio": float(np.mean(target_gain_arr > 0)) if len(target_gain_arr) else 0.0,
        "mean_query_shift_l2": float(np.mean([r["query_shift_l2"] for r in summary_rows])) if summary_rows else 0.0,
        "important_note": "Residual demand changes query-conditioned explicit representations, not static API node embeddings.",
    }
    (out_dir / f"{cli_args.query_split}_aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
