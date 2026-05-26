from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
CSV_PATH = ROOT / "docs" / "outputs_index.csv"
MD_PATH = ROOT / "docs" / "outputs_index.md"


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def compact_bool_flag(value) -> str:
    return "1" if bool(value) else "0"


def build_branch_tag(args: dict) -> str:
    return "E{e}_I{i}_R{r}_T{t}".format(
        e=compact_bool_flag(args.get("w_explicit", 1)),
        i=compact_bool_flag(args.get("w_implicit", 1)),
        r=compact_bool_flag(args.get("w_residual", 1)),
        t=compact_bool_flag(args.get("w_tss", 1)),
    )


def find_run_dirs() -> list[Path]:
    run_dirs = []
    for path in OUTPUTS_DIR.rglob("run_meta.json"):
        run_dirs.append(path.parent)
    return sorted(set(run_dirs))


def summarize_run(run_dir: Path) -> dict:
    rel_dir = run_dir.relative_to(ROOT).as_posix()
    parts = rel_dir.split("/")
    group = parts[1] if len(parts) > 1 else "(root)"
    run_name = "/".join(parts[2:]) if len(parts) > 2 else parts[-1]

    run_meta = load_json(run_dir / "run_meta.json")
    summary = load_json(run_dir / "summary.json")
    test_metrics = load_json(run_dir / "test_metrics.json")

    args = run_meta.get("args", {})
    protocol = run_meta.get("protocol_report", {}).get("standard_protocol", {})

    metrics = summary.get("test_metrics") or test_metrics

    residual_csv = args.get("residual_text_csv", "") or ""
    residual_name = Path(residual_csv).name if residual_csv else "(none)"

    row = {
        "group": group,
        "run_name": run_name,
        "run_dir": rel_dir,
        "training_mode": summary.get("training_mode", "(unknown)"),
        "best_stage": summary.get("best_stage", "(unknown)"),
        "split_mode": args.get("split_mode", "(unknown)"),
        "protocol": protocol.get("name", "(n/a)"),
        "seed": args.get("seed", ""),
        "encoder": args.get("encoder_backend", ""),
        "model_name": args.get("encoder_model", ""),
        "residual_csv": residual_name,
        "branch_tag": build_branch_tag(args),
        "fusion_mode": args.get("fusion_mode", ""),
        "joint_loss_variant": args.get("joint_loss_variant", ""),
        "context_loss_type": args.get("context_loss_type", ""),
        "negative_mode": args.get("negative_mode", ""),
        "graph_mode": args.get("graph_mode", ""),
        "emb_dim": args.get("emb_dim", ""),
        "n_layers": args.get("n_layers", ""),
        "batch_size": args.get("batch_size", ""),
        "lr": args.get("lr", ""),
        "lambda_exp": args.get("lambda_exp", ""),
        "lambda_imp": args.get("lambda_imp", ""),
        "lambda_tss": args.get("lambda_tss", ""),
        "best_epoch": summary.get("best_epoch", ""),
        "best_val_metric": summary.get("best_val_metric", ""),
        "MRR": metrics.get("MRR", ""),
        "Recall@5": metrics.get("Recall@5", ""),
        "Recall@10": metrics.get("Recall@10", ""),
        "NDCG@10": metrics.get("NDCG@10", ""),
    }
    return row


def write_csv(rows: list[dict]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "run_name",
        "run_dir",
        "training_mode",
        "best_stage",
        "split_mode",
        "protocol",
        "seed",
        "encoder",
        "model_name",
        "residual_csv",
        "branch_tag",
        "fusion_mode",
        "joint_loss_variant",
        "context_loss_type",
        "negative_mode",
        "graph_mode",
        "emb_dim",
        "n_layers",
        "batch_size",
        "lr",
        "lambda_exp",
        "lambda_imp",
        "lambda_tss",
        "best_epoch",
        "best_val_metric",
        "MRR",
        "Recall@5",
        "Recall@10",
        "NDCG@10",
        "group_best",
    ]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict]) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)

    lines = []
    lines.append("# Outputs Index")
    lines.append("")
    lines.append("This file summarizes experiment runs under `outputs/`.")
    lines.append("")
    lines.append("## Column Notes")
    lines.append("")
    lines.append("- `branch_tag`: `E/I/R/T` means explicit branch, implicit branch, residual text, and TSS are enabled (`1`) or disabled (`0`).")
    lines.append("- `group_best=1`: best run within the same top-level output group, ranked by `Recall@10` then `MRR`.")
    lines.append("")

    lines.append("## Group Summary")
    lines.append("")
    lines.append("| Group | #Runs | Best Run | Recall@10 | Recall@5 | MRR |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for group, group_rows in sorted(grouped.items()):
        best = max(group_rows, key=lambda x: (to_float(x["Recall@10"]), to_float(x["MRR"])))
        lines.append(
            f"| {group} | {len(group_rows)} | `{best['run_name']}` | "
            f"{to_float(best['Recall@10']):.4f} | {to_float(best['Recall@5']):.4f} | {to_float(best['MRR']):.4f} |"
        )
    lines.append("")

    top_rows = sorted(rows, key=lambda x: (to_float(x["Recall@10"]), to_float(x["MRR"])), reverse=True)[:30]
    lines.append("## Top 30 Runs by Recall@10")
    lines.append("")
    lines.append("| Rank | Group | Run | Train | Branch | Fusion | Loss | Recall@10 | Recall@5 | MRR |")
    lines.append("|---:|---|---|---|---|---|---|---:|---:|---:|")
    for idx, row in enumerate(top_rows, start=1):
        loss_tag = f"{row['joint_loss_variant']} / {row['context_loss_type']}"
        lines.append(
            f"| {idx} | {row['group']} | `{row['run_name']}` | {row['training_mode']} | "
            f"{row['branch_tag']} | {row['fusion_mode']} | {loss_tag} | "
            f"{to_float(row['Recall@10']):.4f} | {to_float(row['Recall@5']):.4f} | {to_float(row['MRR']):.4f} |"
        )
    lines.append("")

    lines.append("## Full Inventory")
    lines.append("")
    lines.append("| Group | Run | Train | Split | Seed | Branch | Fusion | Residual CSV | Recall@10 | MRR | Best |")
    lines.append("|---|---|---|---|---:|---|---|---|---:|---:|---:|")
    for row in sorted(rows, key=lambda x: (x["group"], -to_float(x["Recall@10"]), -to_float(x["MRR"]), x["run_name"])):
        lines.append(
            f"| {row['group']} | `{row['run_name']}` | {row['training_mode']} | {row['split_mode']} | "
            f"{row['seed']} | {row['branch_tag']} | {row['fusion_mode']} | {row['residual_csv']} | "
            f"{to_float(row['Recall@10']):.4f} | {to_float(row['MRR']):.4f} | {row['group_best']} |"
        )

    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = [summarize_run(run_dir) for run_dir in find_run_dirs()]
    best_per_group = {}
    for row in rows:
        best = best_per_group.get(row["group"])
        if best is None or (to_float(row["Recall@10"]), to_float(row["MRR"])) > (
            to_float(best["Recall@10"]),
            to_float(best["MRR"]),
        ):
            best_per_group[row["group"]] = row

    for row in rows:
        row["group_best"] = 1 if best_per_group.get(row["group"]) is row else 0

    rows.sort(key=lambda x: (x["group"], -to_float(x["Recall@10"]), -to_float(x["MRR"]), x["run_name"]))
    write_csv(rows)
    write_markdown(rows)
    print(f"Indexed {len(rows)} runs.")
    print(f"CSV: {CSV_PATH}")
    print(f"MD : {MD_PATH}")


if __name__ == "__main__":
    main()
