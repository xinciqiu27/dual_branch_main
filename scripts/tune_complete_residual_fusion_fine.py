from __future__ import annotations

import argparse
import itertools
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fine hyperparameter search for complete-residual fusion. "
            "Every config is evaluated with all requested seeds."
        )
    )
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--base-save-dir", type=str, default="outputs/tune_complete_residual_fusion_fine")
    p.add_argument("--python-exe", type=str, default=sys.executable)
    p.add_argument("--encoder-model", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--residual-text-csv", type=str, default="data/llm/residual_text_deepseek.csv")
    p.add_argument("--seeds", type=str, default="42,52,62")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--max-configs", type=int, default=0, help="0 means full search space.")
    p.add_argument("--selection-metric", type=str, default="Recall@10")
    p.add_argument("--focus-metrics", type=str, default="MRR,Recall@2,Recall@4,Recall@5,Recall@6,Recall@8,Recall@20")
    p.add_argument("--eval-ks", type=str, default="2,4,5,6,8,10,20")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_metric_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def build_search_space() -> list[dict[str, object]]:
    # Ordered from the strongest observed region outward.
    grid = {
        "lr": [5e-4, 3e-4],
        "dropout": [0.1],
        "emb-dim": [128],
        "n-layers": [2],
        "lambda-exp": [0.2, 0.25, 0.15],
        "lambda-imp": [0.0],
        "lambda-tss": [0.25, 0.1, 0.2, 0.15],
        "negative-mode": ["strict_global"],
        "fusion-mode": ["mlp"],
        "batch-size": [256, 128],
        "joint-loss-variant": ["no_imp_aux", "full"],
        "context-loss-type": ["bce", "bpr"],
        "stage1-epochs": [10],
        "stage2-epochs": [10],
        "stage3-epochs": [20],
    }
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def config_to_name(idx: int, cfg: dict[str, object]) -> str:
    loss_tag = {
        "full": "full",
        "no_imp_aux": "noimp",
        "total_only": "total",
    }[str(cfg["joint-loss-variant"])]
    ctx_tag = "bce" if cfg["context-loss-type"] == "bce" else "bpr"
    return (
        f"f{idx:03d}"
        f"_lr{str(cfg['lr']).replace('0.', '').replace('.', '')}"
        f"_d{cfg['emb-dim']}"
        f"_l{cfg['n-layers']}"
        f"_do{str(cfg['dropout']).replace('0.', '')}"
        f"_b{cfg['batch-size']}"
        f"_e{str(cfg['lambda-exp']).replace('0.', '').replace('.', '')}"
        f"_i{str(cfg['lambda-imp']).replace('0.', '').replace('.', '')}"
        f"_t{str(cfg['lambda-tss']).replace('0.', '').replace('.', '')}"
        f"_j{loss_tag}"
        f"_c{ctx_tag}"
    )


def build_command(args: argparse.Namespace, cfg: dict[str, object], seed: int, save_dir: Path) -> list[str]:
    cmd = [
        args.python_exe,
        "train.py",
        "--data-dir",
        args.data_dir,
        "--save-dir",
        str(save_dir),
        "--device",
        args.device,
        "--encoder-backend",
        "sbert",
        "--encoder-model",
        args.encoder_model,
        "--residual-text-csv",
        args.residual_text_csv,
        "--selection-metric",
        args.selection_metric,
        "--eval-ks",
        args.eval_ks,
        "--seed",
        str(seed),
    ]
    for key, value in cfg.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def load_summary(save_dir: Path) -> dict[str, object]:
    with open(save_dir / "summary.json", "r", encoding="utf-8") as f:
        return json.load(f)


def mean_metric(metrics: dict[str, float], names: list[str]) -> float:
    values = [float(metrics.get(name, 0.0)) for name in names]
    return sum(values) / len(values)


def stdev_or_zero(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.pstdev(values)


def run_trial(cmd: list[str], save_dir: Path, dry_run: bool) -> bool:
    if dry_run:
        print(" ".join(cmd))
        return False
    if (save_dir / "summary.json").exists():
        return True
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode == 0


def metric_mean(rows: list[dict[str, object]], section: str, metric: str) -> float:
    return statistics.mean([float(r[section].get(metric, 0.0)) for r in rows])


def metric_max(rows: list[dict[str, object]], section: str, metric: str) -> float:
    return max(float(r[section].get(metric, 0.0)) for r in rows)


def best_row(rows: list[dict[str, object]], metric: str) -> dict[str, object]:
    return max(
        rows,
        key=lambda r: (
            float(r["test_metrics"].get(metric, 0.0)),
            float(r["test_metrics"].get("Recall@10", 0.0)),
            float(r["test_metrics"].get("MRR", 0.0)),
        ),
    )


def main() -> None:
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    focus_metrics = parse_metric_list(args.focus_metrics)

    base_save_dir = (ROOT / args.base_save_dir).resolve()
    run_dir = base_save_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    configs = build_search_space()
    if args.max_configs > 0:
        configs = configs[: args.max_configs]

    all_runs: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    total_jobs = len(configs) * len(seeds)
    job_idx = 0
    for idx, cfg in enumerate(configs, start=1):
        cfg_name = config_to_name(idx, cfg)
        for seed in seeds:
            job_idx += 1
            save_dir = run_dir / f"{cfg_name}__seed_{seed}"
            cmd = build_command(args, cfg, seed, save_dir)
            print(f"[run {job_idx}/{total_jobs}] {cfg_name} seed={seed}")
            ok = run_trial(cmd, save_dir, args.dry_run)
            if args.dry_run:
                continue
            if not ok:
                failures.append({"config_name": cfg_name, "seed": seed, "config": cfg})
                continue
            summary = load_summary(save_dir)
            best_val_metrics = summary.get("best_val_metrics", {})
            all_runs.append(
                {
                    "config_name": cfg_name,
                    "config": cfg,
                    "seed": seed,
                    "save_dir": str(save_dir),
                    "best_val_metric": float(summary.get("best_val_metric", summary.get("best_val_recall10", 0.0))),
                    "best_val_metrics": best_val_metrics,
                    "val_focus_score": mean_metric(best_val_metrics, focus_metrics),
                    "test_metrics": summary["test_metrics"],
                    "test_focus_score": mean_metric(summary["test_metrics"], focus_metrics),
                }
            )

    if args.dry_run:
        return

    config_groups: list[dict[str, object]] = []
    for idx, cfg in enumerate(configs, start=1):
        cfg_name = config_to_name(idx, cfg)
        rows = [r for r in all_runs if r["config_name"] == cfg_name]
        if not rows:
            continue
        peak_r5_row = best_row(rows, "Recall@5")
        peak_r10_row = best_row(rows, "Recall@10")
        peak_mrr_row = best_row(rows, "MRR")
        config_groups.append(
            {
                "config_name": cfg_name,
                "config": cfg,
                "num_success_runs": len(rows),
                "seeds_requested": seeds,
                "seeds_completed": [r["seed"] for r in rows],
                "best_run_by_recall5": peak_r5_row,
                "best_run_by_recall10": peak_r10_row,
                "best_run_by_mrr": peak_mrr_row,
                "peak_test_metrics": {
                    "MRR": metric_max(rows, "test_metrics", "MRR"),
                    "Recall@2": metric_max(rows, "test_metrics", "Recall@2"),
                    "Recall@4": metric_max(rows, "test_metrics", "Recall@4"),
                    "Recall@5": metric_max(rows, "test_metrics", "Recall@5"),
                    "Recall@6": metric_max(rows, "test_metrics", "Recall@6"),
                    "Recall@8": metric_max(rows, "test_metrics", "Recall@8"),
                    "Recall@10": metric_max(rows, "test_metrics", "Recall@10"),
                    "Recall@20": metric_max(rows, "test_metrics", "Recall@20"),
                },
                "mean_test_metrics": {
                    "MRR": metric_mean(rows, "test_metrics", "MRR"),
                    "Recall@2": metric_mean(rows, "test_metrics", "Recall@2"),
                    "Recall@4": metric_mean(rows, "test_metrics", "Recall@4"),
                    "Recall@5": metric_mean(rows, "test_metrics", "Recall@5"),
                    "Recall@6": metric_mean(rows, "test_metrics", "Recall@6"),
                    "Recall@8": metric_mean(rows, "test_metrics", "Recall@8"),
                    "Recall@10": metric_mean(rows, "test_metrics", "Recall@10"),
                    "Recall@20": metric_mean(rows, "test_metrics", "Recall@20"),
                },
                "std_test_metrics": {
                    "MRR": stdev_or_zero([float(r["test_metrics"].get("MRR", 0.0)) for r in rows]),
                    "Recall@5": stdev_or_zero([float(r["test_metrics"].get("Recall@5", 0.0)) for r in rows]),
                    "Recall@10": stdev_or_zero([float(r["test_metrics"].get("Recall@10", 0.0)) for r in rows]),
                },
                "mean_val_focus_score": statistics.mean([float(r["val_focus_score"]) for r in rows]),
                "mean_test_focus_score": statistics.mean([float(r["test_focus_score"]) for r in rows]),
                "runs": rows,
            }
        )

    top_runs_by_recall5 = sorted(
        all_runs,
        key=lambda r: (
            float(r["test_metrics"].get("Recall@5", 0.0)),
            float(r["test_metrics"].get("Recall@10", 0.0)),
            float(r["test_metrics"].get("MRR", 0.0)),
        ),
        reverse=True,
    )
    top_runs_by_recall10 = sorted(
        all_runs,
        key=lambda r: (
            float(r["test_metrics"].get("Recall@10", 0.0)),
            float(r["test_metrics"].get("Recall@5", 0.0)),
            float(r["test_metrics"].get("MRR", 0.0)),
        ),
        reverse=True,
    )

    top_configs_by_peak_recall5 = sorted(
        config_groups,
        key=lambda g: (
            float(g["peak_test_metrics"]["Recall@5"]),
            float(g["peak_test_metrics"]["Recall@10"]),
            float(g["peak_test_metrics"]["MRR"]),
        ),
        reverse=True,
    )
    top_configs_by_mean_recall5 = sorted(
        config_groups,
        key=lambda g: (
            float(g["mean_test_metrics"]["Recall@5"]),
            float(g["mean_test_metrics"]["Recall@10"]),
            float(g["mean_test_metrics"]["MRR"]),
        ),
        reverse=True,
    )
    top_configs_by_mean_focus = sorted(
        config_groups,
        key=lambda g: (
            float(g["mean_test_focus_score"]),
            float(g["mean_test_metrics"]["Recall@10"]),
            float(g["mean_test_metrics"]["MRR"]),
        ),
        reverse=True,
    )

    report = {
        "selection_metric_during_training": args.selection_metric,
        "focus_metrics": focus_metrics,
        "residual_text_csv": args.residual_text_csv,
        "seeds": seeds,
        "num_configs": len(configs),
        "num_total_runs": len(configs) * len(seeds),
        "num_success_runs": len(all_runs),
        "num_failures": len(failures),
        "best_run_by_recall5": top_runs_by_recall5[0] if top_runs_by_recall5 else None,
        "best_run_by_recall10": top_runs_by_recall10[0] if top_runs_by_recall10 else None,
        "best_config_by_peak_recall5": top_configs_by_peak_recall5[0] if top_configs_by_peak_recall5 else None,
        "best_config_by_mean_recall5": top_configs_by_mean_recall5[0] if top_configs_by_mean_recall5 else None,
        "best_config_by_mean_focus": top_configs_by_mean_focus[0] if top_configs_by_mean_focus else None,
        "top_runs_by_recall5": top_runs_by_recall5[: args.top_n],
        "top_runs_by_recall10": top_runs_by_recall10[: args.top_n],
        "top_configs_by_peak_recall5": top_configs_by_peak_recall5[: args.top_n],
        "top_configs_by_mean_recall5": top_configs_by_mean_recall5[: args.top_n],
        "top_configs_by_mean_focus": top_configs_by_mean_focus[: args.top_n],
        "failures": failures,
    }

    with open(base_save_dir / "leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved leaderboard to {base_save_dir / 'leaderboard.json'}")
    if report["best_config_by_peak_recall5"] is not None:
        print(json.dumps(report["best_config_by_peak_recall5"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
