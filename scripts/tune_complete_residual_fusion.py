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
        description="Coarse hyperparameter search for complete-residual fusion experiments."
    )
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--base-save-dir", type=str, default="outputs/tune_complete_residual_fusion")
    p.add_argument("--python-exe", type=str, default=sys.executable)
    p.add_argument("--encoder-model", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--residual-text-csv", type=str, default="data/llm/residual_text_deepseek.csv")
    p.add_argument("--search-seed", type=int, default=42)
    p.add_argument("--confirm-seeds", type=str, default="42,52,62")
    p.add_argument("--top-n", type=int, default=6)
    p.add_argument("--max-trials", type=int, default=0, help="0 means full search space.")
    p.add_argument("--selection-metric", type=str, default="Recall@10")
    p.add_argument("--eval-metrics", type=str, default="MRR,Recall@2,Recall@4,Recall@5,Recall@6,Recall@8,Recall@20")
    p.add_argument("--eval-ks", type=str, default="2,4,5,6,8,10,20")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_metric_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def build_search_space() -> list[dict[str, object]]:
    # Order matters because --max-trials truncates from the front.
    # The values below prioritize the region that already looked strong:
    # strict_global + mlp + lambda_exp around 0.2 + batch 256/128.
    grid = {
        "lr": [5e-4],
        "dropout": [0.1],
        "emb-dim": [128],
        "n-layers": [2],
        "lambda-exp": [0.2, 0.1, 0.3, 0.4],
        "lambda-imp": [0.0, 0.1],
        "lambda-tss": [0.25, 0.1],
        "negative-mode": ["strict_global"],
        "fusion-mode": ["mlp"],
        "batch-size": [256, 128],
        "joint-loss-variant": ["full", "no_imp_aux"],
        "context-loss-type": ["bpr", "bce"],
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
        f"t{idx:03d}"
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


def main() -> None:
    args = parse_args()
    metric_names = parse_metric_list(args.eval_metrics)
    confirm_seeds = parse_int_list(args.confirm_seeds)

    base_save_dir = (ROOT / args.base_save_dir).resolve()
    search_dir = base_save_dir / "search_trials"
    confirm_dir = base_save_dir / "confirm_trials"
    search_dir.mkdir(parents=True, exist_ok=True)
    confirm_dir.mkdir(parents=True, exist_ok=True)

    trials = build_search_space()
    if args.max_trials > 0:
        trials = trials[: args.max_trials]

    search_results = []
    failures = []

    for idx, cfg in enumerate(trials, start=1):
        trial_name = config_to_name(idx, cfg)
        save_dir = search_dir / f"{trial_name}__seed_{args.search_seed}"
        cmd = build_command(args, cfg, args.search_seed, save_dir)
        print(f"[search {idx}/{len(trials)}] {trial_name}")
        ok = run_trial(cmd, save_dir, args.dry_run)
        if args.dry_run:
            continue
        if not ok:
            failures.append({"trial": trial_name, "phase": "search", "seed": args.search_seed, "config": cfg})
            continue
        summary = load_summary(save_dir)
        best_val_metrics = summary.get("best_val_metrics", {})
        search_results.append(
            {
                "trial": trial_name,
                "config": cfg,
                "save_dir": str(save_dir),
                "seed": args.search_seed,
                "best_val_metric": float(summary.get("best_val_metric", summary.get("best_val_recall10", 0.0))),
                "best_val_metrics": best_val_metrics,
                "val_focus_score": mean_metric(best_val_metrics, metric_names),
                "test_metrics": summary["test_metrics"],
                "test_focus_score": mean_metric(summary["test_metrics"], metric_names),
            }
        )

    if args.dry_run:
        return

    search_results.sort(
        key=lambda x: (
            x["val_focus_score"],
            x["best_val_metrics"].get("Recall@10", 0.0),
            x["best_val_metrics"].get("MRR", 0.0),
        ),
        reverse=True,
    )
    top_trials = search_results[: args.top_n]

    confirm_results = []
    for rank, item in enumerate(top_trials, start=1):
        cfg = item["config"]
        trial_name = item["trial"]
        for seed in confirm_seeds:
            save_dir = confirm_dir / f"{trial_name}__seed_{seed}"
            cmd = build_command(args, cfg, seed, save_dir)
            print(f"[confirm {rank}/{len(top_trials)} seed={seed}] {trial_name}")
            ok = run_trial(cmd, save_dir, False)
            if not ok:
                failures.append({"trial": trial_name, "phase": "confirm", "seed": seed, "config": cfg})
                continue
            summary = load_summary(save_dir)
            best_val_metrics = summary.get("best_val_metrics", {})
            confirm_results.append(
                {
                    "trial": trial_name,
                    "config": cfg,
                    "save_dir": str(save_dir),
                    "seed": seed,
                    "best_val_metric": float(summary.get("best_val_metric", summary.get("best_val_recall10", 0.0))),
                    "best_val_metrics": best_val_metrics,
                    "val_focus_score": mean_metric(best_val_metrics, metric_names),
                    "test_metrics": summary["test_metrics"],
                    "test_focus_score": mean_metric(summary["test_metrics"], metric_names),
                }
            )

    aggregate = []
    for item in top_trials:
        trial_name = item["trial"]
        rows = [r for r in confirm_results if r["trial"] == trial_name]
        if not rows:
            continue
        agg = {
            "trial": trial_name,
            "config": item["config"],
            "seeds": [r["seed"] for r in rows],
            "val_focus_score_mean": statistics.mean([r["val_focus_score"] for r in rows]),
            "val_focus_score_std": stdev_or_zero([r["val_focus_score"] for r in rows]),
            "test_focus_score_mean": statistics.mean([r["test_focus_score"] for r in rows]),
            "test_focus_score_std": stdev_or_zero([r["test_focus_score"] for r in rows]),
            "val_metrics_mean": {},
            "test_metrics_mean": {},
        }
        for metric in metric_names:
            agg["val_metrics_mean"][metric] = statistics.mean(
                [float(r["best_val_metrics"].get(metric, 0.0)) for r in rows]
            )
            agg["test_metrics_mean"][metric] = statistics.mean(
                [float(r["test_metrics"].get(metric, 0.0)) for r in rows]
            )
        aggregate.append(agg)

    aggregate.sort(
        key=lambda x: (
            x["val_focus_score_mean"],
            x["val_metrics_mean"].get("Recall@10", 0.0),
            x["val_metrics_mean"].get("MRR", 0.0),
        ),
        reverse=True,
    )

    report = {
        "selection_metric_during_training": args.selection_metric,
        "focus_metrics": metric_names,
        "residual_text_csv": args.residual_text_csv,
        "search_seed": args.search_seed,
        "confirm_seeds": confirm_seeds,
        "num_trials": len(trials),
        "num_success_search": len(search_results),
        "num_success_confirm": len(confirm_results),
        "num_failures": len(failures),
        "best_trial_search": search_results[0] if search_results else None,
        "best_trial_confirm": aggregate[0] if aggregate else None,
        "top_trials_search": search_results[: args.top_n],
        "top_trials_confirm": aggregate[: args.top_n],
        "failures": failures,
    }

    with open(base_save_dir / "leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved leaderboard to {base_save_dir / 'leaderboard.json'}")
    if aggregate:
        print(json.dumps(aggregate[0], indent=2, ensure_ascii=False))
    elif search_results:
        print(json.dumps(search_results[0], indent=2, ensure_ascii=False))
    else:
        print("No successful trials.")


if __name__ == "__main__":
    main()
