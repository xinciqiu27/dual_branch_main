from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hyperparameter search for the SBERT-based dual-branch model.")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--base-save-dir", type=str, default="outputs/tuning_sbert")
    p.add_argument("--python-exe", type=str, default=sys.executable)
    p.add_argument("--encoder-model", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--max-trials", type=int, default=0, help="0 means run the full search space.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_search_space() -> list[dict[str, object]]:
    # 这里是穷举网格搜索，不是贝叶斯优化或随机搜索。
    grid = {
        "lr": [5e-4, 1e-3],
        "dropout": [0.1, 0.2],
        "emb-dim": [128, 256],
        "n-layers": [2, 3],
        "lambda-exp": [0.25, 0.5],
        "lambda-imp": [0.25, 0.5],
        "lambda-tss": [0.1, 0.25],
        "negative-mode": ["strict_global", "hard"],
        "fusion-mode": ["avg", "learnable"],
        "batch-size": [128, 256],
        "stage1-epochs": [10],
        "stage2-epochs": [10],
        "stage3-epochs": [20],
        "seed": [42],
    }
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    trials = []
    for combo in itertools.product(*values):
        trials.append(dict(zip(keys, combo)))
    return trials


def config_to_name(idx: int, cfg: dict[str, object]) -> str:
    # 把关键超参编码进目录名，方便人工回看实验结果。
    parts = [
        f"trial_{idx:03d}",
        f"lr_{cfg['lr']}",
        f"dim_{cfg['emb-dim']}",
        f"layer_{cfg['n-layers']}",
        f"drop_{cfg['dropout']}",
        f"bs_{cfg['batch-size']}",
        f"tss_{cfg['lambda-tss']}",
        f"neg_{cfg['negative-mode']}",
        f"fuse_{cfg['fusion-mode']}",
    ]
    return "__".join(str(x).replace("/", "_") for x in parts)


def build_command(args: argparse.Namespace, cfg: dict[str, object], save_dir: Path) -> list[str]:
    cmd = [
        args.python_exe,
        "train.py",
        "--data-dir",
        args.data_dir,
        "--save-dir",
        str(save_dir),
        "--encoder-backend",
        "sbert",
        "--encoder-model",
        args.encoder_model,
    ]
    for key, value in cfg.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def load_summary(save_dir: Path) -> dict[str, object]:
    with open(save_dir / "summary.json", "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    base_save_dir = (ROOT / args.base_save_dir).resolve()
    base_save_dir.mkdir(parents=True, exist_ok=True)

    trials = build_search_space()
    if args.max_trials > 0:
        trials = trials[: args.max_trials]

    results = []
    failures = []
    for idx, cfg in enumerate(trials, start=1):
        trial_name = config_to_name(idx, cfg)
        save_dir = base_save_dir / trial_name
        cmd = build_command(args, cfg, save_dir)
        print(f"[{idx}/{len(trials)}] {trial_name}")
        if args.dry_run:
            print(" ".join(cmd))
            continue
        # 每个 trial 都通过子进程调用 train.py，彼此相互独立。
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0:
            failures.append({"trial": trial_name, "returncode": proc.returncode, "config": cfg})
            continue
        try:
            summary = load_summary(save_dir)
        except FileNotFoundError:
            failures.append({"trial": trial_name, "returncode": "missing_summary", "config": cfg})
            continue
        results.append(
            {
                "trial": trial_name,
                "save_dir": str(save_dir),
                "config": cfg,
                "best_val_recall10": float(summary["best_val_recall10"]),
                "best_epoch": int(summary["best_epoch"]),
                "best_stage": summary.get("best_stage", ""),
                "test_metrics": summary["test_metrics"],
            }
        )

    if args.dry_run:
        return

    results.sort(key=lambda x: x["best_val_recall10"], reverse=True)
    leaderboard = {
        "selection_metric": "best_val_recall10",
        "encoder_backend": "sbert",
        "encoder_model": args.encoder_model,
        "num_trials": len(trials),
        "num_success": len(results),
        "num_failures": len(failures),
        "best_trial": results[0] if results else None,
        "top_trials": results[: args.top_n],
        "failures": failures,
    }
    with open(base_save_dir / "leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2, ensure_ascii=False)

    print(f"Saved leaderboard to {base_save_dir / 'leaderboard.json'}")
    if results:
        print(json.dumps(results[0], indent=2, ensure_ascii=False))
    else:
        print("No successful trials.")


if __name__ == "__main__":
    main()
