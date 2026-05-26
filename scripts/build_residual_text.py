from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm

from src.data_utils import generate_template_residual_text, load_optional_llm_csvs, load_tssgcf_style_data
from src.llm_generation import (
    ChatAPIConfig,
    LocalHFChatConfig,
    LocalHFChatGenerator,
    OpenAICompatibleChatGenerator,
)
from src.sampling import (
    build_cold_api_eval_queries,
    build_eval_queries,
    build_holdout_eval_queries,
    build_train_samples,
)
from src.splits import cold_api_split, cold_mashup_split, standard_split_from_provided


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--output-csv", type=str, required=True)
    p.add_argument("--mode", type=str, default="template", choices=["template", "api", "local_hf"])
    p.add_argument("--max-selected", type=int, default=3)
    p.add_argument("--max-subsets-per-positive", type=int, default=3)
    p.add_argument("--max-queries-per-mashup", type=int, default=6)
    p.add_argument("--split-mode", type=str, default="standard", choices=["standard", "cold_mashup", "cold_api"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cold-api-ratio", type=float, default=0.2)
    p.add_argument("--query-scope", type=str, default="split", choices=["split", "all_samples"])
    p.add_argument("--api-base-url", type=str, default="https://api.openai.com/v1")
    p.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    p.add_argument("--api-key", type=str, default="")
    p.add_argument("--model-name", type=str, default="gpt-4.1-mini")
    p.add_argument("--hf-token-env", type=str, default="HF_TOKEN")
    p.add_argument("--local-max-new-tokens", type=int, default=96)
    p.add_argument("--local-temperature", type=float, default=0.2)
    p.add_argument("--local-top-p", type=float, default=0.9)
    p.add_argument("--local-device-map", type=str, default="auto")
    p.add_argument("--local-torch-dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--local-trust-remote-code", action="store_true")
    p.add_argument("--max-mashup-chars", type=int, default=1200)
    p.add_argument("--max-api-chars", type=int, default=400)
    p.add_argument("--start-from", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--error-csv", type=str, default="")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def clip_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def load_completed_keys(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()
    df = pd.read_csv(output_csv)
    if "query_key" not in df.columns:
        return set()
    return {str(x) for x in df["query_key"].dropna().tolist()}


def append_rows(rows: list[dict], output_csv: Path) -> None:
    if not rows:
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["query_key", "mashup_id", "selected_api_ids", "residual_text"]
    write_header = not output_csv.exists() or output_csv.stat().st_size == 0
    with output_csv.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def append_error_row(row: dict, error_csv: Path | None) -> None:
    if error_csv is None:
        return
    error_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["query_key", "mashup_id", "selected_api_ids", "error"]
    write_header = not error_csv.exists() or error_csv.stat().st_size == 0
    with error_csv.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    output_csv = Path(args.output_csv)
    error_csv = Path(args.error_csv) if args.error_csv else output_csv.with_name(output_csv.stem + "_errors.csv")

    if args.overwrite and output_csv.exists():
        output_csv.unlink()
    if args.overwrite and error_csv.exists():
        error_csv.unlink()

    bundle = load_tssgcf_style_data(args.data_dir)
    llm_texts = load_optional_llm_csvs(args.data_dir, bundle["mid2idx"], bundle["aid2idx"])
    if args.query_scope == "all_samples":
        samples = build_train_samples(
            bundle["all_invocations"],
            sorted(bundle["all_invocations"].keys()),
            max_selected=args.max_selected,
            max_subsets_per_positive=args.max_subsets_per_positive,
        )
    else:
        if args.split_mode == "standard":
            split = standard_split_from_provided(
                bundle["train_invocations"],
                bundle["test_invocations"],
                val_ratio=0.125,
                seed=args.seed,
            )
        elif args.split_mode == "cold_mashup":
            split = cold_mashup_split(bundle["all_invocations"], seed=args.seed)
        else:
            split = cold_api_split(
                bundle["all_invocations"],
                bundle["api_freq"],
                cold_api_ratio=args.cold_api_ratio,
                seed=args.seed,
            )
        cold_apis = set(split.cold_apis or [])
        train_samples = build_train_samples(
            invocations=bundle["all_invocations"] if args.split_mode != "standard" else bundle["train_invocations"],
            mashup_ids=split.train_mashups,
            max_selected=args.max_selected,
            max_subsets_per_positive=args.max_subsets_per_positive,
            cold_apis=cold_apis if args.split_mode == "cold_api" else None,
        )
        if args.split_mode == "standard":
            import numpy as np

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
        elif args.split_mode == "cold_api":
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
        samples = train_samples + val_queries + test_queries

    generator = None
    system_prompt = None
    if args.mode == "api":
        generator = OpenAICompatibleChatGenerator(
            ChatAPIConfig(
                base_url=args.api_base_url,
                api_key_env=args.api_key_env,
                api_key=args.api_key,
                model_name=args.model_name,
            )
        )
        system_prompt = (
            "You infer the remaining unmet functional requirements of a mashup by comparing the mashup goal with "
            "the capabilities already covered by selected APIs.\n"
            "Your task is semantic subtraction: keep only capabilities that are still necessary and not yet covered.\n"
            "Rules:\n"
            "1. Do not restate the whole mashup requirement.\n"
            "2. Do not repeat selected API descriptions unless needed to clarify a missing capability.\n"
            "3. Do not mention candidate APIs or recommend products.\n"
            "4. Output only the still-missing capabilities, as a concise retrieval-friendly summary.\n"
            "5. Prefer short noun phrases or compact sentences about unmet functions, constraints, data needs, or integration needs.\n"
            "6. Keep the output under 80 words.\n"
            "7. If the selected APIs already appear to cover the mashup well, output: No clear unmet capability remains."
        )
    elif args.mode == "local_hf":
        generator = LocalHFChatGenerator(
            LocalHFChatConfig(
                model_name=args.model_name,
                hf_token_env=args.hf_token_env,
                max_new_tokens=args.local_max_new_tokens,
                temperature=args.local_temperature,
                top_p=args.local_top_p,
                device_map=args.local_device_map,
                trust_remote_code=args.local_trust_remote_code,
                torch_dtype=args.local_torch_dtype,
            )
        )
        system_prompt = (
            "You infer the remaining unmet functional requirements of a mashup by comparing the mashup goal with "
            "the capabilities already covered by selected APIs.\n"
            "Your task is semantic subtraction: keep only capabilities that are still necessary and not yet covered.\n"
            "Rules:\n"
            "1. Do not restate the whole mashup requirement.\n"
            "2. Do not repeat selected API descriptions unless needed to clarify a missing capability.\n"
            "3. Do not mention candidate APIs or recommend products.\n"
            "4. Output only the still-missing capabilities, as a concise retrieval-friendly summary.\n"
            "5. Prefer short noun phrases or compact sentences about unmet functions, constraints, data needs, or integration needs.\n"
            "6. Keep the output under 80 words.\n"
            "7. If the selected APIs already appear to cover the mashup well, output: No clear unmet capability remains."
        )

    mashup_df = bundle["mashup_df"]
    api_df = bundle["api_df"]
    rows_buffer: list[dict] = []
    existing_keys = load_completed_keys(output_csv)
    seen = set(existing_keys)
    skipped_existing = 0
    generated_count = 0
    failed_count = 0

    if args.start_from > 0:
        samples = samples[args.start_from:]
    if args.limit > 0:
        samples = samples[:args.limit]

    for s in tqdm(samples, desc="Build residual texts"):
        key = s["query_key"]
        if key in seen:
            skipped_existing += 1
            continue
        seen.add(key)
        mid = s["mashup_id"]
        selected = s["selected_api_ids"]
        mashup_name = str(mashup_df.iloc[mid]["mashup_name"])
        mashup_desc = clip_text(
            llm_texts["mashup_expanded"].get(mid, str(mashup_df.iloc[mid]["description"])),
            args.max_mashup_chars,
        )
        names = [str(api_df.iloc[a]["api_name"]) for a in selected]
        descs = [
            clip_text(
                llm_texts["api_expanded"].get(a, str(api_df.iloc[a]["description"])),
                args.max_api_chars,
            )
            for a in selected
        ]
        try:
            if args.mode in {"api", "local_hf"}:
                user_prompt = (
                    f"Mashup name: {mashup_name}\n"
                    f"Mashup goal / description:\n{mashup_desc}\n\n"
                    f"Selected APIs and already covered capabilities:\n"
                    + ("\n".join([f"- {n}: {d}" for n, d in zip(names, descs)]) if names else "- None")
                    + "\n\n"
                    "Instructions:\n"
                    "- Compare the mashup goal against the covered capabilities above.\n"
                    "- Remove capabilities that are already satisfied by the selected APIs.\n"
                    "- Keep only capabilities that are still missing but necessary for the mashup.\n"
                    "- Do not rewrite the full mashup description.\n"
                    "- Do not summarize selected APIs again.\n"
                    "- Return only a short missing-capability summary."
                )
                residual = generator.generate(system_prompt, user_prompt)
            else:
                residual = generate_template_residual_text(mashup_name, mashup_desc, names, descs)
        except Exception as e:
            failed_count += 1
            append_error_row(
                {
                    "query_key": key,
                    "mashup_id": mid,
                    "selected_api_ids": " ".join(map(str, selected)),
                    "error": str(e),
                },
                error_csv,
            )
            tqdm.write(f"[WARN] Failed query_key={key}: {e}")
            continue

        rows_buffer.append(
            {
                "query_key": key,
                "mashup_id": mid,
                "selected_api_ids": " ".join(map(str, selected)),
                "residual_text": residual,
            }
        )
        generated_count += 1

        if len(rows_buffer) >= max(1, args.save_every):
            append_rows(rows_buffer, output_csv)
            rows_buffer.clear()

    append_rows(rows_buffer, output_csv)
    print(f"Saved to {output_csv}")
    print(
        f"Generated {generated_count} rows, skipped {skipped_existing} existing rows, "
        f"failed {failed_count} rows."
    )
    if failed_count > 0:
        print(f"Failed rows logged to {error_csv}")


if __name__ == "__main__":
    main()
