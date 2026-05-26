from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # 允许脚本直接从项目根目录解析 src 包。
    sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm

from src.llm_generation import ChatAPIConfig, OpenAICompatibleChatGenerator, save_rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", type=str, required=True)
    p.add_argument("--output-csv", type=str, required=True)
    p.add_argument("--id-column", type=str, default="")
    p.add_argument("--name-column", type=str, default="")
    p.add_argument("--text-column", type=str, default="description")
    p.add_argument("--entity-type", type=str, default="api", choices=["api", "mashup"])
    p.add_argument("--mode", type=str, default="existing", choices=["existing", "api"])
    p.add_argument("--api-base-url", type=str, default="https://api.openai.com/v1")
    p.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    p.add_argument("--api-key", type=str, default="")
    p.add_argument("--model-name", type=str, default="gpt-4.1-mini")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    # 允许用户不显式指定列名时做自动推断。
    id_col = args.id_column or df.columns[0]
    name_col = args.name_column or ("api_name" if "api_name" in df.columns else "mashup_name" if "mashup_name" in df.columns else df.columns[0])
    text_col = args.text_column
    rows = []

    generator = None
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
            "You expand short service descriptions into concise, factual, retrieval-friendly technical summaries. "
            "Do not invent unsupported features. Keep the output under 120 words."
        )

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Build expanded text"):
        entity_id = row[id_col]
        name = str(row[name_col])
        text = str(row[text_col])
        if args.mode == "api":
            user_prompt = f"Entity type: {args.entity_type}\nName: {name}\nOriginal description:\n{text}\n\nReturn an expanded but faithful description."
            expanded = generator.generate(system_prompt, user_prompt)
        else:
            # existing 模式相当于直接原样透传 description。
            expanded = text
        rows.append({id_col: entity_id, name_col: name, "expanded_text": expanded})
    save_rows(rows, args.output_csv)
    print(f"Saved to {args.output_csv}")


if __name__ == "__main__":
    main()
