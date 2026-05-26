# Dual-Branch Web API Recommendation

Research code for a dual-branch Web API recommendation framework with:

- an explicit branch for residual-demand matching
- an implicit branch for complementary context modeling
- multiple text encoder backends
- multiple data split and negative sampling strategies
- joint and staged training variants for comparison

This repository contains source code only. Local datasets, generated texts, training outputs, and experiment archives are intentionally excluded from version control.

## Repository Structure

```text
dual_branch_main/
+-- src/               # core model, data pipeline, losses, metrics, training utilities
+-- scripts/           # data preparation, LLM text generation, tuning, and analysis helpers
+-- train.py           # main training entry
+-- requirements.txt   # Python dependencies
`-- README.md
```

## Environment

Recommended:

- Python 3.9+
- PyTorch 2.1+
- CUDA optional

Install dependencies:

```bash
pip install -r requirements.txt
```

## Expected Data Layout

The repository does not include the dataset. Place your local data under `data/` using the following layout:

```text
data/
+-- raw/
|   +-- Mashup_desc.csv
|   +-- API_desc.csv
|   +-- train.txt
|   `-- test.txt
+-- llm/
|   +-- mashup_expanded.csv      # optional
|   +-- api_expanded.csv         # optional
|   `-- residual_text.csv        # optional
`-- cache/
```

Required columns:

- `Mashup_desc.csv`: `mashup_id`, `description`, `mashup_name`
- `API_desc.csv`: `api_id`, `description`, `api_name`

## Quick Start

Default training command:

```bash
python train.py --data-dir data
```

Equivalent explicit command:

```bash
python train.py --data-dir data --training-mode joint
```

Common alternatives:

```bash
python train.py --data-dir data --training-mode staged
python train.py --data-dir data --training-mode no_stage2
python train.py --data-dir data --encoder-backend sbert --encoder-model all-MiniLM-L6-v2
python train.py --data-dir data --graph-mode topk --mashup-topk 20 --api-topk 20
```

By default, training outputs are written under `outputs/`, which is ignored by Git.

## Main Configurable Options

### Encoder backends

- `sbert`
- `llm_hf`
- `llm_api`

Examples:

```bash
python train.py --data-dir data --encoder-backend sbert --encoder-model all-MiniLM-L6-v2
python train.py --data-dir data --encoder-backend llm_hf --encoder-model BAAI/bge-large-en-v1.5 --max-length 512
python train.py --data-dir data --encoder-backend llm_api --encoder-model text-embedding-3-large --api-base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY
```

### Split modes

- `standard`
- `cold_mashup`
- `cold_api`

### Negative sampling modes

- `random`
- `strict_global`
- `hard`

### Fusion and ablation controls

- `--w-explicit`
- `--w-implicit`
- `--w-residual`
- `--w-tss`
- `--fusion-mode avg|learnable|mlp`

Examples:

```bash
python train.py --data-dir data --w-implicit 0
python train.py --data-dir data --w-tss 0
python train.py --data-dir data --fusion-mode avg
```

## Text Generation Utilities

The scripts directory includes helpers for building expanded text and residual text from raw descriptions.

Examples:

```bash
python scripts/build_expanded_text.py --input-csv data/raw/API_desc.csv --output-csv data/llm/api_expanded.csv --id-column api_id --name-column api_name --text-column description --entity-type api --mode existing
python scripts/build_residual_text.py --data-dir data --output-csv data/llm/residual_text.csv --mode template
```

For API-based generation, provide a compatible endpoint and key, for example via `OPENAI_API_KEY`.

## Notes

- This is a research prototype rather than a polished benchmark release.
- Exact reproducibility depends on your local dataset, preprocessing choices, and external model or API settings.
- The repository excludes datasets, generated text files, cached embeddings, logs, checkpoints, and experiment outputs.

## License

Add a license file if you plan to make reuse terms explicit.
