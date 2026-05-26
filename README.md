# Dual-Branch Web API Recommendation v3

This project implements a more complete research prototype for your proposed framework:

- explicit branch: residual-demand matching
- implicit branch: candidate-aware complementary context modeling
- switchable text embedding backends:
  - `sbert`
  - `llm_hf`
  - `llm_api`
- switchable negative sampling:
  - `random`
  - `strict_global`
  - `hard`
- switchable text-similarity graph construction:
  - `threshold`
  - `topk`
- switchable split protocol:
  - `standard`
  - `cold_mashup`
  - `cold_api`
- ablation controls:
  - `--w-explicit`
  - `--w-implicit`
  - `--w-residual`
  - `--w-tss`
  - `--fusion-mode avg|learnable|mlp`

Two training variants are provided:
- `joint`: end-to-end training, recommended default for the current `standard` protocol
- `staged`: retained for ablation / comparison
- `no_stage2`: stage1 -> stage3 comparison path

## Environment

Recommended:
- Python 3.9+
- PyTorch 2.1+
- CUDA optional

Install:
```bash
pip install -r requirements.txt
```

## Data layout

```text
project_root/
├─ data/
│  ├─ raw/
│  │  ├─ Mashup_desc.csv
│  │  ├─ API_desc.csv
│  │  ├─ train.txt
│  │  └─ test.txt
│  ├─ llm/
│  │  ├─ mashup_expanded.csv   # optional
│  │  ├─ api_expanded.csv      # optional
│  │  └─ residual_text.csv     # optional
│  └─ cache/
├─ scripts/
├─ src/
├─ train.py
└─ requirements.txt
```

`Mashup_desc.csv` must contain:
- `mashup_id`
- `description`
- `mashup_name`

`API_desc.csv` must contain:
- `api_id`
- `description`
- `api_name`

## LLM text generation scripts

Build expanded text from existing descriptions:
```bash
python scripts/build_expanded_text.py   --input-csv data/raw/API_desc.csv   --output-csv data/llm/api_expanded.csv   --id-column api_id   --name-column api_name   --text-column description   --entity-type api   --mode existing
```

Call an OpenAI-compatible chat API to generate expanded text:
```bash
python scripts/build_expanded_text.py   --input-csv data/raw/API_desc.csv   --output-csv data/llm/api_expanded.csv   --id-column api_id   --name-column api_name   --text-column description   --entity-type api   --mode api   --api-base-url https://api.openai.com/v1   --api-key-env OPENAI_API_KEY   --model-name gpt-4.1-mini
```

Build residual text:
```bash
python scripts/build_residual_text.py   --data-dir data   --output-csv data/llm/residual_text.csv   --mode template
```

Or via chat API:
```bash
python scripts/build_residual_text.py   --data-dir data   --output-csv data/llm/residual_text.csv   --mode api   --api-base-url https://api.openai.com/v1   --api-key-env OPENAI_API_KEY   --model-name gpt-4.1-mini
```

## Text encoder backends

### SBERT
```bash
python train.py --data-dir data   --encoder-backend sbert   --encoder-model all-MiniLM-L6-v2
```

### Local HuggingFace embedding model
```bash
python train.py --data-dir data   --encoder-backend llm_hf   --encoder-model BAAI/bge-large-en-v1.5   --max-length 512
```

### OpenAI-compatible embedding API
```bash
python train.py --data-dir data   --encoder-backend llm_api   --encoder-model text-embedding-3-large   --api-base-url https://api.openai.com/v1   --api-key-env OPENAI_API_KEY
```

## Negative sampling modes

- `random`: sample any API not in the current mashup
- `strict_global`: current selected set `S` cannot be extended with the sampled API to match any observed positive composition in training
- `hard`: choose from the strict pool, biased toward text-similar negatives

## Split modes

- `standard`: warm-start hold-out evaluation with the provided `train.txt` / `test.txt`
  - matches the original TSSGCF strictness more closely:
  - test mashups remain the full `test.txt` mashup set
  - graph propagation and TSS regularization can use all standard-mode nodes
  - use this if you want direct protocol comparability with the released TSSGCF codebase
  - training supervision uses only `train.txt` on `train_mashups`
  - validation mashups are carved from `train.txt`; one observed train API is held out per validation mashup
  - final test uses `selected_api_ids = train.txt[mid]` and `target_api_ids = test.txt[mid]`
  - this mode is a warm-start protocol, not a cold-start protocol
- `cold_mashup`: merge all data and split by mashup
- `cold_api`: hold out a subset of APIs as cold APIs and evaluate on them

In other words, `standard` follows the usual observed-context -> held-out-target protocol, while
`cold_mashup` / `cold_api` use enumerated complementary-completion queries.

## Query / residual features

- query text embeddings are rebuilt per run from the current split's `train_samples + val_queries + test_queries`
- external `residual_text.csv` entries are restricted to the current split's query keys before encoding
- missing residual entries fall back to the template residual text already constructed from the mashup description and selected APIs
- `run_meta.json` records the query-feature coverage for the active split

If you want to regenerate residual text offline, prefer:
```bash
python scripts/build_residual_text.py --data-dir data --output-csv data/llm/residual_text.csv --query-scope split --split-mode standard --seed 42
```

## Graph modes

- `--graph-mode threshold`
  - build an undirected text-similarity graph by thresholding cosine similarity
- `--graph-mode topk`
  - keep top-k text neighbors per node, then symmetrize and normalize

Example:
```bash
python train.py --data-dir data --graph-mode topk --mashup-topk 20 --api-topk 20
```

## Ablation examples

Disable the implicit branch:
```bash
python train.py --data-dir data --w-implicit 0
```

Disable TSS:
```bash
python train.py --data-dir data --w-tss 0
```

Average fusion:
```bash
python train.py --data-dir data --fusion-mode avg
```

## Notes

This is a research prototype, not a fully benchmarked final release:
- it supports API-based text generation, but you still need a working external service key to actually generate texts
- hard negatives are currently static text-similarity hard negatives, not full online model-score mining
- cold API split is implemented, but you may still refine the exact protocol for your paper
- exact reproducibility on your dataset still depends on your data files and external LLM service settings

## Default training entry

This package now runs `joint` training by default:
```bash
python train.py --data-dir data
```

Equivalent explicit command:
```bash
python train.py --data-dir data --training-mode joint
```

Comparison runs:
```bash
python train.py --data-dir data --training-mode no_stage2
python train.py --data-dir data --training-mode staged
```

For the current codebase and the provided `standard` protocol, `joint` is the recommended main training route, while `staged` is kept mainly for comparison / ablation.
