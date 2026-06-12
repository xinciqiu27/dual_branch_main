# Complementarity-Aware Web API Recommendation via Explicit and Implicit Requirement Modeling

Code for the paper "Complementarity-Aware Web API Recommendation via Explicit and Implicit Requirement Modeling".

The task is to recommend additional Web APIs that are both relevant to the Mashup requirement and complementary to the already selected Web API set.

## Install

```bash
pip install -r requirements.txt
```

## Data

Prepare:

```text
data/
+-- raw/
|   +-- Mashup_desc.csv
|   +-- API_desc.csv
|   +-- train.txt
|   `-- test.txt
`-- llm/
```

## Default Pipeline

1. Set API key for selected-API-aware residual-text generation.

PowerShell:

```powershell
$env:YOUR_API_KEY_ENV="your_key_here"
```

2. Generate a selected-API-aware residual-text file, e.g. `data/llm/residual_text_llm.csv`.

```bash
python scripts/build_residual_text.py --data-dir data --output-csv data/llm/residual_text_llm.csv --mode api --model-name your-chat-model --api-key-env YOUR_API_KEY_ENV --api-base-url https://your-api-base/v1
```

This step calls an OpenAI-compatible chat API to infer supplementary capability information for each query from the Mashup description and the already selected Web APIs.

3. Run the default training command.

```bash
python train.py --data-dir data --residual-text-csv data/llm/residual_text_llm.csv
```

The generated residual-text file is used by the explicit requirement modeling component in the default pipeline.
