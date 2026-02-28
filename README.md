# int20h: Synthetic Dialogue Generation + LLM Council Evaluation

## Overview
This project has two pipelines:
- `generate`: builds synthetic customer-support dialogue datasets.
- `analyze`: evaluates dialogues with LLM councils (intent, quality, satisfaction, mistakes).

Standard workflow:
1. Generate dataset JSONL.
2. Analyze dataset JSONL.
3. Inspect outputs in JSONL files.

## Setup
Requirements:
- Python 3.12+
- OpenAI API key

Install:
```bash
pip install -r requirements.txt
```

Environment (`.env` in project root):
```dotenv
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

## Quick Start (Working Defaults)
These commands are the recommended default flow for this repo:

```bash
python -m generate.run --out datasets/dataset.jsonl --manifest results/generation/manifest.json
python -m analyze.run --dataset datasets/dataset.jsonl --output results/analysis/analysis_dataset.jsonl --overwrite
```

Notes:
- Generator keeps default values for omitted flags: `--n 500`, `--seed 42`, `--max_retries 3`, `--model gpt-4o-mini`.
- Analyzer command above writes fresh results each run because of `--overwrite`.

## Generate 
Entrypoint:
```bash
python -m generate.run
```

What generation does:
- samples scenario/complexity/outcome/mistake specs
- calls LLM (or dummy mode)
- validates role alternation and message count
- writes:
  - dataset JSONL (`--out`)
  - generation manifest JSON (`--manifest`)

Common commands:
```bash
# explicit dataset + manifest paths (recommended)
python -m generate.run --n 200 --seed 42 --out datasets/my_dataset.jsonl --manifest results/generation/my_dataset_manifest.json

# smoke test dataset
python -m generate.run --n 10 --seed 43 --out datasets/sample_10.jsonl --manifest results/generation/sample_10_manifest.json

# dummy mode (no OpenAI calls)
python -m generate.run --n 50 --dummy_text --out datasets/dummy_50.jsonl --manifest results/generation/dummy_50_manifest.json
```

Important flags:
- `--n`: number of dialogues (default `500`)
- `--seed`: deterministic sampling seed (default `42`)
- `--out`: output dataset path
- `--manifest`: manifest path
- `--model`: OpenAI model (default from `OPENAI_MODEL` or `gpt-4o-mini`)
- `--temperature`: generation temperature (default `0.0`)
- `--max_retries`: retries per failed record (default `3`)
- `--allow_partial`: do not fail process if generated count is below requested `n`
- `--dummy_text`: skip LLM calls and write placeholder text

## Analyze 
Entrypoint:
```bash
python -m analyze.run
```

What analysis does:
- reads dataset records (`.json` or `.jsonl`)
- supports either `messages` or `turns`
- normalizes roles to `CLIENT`/`AGENT`
- runs councils/evaluators in parallel
- writes one JSONL result row per analyzed dialogue

Default behavior:
- `python -m analyze.run` analyzes all files in `./datasets`
- output file default is `results/analysis_results.jsonl`

Common commands:
```bash
# analyze one dataset
python -m analyze.run --dataset datasets/my_dataset.jsonl --output results/analysis/my_dataset_analysis.jsonl --overwrite

# analyze one chat
python -m analyze.run --dataset datasets/my_dataset.jsonl --chat-index 0

# analyze random sample (default size 10)
python -m analyze.run --dataset datasets/my_dataset.jsonl --random-sample

# analyze random sample (custom size)
python -m analyze.run --dataset datasets/my_dataset.jsonl --random-sample 25

# analyze all datasets with strict failure policy
python -m analyze.run --all-datasets --strict
```

Important flags:
- `--dataset`: analyze one dataset file
- `--all-datasets`: analyze all files in `datasets/`
- `--chat-index`: analyze one row by 0-based index (requires `--dataset`)
- `--limit`: max chats per dataset
- `--random-sample [N]`: random sample per dataset (`N=10` if no value provided)
- `--output`: output JSONL path
- `--overwrite`: truncate output file before writing
- `--model`: evaluator model (default from `OPENAI_MODEL` or `gpt-4o-mini`)
- `--strict`: fail command if any chat errors occur

## Project Structure
```text
|- requirements.txt
|- Dockerfile
|- docker-compose.yaml
|- datasets/
|- results/
|- generate/
|  |- run.py
|  |- config.py
|  |- README.md
|  `- utils/
|- analyze/
|  |- run.py
|  |- README.md
|  `- llm_council/
`- shared/
   |- chatgpt.py
   `- prompts/
```
