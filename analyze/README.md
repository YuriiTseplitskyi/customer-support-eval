# Analyze Module

## Aim
The `analyze` module evaluates support dialogues with an LLM-council pipeline and returns one structured result per dialogue.

It evaluates:
- customer intent
- agent quality score
- customer satisfaction
- agent mistakes

Primary entrypoint:
```bash
python -m analyze.run
```

## Prerequisites
- Python 3.12+
- installed dependencies from root `requirements.txt`
- `OPENAI_API_KEY` in environment or `.env`

Optional:
- `OPENAI_MODEL` (default: `gpt-4o-mini`)

## Default End-to-End Run (Generate + Analyze)
Use this as a working default pipeline from project root:

```bash
python -m generate.run --out datasets/dataset.jsonl --manifest results/generation/manifest.json
python -m analyze.run --dataset datasets/dataset.jsonl --output results/analysis/analysis_dataset.jsonl --overwrite
```

Why this is the default recommendation:
- generator default output path is not in `datasets/`, so we explicitly set `--out datasets/...`
- analyzer reads from `datasets/` by default, so these paths stay consistent

## Analyzer CLI Workflow
`analyze.run` performs:
1. Resolve input files (`--dataset`, `--all-datasets`, or default `datasets/*.json*`)
2. Load `.json`/`.jsonl`
3. Extract dialogue from `messages` or `turns`
4. Normalize roles to `CLIENT`/`AGENT`
5. Run councils/evaluators
6. Write JSONL output (default: `results/analysis_results.jsonl`)
7. Print dataset summary and any per-record errors

## Common Commands
Analyze all datasets from `datasets/` (default behavior):
```bash
python -m analyze.run
```

Analyze one dataset:
```bash
python -m analyze.run --dataset datasets/my_dataset.jsonl
```

Analyze one chat:
```bash
python -m analyze.run --dataset datasets/my_dataset.jsonl --chat-index 0
```

Analyze random sample (default size = 10):
```bash
python -m analyze.run --dataset datasets/my_dataset.jsonl --random-sample
```

Analyze random sample (custom size):
```bash
python -m analyze.run --dataset datasets/my_dataset.jsonl --random-sample 25
```

Write to custom output and overwrite previous content:
```bash
python -m analyze.run --dataset datasets/my_dataset.jsonl --output results/analysis/my_dataset_analysis.jsonl --overwrite
```

Analyze all datasets and fail on any record error:
```bash
python -m analyze.run --all-datasets --strict
```

## Key CLI Flags
- `--dataset`: analyze one dataset file
- `--all-datasets`: analyze all datasets in `datasets/`
- `--chat-index`: analyze one row by index (requires `--dataset`)
- `--limit`: max rows per dataset
- `--random-sample [N]`: analyze random sample per dataset; if value omitted, `N=10`
- `--output`: output JSONL path
- `--overwrite`: truncate output file before writing
- `--model`: model override for analyzer calls
- `--strict`: non-zero exit if any row fails

Flag compatibility:
- `--chat-index` cannot be combined with `--all-datasets`
- `--chat-index` cannot be combined with `--random-sample`
- `--limit` cannot be combined with `--random-sample`

## Council Structure
Intent council:
- 4 judges in parallel
- 1 aggregator
- open-set intent label (`snake_case`, not hardcoded enum)

Quality council:
- 3 reviewers in parallel (`correctness`, `resolution_effectiveness`, `professional_communication`)
- 1 aggregator
- output includes final score, per-lens breakdown, reasons, and mistake evidence

Satisfaction council:
- 2 reviewers in parallel (`sentiment_outcome`, `friction_effort`)
- 1 aggregator
- output includes final label and hidden dissatisfaction signals

Mistakes evaluator:
- single LLM detector
- Python post-filter rejects unsupported detections (for example missing evidence turns)

## Output Shape (High-Level)
Each analyzed chat outputs:
- `intent`
- `quality`
- `satisfaction`
- `mistakes`

This is written as one JSON object per line in the output JSONL file.
