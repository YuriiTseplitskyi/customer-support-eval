# int20h: Synthetic Dialogue Generation + LLM Council Evaluation

## What This Project Does
This project has two core parts:
- `generate`: creates synthetic customer-support dialogue datasets.
- `analyze`: evaluates dialogues with an LLM-council pipeline.

Typical workflow:
1. Generate a dataset (`.jsonl`) with controlled scenario/quality distributions.
2. Run multi-council analysis for intent, quality score, satisfaction, and agent mistakes.
3. Review results in the Streamlit UI or consume structured output programmatically.

## Main Modules

### `generate`
Goal: deterministic synthetic dataset generation for support chats.

What it does:
- samples dialogue specs (scenario, complexity, outcome, conflict, mistakes)
- generates messages (LLM mode or dummy mode)
- validates structure (roles, alternation, length)
- writes dataset JSONL + manifest JSON

Entrypoint:
- `python -m generate.run ...`

Details:
- `generate/README.md`

### `analyze`
Goal: evaluate dialogue quality using LLM councils and structured outputs.

What it does:
- reads dataset files (`.json` / `.jsonl`)
- supports both dialogue formats (`turns` and `messages`)
- normalizes transcript
- runs evaluators in parallel:
  - intent council
  - quality council
  - satisfaction council
  - mistakes evaluator
- writes analysis records to JSONL output (default: `results/analysis_results.jsonl`)
- returns one unified structured result object per chat

Entrypoint:
- `python analyze/run.py`

Details:
- `analyze/README.md`

## How To Run

### Prerequisites
- Python 3.12+
- OpenAI API key

Install dependencies:
```bash
pip install -r requirements.txt
```

Environment variables:
```bash
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
```

### Generate dataset (CLI)
```bash
python -m generate.run --n 200 --seed 42 --out datasets/my_dataset.jsonl --manifest results/generation/my_dataset_manifest.json
```

Dummy generation (no LLM calls):
```bash
python -m generate.run --n 50 --seed 42 --dummy_text
```

### Analyze (CLI)
```bash
python analyze/run.py
```

Analyze one dataset file:
```bash
python analyze/run.py --dataset datasets/my_dataset.jsonl
```

Analyze one chat:
```bash
python analyze/run.py --dataset datasets/my_dataset.jsonl --chat-index 0
```

Custom output + strict mode:
```bash
python analyze/run.py --all-datasets --output results/analysis_results.jsonl --strict
```

### Run web app
```bash
streamlit run app.py
```

The app includes:
- Dataset Generation tab
- Dataset Analyze tab
- Dataset browser
- Saved results viewer

## Project Structure
```text
|- app.py                          # Streamlit UI integrating generate + analyze
|- requirements.txt                # Shared dependencies
|- Dockerfile
|- docker-compose.yaml
|- datasets/                       # Generated/uploaded datasets
|- results/                        # Analysis + generation artifacts
|- generate/
|  |- run.py                       # Generator CLI
|  |- config.py                    # Distributions, scenario/mistake config
|  |- utils/                       # Dataset utility scripts
|  `- README.md
|- analyze/
|  |- run.py                       # Analyze CLI (datasets -> results JSONL)
|  |- llm_council/
|  |  |- evaluator.py              # Top-level orchestrator
|  |  |- engine.py                 # OpenAI structured-output engine
|  |  |- intent.py                 # Intent council
|  |  |- quality.py                # Quality council
|  |  |- satisfaction.py           # Satisfaction council
|  |  |- mistakes.py               # Mistakes evaluator
|  |  `- common.py                 # Shared types/transcript normalization
|  `- README.md
`- shared/
   |- chatgpt.py                   # Shared OpenAI wrapper for generator
   `- prompts/
      |- generation.py
      |- analysis.py
      |- intent_council/
      |- quality_council/
      `- satisfaction_council/
```

## Output Summary
Main analyzer output shape:
- `intent`
- `quality`
- `satisfaction`
- `mistakes`
