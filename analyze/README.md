# Analyze Module

## Aim
The `analyze` folder evaluates support dialogues with an LLM-council pipeline.

It is used to:
- classify customer intent
- estimate agent quality score
- estimate customer satisfaction
- detect agent mistakes
- return one unified structured evaluation object

Primary runtime entrypoint: `analyze/run.py`.
Main API classes: `analyze.llm_council.ChatEvaluatorAsync` and `analyze.llm_council.ChatEvaluator`.

## Architecture
Core package: `analyze/llm_council`

Main components:
- engine: shared OpenAI client wrapper with built-in structured output parsing (`chat.completions.parse`).
- common types: transcript normalization and shared enums/types.
- intent council: intent classification with multi-judge aggregation.
- quality council: quality scoring with multi-reviewer aggregation.
- satisfaction council: customer satisfaction estimation with multi-reviewer aggregation.
- mistakes evaluator: mistake detection and evidence filtering.
- orchestrator: top-level coordinator that runs evaluators in parallel.

Prompt sources live in `shared/prompts`:
- `shared/prompts/intent_council`
- `shared/prompts/quality_council`
- `shared/prompts/satisfaction_council`
- `shared/prompts/analysis` (currently mistakes prompt)

## End-to-End Workflow
1. Input turns are validated and normalized into indexed transcript lines:
   - `[0] CLIENT: ...`
   - `[1] AGENT: ...`
2. Orchestrator (`ChatEvaluatorAsync.evaluate_async`) runs evaluators concurrently:
   - intent
   - quality
   - satisfaction
   - mistakes
3. Each evaluator uses structured output with Pydantic models.

## `run.py` CLI Workflow
`analyze/run.py` is a real dataset analyzer CLI (not a mock runner).

It does the following:
1. Resolves input datasets:
   - `--dataset <path>` for one file
   - `--all-datasets` for all files in `./datasets`
   - default behavior: analyze all files in `./datasets`
2. Loads dataset records from `.json` or `.jsonl`.
3. Extracts dialogue from either `turns` or `messages`.
4. Normalizes roles into `CLIENT` / `AGENT`.
5. Runs council evaluation per chat.
6. Writes JSONL records to output (default: `results/analysis_results.jsonl`).
7. Prints dataset-level summary and errors to stdout.

Supported important flags:
- `--chat-index` (single chat in one dataset)
- `--limit` (max chats per dataset)
- `--output` (custom output path)
- `--overwrite` (replace output file instead of append)
- `--model` (override model name)
- `--strict` (fail process if any chat failed)

## Councils and Structures

### 1) Intent Council
- Type: true multi-judge council
- Structure: `4 judges in parallel -> 1 aggregator`
- Judges:
  - `problem_statement`
  - `action_request`
  - `blocker`
  - `evidence_only`
- Aggregator inputs:
  - transcript
  - judge votes
  - judge reasons
- Final intent is open-set (not hardcoded taxonomy)

### 2) Quality Council
- Type: true multi-judge council
- Structure: `3 reviewers in parallel -> 1 aggregator`
- Reviewers:
  - `correctness`
  - `resolution_effectiveness`
  - `professional_communication`
- Each reviewer outputs:
  - `quality_score` (1..5)
  - `reason`
- Aggregator outputs:
  - final `quality_score`
  - `reason`
  - `breakdown`
  - `judge_reasons`
  - `agent_mistakes`
  - `evidence`
- Post-processing normalizes evidence turn indices and merges mistakes referenced by evidence.

### 3) Satisfaction Council
- Type: true multi-judge council
- Structure: `2 reviewers in parallel -> 1 aggregator`
- Reviewers:
  - `sentiment_outcome`
  - `friction_effort`
- Aggregator outputs:
  - `satisfaction` (`satisfied|neutral|unsatisfied`)
  - `reason`
  - `hidden_unsat_signals`
  - `evidence_turns`

### 4) Mistakes Evaluator
- Type: single-pass evaluator (not multi-judge council yet)
- Structure: `1 LLM detector -> Python gatekeeper`
- LLM outputs detected mistakes with reasons and evidence turns.
- Gatekeeper rules:
  - reject mistakes without evidence turns
  - deduplicate and sort evidence turns
  - return accepted mistakes + rejected map


## Final Output Schema
Top-level output from `ChatEvaluator*`:
- `intent`
- `quality`
- `satisfaction`
- `mistakes`

High-level shape:
```json
{
  "intent": {
    "intent": "snake_case_label",
    "reason": "...",
    "evidence_turns": [0, 2],
    "judge_votes": {"problem_statement": "..."},
    "judge_reasons": {"problem_statement": "..."}
  },
  "quality": {
    "quality_score": 4,
    "reason": "...",
    "breakdown": {
      "correctness": 4,
      "resolution_effectiveness": 4,
      "professional_communication": 5
    },
    "judge_reasons": {
      "correctness": "...",
      "resolution_effectiveness": "...",
      "professional_communication": "..."
    },
    "agent_mistakes": ["ignored_question"],
    "evidence": {"ignored_question": [3]}
  },
  "satisfaction": {
    "satisfaction": "neutral",
    "reason": "...",
    "hidden_unsat_signals": ["repeat_request"],
    "evidence_turns": [2, 4]
  },
  "mistakes": {
    "agent_mistakes": ["ignored_question"],
    "mistake_reasons": {"ignored_question": "..."},
    "evidence": {"ignored_question": [3]},
    "rejected": {}
  }
}
```

## Runtime Notes
- Required env var: `OPENAI_API_KEY`
- Optional env var: `OPENAI_MODEL` (default: `gpt-4o-mini`)
- Analyze all datasets from `./datasets`:
```bash
python analyze/run.py
```

- Analyze one dataset file:
```bash
python analyze/run.py --dataset datasets/my_dataset.jsonl
```

- Analyze one chat by index:
```bash
python analyze/run.py --dataset datasets/my_dataset.jsonl --chat-index 0
```

- Custom output file:
```bash
python analyze/run.py --all-datasets --output results/analysis_results.jsonl
```
