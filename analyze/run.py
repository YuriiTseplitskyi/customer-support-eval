import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

try:
    from analyze.llm_council import ChatEvaluatorAsync
except ModuleNotFoundError:
    from llm_council import ChatEvaluatorAsync  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM-council analysis for dataset files.")
    parser.add_argument(
        "--dataset",
        help="Path to one dataset file (.json or .jsonl). If omitted, analyzes all datasets in ./datasets.",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Analyze all files from ./datasets.",
    )
    parser.add_argument(
        "--chat-index",
        type=int,
        default=None,
        help="Analyze only one chat by 0-based index (valid only with --dataset).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of chats to analyze per dataset.",
    )
    parser.add_argument(
        "--output",
        default="results/analysis_results.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file instead of appending.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any record cannot be analyzed.",
    )
    return parser.parse_args()


def list_dataset_files(data_dir: Path) -> List[Path]:
    files = list(data_dir.glob("*.json")) + list(data_dir.glob("*.jsonl"))
    return sorted(files, key=lambda path: path.name.lower())


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON format in {path}")


def normalize_role(role: str) -> str:
    value = (role or "").strip().upper()
    if value in {"CLIENT", "CUSTOMER", "USER"}:
        return "CLIENT"
    if value in {"AGENT", "ASSISTANT", "SUPPORT"}:
        return "AGENT"
    return value


def extract_turns(record: Dict[str, Any]) -> List[Dict[str, str]]:
    raw_turns = record.get("turns")
    if not isinstance(raw_turns, list):
        raw_turns = record.get("messages")
    if not isinstance(raw_turns, list):
        return []

    turns: List[Dict[str, str]] = []
    for item in raw_turns:
        if not isinstance(item, dict):
            continue
        role = normalize_role(str(item.get("role", "")))
        text = str(item.get("text", "")).strip()
        if role not in {"CLIENT", "AGENT"} or not text:
            continue
        turns.append({"role": role, "text": text})
    return turns


def record_label(record: Dict[str, Any], index: int) -> str:
    return str(record.get("id") or record.get("dialogue_id") or f"row-{index}")


def summarize_result(result: Dict[str, Any], label: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    mistakes = result.get("mistakes", {}).get("agent_mistakes", []) or []
    summary = {
        "chat": label,
        "intent": result["intent"]["intent"],
        "quality_score": result["quality"]["quality_score"],
        "satisfaction": result["satisfaction"]["satisfaction"],
        "mistakes": ", ".join(mistakes) if mistakes else "-",
    }
    reasons = {
        "intent": result["intent"]["reason"],
        "quality": result["quality"]["reason"],
        "satisfaction": result["satisfaction"]["reason"],
        "mistakes": result.get("mistakes", {}).get("mistake_reasons", {}),
    }
    return summary, reasons


def build_output_record(dataset_name: str, mode: str, summary: Dict[str, Any], reasons: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        "mode": mode,
        **summary,
        "reasons": reasons,
    }


async def analyze_dataset(
    evaluator: ChatEvaluatorAsync,
    dataset_path: Path,
    chat_index: int | None,
    limit: int | None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    dataset = load_dataset(dataset_path)
    records: List[Dict[str, Any]] = []
    errors: List[str] = []
    mode = "single_cli" if chat_index is not None else "dataset_cli"

    if chat_index is not None and (chat_index < 0 or chat_index >= len(dataset)):
        errors.append(f"{dataset_path.name}: chat-index {chat_index} is out of range [0, {len(dataset)-1}]")
        return records, errors

    analyzed_count = 0
    for idx, record in enumerate(dataset):
        if chat_index is not None and idx != chat_index:
            continue
        if limit is not None and analyzed_count >= limit:
            break

        label = record_label(record, idx)
        turns = extract_turns(record)
        if not turns:
            errors.append(f"{dataset_path.name}:{label}: no valid turns")
            continue

        try:
            result = await evaluator.evaluate_async(turns)
            summary, reasons = summarize_result(result, label)
            records.append(build_output_record(dataset_path.name, mode, summary, reasons))
            analyzed_count += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{dataset_path.name}:{label}: {exc}")

    return records, errors


async def main() -> None:
    load_dotenv()
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in .env or environment variables.")

    if args.chat_index is not None and args.dataset is None:
        raise ValueError("--chat-index requires --dataset.")
    if args.chat_index is not None and args.all_datasets:
        raise ValueError("--chat-index cannot be used with --all-datasets.")

    data_dir = Path("datasets")
    dataset_paths: List[Path]
    if args.dataset:
        dataset_paths = [Path(args.dataset)]
    elif args.all_datasets:
        dataset_paths = list_dataset_files(data_dir)
    else:
        dataset_paths = list_dataset_files(data_dir)

    if not dataset_paths:
        raise RuntimeError("No dataset files found. Provide --dataset or place files in ./datasets.")

    missing = [str(path) for path in dataset_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Dataset file(s) not found: {missing}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        output_path.write_text("", encoding="utf-8")

    evaluator = ChatEvaluatorAsync(api_key=api_key, model_name=args.model)

    all_records: List[Dict[str, Any]] = []
    all_errors: List[str] = []
    for path in dataset_paths:
        records, errors = await analyze_dataset(
            evaluator=evaluator,
            dataset_path=path,
            chat_index=args.chat_index,
            limit=args.limit,
        )
        all_records.extend(records)
        all_errors.extend(errors)
        print(f"[DATASET] {path.name}: analyzed={len(records)} errors={len(errors)}")

    if all_records:
        with output_path.open("a", encoding="utf-8") as file:
            for record in all_records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[DONE] analyzed={len(all_records)} errors={len(all_errors)} output={output_path}")
    if all_errors:
        print("[ERRORS]")
        for err in all_errors:
            print(f"- {err}")

    if args.strict and all_errors:
        raise RuntimeError(f"Strict mode failed: {len(all_errors)} record(s) had errors.")


if __name__ == "__main__":
    asyncio.run(main())
