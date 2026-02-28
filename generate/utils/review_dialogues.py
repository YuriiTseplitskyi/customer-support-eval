import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def print_stats(record: Dict[str, Any], idx: int, total: int) -> None:
    spec = record.get("generation_spec", {})
    gt = record.get("ground_truth", {})
    messages = record.get("messages", [])

    print("=" * 80)
    print(f"Record {idx + 1}/{total}")
    print(f"dialogue_id: {record.get('dialogue_id')}")
    print(f"scenario: {record.get('scenario')}")
    print(f"sub_scenario: {record.get('sub_scenario')}")
    print(f"complexity: {record.get('complexity')}")
    print(f"tags: {record.get('tags', [])}")
    print(f"message_count: {len(messages)}")
    print("-" * 80)
    print(f"outcome: {spec.get('outcome')}")
    print(f"conflict_level: {spec.get('conflict_level')}")
    print(f"mistakes_present: {spec.get('mistakes_present')}")
    print(f"num_mistakes: {spec.get('num_mistakes')}")
    print(f"agent_mistakes_main: {spec.get('agent_mistakes_main', [])}")
    print(f"agent_mistakes_sub: {spec.get('agent_mistakes_sub', [])}")
    print(f"hidden_dissatisfaction: {spec.get('hidden_dissatisfaction')}")
    print(f"agent_tone: {spec.get('agent_tone')}")
    print("-" * 80)
    print(f"ground_truth.intent: {gt.get('intent')}")
    print(f"ground_truth.satisfaction: {gt.get('satisfaction')}")
    print(f"ground_truth.quality_score: {gt.get('quality_score')}")
    print(f"ground_truth.hidden_dissatisfaction: {gt.get('hidden_dissatisfaction')}")
    print(f"ground_truth.agent_mistakes_main: {gt.get('agent_mistakes_main', [])}")
    print("=" * 80)


def print_dialogue(record: Dict[str, Any]) -> None:
    print("DIALOGUE")
    print("-" * 80)
    for i, msg in enumerate(record.get("messages", []), start=1):
        role = msg.get("role", "unknown")
        text = msg.get("text", "")
        print(f"{i:02d}. [{role}] {text}")
    print("-" * 80)


def review(records: List[Dict[str, Any]]) -> None:
    total = len(records)
    if total == 0:
        print("No records found.")
        return

    for idx, record in enumerate(records):
        print_stats(record, idx, total)
        print_dialogue(record)

        while True:
            answer = input("Enter 1-next, 0-stop: ").strip()
            if answer == "1":
                break
            if answer == "0":
                print("Stopped by user.")
                return
            print("Invalid input. Please enter 1 or 0.")

    print("Reached end of dataset.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive dialogue reviewer for JSONL datasets")
    parser.add_argument(
        "--dataset",
        default="generate/jsons/dataset_openai_10.jsonl",
        help="Path to dataset JSONL file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    records = load_jsonl(dataset_path)
    review(records)


if __name__ == "__main__":
    main()
