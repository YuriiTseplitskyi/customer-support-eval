import json
from pathlib import Path


def count_jsonl_lines(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> int:
    base = Path(__file__).resolve().parents[1] / "jsons"
    manifests = sorted(base.glob("manifest*.json"))

    if not manifests:
        print("No manifest*.json files found.")
        return 1

    total = len(manifests)
    failed = []

    print(f"Scanned folder: {base}")
    print(f"Found manifests: {total}\n")

    for manifest_path in manifests:
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        requested = data.get("n_records_requested")
        generated = data.get("n_records_generated")
        failed_records = data.get("failed_records") or []

        outputs = data.get("outputs") or {}
        dataset_path_raw = outputs.get("dataset_jsonl")
        dataset_lines = None
        dataset_exists = None

        if dataset_path_raw:
            dataset_path = Path(dataset_path_raw)
            if not dataset_path.is_absolute():
                dataset_path = (base.parent.parent / dataset_path).resolve()
            dataset_exists = dataset_path.exists()
            dataset_lines = count_jsonl_lines(dataset_path) if dataset_exists else None
        else:
            dataset_path = None

        issues = []

        if isinstance(requested, int) and isinstance(generated, int) and generated < requested:
            issues.append(f"generated<{requested} (actual {generated})")

        if failed_records:
            issues.append(f"failed_records={len(failed_records)}")

        if dataset_path_raw and dataset_exists is False:
            issues.append("dataset_jsonl_missing")

        if isinstance(generated, int) and isinstance(dataset_lines, int) and dataset_lines != generated:
            issues.append(f"dataset_lines_mismatch (lines={dataset_lines}, generated={generated})")

        print(f"- {manifest_path.name}")
        print(f"  requested={requested}, generated={generated}, failed_records={len(failed_records)}")
        if dataset_path_raw:
            print(f"  dataset={dataset_path_raw}")
            print(f"  dataset_exists={dataset_exists}, dataset_lines={dataset_lines}")

        if issues:
            print(f"  status=FAILED ({'; '.join(issues)})")
            failed.append(
                {
                    "manifest": manifest_path.name,
                    "requested": requested,
                    "generated": generated,
                    "failed_records": len(failed_records),
                    "issues": issues,
                }
            )
        else:
            print("  status=OK")

    print("\nSummary")
    print(f"- manifests_scanned={total}")
    print(f"- manifests_with_issues={len(failed)}")
    print(f"- manifests_ok={total - len(failed)}")

    if failed:
        print("\nFailed manifests:")
        for item in failed:
            print(
                f"- {item['manifest']}: requested={item['requested']}, "
                f"generated={item['generated']}, failed_records={item['failed_records']}, "
                f"issues={', '.join(item['issues'])}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
