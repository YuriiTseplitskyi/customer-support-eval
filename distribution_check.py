import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from generator_config import DISTR, SUB_SCENARIOS


ALLOWED_DEVIATION = 0.05


def build_targets_manifest() -> Dict[str, Any]:
    sub_global: Dict[str, float] = {}
    sub_uniform_within_scenario: Dict[str, Dict[str, float]] = {}
    for scenario, subs in SUB_SCENARIOS.items():
        p_scenario = DISTR["scenario"][scenario]
        per_sub_global = p_scenario / len(subs)
        sub_uniform_within_scenario[scenario] = {}
        for sub in subs:
            sub_global[sub] = per_sub_global
            sub_uniform_within_scenario[scenario][sub] = 1.0 / len(subs)

    return {
        "allowed_absolute_deviation": ALLOWED_DEVIATION,
        "desired_distributions": {
            "scenario": DISTR["scenario"],
            "sub_scenario_global": sub_global,
            "sub_scenario_within_scenario": sub_uniform_within_scenario,
            "complexity": DISTR["complexity"],
            "outcome": DISTR["outcome"],
            "conflict_level": DISTR["conflict_level"],
            "mistakes_present": {"false": 0.80, "true": 0.20},
            "num_mistakes_if_present": {"1": 0.60, "2": 0.30, "3": 0.10},
            "hidden_dissatisfaction_given_resolved": {"true": 0.15, "false": 0.85},
        },
    }


def bump(counter: Dict[str, int], key: Any) -> None:
    k = str(key)
    counter[k] = counter.get(k, 0) + 1


def normalized(counter: Dict[str, int], total: int) -> Dict[str, float]:
    if total <= 0:
        return {k: 0.0 for k in counter}
    return {k: v / total for k, v in counter.items()}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_achieved(dataset_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    c_scenario: Dict[str, int] = {}
    c_sub_scenario: Dict[str, int] = {}
    c_complexity: Dict[str, int] = {}
    c_outcome: Dict[str, int] = {}
    c_conflict: Dict[str, int] = {}
    c_mistakes_present: Dict[str, int] = {"true": 0, "false": 0}
    c_num_mistakes_if_present: Dict[str, int] = {}
    c_hidden_given_resolved: Dict[str, int] = {"true": 0, "false": 0}
    c_sub_within_scenario: Dict[str, Dict[str, int]] = {s: {} for s in SUB_SCENARIOS}

    n = len(dataset_rows)
    resolved_total = 0
    mistakes_present_total = 0

    for row in dataset_rows:
        spec = row["generation_spec"]
        scenario = row["scenario"]
        sub = row["sub_scenario"]
        bump(c_scenario, scenario)
        bump(c_sub_scenario, sub)
        bump(c_sub_within_scenario[scenario], sub)
        bump(c_complexity, row["complexity"])
        bump(c_outcome, spec["outcome"])
        bump(c_conflict, spec["conflict_level"])
        mp = str(spec["mistakes_present"]).lower()
        bump(c_mistakes_present, mp)
        if mp == "true":
            mistakes_present_total += 1
            # spec stores 3 as the capped bucket for 3+
            bump(c_num_mistakes_if_present, int(spec["num_mistakes"]))
        if spec["outcome"] == "resolved":
            resolved_total += 1
            bump(c_hidden_given_resolved, str(spec["hidden_dissatisfaction"]).lower())

    return {
        "n_records": n,
        "scenario": normalized(c_scenario, n),
        "sub_scenario_global": normalized(c_sub_scenario, n),
        "sub_scenario_within_scenario": {
            s: normalized(c_sub_within_scenario[s], sum(c_sub_within_scenario[s].values()))
            for s in c_sub_within_scenario
        },
        "complexity": normalized(c_complexity, n),
        "outcome": normalized(c_outcome, n),
        "conflict_level": normalized(c_conflict, n),
        "mistakes_present": normalized(c_mistakes_present, n),
        "num_mistakes_if_present": normalized(c_num_mistakes_if_present, mistakes_present_total),
        "hidden_dissatisfaction_given_resolved": normalized(c_hidden_given_resolved, resolved_total),
        "counts": {
            "resolved_total": resolved_total,
            "mistakes_present_total": mistakes_present_total,
        },
    }


def compare_flat_distribution(
    desired: Dict[str, float], achieved: Dict[str, float], threshold: float
) -> Dict[str, Dict[str, Any]]:
    keys = sorted(set(desired) | set(achieved))
    out: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        dv = float(desired.get(key, 0.0))
        av = float(achieved.get(key, 0.0))
        diff = abs(av - dv)
        out[key] = {
            "desired": dv,
            "achieved": av,
            "abs_diff": diff,
            "within_threshold": diff <= threshold,
            "status": "🟩" if diff <= threshold else "❌",
        }
    return out


def build_comparison(targets_manifest: Dict[str, Any], achieved: Dict[str, Any]) -> Dict[str, Any]:
    threshold = float(targets_manifest["allowed_absolute_deviation"])
    desired = targets_manifest["desired_distributions"]
    comparisons: Dict[str, Any] = {
        "threshold": threshold,
        "summary": {},
        "comparison": {},
    }

    flat_sections = [
        "scenario",
        "sub_scenario_global",
        "complexity",
        "outcome",
        "conflict_level",
        "mistakes_present",
        "num_mistakes_if_present",
        "hidden_dissatisfaction_given_resolved",
    ]
    for section in flat_sections:
        rows = compare_flat_distribution(desired[section], achieved.get(section, {}), threshold)
        comparisons["comparison"][section] = rows
        comparisons["summary"][section] = {
            "passed": all(r["within_threshold"] for r in rows.values()),
            "failed_items": [k for k, r in rows.items() if not r["within_threshold"]],
        }

    by_scenario: Dict[str, Any] = {}
    for scenario, desired_sub in desired["sub_scenario_within_scenario"].items():
        rows = compare_flat_distribution(
            desired_sub,
            achieved.get("sub_scenario_within_scenario", {}).get(scenario, {}),
            threshold,
        )
        by_scenario[scenario] = rows
    comparisons["comparison"]["sub_scenario_within_scenario"] = by_scenario
    comparisons["summary"]["sub_scenario_within_scenario"] = {
        "passed": all(
            r["within_threshold"]
            for section in by_scenario.values()
            for r in section.values()
        ),
        "failed_items": {
            s: [k for k, r in rows.items() if not r["within_threshold"]]
            for s, rows in by_scenario.items()
            if any(not r["within_threshold"] for r in rows.values())
        },
    }

    comparisons["achieved_meta"] = {
        "n_records": achieved["n_records"],
        **achieved["counts"],
    }
    return comparisons


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare desired vs achieved dataset distributions")
    p.add_argument("--dataset", default="dist_check_dataset.jsonl")
    p.add_argument("--targets_out", default="distribution_targets_manifest.json")
    p.add_argument("--achieved_out", default="dist_check_achieved.json")
    p.add_argument("--comparison_out", default="dist_check_comparison.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    targets = build_targets_manifest()
    Path(args.targets_out).write_text(json.dumps(targets, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = load_jsonl(Path(args.dataset))
    achieved = compute_achieved(rows)
    Path(args.achieved_out).write_text(json.dumps(achieved, ensure_ascii=False, indent=2), encoding="utf-8")

    comparison = build_comparison(targets, achieved)
    Path(args.comparison_out).write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
