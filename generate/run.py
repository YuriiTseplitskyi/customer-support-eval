import argparse
import json
import os
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Sequence, Tuple

from pydantic import BaseModel, Field, ValidationInfo, field_validator

try:
    from generate.config import (
        DISTR,
        DIALOGUE_LENGTH_BOUNDS,
        GENERATOR_VERSION,
        MAJOR_MISTAKES,
        MISTAKE_CATEGORY_WEIGHTS,
        MISTAKE_POOLS,
        SCENARIO_MAIN_BIAS,
        SUB_SCENARIOS,
        SUB_TO_MAIN,
    )
except ModuleNotFoundError:
    from config import (  # type: ignore
        DISTR,
        DIALOGUE_LENGTH_BOUNDS,
        GENERATOR_VERSION,
        MAJOR_MISTAKES,
        MISTAKE_CATEGORY_WEIGHTS,
        MISTAKE_POOLS,
        SCENARIO_MAIN_BIAS,
        SUB_SCENARIOS,
        SUB_TO_MAIN,
    )

try:
    from shared.chatgpt import ChatGPTWrapper, load_dotenv_file
    from shared.prompts import GENERATION_SYSTEM_PROMPT
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from shared.chatgpt import ChatGPTWrapper, load_dotenv_file  # type: ignore
    from shared.prompts import GENERATION_SYSTEM_PROMPT  # type: ignore



TARGET_MISTAKES_PRESENT = 0.20
TARGET_OUTCOME_NOT_RESOLVED = DISTR["outcome"]["not_resolved"]
MISTAKES_PRESENT_IF_NOT_NOT_RESOLVED = (TARGET_MISTAKES_PRESENT - TARGET_OUTCOME_NOT_RESOLVED) / (
    1.0 - TARGET_OUTCOME_NOT_RESOLVED
)
NUM_MISTAKES_IF_NON_LOW = {"1": 0.20, "2": 0.60, "3": 0.20}
load_dotenv_file(".env")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class DialogueMessageModel(BaseModel):
    role: Literal["client", "agent"]
    text: str = Field(min_length=1, max_length=1200)


class DialogueJSONModel(BaseModel):
    messages: List[DialogueMessageModel]

    @field_validator("messages")
    @classmethod
    def validate_messages_shape(cls, v: List[DialogueMessageModel], info: ValidationInfo) -> List[DialogueMessageModel]:
        bounds = (info.context or {}).get("length_bounds", (1, 100))
        lo, hi = int(bounds[0]), int(bounds[1])
        if not (lo <= len(v) <= hi):
            raise ValueError(f"messages length must be in [{lo}, {hi}]")
        if not v:
            raise ValueError("messages must not be empty")
        if v[0].role != "client":
            raise ValueError("first message must be client")
        for i in range(1, len(v)):
            if v[i - 1].role == v[i].role:
                raise ValueError("messages must strictly alternate roles")
        return v


def make_rng(seed: int) -> random.Random:
    return random.Random(seed)


def weighted_choice(rng: random.Random, weights_dict: Dict[str, float]) -> str:
    total = sum(weights_dict.values())
    x = rng.random() * total
    cumulative = 0.0
    for key, weight in weights_dict.items():
        cumulative += weight
        if x <= cumulative:
            return key
    return next(reversed(weights_dict))


def weighted_choice_from_pairs(rng: random.Random, pairs: Sequence[Tuple[str, float]]) -> str:
    total = sum(w for _, w in pairs if w > 0)
    if total <= 0:
        return pairs[0][0]
    x = rng.random() * total
    cumulative = 0.0
    for key, weight in pairs:
        if weight <= 0:
            continue
        cumulative += weight
        if x <= cumulative:
            return key
    return pairs[-1][0]


def bernoulli(rng: random.Random, p: float) -> bool:
    return rng.random() < p


def choice(rng: random.Random, items: Sequence[str]) -> str:
    return items[rng.randrange(len(items))]


def flatten_mistakes() -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for category, subs in MISTAKE_POOLS.items():
        for sub in subs:
            rows.append((category, sub))
    return rows


ALL_SUB_MISTAKES = flatten_mistakes()
SUB_TO_CATEGORY = {sub: category for category, sub in ALL_SUB_MISTAKES}


def build_main_to_subs() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for _category, sub in ALL_SUB_MISTAKES:
        out.setdefault(SUB_TO_MAIN[sub], []).append(sub)
    return out


MAIN_TO_SUBS = build_main_to_subs()


def scenario_bias_multiplier(scenario: str, main_mistake: str) -> float:
    return SCENARIO_MAIN_BIAS.get(scenario, {}).get(main_mistake, 1.0)


def sample_sub_mistakes(rng: random.Random, k: int, scenario: str) -> List[str]:
    selected: List[str] = []
    used = set()
    while len(selected) < max(1, k):
        pairs: List[Tuple[str, float]] = []
        for category, sub in ALL_SUB_MISTAKES:
            if sub in used:
                continue
            weight = MISTAKE_CATEGORY_WEIGHTS[category] * scenario_bias_multiplier(scenario, SUB_TO_MAIN[sub])
            pairs.append((sub, weight))
        if not pairs:
            break
        sub = weighted_choice_from_pairs(rng, pairs)
        selected.append(sub)
        used.add(sub)
    return selected


def sample_distinct_main_mistakes(rng: random.Random, k: int, scenario: str) -> List[str]:
    selected: List[str] = []
    available = list(MAIN_TO_SUBS.keys())
    for _ in range(min(k, len(available))):
        pairs: List[Tuple[str, float]] = []
        for main in available:
            # Approximate base weight from all sub-mistake/category weights that map to this main.
            base = 0.0
            for sub in MAIN_TO_SUBS[main]:
                base += MISTAKE_CATEGORY_WEIGHTS[SUB_TO_CATEGORY[sub]]
            pairs.append((main, base * scenario_bias_multiplier(scenario, main)))
        main_pick = weighted_choice_from_pairs(rng, pairs)
        selected.append(main_pick)
        available.remove(main_pick)
    return selected


def sample_subs_for_selected_mains(rng: random.Random, selected_mains: List[str], scenario: str) -> List[str]:
    subs: List[str] = []
    used = set()
    for main in selected_mains:
        candidates = [s for s in MAIN_TO_SUBS[main] if s not in used]
        pairs: List[Tuple[str, float]] = []
        for sub in candidates:
            weight = MISTAKE_CATEGORY_WEIGHTS[SUB_TO_CATEGORY[sub]] * scenario_bias_multiplier(
                scenario, main
            )
            pairs.append((sub, weight))
        if not pairs:
            continue
        pick = weighted_choice_from_pairs(rng, pairs)
        subs.append(pick)
        used.add(pick)
    return subs


def unique_extend_main_mistakes(
    rng: random.Random, selected_sub: List[str], target_k: int, scenario: str
) -> Tuple[List[str], List[str]]:
    target_k = max(1, min(3, target_k))
    sub_list = list(selected_sub)
    used = set(sub_list)
    for _ in range(100):
        ordered_mains: List[str] = []
        seen_mains = set()
        for sub in sub_list:
            main = SUB_TO_MAIN[sub]
            if main not in seen_mains:
                ordered_mains.append(main)
                seen_mains.add(main)
        if len(ordered_mains) >= target_k:
            keep_subs: List[str] = []
            seen = set()
            for sub in sub_list:
                main = SUB_TO_MAIN[sub]
                if main in seen:
                    continue
                keep_subs.append(sub)
                seen.add(main)
                if len(keep_subs) == target_k:
                    break
            return keep_subs, [SUB_TO_MAIN[s] for s in keep_subs]
        extra = sample_sub_mistakes(rng, 1, scenario)[0]
        if extra in used:
            continue
        sub_list.append(extra)
        used.add(extra)
    dedup_subs: List[str] = []
    seen = set()
    for sub in sub_list:
        main = SUB_TO_MAIN[sub]
        if main in seen:
            continue
        dedup_subs.append(sub)
        seen.add(main)
        if len(dedup_subs) == target_k:
            break
    return dedup_subs, [SUB_TO_MAIN[s] for s in dedup_subs]


def sample_mistakes_bundle(rng: random.Random, spec: Dict[str, Any]) -> Tuple[bool, List[str], List[str], int]:
    if spec.get("outcome") == "not_resolved":
        mistakes_present = True
    else:
        mistakes_present = bernoulli(rng, MISTAKES_PRESENT_IF_NOT_NOT_RESOLVED)
    if not mistakes_present:
        return False, [], [], 0
    if spec.get("complexity") == "low":
        k = 1
    else:
        k = min(3, int(weighted_choice(rng, NUM_MISTAKES_IF_NON_LOW)))
    selected_mains = sample_distinct_main_mistakes(rng, k, spec["scenario"])
    selected_subs = sample_subs_for_selected_mains(rng, selected_mains, spec["scenario"])
    # Fallback if any main failed to receive a sub due to edge-case collisions.
    sub, main = unique_extend_main_mistakes(rng, selected_subs, k, spec["scenario"])
    return True, sub, main, len(main)


def resample_mistakes_only(rng: random.Random, spec: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(spec)
    present, subs, mains, num = sample_mistakes_bundle(rng, updated)
    updated["mistakes_present"] = present
    updated["agent_mistakes_sub"] = subs
    updated["agent_mistakes_main"] = mains
    updated["num_mistakes"] = num
    return updated


def apply_constraints_and_fix(rng: random.Random, spec: Dict[str, Any]) -> Dict[str, Any]:
    fixed = dict(spec)
    if fixed["hidden_dissatisfaction"] and fixed["outcome"] not in {"resolved", "escalated"}:
        fixed["hidden_dissatisfaction"] = False
    for _ in range(20):
        mains = fixed["agent_mistakes_main"]
        ok = True
        if fixed["outcome"] == "resolved" and "no_resolution" in mains:
            ok = False
        if fixed["conflict_level"] == "low" and "rude_tone" in mains:
            ok = False
        if fixed["outcome"] == "not_resolved" and not ({"no_resolution", "ignored_question"} & set(mains)):
            ok = False
        if fixed["complexity"] == "low" and len(mains) > 1:
            ok = False
        if fixed["mistakes_present"] and not mains:
            ok = False
        if (not fixed["mistakes_present"]) and mains:
            ok = False
        if ok:
            fixed["num_mistakes"] = len(mains)
            return fixed
        fixed = resample_mistakes_only(rng, fixed)
    fixed["num_mistakes"] = len(fixed["agent_mistakes_main"])
    return fixed


def derive_satisfaction(outcome: str, hidden: bool, main_mistakes: List[str]) -> str:
    if outcome == "resolved":
        satisfaction = "satisfied"
    elif outcome == "escalated":
        satisfaction = "neutral"
    else:
        satisfaction = "unsatisfied"
    has_major = any(m in MAJOR_MISTAKES for m in main_mistakes)
    if has_major:
        if outcome == "resolved":
            satisfaction = "neutral"
        else:
            satisfaction = "unsatisfied"
    if hidden:
        if outcome == "resolved":
            satisfaction = "neutral"
        else:
            satisfaction = "unsatisfied"
    return satisfaction


def derive_quality_score(
    outcome: str, hidden: bool, conflict_level: str, main_mistakes: List[str], satisfaction: str
) -> int:
    base = {"resolved": 5, "escalated": 3, "not_resolved": 2}[outcome]
    penalties = {
        "rude_tone": 2,
        "incorrect_info": 2,
        "no_resolution": 2,
        "ignored_question": 1,
        "unnecessary_escalation": 1,
    }
    score = base
    for m in set(main_mistakes):
        score -= penalties.get(m, 0)
    if hidden:
        score -= 1
    if conflict_level == "high" and ({"rude_tone", "ignored_question"} & set(main_mistakes)):
        score -= 1
    score = max(1, min(5, score))
    if satisfaction == "satisfied":
        score = max(score, 4)
    if satisfaction == "unsatisfied":
        score = min(score, 2)
    return score


def sample_generation_spec(rng: random.Random, dialogue_id: str) -> Dict[str, Any]:
    scenario = weighted_choice(rng, DISTR["scenario"])
    sub_scenario = choice(rng, SUB_SCENARIOS[scenario])
    complexity = weighted_choice(rng, DISTR["complexity"])
    outcome = weighted_choice(rng, DISTR["outcome"])
    conflict_level = weighted_choice(rng, DISTR["conflict_level"])
    agent_tone = weighted_choice(rng, DISTR["agent_tone"])
    spec: Dict[str, Any] = {
        "dialogue_id": dialogue_id,
        "scenario": scenario,
        "sub_scenario": sub_scenario,
        "complexity": complexity,
        "outcome": outcome,
        "conflict_level": conflict_level,
        "agent_tone": agent_tone,
        "hidden_dissatisfaction": False,
        "mistakes_present": False,
        "num_mistakes": 0,
        "agent_mistakes_sub": [],
        "agent_mistakes_main": [],
        "length_bounds": list(DIALOGUE_LENGTH_BOUNDS[complexity]),
        "length_target": dialogue_message_count(rng, complexity, False),
    }
    present, subs, mains, num = sample_mistakes_bundle(rng, spec)
    spec["mistakes_present"] = present
    spec["agent_mistakes_sub"] = subs
    spec["agent_mistakes_main"] = mains
    spec["num_mistakes"] = num
    spec["hidden_dissatisfaction"] = bernoulli(rng, 0.15) if outcome == "resolved" else False
    spec["length_target"] = dialogue_message_count(rng, complexity, spec["hidden_dissatisfaction"])
    spec = apply_constraints_and_fix(rng, spec)
    satisfaction = derive_satisfaction(spec["outcome"], spec["hidden_dissatisfaction"], spec["agent_mistakes_main"])
    quality_score = derive_quality_score(
        spec["outcome"],
        spec["hidden_dissatisfaction"],
        spec["conflict_level"],
        spec["agent_mistakes_main"],
        satisfaction,
    )
    spec["ground_truth"] = {
        "intent": spec["scenario"],
        "satisfaction": satisfaction,
        "hidden_dissatisfaction": spec["hidden_dissatisfaction"],
        "quality_score": quality_score,
        "agent_mistakes_main": list(spec["agent_mistakes_main"]),
    }
    return spec


def build_tags(spec: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    if spec["hidden_dissatisfaction"]:
        tags.append("hidden_dissatisfaction")
    if spec["mistakes_present"]:
        tags.append("agent_mistake_present")
    if spec["conflict_level"] != "low":
        tags.append("conflict")
    return tags


def dialogue_message_count(rng: random.Random, complexity: str, hidden: bool) -> int:
    lo, hi = DIALOGUE_LENGTH_BOUNDS[complexity]
    counts = list(range(lo, hi + 1))
    if hidden:
        odd_counts = [n for n in counts if n % 2 == 1]
        if odd_counts:
            counts = odd_counts
    return choice(rng, counts)


def build_user_prompt(spec: Dict[str, Any]) -> str:
    return (
        "Generate one dialogue JSON for this spec.\n"
        f"- scenario: {spec['scenario']}\n"
        f"- sub_scenario: {spec['sub_scenario']}\n"
        f"- complexity: {spec['complexity']}\n"
        f"- outcome: {spec['outcome']}\n"
        f"- conflict_level: {spec['conflict_level']}\n"
        f"- agent_tone: {spec['agent_tone']}\n"
        f"- hidden_dissatisfaction: {str(spec['hidden_dissatisfaction']).lower()}\n"
        f"- mistakes_present: {str(spec['mistakes_present']).lower()}\n"
        f"- agent_mistakes_sub: {spec['agent_mistakes_sub']}\n"
        f"- agent_mistakes_main: {spec['agent_mistakes_main']}\n"
        f"- target_message_count: {spec['length_target']}\n"
        "Behavior constraints (mandatory):\n"
        f"- EXACT message count required: {spec['length_target']}.\n"
        f"- Let N={spec['length_target']}. Internally plan turn indices 1..N, then output exactly N messages.\n"
        "- Respect outcome and conflict level in actual conversation flow.\n"
        "- Keep strict role alternation starting with client.\n"
        "- If hidden_dissatisfaction=true, make final client line neutral-positive.\n"
        "- If mistakes_present=true, reflect listed mistakes naturally in agent replies.\n"
        "- If mistakes_present=false, avoid mistake-like behavior.\n"
        "- Keep text concise and realistic for support chat.\n"
        "Output now as JSON only.\n"
    )


def generate_dummy_messages(spec: Dict[str, Any]) -> List[Dict[str, str]]:
    total_messages = spec["length_target"]
    messages: List[Dict[str, str]] = []
    for i in range(total_messages):
        role = "client" if i % 2 == 0 else "agent"
        messages.append({"role": role, "text": "dummy"})
    return messages


def generate_messages_from_spec(
    llm: ChatGPTWrapper | None,
    spec: Dict[str, Any],
    attempt: int = 0,
    dummy_text: bool = False,
    temperature: float = 0.0,
) -> List[Dict[str, str]]:
    if dummy_text:
        return generate_dummy_messages(spec)
    if llm is None:
        raise RuntimeError("ChatGPT client is required when --dummy_text is not set.")

    system_prompt = GENERATION_SYSTEM_PROMPT
    user_prompt = build_user_prompt(spec)
    model = llm.ask_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=DialogueJSONModel,
        temperature=temperature,
        # Allow outputs longer than target; retry logic/validator handles acceptance rules.
        validation_context={"length_bounds": (spec["length_target"], 200)},
    )
    return [{"role": m.role, "text": m.text} for m in model.messages]


def validate_messages(messages: Any, complexity: str) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not isinstance(messages, list):
        return False, ["messages_not_list"]
    if not messages:
        return False, ["messages_empty"]
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            reasons.append(f"bad_item_{i}")
            continue
        if m.get("role") not in {"client", "agent"}:
            reasons.append(f"bad_role_{i}")
        if not str(m.get("text", "")).strip():
            reasons.append(f"empty_text_{i}")
    if reasons:
        return False, reasons
    if messages[0]["role"] != "client":
        reasons.append("first_not_client")
    for i in range(1, len(messages)):
        if messages[i - 1]["role"] == "agent" and messages[i]["role"] == "agent":
            reasons.append("two_agents_in_row")
        if messages[i - 1]["role"] == "client" and messages[i]["role"] == "client":
            reasons.append("two_clients_in_row")
    lo, hi = DIALOGUE_LENGTH_BOUNDS[complexity]
    if not (lo <= len(messages) <= hi):
        reasons.append("length_out_of_bounds")
    return len(reasons) == 0, reasons


def assemble_record(spec: Dict[str, Any], messages: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "dialogue_id": spec["dialogue_id"],
        "scenario": spec["scenario"],
        "sub_scenario": spec["sub_scenario"],
        "complexity": spec["complexity"],
        "tags": build_tags(spec),
        "messages": messages,
        "generation_spec": {
            "dialogue_id": spec["dialogue_id"],
            "scenario": spec["scenario"],
            "sub_scenario": spec["sub_scenario"],
            "complexity": spec["complexity"],
            "outcome": spec["outcome"],
            "conflict_level": spec["conflict_level"],
            "mistakes_present": spec["mistakes_present"],
            "num_mistakes": spec["num_mistakes"],
            "agent_mistakes_sub": list(spec["agent_mistakes_sub"]),
            "agent_mistakes_main": list(spec["agent_mistakes_main"]),
            "hidden_dissatisfaction": spec["hidden_dissatisfaction"],
            "agent_tone": spec["agent_tone"],
            "length_bounds": list(spec["length_bounds"]),
            "length_target": spec["length_target"],
        },
        "ground_truth": {
            **spec["ground_truth"],
            "agent_mistakes": list(spec["ground_truth"]["agent_mistakes_main"]),
        },
    }


def build_manifest(
    seed: int,
    n_requested: int,
    n_generated: int,
    out_path: str,
    manifest_path: str,
    observed_stats: Dict[str, Any],
    generation_mode: str,
    model_name: str,
) -> Dict[str, Any]:
    return {
        "seed": seed,
        "generator_version": GENERATOR_VERSION,
        "distributions": DISTR,
        "n_records_requested": n_requested,
        "n_records_generated": n_generated,
        "dialogue_length_bounds": DIALOGUE_LENGTH_BOUNDS,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {"dataset_jsonl": out_path, "manifest_json": manifest_path},
        "observed_stats": observed_stats,
        "notes": {
            "language": "english",
            "retry_policy": "retry_on_validator_fail_with_fixed_spec",
            "validator": "role_format_turntaking_length",
            "deterministic_sampling": True,
            "generation_mode": generation_mode,
            "llm_model": model_name,
            "structured_output": True,
            "schema_validation": "pydantic",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic synthetic support dialogue generator")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="generate/jsons/dataset.jsonl")
    parser.add_argument("--manifest", default="generate/jsons/manifest.json")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--dummy_text", action="store_true", help="Write 'dummy' as every message text")
    parser.add_argument("--model", default=DEFAULT_OPENAI_MODEL, help="OpenAI model name")
    parser.add_argument("--api_key", default=None, help="OpenAI API key (fallback: OPENAI_API_KEY env)")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Do not raise if generated records are fewer than requested",
    )
    return parser.parse_args()


def init_observed_stats() -> Dict[str, Any]:
    return {
        "scenario": {},
        "complexity": {},
        "outcome": {},
        "conflict_level": {},
        "mistakes_present": {"true": 0, "false": 0},
        "num_mistakes": {},
        "hidden_dissatisfaction": {"true": 0, "false": 0},
        "satisfaction": {},
        "quality_score": {},
    }


def bump(counter: Dict[str, int], key: Any) -> None:
    k = str(key)
    counter[k] = counter.get(k, 0) + 1


def update_observed_stats(stats: Dict[str, Any], spec: Dict[str, Any]) -> None:
    bump(stats["scenario"], spec["scenario"])
    bump(stats["complexity"], spec["complexity"])
    bump(stats["outcome"], spec["outcome"])
    bump(stats["conflict_level"], spec["conflict_level"])
    bump(stats["mistakes_present"], str(spec["mistakes_present"]).lower())
    bump(stats["num_mistakes"], spec["num_mistakes"])
    bump(stats["hidden_dissatisfaction"], str(spec["hidden_dissatisfaction"]).lower())
    bump(stats["satisfaction"], spec["ground_truth"]["satisfaction"])
    bump(stats["quality_score"], spec["ground_truth"]["quality_score"])


def main() -> None:
    args = parse_args()
    rng = make_rng(args.seed)
    llm = None if args.dummy_text else ChatGPTWrapper(model_name=args.model, api_key=args.api_key)
    n_generated = 0
    failed: List[Dict[str, Any]] = []
    observed_stats = init_observed_stats()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as out_f:
        for i in range(args.n):
            dialogue_id = f"dlg_{i:06d}"
            spec = sample_generation_spec(rng, dialogue_id)
            print(
                f"[SPEC] {dialogue_id} "
                + json.dumps(
                    {
                        "scenario": spec["scenario"],
                        "sub_scenario": spec["sub_scenario"],
                        "complexity": spec["complexity"],
                        "outcome": spec["outcome"],
                        "conflict_level": spec["conflict_level"],
                        "mistakes_present": spec["mistakes_present"],
                        "num_mistakes": spec["num_mistakes"],
                        "agent_mistakes_sub": spec["agent_mistakes_sub"],
                        "agent_mistakes_main": spec["agent_mistakes_main"],
                        "hidden_dissatisfaction": spec["hidden_dissatisfaction"],
                        "agent_tone": spec["agent_tone"],
                        "length_bounds": spec["length_bounds"],
                        "length_target": spec["length_target"],
                    },
                    ensure_ascii=False,
                )
            )
            wrote = False
            for attempt in range(args.max_retries + 1):
                try:
                    messages = generate_messages_from_spec(
                        llm=llm,
                        spec=spec,
                        attempt=attempt,
                        dummy_text=args.dummy_text,
                        temperature=args.temperature,
                    )
                except Exception as e:
                    reasons = [f"llm_error:{type(e).__name__}"]
                    if attempt == args.max_retries:
                        failed.append({"dialogue_id": dialogue_id, "reasons": reasons})
                    continue
                ok, reasons = validate_messages(messages, spec["complexity"])
                # If model produced more turns than target, accept it as valid per review requirement.
                if (not ok) and ("length_out_of_bounds" in reasons) and (len(messages) > spec["length_target"]):
                    ok = True
                    reasons = []
                if ok:
                    record = assemble_record(spec, messages)
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_generated += 1
                    update_observed_stats(observed_stats, spec)
                    wrote = True
                    break
                if attempt == args.max_retries:
                    failed.append({"dialogue_id": dialogue_id, "reasons": reasons})
            if not wrote:
                continue

    generation_mode = "dummy" if args.dummy_text else "llm_chatgpt"
    manifest = build_manifest(
        args.seed,
        args.n,
        n_generated,
        args.out,
        args.manifest,
        observed_stats,
        generation_mode=generation_mode,
        model_name=args.model,
    )
    if failed:
        manifest["failed_records"] = failed
    with open(args.manifest, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)

    if not args.dummy_text and not args.allow_partial and n_generated != args.n:
        sample_reasons = [f.get("reasons", []) for f in failed[:3]]
        raise RuntimeError(
            f"LLM generation incomplete: generated {n_generated}/{args.n}. "
            f"Check OPENAI_API_KEY/model/access. Sample failure reasons: {sample_reasons}"
        )


if __name__ == "__main__":
    main()
