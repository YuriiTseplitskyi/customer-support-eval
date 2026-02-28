# Generate Module

## Aim
The `generate` folder builds a synthetic customer-support dialogue dataset in JSONL format.

It is used to:
- sample dialogue scenarios with controlled distributions
- generate dialogues with either an LLM (`ChatGPTWrapper`) or dummy text
- attach `generation_spec` and `ground_truth` labels
- save a manifest with sampling setup and observed counts

Primary entrypoint: `generate/run.py`.

## Generation Process
1. Parse CLI args (`n`, `seed`, output paths, retries, model, temperature).
2. Initialize deterministic RNG from `seed`.
3. For each dialogue id:
- sample `scenario` and `sub_scenario`
- sample `complexity`, `outcome`, `conflict_level`, `agent_tone`
- sample mistake bundle (or force mistakes for `not_resolved`)
- sample `hidden_dissatisfaction` (15% only when `outcome=resolved`)
- apply consistency constraints (resample mistakes if needed)
- build `ground_truth`
4. Generate messages:
- LLM mode (`ChatGPTWrapper`) with structured validation, or
- dummy mode (`--dummy_text`)
5. Validate messages (roles, alternation, non-empty text, length bounds).
6. Save valid record to JSONL and update observed stats.
7. Write manifest JSON with distributions, stats, generator version, and failures.

## Dimensions

### Dimensions meaning
| Dimension | What it means |
|---|---|
| `scenario` | Main customer issue type for the dialogue. |
| `complexity` | How difficult the case is, which also controls expected dialogue length. |
| `outcome` | Final result of support handling (`resolved`, `not_resolved`, `escalated`). |
| `conflict_level` | How tense or confrontational the conversation is. |
| `mistakes_present` | Whether the agent makes any mistakes in the dialogue. |
| `num_mistakes_if_present` | How many distinct main mistake types are injected when mistakes are enabled. |
| `agent_tone` | Target communication tone of the agent (`polite` or `neutral`). |
| `hidden_dissatisfaction` | Whether the client appears neutral-positive at the end but is still not truly satisfied. |
| `dialogue length by complexity` | Allowed message-count range for each complexity level. |
| `satisfaction` | Label in `ground_truth` representing user-level outcome quality. |

### Dimension target distributions
| Dimension | Values (target) |
|---|---|
| `scenario` | `tariff_question` 30%, `payment_issue` 25%, `technical_issue` 20%, `account_access` 15%, `refund_request` 10% |
| `complexity` | `low` 50%, `medium` 35%, `high` 15% |
| `outcome` | `resolved` 75%, `not_resolved` 15%, `escalated` 10% |
| `conflict_level` | `low` 70%, `medium` 20%, `high` 10% |
| `mistakes_present` | `false` 80%, `true` 20% (implemented with conditioning to keep overall target) |
| `num_mistakes_if_present` | `1` 60%, `2` 30%, `3` 10% (non-low complexity uses tuned 20/60/20; low complexity capped at 1) |
| `agent_tone` | `polite` 60%, `neutral` 40% |
| `hidden_dissatisfaction` | 15% of `resolved` dialogues only |
| `dialogue length by complexity` | `low`: 3-5, `medium`: 6-9, `high`: 10-13 messages |
| `satisfaction` | `satisfied` 65%, `neutral` 15%, `unsatisfied` 20% |

## Scenario Catalog

### `payment_issue` (25%)
- `double charge`
- `payment failed`
- `charge without confirmation`
- `incorrect charged amount`
- `payment processing too long`
- `promo code issue`
- `charge after subscription cancellation`

### `technical_issue` (20%)
- `app does not open`
- `error 500 or other system error`
- `specific feature not working`
- `data not updating`
- `freeze or crash`
- `slow system performance`
- `notifications not received`

### `account_access` (15%)
- `forgotten password`
- `2FA code not received`
- `account locked`
- `suspected account breach`
- `wrongful account block`
- `cannot change email`
- `login error`

### `tariff_question` (30%)
- `difference between plans`
- `switching to another plan`
- `feature limitations`
- `subscription issue`
- `auto-renewal`
- `price increase question`
- `free trial terms`

### `refund_request` (10%)
- `refund after service error`
- `refund after subscription cancellation`
- `refund denied`
- `partial refund`
- `late refund request`
- `refund for service not received`

## Mistake Catalog

### Category weights
- `communication`: 30%
- `logical`: 20%
- `process`: 20%
- `informational`: 25%
- `dialogue_structure`: 3%
- `hidden_dissatisfaction`: 2%

### Communication
- `passive_aggression`
- `dry_formal_tone_in_conflict_case`
- `ignoring_customer_emotions`
- `lack_of_empathy`
- `overly_templated_response`
- `blaming_customer`
- `minimizing_the_problem`
- `overly_short_response_without_explanation`
- `overly_long_response_without_specifics`

### Logical
- `contradictory_information_in_same_dialogue`
- `incomplete_answer_to_question`
- `off_topic_answer`
- `partial_ignore_of_multi_part_question`
- `missing_step_in_instructions`
- `repeating_the_same_instruction`
- `incorrect_interpretation_of_request`
- `answer_without_checking_context`
- `suggesting_solution_that_already_failed`

### Process
- `unjustified_escalation`
- `refusal_without_policy_explanation`
- `incorrect_policy_reference`
- `closes_case_without_confirming_resolution`
- `shifts_responsibility`
- `ask_to_contact_later_without_specific_time`
- `inconsistent_procedure`

### Informational
- `incorrect_amount`
- `incorrect_timeframe`
- `incorrect_plan`
- `incorrect_refund_policy`
- `incorrect_technical_instruction`

### Dialogue structure
- `responds_not_to_latest_message`
- `ignores_customer_clarification`
- `interrupts_dialogue_with_standard_phrase`
- `no_solution_summary`
- `ambiguous_answer`

### Hidden dissatisfaction
- `formal_closure_without_real_resolution`
- `temporary_fix_without_explaining_permanent_one`
- `does_not_explain_consequences`
- `shifts_responsibility_to_system`
- `answer_without_result_guarantee`

## Main Mistake Mapping (`sub -> main`)

| Sub mistake | Main mistake |
|---|---|
| `passive_aggression` | `rude_tone` |
| `dry_formal_tone_in_conflict_case` | `rude_tone` |
| `ignoring_customer_emotions` | `rude_tone` |
| `lack_of_empathy` | `rude_tone` |
| `overly_templated_response` | `ignored_question` |
| `blaming_customer` | `rude_tone` |
| `minimizing_the_problem` | `rude_tone` |
| `overly_short_response_without_explanation` | `ignored_question` |
| `overly_long_response_without_specifics` | `no_resolution` |
| `contradictory_information_in_same_dialogue` | `incorrect_info` |
| `incomplete_answer_to_question` | `ignored_question` |
| `off_topic_answer` | `ignored_question` |
| `partial_ignore_of_multi_part_question` | `ignored_question` |
| `missing_step_in_instructions` | `no_resolution` |
| `repeating_the_same_instruction` | `no_resolution` |
| `incorrect_interpretation_of_request` | `ignored_question` |
| `answer_without_checking_context` | `incorrect_info` |
| `suggesting_solution_that_already_failed` | `no_resolution` |
| `unjustified_escalation` | `unnecessary_escalation` |
| `refusal_without_policy_explanation` | `no_resolution` |
| `incorrect_policy_reference` | `incorrect_info` |
| `closes_case_without_confirming_resolution` | `no_resolution` |
| `shifts_responsibility` | `no_resolution` |
| `ask_to_contact_later_without_specific_time` | `no_resolution` |
| `inconsistent_procedure` | `incorrect_info` |
| `incorrect_amount` | `incorrect_info` |
| `incorrect_timeframe` | `incorrect_info` |
| `incorrect_plan` | `incorrect_info` |
| `incorrect_refund_policy` | `incorrect_info` |
| `incorrect_technical_instruction` | `incorrect_info` |
| `responds_not_to_latest_message` | `ignored_question` |
| `ignores_customer_clarification` | `ignored_question` |
| `interrupts_dialogue_with_standard_phrase` | `no_resolution` |
| `no_solution_summary` | `no_resolution` |
| `ambiguous_answer` | `incorrect_info` |
| `formal_closure_without_real_resolution` | `no_resolution` |
| `temporary_fix_without_explaining_permanent_one` | `no_resolution` |
| `does_not_explain_consequences` | `ignored_question` |
| `shifts_responsibility_to_system` | `no_resolution` |
| `answer_without_result_guarantee` | `no_resolution` |

## Consistency Constraints

### Outcome vs mistakes
- If `outcome=resolved`, `no_resolution` is not allowed.
- If `outcome=not_resolved`, main mistakes should include `no_resolution` or `ignored_question`.
- If `outcome=escalated`, mistakes may be empty (valid escalation is allowed).
- `unnecessary_escalation` is more likely when `outcome=escalated`.

### Hidden dissatisfaction
- If `hidden_dissatisfaction=true`, `satisfaction` cannot be `satisfied`.
- `hidden_dissatisfaction=true` is allowed only for `outcome in {resolved, escalated}`.
- If `hidden_dissatisfaction=true`, the final client message should be neutral-positive.

### Conflict level
- `conflict_level=low`: avoid aggressive wording, accusations, ultimatums.
- `conflict_level=high`: include at least one explicit conflict marker.
- `rude_tone` is allowed only when `conflict_level in {medium, high}`.
- `unnecessary_escalation` is more likely when `conflict_level in {medium, high}`.

### Complexity and length
- `low`: 3-5 messages.
- `medium`: 6-9 messages.
- `high`: 10-13 messages.
- `low` complexity should not contain more than one main mistake.

### Scenario and mistake bias (soft)
- `refund_request`: bias to `incorrect_info` or `no_resolution`.
- `payment_issue`: bias to `incorrect_info`.
- `technical_issue`: bias to `no_resolution` or `ignored_question`.
- `account_access`: bias to `incorrect_info` or `no_resolution`.
- `tariff_question`: bias to `incorrect_info`.

### Mistake presence constraints
- `mistakes_present=false` -> `main mistakes=[]`.
- `mistakes_present=true` -> at least one main mistake.
- If `num_mistakes=1` -> exactly one main mistake.
- If `num_mistakes=2` -> two distinct main mistakes.
- If `num_mistakes>=3` -> maximum three distinct main mistakes.

## Generation Settings

### Language and style
- `agent_tone`: `polite` or `neutral`
- `client_style`: abbreviations, slang, and typos are allowed

### Output format
- dataset format: `JSONL`
- each record includes: `messages`, `generation_spec`, `ground_truth`
- manifest includes: `seed`, distributions, generator version, observed stats, output paths, notes

## Safety Constraints
- Do not use real PII (phone numbers, emails, card numbers).
- Use placeholders like `ORDER_12345` and `USER_6789`.

## Post-generation Validator

### Role and format checks
- `messages` must be a list.
- each item must have `role in {client, agent}`.
- each item must have non-empty `text`.

### Turn-taking checks
- first message must be from `client`.
- no two `agent` messages in a row.
- no two `client` messages in a row.

### Length checks
- validate message count against complexity bounds.

## Retry Policy
- If validation fails, retry generation with the same spec.
- No repair prompt is used.

## Sampling Logic Structure (`sample_generation_spec`)
1. Initialize RNG from seed.
2. Sample `scenario` by weights.
3. Sample `sub_scenario` from selected scenario list.
4. Sample `complexity`.
5. Sample `outcome`.
6. Sample `conflict_level`.
7. Sample `mistakes_present`.
8. If mistakes are present, sample `num_mistakes`, then sample and map sub mistakes to distinct main mistakes.
9. Sample `hidden_dissatisfaction` conditionally for `resolved` outcome.
10. Apply consistency constraints and resample mistakes if needed.
11. Build final `generation_spec` for the dialogue.

## Example `generation_spec`
```json
{
  "dialogue_id": "dlg_000123",
  "scenario": "refund_request",
  "sub_scenario": "refund after subscription cancellation",
  "complexity": "medium",
  "outcome": "not_resolved",
  "conflict_level": "low",
  "agent_tone": "polite",
  "mistakes_present": true,
  "num_mistakes": 2,
  "agent_mistakes_sub": [
    "ask_to_contact_later_without_specific_time",
    "formal_closure_without_real_resolution"
  ],
  "agent_mistakes_main": [
    "no_resolution",
    "ignored_question"
  ],
  "hidden_dissatisfaction": false,
  "length_bounds": [6, 9],
  "length_target": 7
}
```

## Example dataset record (JSONL line)
```json
{
  "dialogue_id": "dlg_000123",
  "scenario": "refund_request",
  "sub_scenario": "refund after subscription cancellation",
  "complexity": "medium",
  "tags": ["agent_mistake_present"],
  "messages": [
    {"role": "client", "text": "I canceled yesterday, but I was charged again for ORDER_12345."},
    {"role": "agent", "text": "Thanks for contacting support. Please contact us later."},
    {"role": "client", "text": "Later when exactly? I need the refund now."},
    {"role": "agent", "text": "Everything should be fine now, thank you for your understanding."},
    {"role": "client", "text": "Okay, thanks."}
  ],
  "generation_spec": {
    "dialogue_id": "dlg_000123",
    "scenario": "refund_request",
    "sub_scenario": "refund after subscription cancellation",
    "complexity": "medium",
    "outcome": "not_resolved",
    "conflict_level": "low",
    "mistakes_present": true,
    "num_mistakes": 2,
    "agent_mistakes_sub": [
      "ask_to_contact_later_without_specific_time",
      "formal_closure_without_real_resolution"
    ],
    "agent_mistakes_main": ["no_resolution", "ignored_question"],
    "hidden_dissatisfaction": false,
    "agent_tone": "polite",
    "length_bounds": [6, 9],
    "length_target": 7
  },
  "ground_truth": {
    "intent": "refund_request",
    "satisfaction": "unsatisfied",
    "hidden_dissatisfaction": false,
    "quality_score": 2,
    "agent_mistakes_main": ["no_resolution", "ignored_question"],
    "agent_mistakes": ["no_resolution", "ignored_question"]
  }
}
```

## Quick Run
```bash
python -m generate.run --n 500 --seed 42 --out generate/jsons/dataset.jsonl --manifest generate/jsons/manifest.json
```

Dummy mode (no API):
```bash
python -m generate.run --n 50 --seed 42 --dummy_text
```
