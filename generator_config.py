from typing import Dict, List, Tuple


GENERATOR_VERSION = "0.1.0"


DISTR: Dict[str, Dict[str, float]] = {
    "scenario": {
        "tariff_question": 0.30,
        "payment_issue": 0.25,
        "technical_issue": 0.20,
        "account_access": 0.15,
        "refund_request": 0.10,
    },
    "complexity": {"low": 0.50, "medium": 0.35, "high": 0.15},
    "outcome": {"resolved": 0.75, "not_resolved": 0.15, "escalated": 0.10},
    "conflict_level": {"low": 0.70, "medium": 0.20, "high": 0.10},
    "mistakes_present": {"false": 0.80, "true": 0.20},
    "num_mistakes_if_present": {"1": 0.60, "2": 0.30, "3": 0.10},
    "agent_tone": {"polite": 0.60, "neutral": 0.40},
}


DIALOGUE_LENGTH_BOUNDS: Dict[str, Tuple[int, int]] = {
    "low": (3, 5),
    "medium": (6, 9),
    "high": (10, 13),
}


SUB_SCENARIOS: Dict[str, List[str]] = {
    "payment_issue": [
        "double charge",
        "payment failed",
        "charge without confirmation",
        "incorrect charged amount",
        "payment processing too long",
        "promo code issue",
        "charge after subscription cancellation",
    ],
    "technical_issue": [
        "app does not open",
        "error 500 or other system error",
        "specific feature not working",
        "data not updating",
        "freeze or crash",
        "slow system performance",
        "notifications not received",
    ],
    "account_access": [
        "forgotten password",
        "2FA code not received",
        "account locked",
        "suspected account breach",
        "wrongful account block",
        "cannot change email",
        "login error",
    ],
    "tariff_question": [
        "difference between plans",
        "switching to another plan",
        "feature limitations",
        "subscription issue",
        "auto-renewal",
        "price increase question",
        "free trial terms",
    ],
    "refund_request": [
        "refund after service error",
        "refund after subscription cancellation",
        "refund denied",
        "partial refund",
        "late refund request",
        "refund for service not received",
    ],
}


MISTAKE_POOLS: Dict[str, List[str]] = {
    "communication": [
        "passive_aggression",
        "dry_formal_tone_in_conflict_case",
        "ignoring_customer_emotions",
        "lack_of_empathy",
        "overly_templated_response",
        "blaming_customer",
        "minimizing_the_problem",
        "overly_short_response_without_explanation",
        "overly_long_response_without_specifics",
    ],
    "logical": [
        "contradictory_information_in_same_dialogue",
        "incomplete_answer_to_question",
        "off_topic_answer",
        "partial_ignore_of_multi_part_question",
        "missing_step_in_instructions",
        "repeating_the_same_instruction",
        "incorrect_interpretation_of_request",
        "answer_without_checking_context",
        "suggesting_solution_that_already_failed",
    ],
    "process": [
        "unjustified_escalation",
        "refusal_without_policy_explanation",
        "incorrect_policy_reference",
        "closes_case_without_confirming_resolution",
        "shifts_responsibility",
        "ask_to_contact_later_without_specific_time",
        "inconsistent_procedure",
    ],
    "informational": [
        "incorrect_amount",
        "incorrect_timeframe",
        "incorrect_plan",
        "incorrect_refund_policy",
        "incorrect_technical_instruction",
    ],
    "dialogue_structure": [
        "responds_not_to_latest_message",
        "ignores_customer_clarification",
        "interrupts_dialogue_with_standard_phrase",
        "no_solution_summary",
        "ambiguous_answer",
    ],
    "hidden_dissatisfaction": [
        "formal_closure_without_real_resolution",
        "temporary_fix_without_explaining_permanent_one",
        "does_not_explain_consequences",
        "shifts_responsibility_to_system",
        "answer_without_result_guarantee",
    ],
}


MISTAKE_CATEGORY_WEIGHTS: Dict[str, float] = {
    "communication": 0.30,
    "logical": 0.20,
    "process": 0.20,
    "informational": 0.25,
    "dialogue_structure": 0.03,
    "hidden_dissatisfaction": 0.02,
}


SUB_TO_MAIN: Dict[str, str] = {
    "passive_aggression": "rude_tone",
    "dry_formal_tone_in_conflict_case": "rude_tone",
    "ignoring_customer_emotions": "rude_tone",
    "lack_of_empathy": "rude_tone",
    "overly_templated_response": "ignored_question",
    "blaming_customer": "rude_tone",
    "minimizing_the_problem": "rude_tone",
    "overly_short_response_without_explanation": "ignored_question",
    "overly_long_response_without_specifics": "no_resolution",
    "contradictory_information_in_same_dialogue": "incorrect_info",
    "incomplete_answer_to_question": "ignored_question",
    "off_topic_answer": "ignored_question",
    "partial_ignore_of_multi_part_question": "ignored_question",
    "missing_step_in_instructions": "no_resolution",
    "repeating_the_same_instruction": "no_resolution",
    "incorrect_interpretation_of_request": "ignored_question",
    "answer_without_checking_context": "incorrect_info",
    "suggesting_solution_that_already_failed": "no_resolution",
    "unjustified_escalation": "unnecessary_escalation",
    "refusal_without_policy_explanation": "no_resolution",
    "incorrect_policy_reference": "incorrect_info",
    "closes_case_without_confirming_resolution": "no_resolution",
    "shifts_responsibility": "no_resolution",
    "ask_to_contact_later_without_specific_time": "no_resolution",
    "inconsistent_procedure": "incorrect_info",
    "incorrect_amount": "incorrect_info",
    "incorrect_timeframe": "incorrect_info",
    "incorrect_plan": "incorrect_info",
    "incorrect_refund_policy": "incorrect_info",
    "incorrect_technical_instruction": "incorrect_info",
    "responds_not_to_latest_message": "ignored_question",
    "ignores_customer_clarification": "ignored_question",
    "interrupts_dialogue_with_standard_phrase": "no_resolution",
    "no_solution_summary": "no_resolution",
    "ambiguous_answer": "incorrect_info",
    "formal_closure_without_real_resolution": "no_resolution",
    "temporary_fix_without_explaining_permanent_one": "no_resolution",
    "does_not_explain_consequences": "ignored_question",
    "shifts_responsibility_to_system": "no_resolution",
    "answer_without_result_guarantee": "no_resolution",
}


MAJOR_MISTAKES = {"incorrect_info", "rude_tone", "no_resolution", "ignored_question"}


SCENARIO_MAIN_BIAS: Dict[str, Dict[str, float]] = {
    "refund_request": {"incorrect_info": 1.2, "no_resolution": 1.2},
    "payment_issue": {"incorrect_info": 1.3},
    "technical_issue": {"no_resolution": 1.2, "ignored_question": 1.1},
    "account_access": {"incorrect_info": 1.15, "no_resolution": 1.15},
    "tariff_question": {"incorrect_info": 1.25},
}


TOPIC_HINTS: Dict[str, Dict[str, str]] = {
    "payment_issue": {
        "client_open": "I have a payment issue regarding {sub_scenario} for ORDER_12345.",
        "details": "The charge appeared on my card today and I need clarification.",
        "resolved": "I checked the payment record and the issue is fixed now. You will see the corrected status shortly.",
        "not_resolved": "I cannot complete a fix from chat right now.",
        "escalated": "I am escalating this payment case to the billing team for manual review.",
    },
    "technical_issue": {
        "client_open": "I need help with a technical issue: {sub_scenario} on USER_6789.",
        "details": "I already tried restarting the app but the problem is still happening.",
        "resolved": "Thanks for waiting. I can confirm the issue is resolved after the troubleshooting steps.",
        "not_resolved": "I do not have a working fix from my side at the moment.",
        "escalated": "I will escalate this to technical support for deeper investigation.",
    },
    "account_access": {
        "client_open": "I have an account access problem: {sub_scenario} for USER_6789.",
        "details": "I need access restored because I cannot continue using the service.",
        "resolved": "I verified your account and completed the required access recovery steps.",
        "not_resolved": "I cannot restore access in this chat session right now.",
        "escalated": "I am escalating this account case to the security/access team.",
    },
    "tariff_question": {
        "client_open": "I have a plan question about {sub_scenario}.",
        "details": "Please explain what applies to my current subscription.",
        "resolved": "I clarified the plan details and what applies to your account.",
        "not_resolved": "I cannot give a complete answer from chat right now.",
        "escalated": "I will escalate this plan question to the subscription team.",
    },
    "refund_request": {
        "client_open": "I want help with a refund request: {sub_scenario} for ORDER_12345.",
        "details": "Please review the charge and tell me what refund options I have.",
        "resolved": "I reviewed the request and processed the refund according to the policy.",
        "not_resolved": "I cannot process the refund in this chat right now.",
        "escalated": "I am escalating the refund request to the billing/refunds team.",
    },
}


MISTAKE_TEXT: Dict[str, str] = {
    "passive_aggression": "As I already said, this is a standard process.",
    "dry_formal_tone_in_conflict_case": "Please remain objective. We will proceed according to procedure.",
    "ignoring_customer_emotions": "I understand your message. Proceed to the next step.",
    "lack_of_empathy": "This is a routine case.",
    "overly_templated_response": "Thank you for contacting support. Your request is important to us.",
    "blaming_customer": "This usually happens when the user enters incorrect details.",
    "minimizing_the_problem": "This is a minor issue and should not be a concern.",
    "overly_short_response_without_explanation": "Do it again.",
    "overly_long_response_without_specifics": "There are many possible reasons, internal factors, and processing states, so please wait while the system updates eventually.",
    "contradictory_information_in_same_dialogue": "Earlier I said it was blocked, but actually it is already active.",
    "incomplete_answer_to_question": "I answered part of it, and the rest is standard.",
    "off_topic_answer": "You can also check your profile theme settings.",
    "partial_ignore_of_multi_part_question": "I addressed only the first part.",
    "missing_step_in_instructions": "Please reset it, then log in again.",
    "repeating_the_same_instruction": "Please refresh and try again. Please refresh and try again.",
    "incorrect_interpretation_of_request": "I see you want to cancel, so I will explain pricing.",
    "answer_without_checking_context": "Without checking your account, I can confirm that this is expected.",
    "suggesting_solution_that_already_failed": "Please try the same step again.",
    "unjustified_escalation": "I will escalate this immediately even though no checks are needed.",
    "refusal_without_policy_explanation": "We cannot do that.",
    "incorrect_policy_reference": "Policy says refunds are never possible after one minute.",
    "closes_case_without_confirming_resolution": "I will close this case now.",
    "shifts_responsibility": "This is not our side, so you need to handle it elsewhere.",
    "ask_to_contact_later_without_specific_time": "Please contact us later.",
    "inconsistent_procedure": "The process is to verify after completion, not before.",
    "incorrect_amount": "The charge amount is 1999 USD for your basic plan.",
    "incorrect_timeframe": "This normally takes 30 business days.",
    "incorrect_plan": "Your current plan is Enterprise even if you use Free.",
    "incorrect_refund_policy": "Refunds are always automatic for all cases.",
    "incorrect_technical_instruction": "Delete your operating system cache folder to fix this app issue.",
    "responds_not_to_latest_message": "Regarding your first message, the app is fine.",
    "ignores_customer_clarification": "I will continue with the previous answer.",
    "interrupts_dialogue_with_standard_phrase": "Thank you for contacting support. Have a nice day.",
    "no_solution_summary": "The issue is handled.",
    "ambiguous_answer": "It may be yes, or not, depending.",
    "formal_closure_without_real_resolution": "Everything should be fine now, thank you for your understanding.",
    "temporary_fix_without_explaining_permanent_one": "This workaround should help for now.",
    "does_not_explain_consequences": "Please proceed; I will not cover what happens next.",
    "shifts_responsibility_to_system": "The system decides this automatically.",
    "answer_without_result_guarantee": "This should probably work, but I cannot guarantee the result.",
}
