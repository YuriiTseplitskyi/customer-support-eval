
import asyncio
import json
from typing import Dict, List, Literal

from pydantic import BaseModel, Field

from shared.prompts import INTENT_AGGREGATOR_SYSTEM_PROMPT, INTENT_JUDGE_SYSTEM_PROMPT

from .engine import CouncilEngineAsync

IntentJudge = Literal["problem_statement", "action_request", "blocker", "evidence_only"]


class IntentJudgeOutput(BaseModel):
    judge: IntentJudge
    intent: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_]+$")
    reason: str
    evidence_turns: List[int]


class IntentAggregatorOutput(BaseModel):
    intent: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_]+$")
    reason: str
    evidence_turns: List[int]


class IntentFinalOutput(BaseModel):
    intent: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_]+$")
    reason: str
    evidence_turns: List[int]
    judge_votes: Dict[str, str]
    judge_reasons: Dict[str, str]


INTENT_JUDGE_DEFS = {
    "problem_statement": "Classify by the customer's stated problem/symptoms.",
    "action_request": "Classify by what the customer explicitly wants to achieve.",
    "blocker": "Classify by what most blocks the customer right now.",
    "evidence_only": "Be extremely strict: infer an intent only from explicit evidence.",
}


class IntentCouncilAsync:
    def __init__(self, engine: CouncilEngineAsync):
        self.engine = engine

    async def _run_judge(self, judge: IntentJudge, transcript: str) -> IntentJudgeOutput:
        prompt = (
            INTENT_JUDGE_SYSTEM_PROMPT.replace("__JUDGE__", judge).replace(
                "__DEFINITION__", INTENT_JUDGE_DEFS[judge]
            )
            + f"\n\nTranscript:\n{transcript}"
        ).strip()
        return await self.engine.generate_structured(
            prompt, IntentJudgeOutput, max_output_tokens=300
        )

    async def evaluate(self, transcript: str) -> IntentFinalOutput:
        judges_order: List[IntentJudge] = ["problem_statement", "action_request", "blocker", "evidence_only"]
        tasks = [self._run_judge(judge, transcript) for judge in judges_order]
        results = await asyncio.gather(*tasks)

        judge_outputs = dict(zip(judges_order, results))
        votes = {judge: out.intent for judge, out in judge_outputs.items()}
        reasons = {judge: out.reason for judge, out in judge_outputs.items()}

        agg_prompt = (
            INTENT_AGGREGATOR_SYSTEM_PROMPT.replace("__VOTES__", json.dumps(votes)).replace(
                "__REASONS__", json.dumps(reasons)
            )
        ).strip()
        agg = await self.engine.generate_structured(
            agg_prompt, IntentAggregatorOutput, max_output_tokens=300
        )

        return IntentFinalOutput(
            intent=agg.intent,
            reason=agg.reason,
            evidence_turns=agg.evidence_turns,
            judge_votes=votes,
            judge_reasons=reasons,
        )

