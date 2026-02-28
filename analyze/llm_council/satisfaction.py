
import asyncio
import json
from typing import List, Literal

from pydantic import BaseModel, Field

from shared.prompts import (
    SATISFACTION_AGGREGATOR_SYSTEM_PROMPT,
    SATISFACTION_REVIEWER_FRICTION_SYSTEM_PROMPT,
    SATISFACTION_REVIEWER_SENTIMENT_SYSTEM_PROMPT,
)

from .engine import CouncilEngineAsync

Satisfaction = Literal["satisfied", "neutral", "unsatisfied"]
HiddenSignal = Literal[
    "polite_but_unresolved",
    "repeat_request",
    "asks_for_manager_or_complaint",
    "sarcasm_or_passive_aggression",
    "abandonment",
    "negative_phrase",
    "high_effort_many_steps",
]


class SatisfactionFinalOutput(BaseModel):
    satisfaction: Satisfaction
    reason: str = Field(min_length=1, max_length=500)
    hidden_unsat_signals: List[HiddenSignal] = Field(default_factory=list)
    evidence_turns: List[int] = Field(default_factory=list)


class SatisfactionReviewerOutput(BaseModel):
    reviewer: Literal["sentiment_outcome", "friction_effort"]
    satisfaction: Satisfaction
    reason: str = Field(min_length=1, max_length=500)
    hidden_unsat_signals: List[HiddenSignal] = Field(default_factory=list)
    evidence_turns: List[int] = Field(default_factory=list)


class CustomerSatisfactionCouncilAsync:
    def __init__(self, engine: CouncilEngineAsync):
        self.engine = engine

    async def _run_reviewer(self, prompt: str) -> SatisfactionReviewerOutput:
        return await self.engine.generate_structured(
            prompt, SatisfactionReviewerOutput, max_output_tokens=600
        )

    async def evaluate(self, transcript: str) -> SatisfactionFinalOutput:
        reviewer_a_prompt = (
            f"{SATISFACTION_REVIEWER_SENTIMENT_SYSTEM_PROMPT}\n\nTranscript:\n{transcript}"
        ).strip()
        reviewer_b_prompt = (
            f"{SATISFACTION_REVIEWER_FRICTION_SYSTEM_PROMPT}\n\nTranscript:\n{transcript}"
        ).strip()

        review_a, review_b = await asyncio.gather(
            self._run_reviewer(reviewer_a_prompt),
            self._run_reviewer(reviewer_b_prompt),
        )

        agg_prompt = (
            f"{SATISFACTION_AGGREGATOR_SYSTEM_PROMPT}\n\n"
            f"Transcript:\n{transcript}\n\n"
            f"review_a:\n{json.dumps(review_a.model_dump(), ensure_ascii=False)}\n\n"
            f"review_b:\n{json.dumps(review_b.model_dump(), ensure_ascii=False)}"
        ).strip()

        return await self.engine.generate_structured(
            agg_prompt, SatisfactionFinalOutput, max_output_tokens=700
        )

