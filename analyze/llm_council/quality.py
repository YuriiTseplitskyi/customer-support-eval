
import asyncio
import json
from typing import Dict, List, Literal

from pydantic import BaseModel, Field

from shared.prompts import (
    QUALITY_AGGREGATOR_SYSTEM_PROMPT,
    QUALITY_REVIEWER_COMMUNICATION_SYSTEM_PROMPT,
    QUALITY_REVIEWER_CORRECTNESS_SYSTEM_PROMPT,
    QUALITY_REVIEWER_RESOLUTION_SYSTEM_PROMPT,
)

from .common import AllowedMistake
from .engine import CouncilEngineAsync


QualityReviewer = Literal[
    "correctness",
    "resolution_effectiveness",
    "professional_communication",
]


class QualityReviewerOutput(BaseModel):
    reviewer: QualityReviewer
    quality_score: int = Field(ge=1, le=5)
    reason: str = Field(min_length=1, max_length=500)


class QualityEvidenceItem(BaseModel):
    mistake: AllowedMistake
    evidence_turns: List[int]


class QualityAggregatorOutput(BaseModel):
    quality_score: int = Field(ge=1, le=5)
    reason: str
    breakdown_correctness: int = Field(ge=1, le=5)
    breakdown_resolution_effectiveness: int = Field(ge=1, le=5)
    breakdown_professional_communication: int = Field(ge=1, le=5)
    judge_reason_correctness: str
    judge_reason_resolution_effectiveness: str
    judge_reason_professional_communication: str
    agent_mistakes: List[AllowedMistake]
    evidence: List[QualityEvidenceItem]


class QualityFinalOutput(BaseModel):
    quality_score: int = Field(ge=1, le=5)
    reason: str
    breakdown: Dict[str, int]
    judge_reasons: Dict[str, str]
    agent_mistakes: List[AllowedMistake]
    evidence: Dict[AllowedMistake, List[int]]


class AgentQualityCouncilAsync:
    def __init__(self, engine: CouncilEngineAsync):
        self.engine = engine

    async def _run_reviewer(self, prompt: str, expected_reviewer: QualityReviewer) -> QualityReviewerOutput:
        out = await self.engine.generate_structured(
            prompt, QualityReviewerOutput, max_output_tokens=450
        )
        if out.reviewer != expected_reviewer:
            raise ValueError(f"Reviewer mismatch. Expected '{expected_reviewer}', got '{out.reviewer}'.")
        return out

    @staticmethod
    def _normalize_evidence(evidence_items: List[QualityEvidenceItem]) -> Dict[AllowedMistake, List[int]]:
        merged: Dict[AllowedMistake, set[int]] = {}
        for item in evidence_items:
            bucket = merged.setdefault(item.mistake, set())
            for idx in item.evidence_turns:
                bucket.add(int(idx))

        normalized: Dict[AllowedMistake, List[int]] = {
            mistake: sorted(turns) for mistake, turns in merged.items()
        }
        return normalized

    async def evaluate(self, transcript: str) -> QualityFinalOutput:
        prompt_correctness = (
            f"{QUALITY_REVIEWER_CORRECTNESS_SYSTEM_PROMPT}\n\nTranscript:\n{transcript}"
        ).strip()
        prompt_resolution = (
            f"{QUALITY_REVIEWER_RESOLUTION_SYSTEM_PROMPT}\n\nTranscript:\n{transcript}"
        ).strip()
        prompt_communication = (
            f"{QUALITY_REVIEWER_COMMUNICATION_SYSTEM_PROMPT}\n\nTranscript:\n{transcript}"
        ).strip()

        review_correctness, review_resolution, review_communication = await asyncio.gather(
            self._run_reviewer(prompt_correctness, "correctness"),
            self._run_reviewer(prompt_resolution, "resolution_effectiveness"),
            self._run_reviewer(prompt_communication, "professional_communication"),
        )

        agg_prompt = (
            f"{QUALITY_AGGREGATOR_SYSTEM_PROMPT}\n\n"
            f"Transcript:\n{transcript}\n\n"
            f"review_correctness:\n{json.dumps(review_correctness.model_dump(), ensure_ascii=False)}\n\n"
            f"review_resolution:\n{json.dumps(review_resolution.model_dump(), ensure_ascii=False)}\n\n"
            f"review_communication:\n{json.dumps(review_communication.model_dump(), ensure_ascii=False)}"
        ).strip()

        parsed = await self.engine.generate_structured(
            agg_prompt, QualityAggregatorOutput, max_output_tokens=900
        )

        normalized_evidence = self._normalize_evidence(parsed.evidence)
        merged_mistakes = sorted(set(parsed.agent_mistakes) | set(normalized_evidence.keys()))

        return QualityFinalOutput(
            quality_score=parsed.quality_score,
            reason=parsed.reason,
            breakdown={
                "correctness": parsed.breakdown_correctness,
                "resolution_effectiveness": parsed.breakdown_resolution_effectiveness,
                "professional_communication": parsed.breakdown_professional_communication,
            },
            judge_reasons={
                "correctness": parsed.judge_reason_correctness,
                "resolution_effectiveness": parsed.judge_reason_resolution_effectiveness,
                "professional_communication": parsed.judge_reason_professional_communication,
            },
            agent_mistakes=merged_mistakes,
            evidence=normalized_evidence,
        )

