
from typing import Dict, List

from pydantic import BaseModel, Field

from shared.prompts import MISTAKES_SYSTEM_PROMPT

from .common import AllowedMistake
from .engine import CouncilEngineAsync


class DetectedMistake(BaseModel):
    mistake: AllowedMistake
    reason: str = Field(min_length=1, max_length=500)
    evidence_turns: List[int] = Field(default_factory=list)


class UnifiedMistakesOutput(BaseModel):
    detected_mistakes: List[DetectedMistake] = Field(default_factory=list)


class MistakeFinalOutput(BaseModel):
    agent_mistakes: List[AllowedMistake]
    mistake_reasons: Dict[str, str]
    evidence: Dict[AllowedMistake, List[int]]
    rejected: Dict[str, str] = Field(default_factory=dict)


class AgentMistakeCouncilAsync:
    def __init__(self, engine: CouncilEngineAsync):
        self.engine = engine

    async def evaluate(self, transcript: str) -> MistakeFinalOutput:
        prompt = f"{MISTAKES_SYSTEM_PROMPT}\n\nTranscript:\n{transcript}".strip()
        parsed = await self.engine.generate_structured(
            prompt, UnifiedMistakesOutput, max_output_tokens=600
        )

        accepted_mistakes = []
        reasons = {}
        evidence = {}
        rejected = {}

        for det in parsed.detected_mistakes:
            if not det.evidence_turns:
                rejected[det.mistake] = "Rejected: no evidence_turns provided."
                continue
            accepted_mistakes.append(det.mistake)
            reasons[det.mistake] = det.reason
            evidence[det.mistake] = sorted(set(int(i) for i in det.evidence_turns))

        accepted_mistakes.sort()
        return MistakeFinalOutput(
            agent_mistakes=accepted_mistakes,
            mistake_reasons=reasons,
            evidence=evidence,
            rejected=rejected,
        )

