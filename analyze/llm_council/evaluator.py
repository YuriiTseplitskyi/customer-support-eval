
import asyncio
import threading
from typing import Any, Dict, List

from pydantic import BaseModel

from .common import ChatTurn, normalize_chat
from .engine import CouncilEngineAsync
from .intent import IntentCouncilAsync, IntentFinalOutput
from .mistakes import AgentMistakeCouncilAsync, MistakeFinalOutput
from .quality import AgentQualityCouncilAsync, QualityFinalOutput
from .satisfaction import CustomerSatisfactionCouncilAsync, SatisfactionFinalOutput


class ChatEvaluatorOutput(BaseModel):
    intent: IntentFinalOutput
    quality: QualityFinalOutput
    satisfaction: SatisfactionFinalOutput
    mistakes: MistakeFinalOutput


class ChatEvaluatorAsync:
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        engine = CouncilEngineAsync(api_key=api_key, model_name=model_name)
        self.intent = IntentCouncilAsync(engine)
        self.quality = AgentQualityCouncilAsync(engine)
        self.satisfaction = CustomerSatisfactionCouncilAsync(engine)
        self.mistakes = AgentMistakeCouncilAsync(engine)

    async def evaluate_async(self, turns: List[Dict[str, str]]) -> Dict[str, Any]:
        parsed_turns = [ChatTurn.model_validate(t) for t in turns]
        transcript = normalize_chat(parsed_turns)

        results = await asyncio.gather(
            self.intent.evaluate(transcript),
            self.quality.evaluate(transcript),
            self.satisfaction.evaluate(transcript),
            self.mistakes.evaluate(transcript),
        )

        out = ChatEvaluatorOutput(
            intent=results[0],
            quality=results[1],
            satisfaction=results[2],
            mistakes=results[3],
        )
        return out.model_dump()


class ChatEvaluator:
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        self._async_evaluator = ChatEvaluatorAsync(api_key=api_key, model_name=model_name)

    def evaluate(self, turns: List[Dict[str, str]]) -> Dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._async_evaluator.evaluate_async(turns))

        result_holder: Dict[str, Any] = {}
        error_holder: Dict[str, BaseException] = {}

        def _runner() -> None:
            try:
                result_holder["result"] = asyncio.run(self._async_evaluator.evaluate_async(turns))
            except BaseException as exc:  # noqa: BLE001
                error_holder["error"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()

        if "error" in error_holder:
            raise error_holder["error"]
        return result_holder["result"]


