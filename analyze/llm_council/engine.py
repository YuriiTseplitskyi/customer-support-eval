
import asyncio
from typing import Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)


class CouncilEngineAsync:
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    @retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
    def generate_structured_sync(
        self,
        prompt: str,
        response_model: Type[StructuredModelT],
        max_output_tokens: int = 450,
    ) -> StructuredModelT:
        completion = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Return only the requested output."},
                {"role": "user", "content": prompt},
            ],
            response_format=response_model,
            temperature=0.0,
            max_tokens=max_output_tokens,
        )

        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise ValueError(f"Model refusal: {message.refusal}")

        if getattr(message, "parsed", None) is None:
            raise ValueError("Structured output parsing failed: empty parsed payload.")

        parsed = message.parsed
        if isinstance(parsed, response_model):
            return parsed
        if hasattr(parsed, "model_dump"):
            payload = parsed.model_dump()
        else:
            payload = parsed
        return response_model.model_validate(payload)

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[StructuredModelT],
        max_output_tokens: int = 450,
    ) -> StructuredModelT:
        return await asyncio.to_thread(
            self.generate_structured_sync, prompt, response_model, max_output_tokens
        )

