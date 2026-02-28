import json
import os
from typing import Any, Type

from pydantic import BaseModel


def load_dotenv_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value


class ChatGPTWrapper:
    def __init__(self, model_name: str, api_key: str | None = None, dotenv_path: str = ".env"):
        load_dotenv_file(dotenv_path)
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError(
                "OpenAI SDK is not available. Install `openai` in the active environment."
            ) from e
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def ask_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        temperature: float = 0.0,
        validation_context: dict[str, Any] | None = None,
    ) -> BaseModel:
        completion = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_model,
            temperature=temperature,
        )

        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise RuntimeError(f"Model refusal: {message.refusal}")
        if getattr(message, "parsed", None) is not None:
            parsed = message.parsed
            if isinstance(parsed, response_model):
                return parsed
            if hasattr(parsed, "model_dump"):
                payload = parsed.model_dump()
            else:
                payload = parsed
            return response_model.model_validate(payload, context=validation_context or {})

        content = message.content or ""
        if isinstance(content, str):
            payload = json.loads(content)
        else:
            payload = content
        return response_model.model_validate(payload, context=validation_context or {})
