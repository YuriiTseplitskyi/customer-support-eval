
from typing import List, Literal

from pydantic import BaseModel

Role = Literal["CLIENT", "AGENT"]
AllowedMistake = Literal[
    "ignored_question",
    "incorrect_info",
    "rude_tone",
    "no_resolution",
    "unnecessary_escalation",
]


class ChatTurn(BaseModel):
    role: Role
    text: str


def normalize_chat(turns: List[ChatTurn]) -> str:
    lines = []
    for i, t in enumerate(turns):
        text = (t.text or "").strip().replace("\r\n", "\n").replace("\r", "\n")
        lines.append(f"[{i}] {t.role}: {text}")
    return "\n".join(lines)

