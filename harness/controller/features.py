"""Deterministic request features for the controller ladder.

E1 starts with a frozen encoder plus a small head.  These features are the
model-free side of that input: they are cheap, inspectable, and useful as a
baseline even before an encoder is attached.  They are not an embedding and do
not pretend to be one.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

_CODE_MARKERS = re.compile(
    r"(^|\n)\s*(def |class |import |from |fn |func |let |const |var |"
    r"#include|package |public |private |SELECT |CREATE |async def )",
    re.MULTILINE,
)
_URL = re.compile(r"https?://|www\.", re.IGNORECASE)
_MATH = re.compile(r"\b(prove|derive|calculate|compute|integral|theorem|equation)\b",
                   re.IGNORECASE)
_RISK = re.compile(
    r"\b(delete|rm -rf|drop table|truncate|force push|deploy|chmod|chown|"
    r"sudo|credential|password|secret|production)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TextFeatures:
    chars: int
    lines: int
    words: int
    estimated_tokens: int
    code_markers: int
    has_url: bool
    is_question: bool
    intent: str
    context_pressure: str
    risk: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "chars": self.chars,
            "lines": self.lines,
            "words": self.words,
            "estimated_tokens": self.estimated_tokens,
            "code_markers": self.code_markers,
            "has_url": self.has_url,
            "is_question": self.is_question,
            "intent": self.intent,
            "context_pressure": self.context_pressure,
            "risk": self.risk,
        }


def text_features(text: str) -> TextFeatures:
    """Classify coarse request shape without a model or a user prompt upload."""
    value = str(text or "")
    chars = len(value)
    lines = value.count("\n") + 1 if value else 0
    words = len(re.findall(r"\S+", value))
    # A deliberately rough tokenizer estimate: roughly four characters per
    # token for English/code.  The exact tokenizer belongs to the model, not
    # this feature extractor.
    estimated = max(0, math.ceil(chars / 4)) if chars else 0
    code = len(_CODE_MARKERS.findall(value))
    has_url = bool(_URL.search(value))
    question = value.strip().endswith("?") or bool(
        re.match(r"^\s*(what|why|how|when|where|who|which|can|does|is|are)\b",
                 value, re.IGNORECASE))
    if code >= 2 or any(marker in value for marker in ("```", "def ", "class ")):
        intent = "code"
    elif _MATH.search(value):
        intent = "reasoning"
    elif question:
        intent = "qa"
    else:
        intent = "chat"
    if estimated >= 6000:
        pressure = "long"
    elif estimated >= 1500:
        pressure = "medium"
    else:
        pressure = "short"
    risk = "high" if _RISK.search(value) else ("medium" if intent == "code" else "low")
    return TextFeatures(
        chars=chars,
        lines=lines,
        words=words,
        estimated_tokens=estimated,
        code_markers=code,
        has_url=has_url,
        is_question=question,
        intent=intent,
        context_pressure=pressure,
        risk=risk,
    )
