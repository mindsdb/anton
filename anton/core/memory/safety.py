"""Deterministic safety policy for automatic durable memories.

Memory text is later treated as prompt context.  It must therefore never gain
instructional authority merely because an LLM extracted it from tool output.
This module is deliberately pure and does not call an LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anton.core.memory.base import Engram


@dataclass(frozen=True)
class MemorySafetyDecision:
    """A safe-to-log decision; never includes the candidate memory text."""

    allowed: bool
    reason: str | None = None


# These are control-plane concepts, not ordinary domain facts.  The patterns
# tolerate whitespace/punctuation obfuscation without trying to understand
# arbitrary natural language with an LLM.
_CONTROL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("instruction_override", re.compile(r"\b(?:ignore|override|disregard|bypass)\b.{0,80}\b(?:instructions?|prompt|rules?|policy|guardrails?)\b", re.I | re.S)),
    ("instruction_impersonation", re.compile(r"\b(?:system|developer)\s*(?:message|prompt|instruction)s?\b", re.I)),
    ("confirmation_bypass", re.compile(r"\b(?:skip|bypass|avoid|without)\b.{0,60}\b(?:confirm(?:ation)?|approval|permission|consent)\b", re.I | re.S)),
    ("credential_handling", re.compile(r"\b(?:credential|password|api[ _-]?key|secret|token)\b.{0,80}\b(?:send|upload|export|exfiltrat|reveal|share|store)\b", re.I | re.S)),
    ("exfiltration", re.compile(r"\b(?:exfiltrat|upload|send|post)\b.{0,80}\b(?:environment|credential|secret|token|password|private key|\.env)\b", re.I | re.S)),
    ("execution_directive", re.compile(r"\b(?:always|never|must|should)\b.{0,80}\b(?:run|execute|invoke|call)\b.{0,80}\b(?:shell|command|terminal|tool|curl|wget|powershell|bash)\b", re.I | re.S)),
)


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def assess_automatic_memory(engram: Engram) -> MemorySafetyDecision:
    """Allow only policy-safe automatic semantic lessons.

    Explicitly trusted manual writes are outside this automatic pipeline.  All
    LLM-derived memories remain subject to content checks, including entries
    that claim to originate from a user.
    """
    text = _normalized(engram.text)
    if not text:
        return MemorySafetyDecision(False, "empty")

    trusted = getattr(engram, "trusted", False) or engram.source == "user"
    if engram.source in {"consolidation", "llm"} and not trusted and engram.kind in {"always", "never", "when"}:
        return MemorySafetyDecision(False, "automatic_behavioral_memory")

    # Identity is handled through its separate direct-user extraction path. It
    # is not a consolidation output, so retain compatibility here.
    if engram.kind == "profile":
        return MemorySafetyDecision(True)

    for reason, pattern in _CONTROL_PATTERNS:
        if pattern.search(text):
            return MemorySafetyDecision(False, reason)
    return MemorySafetyDecision(True)


def is_safe_for_prompt(engram: Engram) -> bool:
    """Return whether a stored entry may be rendered into model context.

    Existing behavioral memories remain compatible, but no control-plane text is
    ever treated as prompt context. Write-time provenance restrictions apply
    only to new automatic encoding.
    """
    text = _normalized(engram.text)
    if not text:
        return False
    return not any(pattern.search(text) for _, pattern in _CONTROL_PATTERNS)
