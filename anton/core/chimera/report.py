"""What a delegation hands back to the main agent."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal

SidekickStatus = Literal["done", "needs_main", "budget_exhausted", "failed"]
SIDEKICK_STATUSES: tuple[str, ...] = ("done", "needs_main", "budget_exhausted", "failed")

# What the main agent should do with each status. Sent with the report so the
# instruction sits beside the evidence it applies to, not only in the tool's
# prompt thousands of tokens back.
_NEXT = {
    "done": (
        "Review the outputs and evidence against done_when before relying on them: "
        "check one or two claims yourself. Then carry on with your plan."
    ),
    "needs_main": (
        "The sidekick handed this back. Read open_questions, decide or ask the "
        "user yourself, then finish the work or delegate again with a sharper brief."
    ),
    "budget_exhausted": (
        "The sidekick ran out of rounds before finishing. Use what it got done, "
        "then finish the rest yourself or delegate a smaller remainder."
    ),
    "failed": "The delegation failed. Do the work yourself; do not retry the same brief.",
}


@dataclass
class SidekickReport:
    """One delegation's result. ``tokens`` counts every model call the
    delegation made, Jev's checks and the tools' own calls included;
    ``cost_usd`` is None when no price is known for the models (anton keeps no
    price table)."""

    status: SidekickStatus
    summary: str
    outputs: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    rounds: int = 0
    tokens: int = 0
    cost_usd: float | None = None

    def to_tool_content(self) -> str:
        """The compact JSON the main agent sees as the tool result: empty lists
        and an unknown cost left out, plus what to do next for this status."""
        fields = {
            key: value for key, value in asdict(self).items()
            if value not in ([], None)
        }
        fields["next"] = _NEXT[self.status]
        return json.dumps(fields, ensure_ascii=False, separators=(",", ":"))


def str_list(value: object, *, limit: int = 20, chars: int = 2000) -> list[str]:
    """A model-supplied list of strings, cleaned: a lone string becomes one
    item, blanks and non-strings go, and each item and the list are capped so
    one runaway field cannot flood the main agent's context."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    items = [str(item).strip() for item in value if isinstance(item, str | int | float) and str(item).strip()]
    return [item[:chars] for item in items[:limit]]
