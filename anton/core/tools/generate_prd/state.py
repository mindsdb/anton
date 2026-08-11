"""State carried across generate_prd's two phases.

One `PrdState` instance lives for the whole tool call — both the gathering
loop (phase 1, `engine.py`) and the brief/confirm/write sequence (phase 2,
`orchestrator.py`) mutate the SAME instance and append to the SAME
`messages` list, so phase 2 sees everything phase 1 did without a separate
hand-off object (see prd-design.md, "Context handoff between phases").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .debug_trace import NullTrace, PrdTrace  # noqa: F401  (PrdTrace re-exported for typing)

if TYPE_CHECKING:
    from anton.chat_session import ChatSession


# Reserved out of MAX_QUESTIONS_PER_TURN for phase 2: one `show_and_confirm`
# call plus up to two "revise brief, show again" cycles. Not a separate hard
# cap — the shared budget itself is what eventually stops the revise loop
# (elicit() returns "limit"), this only decides how many of the turn's
# questions phase 1 is allowed to spend before that.
PHASE2_RESERVED_QUESTIONS = 3


@dataclass
class PrdState:
    session: "ChatSession | Any"
    slug: str
    artifact_path: Path
    artifact_type: str
    user_request: str
    agent_understanding: str
    known_data: str
    user_preferences: str
    messages: list[dict] = field(default_factory=list)
    qa_log: list[str] = field(default_factory=list)
    final_artifact_type: str = ""
    gathering_notes: str = ""
    brief_markdown: str = ""
    trace_log: "PrdTrace | NullTrace" = field(default_factory=NullTrace)

    def record_qa(self, question: str, answer_summary: str) -> None:
        self.qa_log.append(f"- **Q:** {question}\n  **A:** {answer_summary}")

    def qa_log_markdown(self) -> str:
        return "\n".join(self.qa_log) if self.qa_log else "(no questions were asked)"


def gathering_question_budget(session: "ChatSession | Any") -> int:
    """How many `ask_user` calls phase 1 may make this call to
    `run_gathering_loop`, reserving `PHASE2_RESERVED_QUESTIONS` for phase 2.
    Never negative. Recomputed on every call (not cached on `PrdState`)
    because `session.question_count` keeps changing as questions are asked.
    """
    from anton.core.interaction.elicit import MAX_QUESTIONS_PER_TURN

    remaining = MAX_QUESTIONS_PER_TURN - getattr(session, "question_count", 0)
    return max(0, remaining - PHASE2_RESERVED_QUESTIONS)
