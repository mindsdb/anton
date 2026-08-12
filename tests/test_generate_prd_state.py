"""PrdState and the phase-2 question-budget reservation."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from anton.core.interaction.elicit import MAX_QUESTIONS_PER_TURN
from anton.core.tools.generate_prd.state import (
    PHASE2_RESERVED_QUESTIONS,
    PrdState,
    gathering_question_budget,
)


def _state(**over) -> PrdState:
    base = dict(
        session=SimpleNamespace(question_count=0),
        slug="s",
        artifact_path=Path("/tmp/s"),
        artifact_type="html-app",
        user_request="build a dashboard",
        agent_understanding="a dashboard",
        known_data="",
        user_preferences="",
    )
    base.update(over)
    return PrdState(**base)


def test_reserved_questions_is_three():
    assert PHASE2_RESERVED_QUESTIONS == 3


def test_budget_reserves_three_for_phase_two():
    # Against the imported constants, not a hardcoded 8 - 3: a hardcoded
    # number here would silently stop reflecting reality the next time
    # MAX_QUESTIONS_PER_TURN changes — precisely the problem Task 2 fixed
    # for the pinned budget test itself.
    session = SimpleNamespace(question_count=0)
    assert gathering_question_budget(session) == MAX_QUESTIONS_PER_TURN - PHASE2_RESERVED_QUESTIONS


def test_budget_accounts_for_questions_already_spent_by_the_main_agent():
    session = SimpleNamespace(question_count=2)
    assert gathering_question_budget(session) == MAX_QUESTIONS_PER_TURN - 2 - PHASE2_RESERVED_QUESTIONS


def test_budget_never_goes_negative():
    session = SimpleNamespace(question_count=MAX_QUESTIONS_PER_TURN)
    assert gathering_question_budget(session) == 0


def test_record_qa_and_markdown_rendering():
    state = _state()
    assert state.qa_log_markdown() == "(no questions were asked)"
    state.record_qa("Which theme?", "dark")
    md = state.qa_log_markdown()
    assert "Which theme?" in md
    assert "dark" in md
