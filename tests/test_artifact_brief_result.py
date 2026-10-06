"""What the calling agent learns about the brief from the tool result."""
from __future__ import annotations

from types import SimpleNamespace

from anton.core.tools.generate_artifact import orchestrator
from anton.core.tools.generate_artifact.discovery import checkpoint as cp
from anton.core.tools.generate_artifact.state import GenState


def _state(tmp_path, **over) -> GenState:
    return GenState(
        session=SimpleNamespace(), artifact_type="html-app",
        artifact_path=tmp_path, slug="s", **over,
    )


def test_generated_reports_a_shown_brief(tmp_path):
    result = orchestrator._finish(_state(tmp_path, act_first=True, brief="## B", brief_shown=True))
    assert result["brief_shown"] is True
    assert result["brief_summary"] == "## B"


def test_generated_hands_over_an_unshown_act_first_brief(tmp_path):
    result = orchestrator._finish(_state(tmp_path, act_first=True, brief="## B"))
    assert "brief_shown" not in result
    assert result["brief_summary"] == "## B"


def test_ask_first_generated_carries_no_brief(tmp_path):
    result = orchestrator._finish(_state(tmp_path, brief="## B"))
    assert "brief_shown" not in result
    assert "brief_summary" not in result


def test_a_budget_stop_says_the_brief_was_shown(tmp_path):
    result = orchestrator._stopped_over_budget(
        _state(tmp_path, act_first=True, brief="## B", brief_shown=True), "x",
    )
    assert result["brief_shown"] is True
    assert result["brief_summary"] == "## B"


def test_a_resumed_act_first_run_does_not_retell_the_brief(tmp_path):
    """The user saw the brief a turn ago: turn 1 showed it and then stopped
    over budget (before the PRD for ENTRY_CONFIRM, after it otherwise)."""
    for entry in (cp.ENTRY_CONFIRM, cp.ENTRY_SPEC, cp.ENTRY_GENERATE):
        state = _state(tmp_path, act_first=True, brief="## B", entry=entry)
        result = orchestrator._finish(state)
        assert "brief_summary" not in result, entry
        assert "brief_shown" not in result, entry
        assert orchestrator._stopped_over_budget(state, "x")["brief_shown"] is True, entry


def test_a_resumed_ask_first_budget_stop_is_unchanged(tmp_path):
    result = orchestrator._stopped_over_budget(
        _state(tmp_path, brief="## B", entry=cp.ENTRY_SPEC), "x",
    )
    assert "brief_shown" not in result
    assert result["brief_summary"] == "## B"


def test_status_instructions_branch_on_brief_shown():
    from anton.core.tools.tool_handlers import _STATUS_INSTRUCTIONS

    for status in ("generated", "stopped_over_budget"):
        assert "brief_shown" in _STATUS_INSTRUCTIONS[status], status
