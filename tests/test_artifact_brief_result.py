"""What the calling agent learns about the brief from the tool result."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from anton.core.tools.generate_artifact import orchestrator
from anton.core.tools.generate_artifact.discovery import checkpoint as cp
from anton.core.tools.generate_artifact.state import GenState


def _state(tmp_path, **over) -> GenState:
    return GenState(
        session=SimpleNamespace(), artifact_type="html-app",
        artifact_path=tmp_path, slug="s", **over,
    )


def test_generated_reports_a_shown_brief_without_resending_it(tmp_path):
    result = orchestrator._finish(_state(tmp_path, act_first=True, brief="## B", brief_shown=True))
    assert result["brief_shown"] is True
    assert "brief_summary" not in result


def test_generated_hands_over_an_unshown_act_first_brief(tmp_path):
    result = orchestrator._finish(_state(tmp_path, act_first=True, brief="## B"))
    assert "brief_shown" not in result
    assert result["brief_summary"] == "## B"


def test_ask_first_generated_carries_no_brief(tmp_path):
    result = orchestrator._finish(_state(tmp_path, brief="## B"))
    assert "brief_shown" not in result
    assert "brief_summary" not in result


def test_a_budget_stop_after_a_shown_brief_does_not_resend_it(tmp_path):
    result = orchestrator._stopped_over_budget(
        _state(tmp_path, act_first=True, brief="## B", brief_shown=True), "x",
    )
    assert result["brief_shown"] is True
    assert "brief_summary" not in result


def test_an_ask_first_budget_stop_still_carries_the_brief(tmp_path):
    result = orchestrator._stopped_over_budget(_state(tmp_path, brief="## B"), "x")
    assert "brief_shown" not in result
    assert result["brief_summary"] == "## B"


async def _skipped(state, *args, **kwargs):
    return None


@pytest.mark.parametrize("entry", [cp.ENTRY_CONFIRM, cp.ENTRY_SPEC, cp.ENTRY_GENERATE])
@pytest.mark.parametrize("act_first", [True, False])
async def test_a_repeat_call_past_the_brief_counts_it_as_seen_only_when_acting_first(
    tmp_path, monkeypatch, entry, act_first,
):
    """An earlier turn showed the brief (for ENTRY_CONFIRM, the budget stop
    that asked to continue), so acting first it is neither resent nor
    re-shown; asking first, nothing changes."""
    async def prd_written(state, *, entry):
        return cp.STAGE_PRD_WRITTEN

    monkeypatch.setattr(orchestrator, "run_discovery", prd_written)
    for name in ("_data_phase", "_write_tech_spec", "_gen_verify_frontend"):
        monkeypatch.setattr(orchestrator, name, _skipped)
    state = _state(tmp_path, act_first=act_first, brief="## B")

    result = await orchestrator.run(state, entry=entry)

    assert result["status"] == "generated"
    assert "brief_summary" not in result
    assert ("brief_shown" in result) is act_first


def test_status_instructions_branch_on_brief_shown():
    from anton.core.tools.tool_handlers import _STATUS_INSTRUCTIONS

    for status in ("generated", "stopped_over_budget"):
        assert "brief_shown" in _STATUS_INSTRUCTIONS[status], status
