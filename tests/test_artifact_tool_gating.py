"""Tool availability is enforced in code, not by editing the schema array.

The array has to stay byte-identical across phases A-D for the prefix cache
to hold, so "remove it from `tools`" — the old mechanism — is unavailable.
"""
from __future__ import annotations

from anton.core.tools.generate_artifact.discovery import sub_tools as st


def test_the_pipeline_offers_the_same_five_tools_regardless_of_step():
    names = [t["name"] for t in st.pipeline_tool_schemas()]
    assert names == ["scratchpad", "web_search", "web_fetch", "ask_user", "finish_gathering"]


def test_the_tool_array_is_rebuilt_identically_every_time():
    """It is part of the cached prefix; a reordered or reshaped array is a
    changed prefix, and the cache miss lands on the largest context."""
    assert st.pipeline_tool_schemas() == st.pipeline_tool_schemas()


def test_gathering_allows_everything():
    for name in ("scratchpad", "web_search", "web_fetch", "ask_user", "finish_gathering"):
        assert st.rejection_for(st.STEP_GATHERING, name, questions_left=1) is None


def test_finish_gathering_is_rejected_while_drafting_the_brief():
    reason = st.rejection_for(st.STEP_DRAFT_BRIEF, "finish_gathering", questions_left=1)
    assert reason is not None
    assert "finish_gathering" in reason


def test_finish_gathering_is_allowed_on_the_redraw_step():
    """The redraw re-declares artifact type and data sources after a user
    correction — that is what opens the emergency data loop."""
    assert st.rejection_for(st.STEP_REDRAW_BRIEF, "finish_gathering", questions_left=0) is None


def test_draft_and_redraw_are_distinct_rows():
    """Collapsing them would either break the hot path (where the model
    reaches for finish_gathering unprompted) or break corrections (where the
    call is the only thing that re-declares a new source)."""
    assert st.ALLOWED_TOOLS_BY_STEP[st.STEP_DRAFT_BRIEF] != st.ALLOWED_TOOLS_BY_STEP[
        st.STEP_REDRAW_BRIEF
    ]


def test_spec_steps_allow_no_tools_at_all():
    for step in (st.STEP_TECH_SPEC, st.STEP_API_SPEC):
        for name in ("scratchpad", "web_search", "finish_gathering", "ask_user"):
            assert st.rejection_for(step, name, questions_left=1) is not None


def test_ask_user_is_rejected_by_code_when_the_budget_is_gone():
    """The tool stays in the array — dropping it would change the prefix —
    so the budget check moves into the gate."""
    assert "ask_user" in [t["name"] for t in st.pipeline_tool_schemas()]
    reason = st.rejection_for(st.STEP_GATHERING, "ask_user", questions_left=0)
    assert reason is not None
    assert "budget" in reason.lower()


def test_an_unknown_tool_is_rejected_on_every_step():
    assert st.rejection_for(st.STEP_GATHERING, "delete_everything", questions_left=1) is not None


def test_an_unknown_step_rejects_everything():
    """A step nobody registered must not silently inherit gathering's rights."""
    assert st.rejection_for("some_new_step", "scratchpad", questions_left=1) is not None
