"""Prompt builders for the discovery phases.

The per-phase system prompts are gone: phases A-D share one immutable system
prompt so the provider's prefix cache holds across the whole region, and what
used to differ between them now arrives as a step message. What each builder
must still carry is asserted here; the stability of the shared prefix itself
is locked in test_artifact_prefix_stability.py.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from anton.core.artifacts.models import ARTIFACT_TYPES
from anton.core.tools.generate_artifact.discovery import sub_tools
from anton.core.tools.generate_artifact.discovery.prompts import (
    GATHERING_CONTINUE,
    build_call_kickoff,
    build_pipeline_system_prompt,
    step_message,
)
from anton.core.tools.generate_artifact.state import GenState


def _state(**over) -> GenState:
    base = dict(
        session=SimpleNamespace(question_count=0),
        slug="s",
        artifact_path=Path("/tmp/s"),
        artifact_type="html-app",
        user_request="build a dashboard of BTC price",
        agent_understanding="a live BTC price dashboard",
        known_data="CoinGecko public API",
        user_preferences="prefers dark mode",
    )
    base.update(over)
    return GenState(**base)


def test_pipeline_system_prompt_lists_every_valid_artifact_type():
    prompt = build_pipeline_system_prompt(_state())
    for artifact_type in ARTIFACT_TYPES:
        assert artifact_type in prompt


def test_pipeline_system_prompt_explains_that_a_listed_tool_may_be_unavailable():
    """The array is fixed for the whole region, so "it is in the list" no
    longer implies "you may call it". A model that assumes otherwise spends
    a round finding out."""
    prompt = build_pipeline_system_prompt(_state())
    assert "does NOT mean it is available" in prompt


def test_call_kickoff_has_the_four_sections():
    kickoff = build_call_kickoff(_state())
    for header in (
        "## User request",
        "## Agent's understanding",
        "## Known data",
        "## User preferences",
    ):
        assert header in kickoff


def test_call_kickoff_carries_the_request_understanding_and_context():
    kickoff = build_call_kickoff(_state())
    assert "build a dashboard of BTC price" in kickoff
    assert "a live BTC price dashboard" in kickoff
    assert "CoinGecko" in kickoff
    assert "prefers dark mode" in kickoff


def test_the_gathering_step_asks_for_finish_gathering():
    assert "finish_gathering" in step_message(sub_tools.STEP_GATHERING, _state())


def test_the_continue_message_mentions_finish_gathering():
    assert "finish_gathering" in GATHERING_CONTINUE


def test_the_brief_step_does_not_invite_finish_gathering():
    """It used to be kept out of the tool array on this step because a model
    reaches for it whenever a prompt says "you have finished gathering". The
    array is fixed now, so the guarantee moves to the gate — but the step's
    own instruction should not be inviting the call either."""
    message = step_message(sub_tools.STEP_DRAFT_BRIEF, _state())
    assert "finish_gathering" not in message
    assert sub_tools.rejection_for(
        sub_tools.STEP_DRAFT_BRIEF, "finish_gathering", questions_left=1
    ) is not None


def test_the_redraw_step_requires_finish_gathering():
    """The opposite of the row above, and the reason the two steps are
    separate: after a correction this call is what re-declares the sources."""
    message = step_message(sub_tools.STEP_REDRAW_BRIEF, _state())
    assert "finish_gathering" in message
    assert sub_tools.rejection_for(
        sub_tools.STEP_REDRAW_BRIEF, "finish_gathering", questions_left=0
    ) is None
