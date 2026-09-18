"""System prompt and tool array must not change across phases A-D.

They are the cached prefix. Rewriting either at a phase switch discards the
cache for the whole shared history, and the largest context in the pipeline
is the one that would pay for it.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from anton.core.tools.generate_artifact.discovery import prompts as p
from anton.core.tools.generate_artifact.discovery import sub_tools as st
from anton.core.tools.generate_artifact.state import GenState


def _state(**over) -> GenState:
    base = dict(
        session=SimpleNamespace(question_count=0),
        artifact_type="html-app",
        artifact_path=Path("/tmp/x"),
        slug="x",
        user_request="build a dashboard from https://example.com/a",
        agent_understanding="a dashboard",
    )
    base.update(over)
    return GenState(**base)


def test_the_system_prompt_is_built_once_and_cached_on_the_state():
    state = _state()
    first = state.pipeline_system
    state.final_artifact_type = "fullstack-stateless-app"
    state.declared_sources = ["orders table"]
    assert state.pipeline_system == first


def test_the_tool_array_is_built_once_and_cached_on_the_state():
    state = _state()
    first = state.pipeline_tools
    assert state.pipeline_tools is first


def test_the_system_prompt_carries_no_mutable_values():
    """artifact_type changes mid-run via finish_gathering, and the question
    budget drains as questions are asked. Either one baked into the prompt
    would be stale the moment it changed, and the prompt cannot be rewritten
    without discarding the cache."""
    state = _state(artifact_type="fullstack-stateful-app")
    system = p.build_pipeline_system_prompt(state)
    # The type appears only in the list of valid types, never as "the current
    # type of this artifact".
    assert "Initially registered artifact type" not in system
    assert "questions left" not in system.lower()
    assert "Questions you may still ask" not in system


def test_the_system_prompt_carries_the_immutable_addressing():
    state = _state()
    system = p.build_pipeline_system_prompt(state)
    assert state.slug in system
    assert str(state.artifact_path) in system


def test_every_step_has_a_user_message_carrying_its_instruction():
    state = _state()
    for step in (
        st.STEP_GATHERING,
        st.STEP_DRAFT_BRIEF,
        st.STEP_REDRAW_BRIEF,
        st.STEP_WRITE_PRD,
    ):
        message = p.step_message(step, state)
        assert isinstance(message, str) and message.strip()
        assert message.startswith(f"STEP: {step}")


def test_the_step_message_is_where_the_mutable_values_live():
    state = _state(artifact_type="fullstack-stateful-app")
    message = p.step_message(st.STEP_GATHERING, state)
    assert "fullstack-stateful-app" in message
    assert "Questions you may still ask" in message


def test_the_kickoff_carries_all_four_call_fields():
    """These lived in the phase-1 system prompt, which no longer exists. A
    gathering step that cannot see the request gathers for nothing."""
    state = _state(
        user_request="build a dashboard from https://example.com/a",
        agent_understanding="a dashboard of the article's figures",
        known_data="pad `dash`, cell 3",
        user_preferences="dark theme",
    )
    kickoff = p.build_call_kickoff(state)
    assert "https://example.com/a" in kickoff
    assert "a dashboard of the article's figures" in kickoff
    assert "pad `dash`, cell 3" in kickoff
    assert "dark theme" in kickoff


def test_a_step_with_no_history_is_handed_the_restored_material():
    """Cold start: there is no conversation to continue, so the brief and the
    notes have to arrive in the message itself."""
    state = _state(
        messages=[],
        brief="## Goal\nA dashboard of weekly figures.",
        data_notes="Scratchpad `dash`:\n```python\nrows = q()\n```",
        declared_sources=["orders table"],
    )
    message = p.step_message(st.STEP_REDRAW_BRIEF, state)
    assert "A dashboard of weekly figures." in message
    assert "rows = q()" in message
    assert "orders table" in message


def test_a_step_with_history_does_not_repeat_the_restored_material():
    """On the hot path the same material is already in the shared list;
    sending it twice would grow every cached prefix for nothing."""
    state = _state(
        messages=[{"role": "user", "content": "kickoff"}],
        brief="## Goal\nA dashboard of weekly figures.",
    )
    message = p.step_message(st.STEP_REDRAW_BRIEF, state)
    assert "A dashboard of weekly figures." not in message


def test_the_first_message_carries_the_call_fields_exactly_once():
    """The kickoff is prepended explicitly AND `restored_context` opens with
    the same block. Gathering is therefore excluded from the restore tail —
    without that, the four call fields would sit twice inside the cached
    prefix of every phase A-D call, for the whole run."""
    state = _state(messages=[], user_request="build a dashboard from an article")
    first_message = (
        p.build_call_kickoff(state) + "\n\n"
        + p.step_message(st.STEP_GATHERING, state)
    )
    assert first_message.count("build a dashboard from an article") == 1
    assert first_message.count("## User request") == 1


def test_gathering_never_gets_the_restore_tail():
    """It never runs cold: a repeat call restores phase A from disk, and the
    re-entry path always already has a history."""
    state = _state(messages=[], brief="## Goal\nA dashboard.")
    assert "A dashboard." not in p.step_message(st.STEP_GATHERING, state)
