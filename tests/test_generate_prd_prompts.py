"""Prompt builders for generate_prd's two phases."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from anton.core.tools.generate_prd.prompts import (
    build_gathering_continue_message,
    build_gathering_kickoff,
    build_gathering_system_prompt,
    build_phase2_system_prompt,
)
from anton.core.tools.generate_prd.state import PrdState


def _state(**over) -> PrdState:
    base = dict(
        session=SimpleNamespace(question_count=0),
        slug="s",
        artifact_path=Path("/tmp/s"),
        artifact_type="html-app",
        user_request="build a dashboard of BTC price",
        agent_understanding="a live BTC price dashboard",
        known_data="",
        user_preferences="prefers dark mode",
    )
    base.update(over)
    return PrdState(**base)


def test_gathering_system_prompt_carries_the_request_and_understanding():
    prompt = build_gathering_system_prompt(_state())
    assert "build a dashboard of BTC price" in prompt
    assert "a live BTC price dashboard" in prompt
    assert "finish_gathering" in prompt


def test_gathering_system_prompt_lists_every_valid_artifact_type():
    """Without an explicit closed list, the model can invent an artifact
    type string that later crashes write_prd's `ArtifactStore.update`
    (raises ValueError outside ARTIFACT_TYPES) — see also the schema's own
    `enum` constraint in sub_tools.FINISH_GATHERING_SCHEMA."""
    from anton.core.artifacts.models import ARTIFACT_TYPES

    prompt = build_gathering_system_prompt(_state())
    for artifact_type in ARTIFACT_TYPES:
        assert artifact_type in prompt


def test_gathering_system_prompt_carries_known_data_and_preferences():
    state = _state(known_data="scratchpad `btc`, cell 1: fetched from CoinGecko")
    prompt = build_gathering_system_prompt(state)
    assert "CoinGecko" in prompt
    assert "prefers dark mode" in prompt


def test_gathering_kickoff_has_the_four_sections():
    kickoff = build_gathering_kickoff(_state())
    for header in ("## User request", "## Agent's understanding", "## Known data", "## User preferences"):
        assert header in kickoff


def test_gathering_continue_message_mentions_finish_gathering():
    assert "finish_gathering" in build_gathering_continue_message()


def test_phase2_system_prompt_does_not_mention_the_gathering_tool():
    """Phase 2's `plan()` calls do carry a `tools` list (see orchestrator.py,
    Task 5) — but only so the Anthropic API accepts a `messages` history
    that already contains phase 1's tool_use/tool_result blocks. Mentioning
    `finish_gathering` by name here would suggest the model should call it,
    which it must not do in this phase."""
    prompt = build_phase2_system_prompt(_state())
    assert "finish_gathering" not in prompt
