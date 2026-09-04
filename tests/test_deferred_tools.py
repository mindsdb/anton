"""Deferred tools (`unlock_skill`) — the ENG-764 on-demand tool mechanism.

Covered here:
- allowlist coexistence: a deferred tool listed in `tool_allowlist` must not
  make `_build_tools` raise before it unlocks;
- history replay: a prior `recall_skill` in loaded history re-unlocks its
  bundle on the first build (survives a session/server restart);
- same-turn visibility: a bundle unlocked by a `recall_skill` this round is in
  the tools sent to the follow-up request in the same turn.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from anton.core.llm.provider import LLMResponse, ToolCall, Usage
from anton.core.memory.skills import Skill, SkillStore
from anton.core.tools.tool_defs import ToolDef
from tests.conftest import run_turn


async def _noop_handler(session, tc_input) -> str:  # pragma: no cover - never called
    return ""


def _deferred_tool(name: str = "deferred_tool") -> ToolDef:
    return ToolDef(
        name=name,
        description="deferred",
        input_schema={"type": "object", "properties": {}},
        handler=_noop_handler,
        unlock_skill="test-skill",
    )


def _names(session) -> set[str]:
    return {t["name"] for t in session._build_tools()}


def test_allowlisted_deferred_tool_does_not_raise_and_is_hidden(make_session):
    tool = _deferred_tool()
    session = make_session(tools=[tool], tool_allowlist=frozenset({tool.name}))
    # Before the fix this raised ValueError: the name was in the allowlist but
    # not yet registered, so it counted as "unknown".
    assert tool.name not in _names(session)


def test_deferred_tool_appears_after_its_bundle_unlocks(make_session):
    tool = _deferred_tool()
    session = make_session(tools=[tool], tool_allowlist=frozenset({tool.name}))
    _names(session)  # first build populates _deferred_bundles
    session._register_tool_bundle("test-skill")
    # It survives the allowlist re-enforcement on the next build.
    assert tool.name in _names(session)


def test_deferred_tool_left_out_of_allowlist_stays_filtered(make_session):
    """Deferral only changes *timing*; the allowlist still governs whether an
    unlocked tool is allowed at all."""
    tool = _deferred_tool()
    session = make_session(tools=[tool], tool_allowlist=frozenset({"scratchpad"}))
    _names(session)
    session._register_tool_bundle("test-skill")
    assert tool.name not in _names(session)


def test_bundle_replays_from_prior_recall_in_history(make_session):
    """A `recall_skill` already in the loaded history re-unlocks its bundle on
    the first build — so sticky tools survive a session rebuild (server
    restart) without waiting for the model to recall again."""
    tool = _deferred_tool()
    history = [
        {"role": "user", "content": "connect my db"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tc_hist",
                    "name": "recall_skill",
                    # Canonical label: store.load returns None here (no skill
                    # on disk), so replay falls back to this raw label, which
                    # already equals the tool's `unlock_skill`.
                    "input": {"label": "test-skill"},
                }
            ],
        },
    ]
    session = make_session(tools=[tool], initial_history=history)
    assert tool.name in _names(session)


def test_replay_resolves_non_canonical_label_via_store(make_session, tmp_path):
    """Replay must resolve the raw historical label through the store to the
    canonical skill label, exactly as the live recall paths do. A live
    `recall_skill(label="test_skill")` (underscore) resolves to the `test-skill`
    bundle; after a restart, replay must re-unlock the same bundle from that
    underscore label — not miss because the raw string differs from the key."""
    store = SkillStore(root=tmp_path / "skills")
    store.save(
        Skill(
            label="test-skill",
            name="Test Skill",
            description="unlocks the deferred tool",
            declarative_md="1. Do the thing.",
            created_at="2026-01-01T00:00:00+00:00",
            provenance="host",
        )
    )
    tool = _deferred_tool()  # unlock_skill="test-skill"
    history = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tc_hist",
                    "name": "recall_skill",
                    "input": {"label": "test_skill"},  # underscore, non-canonical
                }
            ],
        },
    ]
    session = make_session(tools=[tool], initial_history=history)
    session._skill_store = store  # bypass settings wiring; mirrors same-turn test
    assert tool.name in _names(session)


async def test_recalled_bundle_reaches_the_same_turn_followup(make_session, tmp_path):
    """The rebuild after tool dispatch must hand the follow-up request the tool
    that this round's `recall_skill` just unlocked — otherwise the model can't
    use it until the next turn."""
    label = "test-skill"
    store = SkillStore(root=tmp_path / "skills")
    store.save(
        Skill(
            label=label,
            name="Test Skill",
            description="unlocks the deferred tool",
            declarative_md="1. Do the thing.",
            created_at="2026-01-01T00:00:00+00:00",
            provenance="host",
        )
    )

    tool = _deferred_tool()
    session = make_session(tools=[tool])
    session._skill_store = store  # bypass settings wiring; mirrors test_skills_e2e
    session._llm.plan = AsyncMock(
        side_effect=[
            LLMResponse(
                content="recalling",
                tool_calls=[
                    ToolCall(id="tc1", name="recall_skill", input={"label": label})
                ],
                usage=Usage(input_tokens=10, output_tokens=20),
                stop_reason="tool_use",
            ),
            LLMResponse(
                content="done",
                tool_calls=[],
                usage=Usage(input_tokens=10, output_tokens=20),
                stop_reason="end_turn",
            ),
        ]
    )

    await run_turn(session, "connect my db")

    def _tool_names(call):
        return {t["name"] for t in call.kwargs["tools"]}

    calls = session._llm.plan.call_args_list
    assert tool.name not in _tool_names(calls[0])  # deferred: hidden up front
    assert tool.name in _tool_names(calls[-1])  # unlocked and visible same turn


if __name__ == "__main__":  # allow a bare `python tests/test_deferred_tools.py`
    raise SystemExit(pytest.main([__file__, "-q"]))
