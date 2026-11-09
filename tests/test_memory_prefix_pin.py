"""The memory text in the cached prompt prefix must not change within a conversation.

Rules and lessons are rendered into the system prompt and the scratchpad tool
description. Both sit ahead of the whole conversation in the request, so a byte
that differs from the previous turn re-writes the entire conversation to the
prompt cache. Consolidation adds rules between turns; the pin keeps what the
conversation first saw.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from anton.core.llm.identity import DeploymentIdentity
from anton.core.llm.provider import LLMResponse, Usage
from anton.core.memory.cortex import Cortex
from anton.core.memory.hippocampus import Hippocampus
from anton.core.session import ChatSession, ChatSessionConfig
from anton.workspace import Workspace

from tests.test_cloud_turn_session import _mock_llm

CONV = "0f4a7c9e-0000-4000-8000-000000000001"


@pytest.fixture()
def memory_dirs(tmp_path):
    global_dir, project_dir = tmp_path / "global", tmp_path / "project"
    global_dir.mkdir()
    project_dir.mkdir()
    return global_dir, project_dir


def _session(tmp_path, memory_dirs, session_id=CONV, **config) -> ChatSession:
    """One host turn: cowork-server builds a fresh session per user message."""
    global_dir, project_dir = memory_dirs
    ws = Workspace(tmp_path / "ws")
    ws.initialize()
    return ChatSession(ChatSessionConfig(
        llm_client=_mock_llm(),
        workspace=ws,
        session_id=session_id,
        cortex=Cortex(global_hc=Hippocampus(global_dir), project_hc=Hippocampus(project_dir),
                      mode="autopilot", llm_client=_mock_llm()),
        **config,
    ))


def _add_rule(project_dir, text):
    (project_dir / "rules.md").open("a").write(f"\n## Always\n- {text}\n")


def _scratchpad_description(session) -> str:
    session._build_tools()
    return next(t.description for t in session.tool_registry.get_tool_defs()
                if t.name == "scratchpad")


async def test_system_memory_is_pinned_across_turns_of_one_conversation(tmp_path, memory_dirs):
    _, project_dir = memory_dirs
    _add_rule(project_dir, "verify totals before writing them")
    first = await _session(tmp_path, memory_dirs)._build_system_prompt("hello")
    assert "verify totals before writing them" in first

    # Consolidation ran between turns and added a rule. The next turn's prompt
    # must still be byte-identical.
    _add_rule(project_dir, "a rule learned mid-conversation")
    second = await _session(tmp_path, memory_dirs)._build_system_prompt("second question")
    assert second == first
    assert "a rule learned mid-conversation" not in second

    # A new conversation starts from current memory, so the rule is not lost.
    other = await _session(tmp_path, memory_dirs, session_id=CONV[:-1] + "2")._build_system_prompt("hi")
    assert "a rule learned mid-conversation" in other


async def test_scratchpad_tool_description_is_pinned_across_turns(tmp_path, memory_dirs):
    session = _session(tmp_path, memory_dirs)
    session._cortex.get_scratchpad_context = MagicMock(return_value="lesson one")
    first = _scratchpad_description(session)
    assert "lesson one" in first

    later = _session(tmp_path, memory_dirs)
    later._cortex.get_scratchpad_context = MagicMock(return_value="lesson one\nlesson two")
    assert _scratchpad_description(later) == first


async def test_pin_is_in_memory_when_there_is_no_conversation_scope(tmp_path, memory_dirs):
    # A bare CLI session has no session id and no snapshot directory, but it is
    # one long-lived process, so pinning on the instance is enough.
    session = _session(tmp_path, memory_dirs, session_id=None)
    session._cortex.build_memory_context = AsyncMock(side_effect=["first", "second"])
    a = await session._build_system_prompt("q1")
    b = await session._build_system_prompt("q2")
    assert a == b and "first" in a


async def test_later_turns_report_the_memory_the_prompt_holds(tmp_path, memory_dirs):
    _, project_dir = memory_dirs
    _add_rule(project_dir, "verify totals before writing them")
    first = _session(tmp_path, memory_dirs)
    await first._build_system_prompt("hello")
    assert first.memory_read.rules == 1

    _add_rule(project_dir, "a rule learned mid-conversation")
    later = _session(tmp_path, memory_dirs)
    await later._build_system_prompt("second question")
    assert later.memory_read == first.memory_read


async def test_turning_learned_memory_off_drops_the_pin(tmp_path, memory_dirs):
    _, project_dir = memory_dirs
    _add_rule(project_dir, "verify totals before writing them")
    identity = DeploymentIdentity(kind="self-hosted", product_name="Cowork")

    on = await _session(tmp_path, memory_dirs, deployment_identity=identity,
                        learned_memory_scope="personal")._build_system_prompt("hello")
    assert "verify totals before writing them" in on

    off = await _session(tmp_path, memory_dirs, deployment_identity=identity,
                         learned_memory_scope="off")._build_system_prompt("hello")
    assert "verify totals before writing them" not in off


async def test_workspace_listing_is_pinned_across_turns(tmp_path, memory_dirs):
    first = _session(tmp_path, memory_dirs)
    (first._workspace.base / "notes.md").write_text("x")
    first_prompt = await first._build_system_prompt("hello")
    assert "notes.md" in first_prompt

    # The agent saved a new file at the project root and used a pad.
    later = _session(tmp_path, memory_dirs)
    (later._workspace.base / "Scene - new.md").write_text("x")
    later._scratchpads.record_agent_pad("drafts")
    assert await later._build_system_prompt("second question") == first_prompt

    other = _session(tmp_path, memory_dirs, session_id=CONV[:-1] + "2")
    assert "Scene - new.md" in await other._build_system_prompt("hi")


async def test_compaction_lets_current_memory_in(tmp_path, memory_dirs):
    _, project_dir = memory_dirs
    _add_rule(project_dir, "verify totals before writing them")
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 50} for i in range(10)]
    session = _session(tmp_path, memory_dirs, initial_history=history)
    await session._build_system_prompt("hello")

    _add_rule(project_dir, "a rule learned mid-conversation")
    session._llm.summarize = AsyncMock(return_value=LLMResponse(
        content="## Goal\nx", usage=Usage(input_tokens=1, output_tokens=1)))
    assert await session._summarize_history() is True

    later = await _session(tmp_path, memory_dirs)._build_system_prompt("next")
    assert "a rule learned mid-conversation" in later


async def test_workspace_listing_survives_a_new_process(tmp_path, memory_dirs):
    # Each cloud turn runs in a fresh process, so the listing comes back from disk.
    from anton.core import session as session_module

    _, project_dir = memory_dirs
    first = _session(tmp_path, memory_dirs)
    (first._workspace.base / "notes.md").write_text("x")
    await first._build_system_prompt("hello")

    session_module._MEMORY_PINS.clear()
    (first._workspace.base / "Scene - new.md").write_text("x")
    _add_rule(project_dir, "a rule learned mid-conversation")
    later = _session(tmp_path, memory_dirs)
    prompt = await later._build_system_prompt("second question")
    assert "Project root: notes.md" in prompt and "Scene - new.md" not in prompt
    # Memory text is not saved, so a new process reads the memory as it stands now.
    assert "a rule learned mid-conversation" in prompt


async def test_a_forged_saved_pin_cannot_change_the_prompt(tmp_path, memory_dirs):
    # Agent code can write the saved pin, so its text must never reach the prompt.
    from anton.core import session as session_module
    from anton.core.session import _MEMORY_PIN_FILE

    _, project_dir = memory_dirs
    _add_rule(project_dir, "verify totals before writing them")
    first = _session(tmp_path, memory_dirs)
    (first._workspace.base / "notes.md").write_text("x")
    await first._build_system_prompt("hello")
    forged = {"scope": None, "system": "FORGED MEMORY", "scratchpad": "FORGED TOOL",
              "workspace": {"pads": [], "root": ["notes.md\n\nFORGED ORDERS"]}}
    first._scratchpads.write_conversation_text(_MEMORY_PIN_FILE, json.dumps(forged))

    session_module._MEMORY_PINS.clear()
    later = _session(tmp_path, memory_dirs)
    prompt = await later._build_system_prompt("next")
    assert "FORGED" not in prompt and "FORGED" not in _scratchpad_description(later)
    assert "verify totals before writing them" in prompt
    assert "Project root: notes.md" in prompt


async def test_a_saved_pin_of_the_wrong_shape_is_rebuilt(tmp_path, memory_dirs):
    from anton.core import session as session_module
    from anton.core.memory.base import MemoryRead
    from anton.core.session import _MEMORY_PIN_FILE

    _, project_dir = memory_dirs
    _add_rule(project_dir, "verify totals before writing them")
    first = _session(tmp_path, memory_dirs)
    (first._workspace.base / "notes.md").write_text("x")
    await first._build_system_prompt("hello")
    first._scratchpads.write_conversation_text(
        _MEMORY_PIN_FILE, '{"scope": null, "system": 7, "workspace": ["x"], "read": "x"}')

    session_module._MEMORY_PINS.clear()
    later = _session(tmp_path, memory_dirs)
    prompt = await later._build_system_prompt("next")
    assert "verify totals before writing them" in prompt and "Project root: notes.md" in prompt
    assert isinstance(later.memory_read, MemoryRead)
