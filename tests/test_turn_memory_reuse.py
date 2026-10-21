"""The memory section is built once per turn, so a retry does not repeat its rule filter model calls."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from anton.core.llm.provider import LLMResponse
from anton.core.session import ChatSession, ChatSessionConfig

from tests.conftest import make_mock_llm, run_turn


def _session() -> ChatSession:
    cortex = MagicMock()
    cortex.build_memory_context = AsyncMock(side_effect=["memory v1", "memory v2", "memory v3"])
    cortex.get_scratchpad_context = MagicMock(return_value="")
    return ChatSession(ChatSessionConfig(llm_client=make_mock_llm(), cortex=cortex))


async def test_retry_reuses_the_memory_section_built_at_turn_start():
    session = _session()
    ok = LLMResponse(content="done", stop_reason="end_turn")
    session._llm.plan = AsyncMock(side_effect=[RuntimeError("provider blip"), ok])

    assert await run_turn(session, "hello") == "done"

    assert session._cortex.build_memory_context.await_count == 1
    systems = [c.kwargs["system"] for c in session._llm.plan.await_args_list]
    assert len(systems) == 2
    assert all("memory v1" in s for s in systems)


async def test_each_turn_builds_the_memory_section_fresh():
    session = _session()
    session._llm.plan = AsyncMock(return_value=LLMResponse(content="done", stop_reason="end_turn"))

    await run_turn(session, "hello")
    await run_turn(session, "hello")

    assert session._cortex.build_memory_context.await_count == 2
    assert "memory v2" in session._llm.plan.await_args_list[-1].kwargs["system"]


async def test_memory_read_counts_the_memory_section_the_turn_built(tmp_path):
    from anton.core.memory.base import MemoryRead
    from anton.core.memory.cortex import Cortex
    from anton.core.memory.hippocampus import Hippocampus

    global_dir, project_dir = tmp_path / "global", tmp_path / "project"
    Hippocampus(global_dir).encode_rule("Use httpx", kind="always", confidence="high", source="user")
    Hippocampus(project_dir).encode_lesson("Project fact")
    Hippocampus(project_dir).encode_lesson("Scratchpad cells time out at 30s")
    cortex = Cortex(global_hc=Hippocampus(global_dir), project_hc=Hippocampus(project_dir), mode="off")
    session = ChatSession(ChatSessionConfig(llm_client=make_mock_llm(), cortex=cortex))
    session._llm.plan = AsyncMock(return_value=LLMResponse(content="done", stop_reason="end_turn"))
    assert session.memory_read is None

    await run_turn(session, "hello")

    assert "Use httpx" in session._llm.plan.await_args_list[-1].kwargs["system"]
    assert session.memory_read == MemoryRead(learned=True, rules=1, lessons=1, scratchpad=1)


def test_a_session_without_a_cortex_reads_no_memory():
    assert ChatSession(ChatSessionConfig(llm_client=make_mock_llm())).memory_read is None
