from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _stub_cortex(**overrides):
    return SimpleNamespace(
        mode="off",
        build_memory_context=lambda: "",
        get_scratchpad_context=lambda: "",
        **overrides,
    )


def _stub_episodic(**overrides):
    return SimpleNamespace(
        enabled=True,
        log_turn=lambda *_a, **_k: None,
        **overrides,
    )


@pytest.mark.asyncio
async def test_chat_session_turn_stream_text_only(monkeypatch, tmp_path):
    from minds.agents.anton_agent.anton.chat_session import ChatSession
    from minds.agents.anton_agent.anton.llm.provider import LLMResponse, StreamComplete, StreamTextDelta, Usage

    monkeypatch.setattr(
        "minds.agents.anton_agent.anton.scratchpad_manager.ScratchpadManager.probe_packages",
        staticmethod(lambda: []),
    )

    class FakeLLM:
        async def plan_stream(self, *, system, messages, tools=None):
            yield StreamTextDelta(text="hello")
            yield StreamComplete(response=LLMResponse(content="hello", tool_calls=[], usage=Usage()))

        async def code(self, *, system, messages, max_tokens=2048):
            return LLMResponse(content="summary")

        @property
        def coding_model(self):
            return "x"

    sess = ChatSession(
        FakeLLM(),
        cortex=_stub_cortex(),
        episodic=_stub_episodic(),
        runtime_context="rc",
        backend="docker",
        coding_provider="anthropic",
        coding_api_key="k",
        coding_model="m",
        workspace_path=tmp_path,
        extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"},
    )

    events = [e async for e in sess.turn_stream("hi")]
    assert any(isinstance(e, StreamTextDelta) and e.text == "hello" for e in events)


@pytest.mark.asyncio
async def test_chat_session_repair_history_and_tool_loop(monkeypatch, tmp_path):
    from minds.agents.anton_agent.anton.chat_session import ChatSession
    from minds.agents.anton_agent.anton.llm.provider import LLMResponse, StreamComplete, ToolCall, Usage

    monkeypatch.setattr(
        "minds.agents.anton_agent.anton.scratchpad_manager.ScratchpadManager.probe_packages",
        staticmethod(lambda: []),
    )
    monkeypatch.setattr("minds.agents.anton_agent.anton.chat_session.dispatch_tool", AsyncMock(return_value="tool ok"))

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def plan_stream(self, *, system, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                yield StreamComplete(
                    response=LLMResponse(
                        content="",
                        tool_calls=[ToolCall(id="t1", name="recall", input={"query": "x"})],
                        usage=Usage(context_pressure=0.0),
                    )
                )
            else:
                yield StreamComplete(
                    response=LLMResponse(content="done", tool_calls=[], usage=Usage(context_pressure=0.0))
                )

        async def code(self, *, system, messages, max_tokens=2048):
            return LLMResponse(content="summary")

        @property
        def coding_model(self):
            return "x"

    sess = ChatSession(
        FakeLLM(),
        cortex=_stub_cortex(),
        episodic=_stub_episodic(),
        runtime_context="rc",
        backend="docker",
        coding_provider="anthropic",
        coding_api_key="k",
        coding_model="m",
        workspace_path=tmp_path,
        extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"},
    )

    sess.load_history(
        [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "recall", "input": {"query": "x"}}],
            }
        ]
    )
    sess.repair_history()
    assert sess.history[-1]["role"] == "user"

    _events = [e async for e in sess.turn_stream("hi")]
    assert sess._llm.calls == 2


def test_chat_session_apply_error_tracking_appends_nudge_and_circuit_breaker():
    from minds.agents.anton_agent.anton.chat_session import _apply_error_tracking

    error_streak: dict[str, int] = {}
    nudged: set[str] = set()
    tool = "scratchpad"

    t1 = _apply_error_tracking("Task failed: boom", tool, error_streak, nudged)
    assert "SYSTEM:" not in t1
    t2 = _apply_error_tracking("failed again", tool, error_streak, nudged)
    assert "SYSTEM: This tool has failed twice" in t2
    for _ in range(10):
        t2 = _apply_error_tracking("failed", tool, error_streak, nudged)
    assert "has failed 5 times" in t2


@pytest.mark.asyncio
async def test_chat_session_context_overflow_triggers_retry(monkeypatch, tmp_path):
    from minds.agents.anton_agent.anton.chat_session import ChatSession
    from minds.agents.anton_agent.anton.llm.provider import ContextOverflowError, LLMResponse, StreamComplete, Usage

    monkeypatch.setattr(
        "minds.agents.anton_agent.anton.scratchpad_manager.ScratchpadManager.probe_packages",
        staticmethod(lambda: []),
    )

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def plan_stream(self, *, system, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                raise ContextOverflowError("too long")
            yield StreamComplete(response=LLMResponse(content="ok", tool_calls=[], usage=Usage(context_pressure=0.0)))

        async def code(self, *, system, messages, max_tokens=2048):
            return LLMResponse(content="summary ok")

        @property
        def coding_model(self):
            return "x"

    sess = ChatSession(
        FakeLLM(),
        cortex=_stub_cortex(),
        episodic=_stub_episodic(),
        runtime_context="rc",
        backend="docker",
        coding_provider="anthropic",
        coding_api_key="k",
        coding_model="m",
        workspace_path=tmp_path,
        extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"},
    )

    events = [e async for e in sess.turn_stream("hi")]
    assert sess._llm.calls == 2
    assert any(isinstance(e, StreamComplete) for e in events)


@pytest.mark.asyncio
async def test_chat_session_build_tools_and_summarize_history(monkeypatch, tmp_path):
    from minds.agents.anton_agent.anton.chat_session import ChatSession
    from minds.agents.anton_agent.anton.llm.provider import LLMResponse, StreamComplete, Usage

    monkeypatch.setattr(
        "minds.agents.anton_agent.anton.scratchpad_manager.ScratchpadManager.probe_packages",
        staticmethod(lambda: ["numpy", "pytest", "requests"]),
    )
    cortex = SimpleNamespace(
        build_memory_context=lambda: "\n\nMEM", get_scratchpad_context=lambda: "- when: use progress()"
    )
    episodic = _stub_episodic()

    class FakeLLM:
        async def plan_stream(self, *, system, messages, tools=None):
            yield StreamComplete(response=LLMResponse(content="ok", tool_calls=[], usage=Usage(context_pressure=0.0)))

        async def code(self, *, system, messages, max_tokens=2048):
            return LLMResponse(content="summary")

        @property
        def coding_model(self):
            return "x"

    sess = ChatSession(
        FakeLLM(),
        runtime_context="rc",
        cortex=cortex,
        episodic=episodic,
        backend="docker",
        coding_provider="anthropic",
        coding_api_key="k",
        coding_model="m",
        workspace_path=tmp_path,
        extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"},
    )

    tools = sess._build_tools()
    assert any("Installed packages" in t.get("description", "") for t in tools)
    assert any("Lessons from past sessions" in t.get("description", "") for t in tools)

    sess.load_history([{"role": "user", "content": "x"}] * 7)
    await sess._summarize_history()
    assert sess.history[0]["role"] == "user"


@pytest.mark.asyncio
async def test_chat_session_scratchpad_exec_tool_path(monkeypatch, tmp_path):
    from minds.agents.anton_agent.anton.backends.base import Cell
    from minds.agents.anton_agent.anton.chat_session import ChatSession
    from minds.agents.anton_agent.anton.llm.provider import (
        LLMResponse,
        StreamComplete,
        StreamTaskProgress,
        ToolCall,
        Usage,
    )

    monkeypatch.setattr(
        "minds.agents.anton_agent.anton.scratchpad_manager.ScratchpadManager.probe_packages",
        staticmethod(lambda: []),
    )

    fake_cell = Cell(code="x", stdout="1", stderr="", error=None, description="", estimated_time="", logs="")

    class FakePad:
        async def execute_streaming(self, code, **_kwargs):
            yield "p1"
            yield fake_cell

    fake_pad = FakePad()
    monkeypatch.setattr(
        "minds.agents.anton_agent.anton.chat_session.prepare_scratchpad_exec",
        AsyncMock(return_value=(fake_pad, "print(1)", "desc", "1s", 1)),
    )
    monkeypatch.setattr("minds.agents.anton_agent.anton.chat_session.format_cell_result", lambda _c: "CELL")

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def plan_stream(self, *, system, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                yield StreamComplete(
                    response=LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="t1",
                                name="scratchpad",
                                input={
                                    "action": "exec",
                                    "code": "print(1)",
                                    "one_line_description": "d",
                                    "estimated_execution_time_seconds": 1,
                                },
                            )
                        ],
                        usage=Usage(context_pressure=0.0),
                    )
                )
            else:
                yield StreamComplete(
                    response=LLMResponse(content="done", tool_calls=[], usage=Usage(context_pressure=0.0))
                )

        async def code(self, *, system, messages, max_tokens=2048):
            return LLMResponse(content="summary")

        @property
        def coding_model(self):
            return "x"

    sess = ChatSession(
        FakeLLM(),
        cortex=_stub_cortex(),
        episodic=_stub_episodic(),
        runtime_context="rc",
        backend="docker",
        coding_provider="anthropic",
        coding_api_key="k",
        coding_model="m",
        workspace_path=tmp_path,
        extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"},
    )
    events = [e async for e in sess.turn_stream("hi")]
    assert any(isinstance(e, StreamTaskProgress) and e.message == "p1" for e in events)


@pytest.mark.asyncio
async def test_chat_session_dump_tool_result_emitted(monkeypatch, tmp_path):
    from minds.agents.anton_agent.anton.chat_session import ChatSession
    from minds.agents.anton_agent.anton.llm.provider import (
        LLMResponse,
        StreamComplete,
        StreamToolResult,
        ToolCall,
        Usage,
    )

    monkeypatch.setattr(
        "minds.agents.anton_agent.anton.scratchpad_manager.ScratchpadManager.probe_packages",
        staticmethod(lambda: []),
    )
    monkeypatch.setattr("minds.agents.anton_agent.anton.chat_session.dispatch_tool", AsyncMock(return_value="DUMP"))

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def plan_stream(self, *, system, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                yield StreamComplete(
                    response=LLMResponse(
                        content="",
                        tool_calls=[ToolCall(id="t1", name="scratchpad", input={"action": "dump"})],
                        usage=Usage(context_pressure=0.0),
                    )
                )
            else:
                yield StreamComplete(
                    response=LLMResponse(content="done", tool_calls=[], usage=Usage(context_pressure=0.0))
                )

        async def code(self, *, system, messages, max_tokens=2048):
            return LLMResponse(content="summary")

        @property
        def coding_model(self):
            return "x"

    sess = ChatSession(
        FakeLLM(),
        cortex=_stub_cortex(),
        episodic=_stub_episodic(),
        runtime_context="rc",
        backend="docker",
        coding_provider="anthropic",
        coding_api_key="k",
        coding_model="m",
        workspace_path=tmp_path,
        extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"},
    )

    events = [e async for e in sess.turn_stream("hi")]
    # Dump content should NOT be streamed to the user.
    assert not any(isinstance(e, StreamToolResult) for e in events)

    # But it should still be returned to the LLM via tool_result blocks in history.
    tool_result_blocks = sess.history[-2]["content"]
    assert isinstance(tool_result_blocks, list)
    assert tool_result_blocks[0]["type"] == "tool_result"
    assert "DUMP" in tool_result_blocks[0]["content"]


@pytest.mark.asyncio
async def test_chat_session_tool_round_limit_branch(monkeypatch, tmp_path):
    from minds.agents.anton_agent.anton.chat_session import ChatSession
    from minds.agents.anton_agent.anton.llm.provider import LLMResponse, StreamComplete, ToolCall, Usage

    monkeypatch.setattr(
        "minds.agents.anton_agent.anton.scratchpad_manager.ScratchpadManager.probe_packages",
        staticmethod(lambda: []),
    )
    monkeypatch.setattr("minds.agents.anton_agent.anton.chat_session.MAX_TOOL_ROUNDS", 0)

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def plan_stream(self, *, system, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                yield StreamComplete(
                    response=LLMResponse(
                        content="c",
                        tool_calls=[ToolCall(id="t1", name="recall", input={"query": "x"})],
                        usage=Usage(context_pressure=0.0),
                    )
                )
            else:
                yield StreamComplete(
                    response=LLMResponse(content="final", tool_calls=[], usage=Usage(context_pressure=0.0))
                )

        async def code(self, *, system, messages, max_tokens=2048):
            return LLMResponse(content="summary")

        @property
        def coding_model(self):
            return "x"

    sess = ChatSession(
        FakeLLM(),
        cortex=_stub_cortex(),
        episodic=_stub_episodic(),
        runtime_context="rc",
        backend="docker",
        coding_provider="anthropic",
        coding_api_key="k",
        coding_model="m",
        workspace_path=tmp_path,
        extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"},
    )
    _events = [e async for e in sess.turn_stream("hi")]
    assert sess._llm.calls >= 2


@pytest.mark.asyncio
async def test_chat_session_turn_stream_triggers_identity_update_task(monkeypatch, tmp_path):
    from minds.agents.anton_agent.anton.chat_session import ChatSession
    from minds.agents.anton_agent.anton.llm.provider import LLMResponse, StreamComplete, Usage

    monkeypatch.setattr(
        "minds.agents.anton_agent.anton.scratchpad_manager.ScratchpadManager.probe_packages",
        staticmethod(lambda: []),
    )

    created: list = []
    monkeypatch.setattr(
        "minds.agents.anton_agent.anton.chat_session.asyncio.create_task", lambda coro: created.append(coro)
    )

    class FakeLLM:
        async def plan_stream(self, *, system, messages, tools=None):
            yield StreamComplete(response=LLMResponse(content="ok", tool_calls=[], usage=Usage(context_pressure=0.0)))

        async def code(self, *, system, messages, max_tokens=2048):
            return LLMResponse(content="summary")

        @property
        def coding_model(self):
            return "x"

    cortex = SimpleNamespace(
        mode="autopilot",
        build_memory_context=lambda: "",
        get_scratchpad_context=lambda: "",
        maybe_update_identity=AsyncMock(),
    )
    episodic = SimpleNamespace(enabled=True, log_turn=lambda *a, **k: None)

    sess = ChatSession(
        FakeLLM(),
        runtime_context="rc",
        cortex=cortex,
        episodic=episodic,
        backend="docker",
        coding_provider="anthropic",
        coding_api_key="k",
        coding_model="m",
        workspace_path=tmp_path,
        extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"},
    )
    sess._turn_count = 4
    _ = [e async for e in sess.turn_stream("hi")]
    assert created


@pytest.mark.asyncio
async def test_chat_session_high_context_pressure_triggers_compaction(monkeypatch, tmp_path):
    from minds.agents.anton_agent.anton.chat_session import ChatSession
    from minds.agents.anton_agent.anton.llm.provider import LLMResponse, StreamComplete, StreamContextCompacted, Usage

    monkeypatch.setattr(
        "minds.agents.anton_agent.anton.scratchpad_manager.ScratchpadManager.probe_packages",
        staticmethod(lambda: []),
    )

    class FakeLLM:
        async def plan_stream(self, *, system, messages, tools=None):
            yield StreamComplete(response=LLMResponse(content="ok", tool_calls=[], usage=Usage(context_pressure=1.0)))

        async def code(self, *, system, messages, max_tokens=2048):
            return LLMResponse(content="summary")

        @property
        def coding_model(self):
            return "x"

    sess = ChatSession(
        FakeLLM(),
        cortex=_stub_cortex(),
        episodic=_stub_episodic(),
        runtime_context="rc",
        backend="docker",
        coding_provider="anthropic",
        coding_api_key="k",
        coding_model="m",
        workspace_path=tmp_path,
        extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"},
    )
    sess._summarize_history = AsyncMock()  # type: ignore[method-assign]
    sess._compact_scratchpads = lambda: True  # type: ignore[method-assign]

    events = [e async for e in sess.turn_stream("hi")]
    assert any(isinstance(e, StreamContextCompacted) for e in events)
    sess._summarize_history.assert_awaited()


@pytest.mark.asyncio
async def test_chat_session_tool_exception_is_captured_and_turn_continues(monkeypatch, tmp_path):
    from minds.agents.anton_agent.anton.chat_session import ChatSession
    from minds.agents.anton_agent.anton.llm.provider import LLMResponse, StreamComplete, ToolCall, Usage

    monkeypatch.setattr(
        "minds.agents.anton_agent.anton.scratchpad_manager.ScratchpadManager.probe_packages",
        staticmethod(lambda: []),
    )

    async def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("minds.agents.anton_agent.anton.chat_session.dispatch_tool", _boom)

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def plan_stream(self, *, system, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                yield StreamComplete(
                    response=LLMResponse(
                        content="",
                        tool_calls=[ToolCall(id="t1", name="recall", input={"query": "x"})],
                        usage=Usage(context_pressure=0.0),
                    )
                )
            else:
                yield StreamComplete(
                    response=LLMResponse(content="done", tool_calls=[], usage=Usage(context_pressure=0.0))
                )

        async def code(self, *, system, messages, max_tokens=2048):
            return LLMResponse(content="summary")

        @property
        def coding_model(self):
            return "x"

    sess = ChatSession(
        FakeLLM(),
        cortex=_stub_cortex(),
        episodic=_stub_episodic(),
        runtime_context="rc",
        backend="docker",
        coding_provider="anthropic",
        coding_api_key="k",
        coding_model="m",
        workspace_path=tmp_path,
        extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"},
    )
    _ = [e async for e in sess.turn_stream("hi")]
    assert sess.history[-1]["role"] == "assistant"
    assert sess.history[-1]["content"] == "done"

    tool_results_msg = next(m for m in reversed(sess.history) if m["role"] == "user" and isinstance(m["content"], list))
    assert "Tool 'recall' failed: boom" in tool_results_msg["content"][0]["content"]


@pytest.mark.asyncio
async def test_chat_session_consolidation_schedules_and_records_confirmations(monkeypatch, tmp_path):
    from minds.agents.anton_agent.anton.chat_session import ChatSession
    from minds.agents.anton_agent.anton.llm.provider import LLMResponse, StreamComplete, Usage

    monkeypatch.setattr(
        "minds.agents.anton_agent.anton.scratchpad_manager.ScratchpadManager.probe_packages",
        staticmethod(lambda: []),
    )

    import minds.agents.anton_agent.anton.memory.consolidator as consolidator_mod

    class FakeConsolidator:
        def should_replay(self, _cells):
            return True

        async def replay_and_extract(self, _cells, _llm):
            return []

    monkeypatch.setattr(consolidator_mod, "Consolidator", FakeConsolidator)

    scheduled: list[object] = []

    def _create_task(coro):
        scheduled.append(coro)
        return SimpleNamespace()

    monkeypatch.setattr(asyncio, "create_task", _create_task)

    class FakeLLM:
        async def plan_stream(self, *, system, messages, tools=None):
            yield StreamComplete(response=LLMResponse(content="ok", tool_calls=[], usage=Usage(context_pressure=0.0)))

        async def code(self, *, system, messages, max_tokens=2048):
            return LLMResponse(content="summary")

        @property
        def coding_model(self):
            return "x"

    cortex = SimpleNamespace(mode="on", build_memory_context=lambda: "", get_scratchpad_context=lambda: "")
    episodic = _stub_episodic()
    sess = ChatSession(
        FakeLLM(),
        runtime_context="rc",
        cortex=cortex,
        episodic=episodic,
        backend="docker",
        coding_provider="anthropic",
        coding_api_key="k",
        coding_model="m",
        workspace_path=tmp_path,
        extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"},
    )
    sess._scratchpads._pad = SimpleNamespace(cells=[SimpleNamespace(stdout="x")])

    _ = [e async for e in sess.turn_stream("hi")]
    assert scheduled, "expected consolidation task to be scheduled"

    class FakeConsolidator2:
        def should_replay(self, _cells):
            return True

        async def replay_and_extract(self, _cells, _llm):
            return [SimpleNamespace(kind="lesson"), SimpleNamespace(kind="rule")]

    monkeypatch.setattr(consolidator_mod, "Consolidator", FakeConsolidator2)

    cortex2 = SimpleNamespace(
        mode="on",
        encode=AsyncMock(),
        encoding_gate=lambda e: (e.kind == "rule"),
    )
    episodic2 = _stub_episodic()
    sess2 = ChatSession(
        FakeLLM(),
        runtime_context="rc",
        cortex=cortex2,
        episodic=episodic2,
        backend="docker",
        coding_provider="anthropic",
        coding_api_key="k",
        coding_model="m",
        workspace_path=tmp_path,
        extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"},
    )
    await sess2._consolidate([SimpleNamespace(stdout="x")])
    cortex2.encode.assert_awaited()
    assert any(getattr(e, "kind", None) == "rule" for e in sess2._pending_memory_confirmations)
