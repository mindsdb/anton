"""ToolRegistry.dispatch_tool_stream / dispatch_tool — streaming handler
support. See docs/eng-763/2026-08-03-streaming-tool-progress-design.md for
the protocol: a handler may be an async generator yielding ToolProgress
markers followed by exactly one final str/list[dict] result, instead of a
plain coroutine.
"""

from __future__ import annotations

import pytest

from anton.core.tools.progress import ToolProgress
from anton.core.tools.registry import ToolRegistry
from anton.core.tools.tool_defs import ToolDef


def _make_tool(name, handler) -> ToolDef:
    return ToolDef(
        name=name,
        description=f"tool {name}",
        input_schema={"type": "object", "properties": {}},
        handler=handler,
    )


async def _plain_handler(_session, _input):
    return "plain result"


async def _streaming_handler(_session, _input):
    yield ToolProgress("step 1 executing")
    yield ToolProgress("step 1 done")
    yield "final result"


async def _empty_streaming_handler(_session, _input):
    yield ToolProgress("only progress, no result")


async def _multi_result_handler(_session, _input):
    yield ToolProgress("progress")
    yield "first candidate"
    yield "second candidate"


async def _raising_streaming_handler(_session, _input):
    yield ToolProgress("about to fail")
    raise ValueError("boom")


class TestDispatchToolStream:
    async def test_plain_coroutine_yields_single_result(self):
        reg = ToolRegistry()
        reg.register_tool(_make_tool("plain", _plain_handler))
        items = [item async for item in reg.dispatch_tool_stream(None, "plain", {})]
        assert items == ["plain result"]

    async def test_streaming_handler_forwards_progress_then_result(self):
        reg = ToolRegistry()
        reg.register_tool(_make_tool("streaming", _streaming_handler))
        items = [item async for item in reg.dispatch_tool_stream(None, "streaming", {})]
        assert items == [
            ToolProgress("step 1 executing"),
            ToolProgress("step 1 done"),
            "final result",
        ]

    async def test_empty_result_raises_runtime_error(self):
        reg = ToolRegistry()
        reg.register_tool(_make_tool("empty", _empty_streaming_handler))
        with pytest.raises(RuntimeError, match="produced no result"):
            async for _ in reg.dispatch_tool_stream(None, "empty", {}):
                pass

    async def test_exception_propagates_after_partial_progress(self):
        reg = ToolRegistry()
        reg.register_tool(_make_tool("raising", _raising_streaming_handler))
        seen = []
        with pytest.raises(ValueError, match="boom"):
            async for item in reg.dispatch_tool_stream(None, "raising", {}):
                seen.append(item)
        assert seen == [ToolProgress("about to fail")]

    async def test_unknown_tool_raises_value_error(self):
        reg = ToolRegistry()
        with pytest.raises(ValueError, match="not found"):
            async for _ in reg.dispatch_tool_stream(None, "nope", {}):
                pass


class TestDispatchTool:
    async def test_plain_coroutine_unchanged(self):
        reg = ToolRegistry()
        reg.register_tool(_make_tool("plain", _plain_handler))
        outcome = await reg.dispatch_tool(None, "plain", {})
        assert outcome.content == "plain result"

    async def test_streaming_handler_drops_progress_keeps_result(self):
        reg = ToolRegistry()
        reg.register_tool(_make_tool("streaming", _streaming_handler))
        outcome = await reg.dispatch_tool(None, "streaming", {})
        assert outcome.content == "final result"

    async def test_empty_result_raises_runtime_error(self):
        reg = ToolRegistry()
        reg.register_tool(_make_tool("empty", _empty_streaming_handler))
        with pytest.raises(RuntimeError, match="produced no result"):
            await reg.dispatch_tool(None, "empty", {})

    async def test_last_non_progress_item_wins(self):
        reg = ToolRegistry()
        reg.register_tool(_make_tool("multi", _multi_result_handler))
        outcome = await reg.dispatch_tool(None, "multi", {})
        assert outcome.content == "second candidate"


async def _peeking_handler(_session, _input):
    yield ToolProgress("step 1")
    yield ToolProgress("<html>\n<body>", kind="peek")
    yield "final result"


class TestPeekRelay:
    @staticmethod
    def _session(**attrs):
        from types import SimpleNamespace

        events = []

        class _Emitter:
            async def emit(self, event):
                events.append(event)

        return SimpleNamespace(emitter=_Emitter(), **attrs), events

    async def test_a_peek_marker_travels_as_its_own_phase_for_a_host_that_renders_it(self):
        """A step line is printed once and kept; a peek replaces itself in
        the footer. The CLI tells them apart by phase, so the relay must."""
        reg = ToolRegistry()
        reg.register_tool(_make_tool("peeking", _peeking_handler))
        session, events = self._session(live_tool_peek=True)
        outcome = await reg.dispatch_tool(session, "peeking", {}, tool_call_id="tc1")
        assert outcome.content == "final result"
        assert [(e.phase, e.message, e.id) for e in events] == [
            ("tool_progress", "step 1", "tc1"),
            ("tool_peek", "<html>\n<body>", "tc1"),
        ]
        assert ToolProgress("x").kind == "step"

    async def test_a_peek_is_dropped_before_the_wire_unless_the_host_opted_in(self):
        """Review of PR #335: `tool_peek` reached cowork-server and the cloud
        JSONL as an undocumented phase nobody rendered, spending their
        progress throttle window. The step line still travels; the tail does
        not. Opt-in is `ChatSessionConfig.live_tool_peek`; a session double
        without the attribute reads as "not declared", i.e. dropped."""
        reg = ToolRegistry()
        reg.register_tool(_make_tool("peeking", _peeking_handler))
        for session, events in (self._session(), self._session(live_tool_peek=False)):
            outcome = await reg.dispatch_tool(session, "peeking", {}, tool_call_id="tc1")
            assert outcome.content == "final result"
            assert [(e.phase, e.message) for e in events] == [("tool_progress", "step 1")]
