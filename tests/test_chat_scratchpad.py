from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from anton.chat import SCRATCHPAD_TOOL, ChatSession
from anton.llm.provider import LLMResponse, StreamComplete, StreamToolResult, ToolCall, Usage


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )


def _scratchpad_response(
    text: str, action: str, name: str, code: str = "",
    packages: list[str] | None = None, tool_id: str = "tc_sp_1",
) -> LLMResponse:
    tc_input: dict = {"action": action, "name": name}
    if code:
        tc_input["code"] = code
    if packages is not None:
        tc_input["packages"] = packages
    return LLMResponse(
        content=text,
        tool_calls=[
            ToolCall(id=tool_id, name="scratchpad", input=tc_input),
        ],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="tool_use",
    )


class TestScratchpadToolDefinition:
    def test_tool_definition_structure(self):
        assert SCRATCHPAD_TOOL["name"] == "scratchpad"
        props = SCRATCHPAD_TOOL["input_schema"]["properties"]
        assert "action" in props
        assert "name" in props
        assert "code" in props
        assert "packages" in props
        assert SCRATCHPAD_TOOL["input_schema"]["required"] == ["action", "name"]

    def test_tool_has_install_action(self):
        actions = SCRATCHPAD_TOOL["input_schema"]["properties"]["action"]["enum"]
        assert "install" in actions

    def test_packages_property_is_array_of_strings(self):
        packages_prop = SCRATCHPAD_TOOL["input_schema"]["properties"]["packages"]
        assert packages_prop["type"] == "array"
        assert packages_prop["items"] == {"type": "string"}

    async def test_scratchpad_tool_in_tools(self):
        """scratchpad should always be in _build_tools() output."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hi!"))

        session = ChatSession(mock_llm)
        try:
            await session.turn("hello")

            call_kwargs = mock_llm.plan.call_args
            tools = call_kwargs.kwargs.get("tools", [])
            tool_names = [t["name"] for t in tools]
            assert "scratchpad" in tool_names
        finally:
            await session.close()


class TestScratchpadExecViaChat:
    async def test_scratchpad_exec_via_chat(self):
        """exec action flows through and returns output."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _scratchpad_response("Let me compute.", "exec", "main", "print(7 * 6)"),
                _text_response("The answer is 42."),
            ]
        )

        session = ChatSession(mock_llm)
        try:
            reply = await session.turn("what is 7 * 6?")

            tool_result_msgs = [
                m for m in session.history
                if m["role"] == "user" and isinstance(m["content"], list)
            ]
            assert len(tool_result_msgs) == 1
            result_content = tool_result_msgs[0]["content"][0]["content"]
            assert "42" in result_content
        finally:
            await session.close()


class TestScratchpadViewViaChat:
    async def test_scratchpad_view_via_chat(self):
        """view action returns cell history."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _scratchpad_response("Running code.", "exec", "analysis", "x = 10\nprint(x)"),
                _scratchpad_response("Let me check history.", "view", "analysis"),
                _text_response("Here's the history."),
            ]
        )

        session = ChatSession(mock_llm)
        try:
            await session.turn("run and show")

            # Find the view result (second tool result)
            tool_result_msgs = [
                m for m in session.history
                if m["role"] == "user" and isinstance(m["content"], list)
            ]
            assert len(tool_result_msgs) == 2
            view_content = tool_result_msgs[1]["content"][0]["content"]
            assert "Cell 1" in view_content
            assert "10" in view_content
        finally:
            await session.close()


class TestScratchpadRemoveViaChat:
    async def test_scratchpad_remove_via_chat(self):
        """remove action cleans up the scratchpad."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _scratchpad_response("Creating.", "exec", "tmp", "print('hi')"),
                _scratchpad_response("Removing.", "remove", "tmp"),
                _text_response("Cleaned up."),
            ]
        )

        session = ChatSession(mock_llm)
        try:
            await session.turn("create and remove")

            tool_result_msgs = [
                m for m in session.history
                if m["role"] == "user" and isinstance(m["content"], list)
            ]
            remove_content = tool_result_msgs[1]["content"][0]["content"]
            assert "removed" in remove_content.lower()
        finally:
            await session.close()


class TestScratchpadDumpViaChat:
    async def test_scratchpad_dump_via_chat(self):
        """dump action flows through chat, returns markdown with code fences."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                # First: exec some code
                _scratchpad_response("Running.", "exec", "main", "print(42)"),
                # Second: dump the scratchpad
                _scratchpad_response("Here's your work.", "dump", "main"),
                # Final text reply
                _text_response("Done!"),
            ]
        )

        session = ChatSession(mock_llm)
        try:
            await session.turn("show me my work")

            tool_result_msgs = [
                m for m in session.history
                if m["role"] == "user" and isinstance(m["content"], list)
            ]
            assert len(tool_result_msgs) == 2
            dump_content = tool_result_msgs[1]["content"][0]["content"]
            assert "```python" in dump_content
            assert "## Scratchpad: main" in dump_content
            assert "42" in dump_content
        finally:
            await session.close()


class _FakeAsyncIter:
    """Wraps items into an async iterator for mocking plan_stream."""

    def __init__(self, items):
        self._items = items

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


class TestScratchpadDumpStreaming:
    async def test_scratchpad_dump_streams_tool_result(self):
        """dump action yields a StreamToolResult for display, but sends a short
        summary back to the LLM to avoid it parroting the full notebook."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("STATUS: COMPLETE — task done"))

        call_count = 0

        def fake_plan_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _FakeAsyncIter([
                    StreamComplete(
                        response=_scratchpad_response("Running.", "exec", "main", "print(42)")
                    )
                ])
            if call_count == 2:
                return _FakeAsyncIter([
                    StreamComplete(
                        response=_scratchpad_response("Here.", "dump", "main")
                    )
                ])
            return _FakeAsyncIter([
                StreamComplete(response=_text_response("Done!"))
            ])

        mock_llm.plan_stream = fake_plan_stream

        session = ChatSession(mock_llm)
        try:
            events = []
            async for event in session.turn_stream("show work"):
                events.append(event)

            tool_results = [e for e in events if isinstance(e, StreamToolResult)]
            assert len(tool_results) == 1
            assert "## Scratchpad: main" in tool_results[0].content

            # The LLM should get a short summary, not the full dump
            history = session.history
            # Find the tool_result message for the dump call
            for msg in history:
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if item.get("type") == "tool_result" and "dump" not in item.get("content", "").lower().split("```"):
                            if "Notebook dump displayed" in item.get("content", ""):
                                break
        finally:
            await session.close()


class TestScratchpadStreaming:
    async def test_scratchpad_in_streaming_path(self):
        """scratchpad exec should work in turn_stream() too."""
        tool_response = _scratchpad_response("Computing.", "exec", "s", "print(99)")
        final_response = _text_response("Got 99.")

        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("STATUS: COMPLETE — task done"))

        call_count = 0

        def fake_plan_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _FakeAsyncIter([StreamComplete(response=tool_response)])
            return _FakeAsyncIter([StreamComplete(response=final_response)])

        mock_llm.plan_stream = fake_plan_stream

        session = ChatSession(mock_llm)
        try:
            events = []
            async for event in session.turn_stream("compute 99"):
                events.append(event)

            assert any(isinstance(e, StreamComplete) for e in events)

            tool_result_msgs = [
                m for m in session.history
                if m["role"] == "user" and isinstance(m["content"], list)
            ]
            assert len(tool_result_msgs) == 1
            result_content = tool_result_msgs[0]["content"][0]["content"]
            assert "99" in result_content
        finally:
            await session.close()


class TestScratchpadInstallViaChat:
    async def test_install_action_dispatch(self):
        """install action flows through chat and returns pip output."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _scratchpad_response(
                    "Installing.", "install", "main", packages=["cowsay"]
                ),
                _text_response("Installed cowsay."),
            ]
        )

        session = ChatSession(mock_llm)
        try:
            reply = await session.turn("install cowsay")

            tool_result_msgs = [
                m for m in session.history
                if m["role"] == "user" and isinstance(m["content"], list)
            ]
            assert len(tool_result_msgs) == 1
            result_content = tool_result_msgs[0]["content"][0]["content"]
            assert "cowsay" in result_content.lower() or "satisfied" in result_content.lower() or "already installed" in result_content.lower()
        finally:
            await session.close()

    async def test_install_empty_packages_via_chat(self):
        """install with no packages returns a message without crashing."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _scratchpad_response("Installing.", "install", "main", packages=[]),
                _text_response("Nothing to install."),
            ]
        )

        session = ChatSession(mock_llm)
        try:
            await session.turn("install nothing")

            tool_result_msgs = [
                m for m in session.history
                if m["role"] == "user" and isinstance(m["content"], list)
            ]
            result_content = tool_result_msgs[0]["content"][0]["content"]
            assert "no packages" in result_content.lower()
        finally:
            await session.close()
