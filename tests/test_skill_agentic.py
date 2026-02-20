from __future__ import annotations

from unittest.mock import AsyncMock

from anton.llm.provider import LLMResponse, ToolCall, Usage
from anton.skill.agentic import agentic_loop
from anton.skill.context import _current_llm, set_skill_llm


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )


def _tool_response(text: str, tool_name: str, tool_input: dict, tool_id: str = "tc_1") -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[ToolCall(id=tool_id, name=tool_name, input=tool_input)],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="tool_use",
    )


class TestAgenticLoop:
    async def test_no_tools_returns_immediately(self):
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value=_text_response("Done!"))
        set_skill_llm(provider, "m")

        try:
            result = await agentic_loop(
                system="sys",
                user_message="hi",
                tools=[{"name": "t", "description": "d", "input_schema": {"type": "object"}}],
                handle_tool=AsyncMock(),
            )
            assert result == "Done!"
            assert provider.complete.call_count == 1
        finally:
            _current_llm.set(None)

    async def test_single_tool_call_then_done(self):
        provider = AsyncMock()
        provider.complete = AsyncMock(
            side_effect=[
                _tool_response("Let me check.", "get_info", {"query": "test"}),
                _text_response("Here's the answer."),
            ]
        )
        set_skill_llm(provider, "m")

        handler = AsyncMock(return_value="info result")

        try:
            result = await agentic_loop(
                system="sys",
                user_message="look up test",
                tools=[{"name": "get_info", "description": "d", "input_schema": {"type": "object"}}],
                handle_tool=handler,
            )

            assert result == "Here's the answer."
            handler.assert_called_once_with("get_info", {"query": "test"})
            assert provider.complete.call_count == 2

            # Verify the second call includes tool result in messages
            second_call = provider.complete.call_args_list[1]
            messages = second_call.kwargs["messages"]
            assert len(messages) == 3  # user, assistant+tool_use, user+tool_result
            assert messages[2]["content"][0]["type"] == "tool_result"
            assert messages[2]["content"][0]["content"] == "info result"
        finally:
            _current_llm.set(None)

    async def test_multiple_turns(self):
        provider = AsyncMock()
        provider.complete = AsyncMock(
            side_effect=[
                _tool_response("Step 1.", "act", {"n": 1}, "tc_a"),
                _tool_response("Step 2.", "act", {"n": 2}, "tc_b"),
                _text_response("All done."),
            ]
        )
        set_skill_llm(provider, "m")

        calls = []

        async def handler(name, inputs):
            calls.append(inputs["n"])
            return f"ok {inputs['n']}"

        try:
            result = await agentic_loop(
                system="sys",
                user_message="do stuff",
                tools=[{"name": "act", "description": "d", "input_schema": {"type": "object"}}],
                handle_tool=handler,
            )

            assert result == "All done."
            assert calls == [1, 2]
            assert provider.complete.call_count == 3
        finally:
            _current_llm.set(None)

    async def test_max_turns_safety(self):
        provider = AsyncMock()
        # Always returns a tool call — never stops
        provider.complete = AsyncMock(
            return_value=_tool_response("again.", "loop", {})
        )
        set_skill_llm(provider, "m")

        try:
            result = await agentic_loop(
                system="sys",
                user_message="go",
                tools=[{"name": "loop", "description": "d", "input_schema": {"type": "object"}}],
                handle_tool=AsyncMock(return_value="ok"),
                max_turns=3,
            )

            # Should stop after 3 turns, not loop forever
            assert provider.complete.call_count == 3
        finally:
            _current_llm.set(None)

    async def test_tool_handler_error_is_caught(self):
        provider = AsyncMock()
        provider.complete = AsyncMock(
            side_effect=[
                _tool_response("Calling.", "fail_tool", {}),
                _text_response("I see there was an error."),
            ]
        )
        set_skill_llm(provider, "m")

        async def bad_handler(name, inputs):
            raise ValueError("boom")

        try:
            result = await agentic_loop(
                system="sys",
                user_message="go",
                tools=[{"name": "fail_tool", "description": "d", "input_schema": {"type": "object"}}],
                handle_tool=bad_handler,
            )

            assert result == "I see there was an error."
            # Verify the error was sent back as a tool result
            second_call = provider.complete.call_args_list[1]
            messages = second_call.kwargs["messages"]
            tool_result = messages[-1]["content"][0]
            assert "Error: boom" in tool_result["content"]
        finally:
            _current_llm.set(None)
