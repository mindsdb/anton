from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from anton.chat import ChatSession, REQUEST_SECRET_TOOL, UPDATE_CONTEXT_TOOL
from anton.context.self_awareness import SelfAwarenessContext
from anton.llm.provider import LLMResponse, ToolCall, Usage
from anton.workspace import Workspace


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )


def _update_context_response(
    text: str, updates: list[dict], tool_id: str = "tc_ctx_1"
) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[
            ToolCall(
                id=tool_id,
                name="update_context",
                input={"updates": updates},
            ),
        ],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="tool_use",
    )


def _request_secret_response(
    text: str, variable_name: str, prompt_text: str, tool_id: str = "tc_sec_1"
) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[
            ToolCall(
                id=tool_id,
                name="request_secret",
                input={"variable_name": variable_name, "prompt_text": prompt_text},
            ),
        ],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="tool_use",
    )


@pytest.fixture()
def ctx_dir(tmp_path):
    d = tmp_path / "context"
    d.mkdir()
    return d


@pytest.fixture()
def sa(ctx_dir):
    return SelfAwarenessContext(ctx_dir)


@pytest.fixture()
def ws(tmp_path):
    w = Workspace(tmp_path)
    w.initialize()
    return w


class TestUpdateContextTool:
    def test_tool_definition_structure(self):
        assert UPDATE_CONTEXT_TOOL["name"] == "update_context"
        props = UPDATE_CONTEXT_TOOL["input_schema"]["properties"]
        assert "updates" in props

    async def test_update_context_creates_file(self, sa, ctx_dir):
        """When LLM calls update_context, the file is created."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _update_context_response(
                    "I'll note that.",
                    [{"file": "project-info.md", "content": "Uses pytest and black"}],
                ),
                _text_response("Got it, I've noted that for future reference."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run, self_awareness=sa)
        reply = await session.turn("this project uses pytest and black")

        assert "noted" in reply.lower() or "reference" in reply.lower()
        assert (ctx_dir / "project-info.md").read_text() == "Uses pytest and black"

    async def test_update_context_deletes_file(self, sa, ctx_dir):
        """When LLM sends null content, the file is deleted."""
        (ctx_dir / "outdated.md").write_text("old info")

        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _update_context_response(
                    "Removing that.",
                    [{"file": "outdated.md", "content": None}],
                ),
                _text_response("Done, removed the outdated info."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run, self_awareness=sa)
        await session.turn("forget about the outdated conventions")

        assert not (ctx_dir / "outdated.md").exists()

    async def test_context_injected_into_system_prompt(self, sa, ctx_dir):
        """Self-awareness context files are injected into the system prompt."""
        (ctx_dir / "stack.md").write_text("Python 3.11 with asyncio")

        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hello!"))
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run, self_awareness=sa)
        await session.turn("hi")

        # Check that the system prompt passed to plan() includes context
        call_kwargs = mock_llm.plan.call_args
        system_prompt = call_kwargs.kwargs.get("system", "")
        assert "Self-Awareness Context" in system_prompt
        assert "Python 3.11 with asyncio" in system_prompt

    async def test_no_self_awareness_excludes_tool(self):
        """Without self_awareness, update_context tool is not offered."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hi!"))
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run, self_awareness=None)
        await session.turn("hello")

        call_kwargs = mock_llm.plan.call_args
        tools = call_kwargs.kwargs.get("tools", [])
        tool_names = [t["name"] for t in tools]
        assert "update_context" not in tool_names
        assert "execute_task" in tool_names

    async def test_tool_result_in_history(self, sa, ctx_dir):
        """update_context tool result appears in conversation history."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _update_context_response(
                    "Noting.",
                    [{"file": "note.md", "content": "Test note"}],
                ),
                _text_response("Done."),
            ]
        )
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run, self_awareness=sa)
        await session.turn("note this")

        # Find the tool result in history
        tool_result_msgs = [
            m for m in session.history
            if m["role"] == "user" and isinstance(m["content"], list)
        ]
        assert len(tool_result_msgs) == 1
        result_content = tool_result_msgs[0]["content"][0]["content"]
        assert "Context updated" in result_content


class TestRequestSecretTool:
    def test_tool_definition_structure(self):
        assert REQUEST_SECRET_TOOL["name"] == "request_secret"
        props = REQUEST_SECRET_TOOL["input_schema"]["properties"]
        assert "variable_name" in props
        assert "prompt_text" in props

    async def test_request_secret_stores_value(self, sa, ws):
        """request_secret stores the value and returns confirmation."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _request_secret_response(
                    "I need your GitHub token.",
                    "GITHUB_TOKEN",
                    "Enter your GitHub personal access token",
                ),
                _text_response("Great, GITHUB_TOKEN is configured."),
            ]
        )
        mock_run = AsyncMock()

        # Mock console.input to simulate user entering a secret
        mock_console = MagicMock()
        mock_console.input = MagicMock(return_value="ghp_testtoken123")

        session = ChatSession(
            mock_llm, mock_run,
            self_awareness=sa,
            workspace=ws,
            console=mock_console,
        )
        reply = await session.turn("I need to connect to GitHub")

        # Secret should be stored
        assert ws.get_secret("GITHUB_TOKEN") == "ghp_testtoken123"
        # Reply should NOT contain the actual secret
        assert "ghp_testtoken123" not in reply

    async def test_request_secret_already_set(self, sa, ws):
        """If the secret is already set, don't ask again."""
        ws.set_secret("EXISTING_KEY", "already_here")

        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _request_secret_response(
                    "Let me check.",
                    "EXISTING_KEY",
                    "Enter the key",
                ),
                _text_response("It's already configured."),
            ]
        )
        mock_run = AsyncMock()
        mock_console = MagicMock()

        session = ChatSession(
            mock_llm, mock_run,
            self_awareness=sa,
            workspace=ws,
            console=mock_console,
        )
        await session.turn("set up the key")

        # Console.input should NOT have been called since key exists
        mock_console.input.assert_not_called()

    async def test_request_secret_tool_included_with_workspace(self, ws):
        """request_secret tool is offered when workspace is provided."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hi!"))
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run, workspace=ws)
        await session.turn("hello")

        call_kwargs = mock_llm.plan.call_args
        tools = call_kwargs.kwargs.get("tools", [])
        tool_names = [t["name"] for t in tools]
        assert "request_secret" in tool_names

    async def test_request_secret_not_included_without_workspace(self):
        """request_secret tool is NOT offered when no workspace."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hi!"))
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run, workspace=None)
        await session.turn("hello")

        call_kwargs = mock_llm.plan.call_args
        tools = call_kwargs.kwargs.get("tools", [])
        tool_names = [t["name"] for t in tools]
        assert "request_secret" not in tool_names


class TestAntonMdInjection:
    async def test_anton_md_injected_into_system_prompt(self, ws, sa):
        """anton.md content is injected into the system prompt."""
        ws.anton_md_path.write_text("This project uses Django and PostgreSQL")

        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hello!"))
        mock_run = AsyncMock()

        session = ChatSession(
            mock_llm, mock_run,
            self_awareness=sa,
            workspace=ws,
        )
        await session.turn("hi")

        call_kwargs = mock_llm.plan.call_args
        system_prompt = call_kwargs.kwargs.get("system", "")
        assert "Project Context" in system_prompt
        assert "Django and PostgreSQL" in system_prompt

    async def test_empty_anton_md_no_section(self, ws, sa):
        """Empty anton.md doesn't add a section to the prompt."""
        ws.anton_md_path.write_text("")

        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hello!"))
        mock_run = AsyncMock()

        session = ChatSession(
            mock_llm, mock_run,
            self_awareness=sa,
            workspace=ws,
        )
        await session.turn("hi")

        call_kwargs = mock_llm.plan.call_args
        system_prompt = call_kwargs.kwargs.get("system", "")
        assert "Project Context" not in system_prompt


class TestRuntimeContext:
    async def test_runtime_context_injected_into_system_prompt(self):
        """Runtime context (provider/model) appears in the system prompt."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hello!"))
        mock_run = AsyncMock()

        session = ChatSession(
            mock_llm, mock_run,
            runtime_context="- Provider: anthropic\n- Planning model: claude-sonnet-4-6\n- Coding model: claude-opus-4-6",
        )
        await session.turn("hi")

        call_kwargs = mock_llm.plan.call_args
        system_prompt = call_kwargs.kwargs.get("system", "")
        assert "Provider: anthropic" in system_prompt
        assert "claude-sonnet-4-6" in system_prompt
        assert "claude-opus-4-6" in system_prompt

    async def test_system_prompt_warns_not_to_ask_about_llm(self):
        """System prompt includes instruction to never ask which LLM to use."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hello!"))
        mock_run = AsyncMock()

        session = ChatSession(
            mock_llm, mock_run,
            runtime_context="- Provider: anthropic",
        )
        await session.turn("hi")

        call_kwargs = mock_llm.plan.call_args
        system_prompt = call_kwargs.kwargs.get("system", "")
        assert "NEVER ask the user which" in system_prompt

    async def test_conversation_discipline_in_prompt(self):
        """System prompt includes conversation discipline rules."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hello!"))
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run, runtime_context="")
        await session.turn("hi")

        call_kwargs = mock_llm.plan.call_args
        system_prompt = call_kwargs.kwargs.get("system", "")
        assert "WAIT for their reply" in system_prompt
        assert "Never ask a question and call execute_task in the same turn" in system_prompt

    async def test_secret_handling_in_prompt(self):
        """System prompt includes secret handling rules."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hello!"))
        mock_run = AsyncMock()

        session = ChatSession(mock_llm, mock_run, runtime_context="")
        await session.turn("hi")

        call_kwargs = mock_llm.plan.call_args
        system_prompt = call_kwargs.kwargs.get("system", "")
        assert "request_secret" in system_prompt
        assert "NEVER passes through you" in system_prompt
