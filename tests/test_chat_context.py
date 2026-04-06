from __future__ import annotations

import asyncio
import socket
import urllib.error
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from anton.chat import ChatSession, _handle_connect
from anton.minds_client import describe_minds_connection_error as _describe_minds_connection_error
from anton.config.settings import AntonSettings
from anton.tools import MEMORIZE_TOOL
from anton.context.self_awareness import SelfAwarenessContext
from anton.llm.provider import LLMResponse, ToolCall, Usage
from anton.workspace import Workspace
from anton.memory.cortex import Cortex
from anton.memory.hippocampus import Hippocampus


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )


def _memorize_response(
    text: str, entries: list[dict], tool_id: str = "tc_mem_1"
) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[
            ToolCall(
                id=tool_id,
                name="memorize",
                input={"entries": entries},
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


@pytest.fixture()
def memory_dirs(tmp_path):
    global_dir = tmp_path / "global_memory"
    project_dir = tmp_path / "project_memory"
    global_dir.mkdir()
    project_dir.mkdir()
    return global_dir, project_dir


@pytest.fixture()
def cortex(memory_dirs):
    global_dir, project_dir = memory_dirs
    return Cortex(global_dir=global_dir, project_dir=project_dir, mode="autopilot")


class TestMemorizeTool:
    def test_tool_definition_structure(self):
        assert MEMORIZE_TOOL["name"] == "memorize"
        props = MEMORIZE_TOOL["input_schema"]["properties"]
        assert "entries" in props

    async def test_memorize_creates_rule(self, cortex, memory_dirs):
        """When LLM calls memorize, a rule is created in memory."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _memorize_response(
                    "I'll remember that.",
                    [{"text": "Use httpx instead of requests", "kind": "always", "scope": "project"}],
                ),
                _text_response("Got it, I've noted that for future reference."),
            ]
        )

        session = ChatSession(mock_llm, cortex=cortex)
        reply = await session.turn("always use httpx instead of requests")
        await asyncio.sleep(0)  # Let fire-and-forget encode task run

        assert "noted" in reply.lower() or "reference" in reply.lower()

        # Verify the rule was written
        global_dir, project_dir = memory_dirs
        rules_path = project_dir / "rules.md"
        assert rules_path.exists()
        content = rules_path.read_text()
        assert "Use httpx instead of requests" in content

    async def test_memorize_creates_lesson(self, cortex, memory_dirs):
        """When LLM calls memorize with kind=lesson, a lesson is created."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _memorize_response(
                    "Noting that.",
                    [{"text": "CoinGecko rate-limits at 50/min", "kind": "lesson", "scope": "global", "topic": "api-coingecko"}],
                ),
                _text_response("Done."),
            ]
        )

        session = ChatSession(mock_llm, cortex=cortex)
        await session.turn("coingecko rate limits at 50 per minute")
        await asyncio.sleep(0)  # Let fire-and-forget encode task run

        global_dir, project_dir = memory_dirs
        lessons_path = global_dir / "lessons.md"
        assert lessons_path.exists()
        content = lessons_path.read_text()
        assert "CoinGecko rate-limits at 50/min" in content

    async def test_memory_injected_into_system_prompt(self, cortex, memory_dirs):
        """Memory context is injected into the system prompt."""
        global_dir, project_dir = memory_dirs

        # Pre-populate some memory
        hc = Hippocampus(project_dir)
        hc.encode_rule("Use httpx instead of requests", kind="always", confidence="high", source="user")

        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hello!"))

        session = ChatSession(mock_llm, cortex=cortex)
        await session.turn("hi")

        call_kwargs = mock_llm.plan.call_args
        system_prompt = call_kwargs.kwargs.get("system", "")
        assert "Project Rules" in system_prompt
        assert "Use httpx instead of requests" in system_prompt

    async def test_no_cortex_excludes_memorize_tool(self):
        """Without cortex or self_awareness, memorize tool is not offered."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hi!"))

        session = ChatSession(mock_llm, self_awareness=None, cortex=None)
        await session.turn("hello")

        call_kwargs = mock_llm.plan.call_args
        tools = call_kwargs.kwargs.get("tools", [])
        tool_names = [t["name"] for t in tools]
        assert "memorize" not in tool_names
        assert "scratchpad" in tool_names

    async def test_cortex_includes_memorize_tool(self, cortex):
        """With cortex, memorize tool is included."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hi!"))

        session = ChatSession(mock_llm, cortex=cortex)
        await session.turn("hello")

        call_kwargs = mock_llm.plan.call_args
        tools = call_kwargs.kwargs.get("tools", [])
        tool_names = [t["name"] for t in tools]
        assert "memorize" in tool_names
        assert "scratchpad" in tool_names

    async def test_tool_result_in_history(self, cortex, memory_dirs):
        """memorize tool result appears in conversation history."""
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(
            side_effect=[
                _memorize_response(
                    "Noting.",
                    [{"text": "Test memory", "kind": "lesson", "scope": "project"}],
                ),
                _text_response("Done."),
            ]
        )

        session = ChatSession(mock_llm, cortex=cortex)
        await session.turn("note this")

        tool_result_msgs = [
            m for m in session.history
            if m["role"] == "user" and isinstance(m["content"], list)
        ]
        assert len(tool_result_msgs) == 1
        result_content = tool_result_msgs[0]["content"][0]["content"]
        assert "Memory updated" in result_content


class TestAntonMdInjection:
    async def test_anton_md_injected_into_system_prompt(self, ws, cortex):
        """anton.md content is injected into the system prompt."""
        ws.anton_md_path.write_text("This project uses Django and PostgreSQL")

        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hello!"))

        session = ChatSession(
            mock_llm,
            cortex=cortex,
            workspace=ws,
        )
        await session.turn("hi")

        call_kwargs = mock_llm.plan.call_args
        system_prompt = call_kwargs.kwargs.get("system", "")
        assert "Project Context" in system_prompt
        assert "Django and PostgreSQL" in system_prompt

    async def test_empty_anton_md_no_section(self, ws, cortex):
        """Empty anton.md doesn't add a section to the prompt."""
        ws.anton_md_path.write_text("")

        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_text_response("Hello!"))

        session = ChatSession(
            mock_llm,
            cortex=cortex,
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

        session = ChatSession(
            mock_llm,
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

        session = ChatSession(
            mock_llm,
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

        session = ChatSession(mock_llm, runtime_context="")
        await session.turn("hi")

        call_kwargs = mock_llm.plan.call_args
        system_prompt = call_kwargs.kwargs.get("system", "")
        assert "WAIT for their reply" in system_prompt


class TestMindsSetupRecovery:
    def test_describe_minds_http_401_is_factual(self):
        error = urllib.error.HTTPError(
            "https://mdb.ai/api/v1/minds/",
            401,
            "Unauthorized",
            None,
            None,
        )

        headline, advice = _describe_minds_connection_error(error)

        assert headline == "Connection failed (HTTP 401: Unauthorized). The server rejected the request."
        assert "invalid or expired credentials" in advice

    def test_describe_minds_timeout_is_factual(self):
        error = urllib.error.URLError(socket.timeout("timed out"))

        headline, advice = _describe_minds_connection_error(error)

        assert headline == "Connection failed because the request timed out."
        assert "server is slow or unavailable" in advice

    async def test_connect_can_reconfigure_api_key_after_auth_failure(
        self,
        tmp_path,
        monkeypatch,
    ):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))

        prompts = iter(["mdb.ai", "", "1", "new-key", "1"])

        async def fake_prompt(*args, **kwargs):
            return next(prompts)

        monkeypatch.setattr("anton.chat.prompt_or_cancel", fake_prompt)
        monkeypatch.setattr("anton.prompt_utils.prompt_or_cancel", fake_prompt)

        calls: list[tuple[str, str, bool]] = []

        def fake_list_minds(base_url: str, api_key: str, verify: bool = True):
            calls.append((base_url, api_key, verify))
            if len(calls) == 1:
                raise urllib.error.HTTPError(
                    f"{base_url}/api/v1/minds/",
                    401,
                    "Unauthorized",
                    None,
                    None,
                )
            return [{"name": "warehouse", "datasources": []}]

        monkeypatch.setattr("anton.chat.list_minds", fake_list_minds)
        monkeypatch.setattr("anton.chat.test_llm", lambda *args, **kwargs: True)
        rebuilt = object()
        monkeypatch.setattr("anton.chat._rebuild_session", lambda **kwargs: rebuilt)

        workspace_base = tmp_path / "workspace"
        workspace_base.mkdir()
        workspace = Workspace(workspace_base)
        workspace.initialize()
        settings = AntonSettings(minds_api_key="old-key", _env_file=None)
        console = MagicMock()

        result = await _handle_connect(
            console,
            settings,
            workspace,
            state={},
            self_awareness=None,
            cortex=None,
            session=object(),
        )

        assert result is rebuilt
        assert calls == [
            ("https://mdb.ai", "old-key", True),
            ("https://mdb.ai", "new-key", True),
        ]
        assert settings.minds_api_key == "new-key"
        assert "ANTON_MINDS_API_KEY=new-key" in (home / ".anton" / ".env").read_text()
        printed = "\n".join(str(call.args[0]) for call in console.print.call_args_list if call.args)
        assert "The server rejected the request." in printed
        assert "Common reasons: invalid or expired credentials" in printed
        assert "Recovery options:" in printed
        assert "certificate issue" not in printed

    async def test_connect_can_retry_without_ssl_verification_after_timeout(
        self,
        tmp_path,
        monkeypatch,
    ):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))

        prompts = iter(["https://terabase.dev.mdb.ai", "", "2", "1"])

        async def fake_prompt(*args, **kwargs):
            return next(prompts)

        monkeypatch.setattr("anton.chat.prompt_or_cancel", fake_prompt)
        monkeypatch.setattr("anton.prompt_utils.prompt_or_cancel", fake_prompt)

        calls: list[tuple[str, str, bool]] = []

        def fake_list_minds(base_url: str, api_key: str, verify: bool = True):
            calls.append((base_url, api_key, verify))
            if len(calls) == 1:
                raise urllib.error.URLError(socket.timeout("timed out"))
            return [{"name": "warehouse", "datasources": []}]

        monkeypatch.setattr("anton.chat.list_minds", fake_list_minds)
        monkeypatch.setattr("anton.chat.test_llm", lambda *args, **kwargs: True)
        rebuilt = object()
        monkeypatch.setattr("anton.chat._rebuild_session", lambda **kwargs: rebuilt)

        workspace_base = tmp_path / "workspace"
        workspace_base.mkdir()
        workspace = Workspace(workspace_base)
        workspace.initialize()
        settings = AntonSettings(minds_api_key="minds-key", _env_file=None)
        console = MagicMock()

        result = await _handle_connect(
            console,
            settings,
            workspace,
            state={},
            self_awareness=None,
            cortex=None,
            session=object(),
        )

        assert result is rebuilt
        assert calls == [
            ("https://terabase.dev.mdb.ai", "minds-key", True),
            ("https://terabase.dev.mdb.ai", "minds-key", False),
        ]
        assert settings.minds_ssl_verify is False
        assert "ANTON_MINDS_SSL_VERIFY=false" in (home / ".anton" / ".env").read_text()
        printed = "\n".join(str(call.args[0]) for call in console.print.call_args_list if call.args)
        assert "Connection failed because the request timed out." in printed
        assert "server is slow or unavailable" in printed
        assert "Recovery options:" in printed
