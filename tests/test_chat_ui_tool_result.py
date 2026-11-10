from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anton.core.llm.provider import StreamToolResult


@pytest.mark.parametrize("name,action,shown", [
    ("scratchpad", "dump", True),
    ("scratchpad", "exec", False),
    ("scratchpad", None, False),
    ("generate_artifact", "message", True),
    ("generate_artifact", None, False),
])
def test_which_tool_results_the_cli_prints(name, action, shown):
    from anton.chat_ui import is_displayed_tool_result

    assert is_displayed_tool_result(StreamToolResult(name=name, content="x", action=action)) is shown


_SCRATCHPAD_EVENTS = [
    StreamToolResult(name="scratchpad", content="EXEC", action="exec"),
    StreamToolResult(name="scratchpad", content="DUMP", action="dump"),
]


class _QuietEscapeWatcher:
    """Stands in for the terminal key watcher; never cancels."""

    def __init__(self, **_kw):
        self.cancelled = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


async def _scripted_turn(*_a, **_kw):
    for event in _SCRATCHPAD_EVENTS:
        yield event


async def test_goal_loop_prints_only_displayed_tool_results(make_session):
    from anton.commands.goal import run_goal_loop

    session = make_session()
    session.turn_stream = _scripted_turn
    display = MagicMock()
    with patch("anton.chat_ui.EscapeWatcher", _QuietEscapeWatcher):
        await run_goal_loop(MagicMock(), session, display, "objective", 1)

    assert [c.args for c in display.show_tool_result.call_args_list] == [("DUMP",)]


async def test_chat_loop_prints_only_displayed_tool_results(tmp_path):
    import anton.chat as chat
    import anton.chat_session as chat_session
    from tests.conftest import make_mock_llm

    session = MagicMock(_pending_memory_confirmations=[], elicitor=None)
    session.turn_stream = _scripted_turn
    session.close = AsyncMock()
    prompt = MagicMock()
    prompt.prompt_async = AsyncMock(side_effect=["hello", EOFError])
    display = MagicMock()
    settings = MagicMock(
        workspace_path=tmp_path,
        context_dir=str(tmp_path / "context"),
        episodic_memory=False,
        memory_mode="off",
        effective_memory_mode="off",
        minds_api_key=None,
        minds_url=None,
        planning_provider="anthropic",
    )
    vault = MagicMock()
    vault.list_connections.return_value = []
    with (
        patch("anton.core.llm.client.LLMClient.from_settings", return_value=make_mock_llm()),
        patch.object(chat, "LocalDataVault", return_value=vault),
        patch.object(chat, "ChatSession", return_value=session),
        patch.object(chat, "build_runtime_context", lambda _s: ""),
        patch.object(chat, "PromptSession", return_value=prompt),
        patch.object(chat, "attach_image_path_detector", lambda *a: None),
        patch.object(chat_session, "get_runtime_factory", lambda *a, **k: MagicMock()),
        patch("anton.chat_ui.StreamDisplay", return_value=display),
        patch("anton.chat_ui.EscapeWatcher", _QuietEscapeWatcher),
        patch("anton.analytics.send_event"),
    ):
        await chat._chat_loop(MagicMock(), settings)

    assert prompt.prompt_async.await_count == 2
    assert [c.args for c in display.show_tool_result.call_args_list] == [("DUMP",)]
