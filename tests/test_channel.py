from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from anton.channel.terminal import TerminalChannel
from anton.events.types import (
    Phase,
    PromptUser,
    StatusUpdate,
    TaskComplete,
    TaskFailed,
)


@pytest.fixture()
def channel():
    with patch("anton.channel.terminal.Console") as MockConsole:
        ch = TerminalChannel()
        ch.console = MockConsole()
        yield ch


class TestTerminalChannelEmit:
    async def test_emit_status_update_starts_spinner(self, channel):
        event = StatusUpdate(phase=Phase.PLANNING, message="thinking")
        await channel.emit(event)
        channel.console.status.assert_called_once()
        channel.console.status.return_value.start.assert_called_once()

    async def test_emit_status_update_with_eta(self, channel):
        event = StatusUpdate(phase=Phase.EXECUTING, message="running", eta_seconds=5)
        await channel.emit(event)
        call_args = channel.console.status.call_args
        assert "5s" in call_args[0][0]

    async def test_emit_task_complete_prints_panel(self, channel):
        event = TaskComplete(summary="all done")
        await channel.emit(event)
        channel.console.print.assert_called_once()

    async def test_emit_task_failed_prints_panel(self, channel):
        event = TaskFailed(error_summary="broken")
        await channel.emit(event)
        channel.console.print.assert_called_once()

    async def test_emit_prompt_user_prints_question(self, channel):
        event = PromptUser(question="continue?")
        await channel.emit(event)
        channel.console.print.assert_called_once()

    async def test_spinner_stopped_before_new_one(self, channel):
        mock_status = MagicMock()
        channel._status_ctx = mock_status
        event = StatusUpdate(phase=Phase.EXECUTING, message="step 2")
        await channel.emit(event)
        mock_status.stop.assert_called_once()


class TestTerminalChannelClose:
    async def test_close_stops_spinner(self, channel):
        mock_status = MagicMock()
        channel._status_ctx = mock_status
        await channel.close()
        mock_status.stop.assert_called_once()
        assert channel._status_ctx is None

    async def test_close_without_spinner(self, channel):
        # Should not raise
        await channel.close()
