"""Tests for the scratchpad observer dispatch in `handle_scratchpad`.

These tests verify that the observer hooks fire at the right moments
around `pad.execute()` and that the runtime stays untouched. They use
a fake observer (not the real Cerebellum) so we can assert exactly
what was passed and in what order.

The dispatcher pattern: observation is an orchestration concern that
lives at the dispatcher layer. The runtime is a pure execution engine
and never sees observers. These tests pin that contract.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from anton.core.backends.base import Cell
from anton.core.tools.tool_handlers import (
    _fire_post_execute,
    _fire_pre_execute,
    handle_scratchpad,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fake observer that records the cells it sees
# ─────────────────────────────────────────────────────────────────────────────


class _RecordingObserver:
    def __init__(self, *, pre_raises: Exception | None = None,
                 post_raises: Exception | None = None) -> None:
        self.pre_calls: list[Cell] = []
        self.post_calls: list[Cell] = []
        self._pre_raises = pre_raises
        self._post_raises = post_raises

    async def on_pre_execute(self, cell: Cell) -> None:
        self.pre_calls.append(cell)
        if self._pre_raises is not None:
            raise self._pre_raises

    async def on_post_execute(self, cell: Cell) -> None:
        self.post_calls.append(cell)
        if self._post_raises is not None:
            raise self._post_raises


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fake_session(
    observers: list | None = None,
    cell_to_return: Cell | None = None,
) -> MagicMock:
    """Build a minimal session-like mock that handle_scratchpad can drive."""
    session = MagicMock()
    if observers is not None:
        session._scratchpad_observers = observers
    else:
        session._scratchpad_observers = []
    session._record_cell_explainability = MagicMock()

    # The cell that pad.execute() will return when called
    if cell_to_return is None:
        cell_to_return = Cell(
            code="print('hi')",
            stdout="hi",
            stderr="",
            error=None,
            description="say hi",
        )
    pad = MagicMock()
    pad.execute = AsyncMock(return_value=cell_to_return)
    pad.view = MagicMock(return_value="view output")
    pad.render_notebook = MagicMock(return_value="notebook")
    pad.reset = AsyncMock()
    pad.install_packages = AsyncMock(return_value="installed")

    session._scratchpads = MagicMock()
    session._scratchpads.get_or_create = AsyncMock(return_value=pad)
    session._scratchpads.pads = {"main": pad}
    session._scratchpads.remove = AsyncMock(return_value="removed 'main'.")
    return session, pad


# ─────────────────────────────────────────────────────────────────────────────
# _fire_pre_execute / _fire_post_execute helpers — direct unit tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFireHelpers:
    @pytest.mark.asyncio
    async def test_no_observers_is_noop(self):
        session = SimpleNamespace(_scratchpad_observers=[])
        cell = Cell(code="x", stdout="", stderr="", error=None)
        # Should not raise
        await _fire_pre_execute(session, cell)
        await _fire_post_execute(session, cell)

    @pytest.mark.asyncio
    async def test_missing_attribute_is_noop(self):
        # Session that doesn't have _scratchpad_observers at all
        session = SimpleNamespace()
        cell = Cell(code="x", stdout="", stderr="", error=None)
        await _fire_pre_execute(session, cell)
        await _fire_post_execute(session, cell)

    @pytest.mark.asyncio
    async def test_observer_receives_cell(self):
        obs = _RecordingObserver()
        session = SimpleNamespace(_scratchpad_observers=[obs])
        cell = Cell(code="x", stdout="", stderr="", error=None, description="set x")
        await _fire_pre_execute(session, cell)
        assert len(obs.pre_calls) == 1
        assert obs.pre_calls[0].code == "x"
        assert obs.pre_calls[0].description == "set x"

    @pytest.mark.asyncio
    async def test_multiple_observers_all_fire(self):
        obs_a = _RecordingObserver()
        obs_b = _RecordingObserver()
        session = SimpleNamespace(_scratchpad_observers=[obs_a, obs_b])
        cell = Cell(code="y", stdout="", stderr="", error=None)
        await _fire_post_execute(session, cell)
        assert len(obs_a.post_calls) == 1
        assert len(obs_b.post_calls) == 1

    @pytest.mark.asyncio
    async def test_one_observer_raising_does_not_stop_others(self):
        bad = _RecordingObserver(pre_raises=RuntimeError("boom"))
        good = _RecordingObserver()
        session = SimpleNamespace(_scratchpad_observers=[bad, good])
        cell = Cell(code="x", stdout="", stderr="", error=None)
        # Must not raise
        await _fire_pre_execute(session, cell)
        # Good observer still fired
        assert len(good.pre_calls) == 1

    @pytest.mark.asyncio
    async def test_observer_without_method_is_skipped(self):
        # Observer with no on_pre_execute/on_post_execute methods
        bare = SimpleNamespace()
        session = SimpleNamespace(_scratchpad_observers=[bare])
        cell = Cell(code="x", stdout="", stderr="", error=None)
        # Should silently skip — no AttributeError
        await _fire_pre_execute(session, cell)
        await _fire_post_execute(session, cell)


# ─────────────────────────────────────────────────────────────────────────────
# handle_scratchpad: full dispatch flow with observers
# ─────────────────────────────────────────────────────────────────────────────


class TestHandleScratchpadObserverIntegration:
    @pytest.mark.asyncio
    async def test_exec_fires_pre_then_post(self):
        obs = _RecordingObserver()
        cell = Cell(
            code="print('hi')",
            stdout="hi",
            stderr="",
            error=None,
            description="say hi",
        )
        session, pad = _fake_session(observers=[obs], cell_to_return=cell)

        await handle_scratchpad(
            session,
            {
                "action": "exec",
                "name": "main",
                "code": "print('hi')",
                "one_line_description": "say hi",
                "estimated_execution_time_seconds": 1,
            },
        )

        # Pre-execute fired with the prelim cell (no outputs yet)
        assert len(obs.pre_calls) == 1
        prelim = obs.pre_calls[0]
        assert prelim.code == "print('hi')"
        assert prelim.description == "say hi"
        assert prelim.stdout == ""  # not yet executed
        assert prelim.error is None

        # Post-execute fired with the actual cell
        assert len(obs.post_calls) == 1
        actual = obs.post_calls[0]
        assert actual.stdout == "hi"
        assert actual.description == "say hi"

        # The order: pre fires before pad.execute, post fires after
        pad.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exec_passes_error_cell_to_observer(self):
        obs = _RecordingObserver()
        cell = Cell(
            code="x = 1/0",
            stdout="",
            stderr="",
            error="ZeroDivisionError: division by zero",
            description="divide by zero",
        )
        session, _ = _fake_session(observers=[obs], cell_to_return=cell)

        await handle_scratchpad(
            session,
            {
                "action": "exec",
                "name": "main",
                "code": "x = 1/0",
                "one_line_description": "divide by zero",
                "estimated_execution_time_seconds": 1,
            },
        )

        assert len(obs.post_calls) == 1
        assert obs.post_calls[0].error is not None
        assert "ZeroDivisionError" in obs.post_calls[0].error

    @pytest.mark.asyncio
    async def test_exec_with_no_observers_works_unchanged(self):
        # Empty observer list — exec should still work end-to-end
        session, _ = _fake_session(observers=[])
        result = await handle_scratchpad(
            session,
            {
                "action": "exec",
                "name": "main",
                "code": "print('hi')",
                "one_line_description": "say hi",
                "estimated_execution_time_seconds": 1,
            },
        )
        # Result is whatever format_cell_result returns — non-empty string
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_observer_exception_does_not_break_exec(self):
        # An observer that throws on every call
        bad = _RecordingObserver(
            pre_raises=RuntimeError("pre boom"),
            post_raises=RuntimeError("post boom"),
        )
        session, _ = _fake_session(observers=[bad])

        # Exec should still complete and return a result
        result = await handle_scratchpad(
            session,
            {
                "action": "exec",
                "name": "main",
                "code": "print('hi')",
                "one_line_description": "say hi",
                "estimated_execution_time_seconds": 1,
            },
        )
        assert isinstance(result, str)
        # Both observer methods were called even though they raised
        assert len(bad.pre_calls) == 1
        assert len(bad.post_calls) == 1

    @pytest.mark.asyncio
    async def test_non_exec_actions_do_not_fire_observers(self):
        """view, reset, install, dump, remove should not trigger observers."""
        obs = _RecordingObserver()
        session, _ = _fake_session(observers=[obs])

        for action in ("view", "reset", "dump"):
            await handle_scratchpad(
                session, {"action": action, "name": "main"}
            )

        await handle_scratchpad(
            session,
            {"action": "install", "name": "main", "packages": ["x"]},
        )
        await handle_scratchpad(session, {"action": "remove", "name": "main"})

        assert obs.pre_calls == []
        assert obs.post_calls == []
