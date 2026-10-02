"""Scratchpad pre-warm: opt-in, shared with the first real use, and reaped on close."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from anton.chat import ChatSession
from anton.core.backends.manager import ScratchpadManager
from anton.core.llm.provider import LLMResponse, Usage
from anton.core.session import ChatSessionConfig
from tests.conftest import make_mock_llm


class _Pad:
    """Minimal runtime: start() blocks until released; records lifecycle."""

    def __init__(self, gate: asyncio.Event, log: list, **_):
        self._gate, self._log = gate, log

    async def start(self):
        self._log.append("start")
        await self._gate.wait()
        self._log.append("started")

    async def close(self):
        self._log.append("close")

    def set_scratchpad_ds_env(self, env):
        pass


def _manager(gate, log):
    return ScratchpadManager(runtime_factory=lambda **kw: _Pad(gate, log, **kw),
                             coding_provider="", coding_model="", coding_api_key="", coding_base_url="")


def _text(text):
    return LLMResponse(content=text, tool_calls=[], usage=Usage(input_tokens=1, output_tokens=1), stop_reason="end_turn")


async def test_prewarm_is_off_by_default():
    llm = make_mock_llm()
    llm.plan = AsyncMock(return_value=_text("hi"))
    session = ChatSession(ChatSessionConfig(llm_client=llm))
    await session.turn("hello")
    assert not getattr(session._scratchpads, "_starting", None)
    assert not session._scratchpads._pads


async def test_prewarm_is_shared_with_the_first_real_use():
    gate, log = asyncio.Event(), []
    mgr = _manager(gate, log)
    task = mgr.prewarm("main")
    assert task is not None and mgr.prewarm("main") is None  # one boot only
    user = asyncio.ensure_future(mgr.get_or_create("main"))
    await asyncio.sleep(0)
    gate.set()
    pad = await user
    await task
    assert log == ["start", "started"]  # the real use shared the pre-warm's start
    assert mgr._pads["main"] is pad


async def test_close_all_cancels_an_inflight_prewarm_and_reaps_its_process():
    gate, log = asyncio.Event(), []
    mgr = _manager(gate, log)
    task = mgr.prewarm("main")
    await asyncio.sleep(0)
    assert log == ["start"]
    await asyncio.wait_for(mgr.close_all(), timeout=2)
    assert task.cancelled()
    assert log == ["start", "close"]  # the half-started pad was closed, never registered
    assert not mgr._pads


async def test_prewarm_runs_when_the_host_opts_in():
    gate, log = asyncio.Event(), []
    llm = make_mock_llm()
    llm.plan = AsyncMock(return_value=_text("hi"))
    session = ChatSession(ChatSessionConfig(llm_client=llm, prewarm_scratchpad=True,
                                            runtime_factory=lambda **kw: _Pad(gate, log, **kw)))
    await session.turn("hello")
    for _ in range(5):  # the boot is a background task; let it reach start()
        await asyncio.sleep(0)
    assert log == ["start"]
    await asyncio.wait_for(session._scratchpads.close_all(), timeout=2)
    assert log == ["start", "close"]
