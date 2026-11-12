"""The browser is tidied on close only after a turn that ended normally and used it."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import anton.core.browser.tool as browser_tool
from anton.core.llm.provider import ProviderAuthError
from anton.core.session import ChatSession, ChatSessionConfig
from tests.conftest import make_mock_llm


@pytest.fixture()
def workspace():
    base = Path(__file__).resolve().parents[1] / ".pytest-workspace"
    base.mkdir(parents=True, exist_ok=True)
    return MagicMock(base=base)


@pytest.fixture()
def tidied(monkeypatch):
    """Whether close() found the latest turn worth tidying after."""
    seen: list[bool] = []

    async def tidy_after_turn(session):
        seen.append(session._tidy_browser)

    monkeypatch.setattr(browser_tool, "tidy_after_turn", tidy_after_turn)
    return seen


def _session(workspace, mock_llm, *, used=True) -> ChatSession:
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    session._browser_state = SimpleNamespace(used=used, aclose=AsyncMock())
    return session


async def _turn(session: ChatSession) -> None:
    try:
        async for _ in session.turn_stream("find the cheapest flight"):
            pass
    finally:
        await session.close()


async def test_a_completed_turn_that_used_the_browser_is_tidied(workspace, tidied):
    session = _session(workspace, make_mock_llm())
    await _turn(session)
    assert tidied == [True]
    assert session._browser_state.used is False


async def test_a_completed_turn_without_the_browser_is_not(workspace, tidied):
    await _turn(_session(workspace, make_mock_llm(), used=False))
    assert tidied == [False]


def _raising(exc):
    def plan_stream(**kwargs):
        raise exc
    return plan_stream


async def test_a_failed_turn_is_not_tidied(workspace, tidied):
    mock_llm = make_mock_llm()
    mock_llm.plan_stream = _raising(ProviderAuthError("Invalid API key"))
    with pytest.raises(ProviderAuthError):
        await _turn(_session(workspace, mock_llm))
    assert tidied == [False]


async def test_a_cancelled_turn_is_not_tidied(workspace, tidied):
    mock_llm = make_mock_llm()
    mock_llm.plan_stream = _raising(asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await _turn(_session(workspace, mock_llm))
    assert tidied == [False]
