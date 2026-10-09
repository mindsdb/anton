"""Effort ordering and the output budgets derived from it."""
from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from anton.core.llm.effort import (
    MAX_STREAM_BUDGET,
    capped,
    client_stream_budget,
    effort_ceiling_of,
    retry_budget,
    stream_budget_for,
)


@pytest.mark.parametrize(
    "effort, ceiling, expected",
    [
        ("max", "high", "high"),
        ("xhigh", "high", "high"),
        ("high", "high", "high"),
        ("low", "high", "low"),
        (None, "high", None),
        ("turbo", "high", "turbo"),
        ("max", "unknown", "max"),
        ("max", None, "max"),
    ],
)
def test_capped(effort, ceiling, expected):
    assert capped(effort, ceiling) == expected


@pytest.mark.parametrize(
    "effort, default, expected",
    [
        ("high", 8192, 16384),
        ("xhigh", 8192, 32768),
        ("max", 8192, 65536),
        ("medium", 8192, 8192),
        (None, 8192, 8192),
        ("turbo", 8192, 8192),
        ("high", 100000, 100000),
        (MagicMock(), 8192, 8192),
    ],
)
def test_stream_budget_for(effort, default, expected):
    assert stream_budget_for(effort, default) == expected


@pytest.mark.parametrize(
    "budget, expected",
    [(16384, 32768), (32768, 65536), (65536, 65536), (100000, 100000)],
)
def test_retry_budget_never_shrinks(budget, expected):
    assert retry_budget(budget) == expected
    assert MAX_STREAM_BUDGET == 65536


def test_client_stream_budget_uses_the_client_method():
    llm = SimpleNamespace(stream_budget=lambda role: 32768 if role == "coding" else 16384)
    assert client_stream_budget(llm, "planning") == 16384
    assert client_stream_budget(llm, "coding") == 32768


def test_client_stream_budget_falls_back_for_mocks():
    assert client_stream_budget(AsyncMock(max_tokens=1000), "planning") == 1000
    assert client_stream_budget(MagicMock(max_tokens=2000), "planning") == 2000
    assert client_stream_budget(None, "planning") == 8192


def test_effort_ceiling_of_uses_the_client_context_manager():
    entered = []

    @contextlib.contextmanager
    def ceiling(level):
        entered.append(level)
        yield

    with effort_ceiling_of(SimpleNamespace(effort_ceiling=ceiling), "high"):
        pass
    assert entered == ["high"]


@pytest.mark.parametrize("llm", [None, AsyncMock(), SimpleNamespace()])
def test_effort_ceiling_of_is_a_no_op_without_support(llm):
    with effort_ceiling_of(llm, "high"):
        pass
