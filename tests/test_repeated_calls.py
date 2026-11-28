"""Identical-call loop breaker: a call repeated with the same result gets a note
from the third time and loses its tool for one round from the fifth."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.conftest import make_mock_llm

from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall, Usage
from anton.core.repeated_calls import RepeatedCalls, without_tools
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.tools.registry import ToolOutcome

WORK = "import json\nprint(json.dumps({'rows': 3}))"


def test_identical_calls_with_identical_results_are_counted_together():
    calls = RepeatedCalls()
    counts = [calls.record("scratchpad", {"code": WORK}, "rows 3") for _ in range(3)]
    assert counts == [1, 2, 3]


@pytest.mark.parametrize("other", [
    ("scratchpad", {"code": "print(2)"}, "rows 3"),
    ("read_text_file", {"code": WORK}, "rows 3"),
    ("scratchpad", {"code": WORK}, "rows 4"),
])
def test_a_different_call_or_result_is_a_new_count(other):
    calls = RepeatedCalls()
    calls.record("scratchpad", {"code": WORK}, "rows 3")
    assert calls.record(*other) == 1


def test_a_result_that_differs_only_in_a_hex_value_is_a_new_count():
    calls = RepeatedCalls()
    calls.record("scratchpad", {"code": WORK}, "block 0x12a4f3")
    assert calls.record("scratchpad", {"code": WORK}, "block 0x12a4f4") == 1
    calls.record("scratchpad", {"code": WORK}, "1920x1080")
    assert calls.record("scratchpad", {"code": WORK}, "1920x1081") == 1


def test_a_changing_memory_address_is_still_the_same_result():
    calls = RepeatedCalls()
    calls.record("scratchpad", {"code": WORK}, "<Env object at 0x7f3a2b10>")
    assert calls.record("scratchpad", {"code": WORK}, "<Env object at 0x7f99c4e8>") == 2


def test_withholding_never_leaves_a_round_without_tools():
    tools = [{"name": "scratchpad"}, {"name": "read_text_file"}]
    assert without_tools(tools, {"scratchpad"}) == [{"name": "read_text_file"}]
    assert without_tools([{"name": "scratchpad"}], {"scratchpad"}) == [{"name": "scratchpad"}]


@pytest.fixture()
def workspace():
    base = Path(__file__).resolve().parents[1] / ".pytest-workspace"
    base.mkdir(parents=True, exist_ok=True)
    return MagicMock(base=base)


def _usage() -> Usage:
    return Usage(input_tokens=250, output_tokens=250, cache_read_tokens=250, cache_creation_tokens=250)


def _call(i: int) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=f"tc_{i}", name="web_fetch", input={"url": "https://example.com/rows"})],
        usage=_usage(), stop_reason="tool_use",
    )


def _answer() -> LLMResponse:
    return LLMResponse(content="done", tool_calls=[], usage=_usage(), stop_reason="end_turn")


def _session(workspace, rounds: int, outputs, script=None):
    """Session whose LLM repeats one fetch `rounds` times (or plays `script`), then
    answers; runs return `outputs` in order. Records the tools offered per call."""
    mock_llm = make_mock_llm()
    script = list(script) if script is not None else [_call(i) for i in range(rounds)]
    offered: list[set[str]] = []

    def _plan_stream(**kwargs):
        offered.append({t["name"] for t in kwargs.get("tools") or []})

        async def _gen():
            resp = script.pop(0) if script else _answer()
            yield StreamComplete(response=resp)
        return _gen()

    async def _plan(**kwargs):
        return _answer()

    mock_llm.usage_listener = None
    mock_llm.plan_stream = _plan_stream
    mock_llm.plan = _plan
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    results = iter(outputs)

    async def _dispatch(*args, **kwargs):
        out = next(results)
        return out if isinstance(out, ToolOutcome) else ToolOutcome(content=out, ok=True)

    session.tool_registry.dispatch_tool = _dispatch
    return session, offered, script


async def _run(session, message="count the rows"):
    async for _ in session.turn_stream(message):
        pass


def _results(session) -> list[str]:
    return [
        block["content"] for m in session._history if isinstance(m.get("content"), list)
        for block in m["content"] if block.get("type") == "tool_result"
    ]


async def test_the_third_identical_result_carries_a_note(workspace):
    session, _, _ = _session(workspace, 3, ["rows 3"] * 3)
    await _run(session)

    results = _results(session)
    assert results[:2] == ["rows 3", "rows 3"]
    assert results[2].startswith("rows 3") and results[2] != "rows 3"


async def test_the_fifth_identical_result_withholds_the_tool_for_one_round(workspace):
    session, offered, _ = _session(workspace, 5, ["rows 3"] * 5)
    await _run(session)

    assert all("web_fetch" in names for names in offered[:5])
    assert "web_fetch" not in offered[5]
    assert "web_fetch" in offered[6]


async def test_a_repeated_call_with_changing_results_is_left_alone(workspace):
    session, offered, _ = _session(workspace, 6, [f"rows {i}" for i in range(6)])
    await _run(session)

    assert all("web_fetch" in names for names in offered[:7])
    assert _results(session) == [f"rows {i}" for i in range(6)]


async def test_calls_that_never_ran_are_not_counted(workspace):
    declined = ToolOutcome(content="Not run: the user declined.", ok=False, reason="write_not_approved")
    session, offered, _ = _session(workspace, 5, [declined] * 5)
    await _run(session)

    assert all("web_fetch" in names for names in offered[:6])


async def test_each_turn_counts_its_own_repeats(workspace):
    session, _, script = _session(workspace, 4, ["rows 3"] * 8)
    await _run(session)
    script[:] = [_call(i) for i in range(4)]
    await _run(session)

    assert _results(session)[4:6] == ["rows 3", "rows 3"]
