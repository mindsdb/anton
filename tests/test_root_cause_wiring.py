"""`ToolOutcome.reason` reaches the session ledger, and nothing else changes (ENG-1492).

`test_root_cause.py` covers the classifier as pure logic. This covers the part
that was actually broken for two tickets: the value existed and **nothing read
it** — both dispatch sites took `.content` and `.ok` and dropped `.reason` on the
floor. So these drive the real `turn_stream` / `turn` and assert on the ledger.

The second half is the no-behaviour-change contract. This ticket is measurement
only; if it can alter a turn, it has failed regardless of how good the numbers
look.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import make_mock_llm

from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall, Usage
from anton.core.root_cause import TIER_SELF, TIER_WALL
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.tools.registry import ToolOutcome


@pytest.fixture()
def workspace():
    base = Path(__file__).resolve().parents[1] / ".pytest-workspace"
    base.mkdir(parents=True, exist_ok=True)
    return MagicMock(base=base)


def _usage(n: int = 1_000) -> Usage:
    return Usage(input_tokens=n // 2, output_tokens=n // 2)


def _tool_call(i: int) -> LLMResponse:
    return LLMResponse(
        content="working",
        tool_calls=[ToolCall(id=f"tc{i}", name="scratchpad",
                             input={"action": "view", "name": "m"})],
        usage=_usage(), stop_reason="tool_use",
    )


def _text(t: str = "done") -> LLMResponse:
    return LLMResponse(content=t, tool_calls=[], usage=_usage(), stop_reason="end_turn")


def _session(workspace, outcomes, n_tool_calls: int):
    """Session whose tool dispatch returns `outcomes` in order."""
    llm = make_mock_llm()
    llm.usage_listener = None
    seq = {"i": 0}

    def plan_stream(**kw):
        async def gen():
            seq["i"] += 1
            if llm.usage_listener:
                llm.usage_listener("planning", "m", _usage())
            yield StreamComplete(
                response=_tool_call(seq["i"]) if seq["i"] <= n_tool_calls else _text()
            )
        return gen()

    async def plan(**kw):
        seq["i"] += 1
        if llm.usage_listener:
            llm.usage_listener("planning", "m", _usage())
        return _tool_call(seq["i"]) if seq["i"] <= n_tool_calls else _text()

    llm.plan_stream = plan_stream
    llm.plan = plan
    s = ChatSession(ChatSessionConfig(llm_client=llm, workspace=workspace))
    s._max_turn_tokens = 0  # ceiling off — this ticket must be independent of it
    pending = list(outcomes)
    s.tool_registry.dispatch_tool = AsyncMock(
        side_effect=lambda *a, **k: pending.pop(0) if pending
        else ToolOutcome(content="ok", ok=True)
    )
    return s


async def _run(session, prompt="go"):
    async for _ in session.turn_stream(prompt):
        pass


WALL = ToolOutcome(content="[error]\nModuleNotFoundError: No module named 'pyodbc'",
                   ok=False, reason="ModuleNotFoundError: No module named 'pyodbc'")
SELF = ToolOutcome(content="[error]\nNameError: name 'wb' is not defined",
                   ok=False, reason="NameError: name 'wb' is not defined")
OK = ToolOutcome(content="fine", ok=True)


async def test_reason_reaches_the_ledger_through_the_real_loop(workspace):
    """The plumbing this ticket exists for.

    Before it, `.reason` was set by five handlers and read by nobody.
    """
    session = _session(workspace, [WALL, WALL, WALL], n_tool_calls=3)
    with patch("anton.analytics.send_event"):
        await _run(session)

    led = session._root_causes
    assert led.failures == 3
    assert led.tiers[TIER_WALL] == 3
    assert led.max_exact == 3
    assert led.top_class == "missing_dependency"
    assert led.reason_coverage == 1.0


async def test_interleaved_successes_do_not_reset_the_count(workspace):
    """The ENG-1276 lesson one level up, end to end.

    The per-tool streak resets on success, and interleaved false successes are
    exactly why it never counted to five through ENG-836. This must not.
    """
    session = _session(workspace, [WALL, OK, WALL, OK, WALL], n_tool_calls=5)
    with patch("anton.analytics.send_event"):
        await _run(session)

    assert session._root_causes.max_exact == 3, (
        "a success between failures reset the count — the exact defect this "
        "counter exists to avoid"
    )


async def test_self_inflicted_failures_reach_the_ledger_but_no_trip_rung(workspace):
    session = _session(workspace, [SELF, SELF, SELF], n_tool_calls=3)
    with patch("anton.analytics.send_event"):
        await _run(session)

    led = session._root_causes
    assert led.tiers[TIER_SELF] == 3      # measured…
    assert led.max_exact == 0             # …but never trip-eligible
    assert led.max_class == 0


async def test_the_ledger_spans_turns_on_a_reused_session(workspace):
    """Spans turns when the SESSION is reused — which is the CLI, not Cowork.

    Deliberately named for the shape it actually tests. `chat.py` builds one
    ChatSession and loops `turn_stream`, so the ledger accumulates. cowork-server
    calls `_build_chat_session()` inside `stream_response()` — once per HTTP turn
    — so on the primary product this resets every turn instead.

    Keeping the test (the object behaves as designed) but not letting its name
    imply coverage the deployment does not give. See `RootCauseLedger`.
    """
    session = _session(workspace, [WALL], n_tool_calls=1)
    with patch("anton.analytics.send_event"):
        await _run(session, "first")
    session.tool_registry.dispatch_tool = AsyncMock(return_value=WALL)
    session._llm.plan_stream = lambda **kw: _one(_tool_call(9))
    with patch("anton.analytics.send_event"):
        await _run(session, "second")

    assert session._root_causes.failures >= 2
    assert session._root_causes.max_exact >= 2


def _one(resp):
    async def gen():
        yield StreamComplete(response=resp)
    return gen()


async def test_counts_ride_the_turn_completed_event(workspace):
    session = _session(workspace, [WALL, WALL], n_tool_calls=2)
    with patch("anton.analytics.send_event") as send:
        await _run(session)

    kw = send.call_args.kwargs
    assert int(kw["root_cause_failures"]) == 2
    assert int(kw["root_cause_max_class"]) == 2
    assert kw["root_cause_top_class"] == "missing_dependency"
    # Flat scalars only — the collector relays query params, not structures.
    assert all(not isinstance(v, (dict, list)) for v in kw.values())


async def test_a_failure_with_no_reason_is_recorded_as_uncovered(workspace):
    """An unmigrated handler must show up as missing coverage, not as a wall."""
    bare = ToolOutcome(content="Tool 'web_fetch' failed: nope", ok=False)
    session = _session(workspace, [bare, bare], n_tool_calls=2)
    with patch("anton.analytics.send_event"):
        await _run(session)

    led = session._root_causes
    assert led.failures == 2
    assert led.reason_coverage == 0.0
    assert led.max_exact == 0, "text-derived keys must never be trip-eligible"


# ── The no-behaviour-change contract ───────────────────────────────────────


async def test_classification_never_changes_the_turn(workspace):
    """Measurement only: same history, same ending, with and without failures.

    If this ticket can alter a turn it has failed, however good the numbers are.
    """
    a = _session(workspace, [WALL, WALL], n_tool_calls=2)
    with patch("anton.analytics.send_event") as send_a:
        await _run(a)

    b = _session(workspace, [WALL, WALL], n_tool_calls=2)
    b._record_root_cause = lambda *args, **kw: None  # classification disabled
    with patch("anton.analytics.send_event") as send_b:
        await _run(b)

    assert send_a.call_args.kwargs["ended_by"] == send_b.call_args.kwargs["ended_by"]
    assert send_a.call_args.kwargs["rounds"] == send_b.call_args.kwargs["rounds"]
    assert [m.get("role") for m in a._history] == [m.get("role") for m in b._history]


async def test_a_raising_classifier_cannot_break_a_turn(workspace):
    """Reporting is guarded whole — a bad key must never cost a user their turn."""
    session = _session(workspace, [WALL], n_tool_calls=1)
    with patch("anton.core.session.classify_root_cause",
               side_effect=RuntimeError("boom")), \
         patch("anton.analytics.send_event") as send:
        await _run(session)

    assert send.call_args.kwargs["ended_by"] == "completed"


async def test_successes_are_not_recorded_at_all(workspace):
    session = _session(workspace, [OK, OK, OK], n_tool_calls=3)
    with patch("anton.analytics.send_event"):
        await _run(session)
    assert session._root_causes.failures == 0


async def test_the_non_streaming_turn_records_too(workspace):
    """`turn()` is public API and its books are wired, so its failures count."""
    session = _session(workspace, [WALL, WALL], n_tool_calls=2)
    with patch("anton.analytics.send_event"):
        await session.turn("go")
    assert session._root_causes.failures == 2
    assert session._root_causes.max_exact == 2
