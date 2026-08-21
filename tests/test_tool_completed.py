"""The per-tool-call `tool_completed` analytics event (ENG-1486).

Anton computes a definitive per-tool success verdict for the UI's ``tool_done``
marker and, before this, threw it away — nothing could answer "which tools
fail, how often, or how slowly". These drive the real ``turn_stream`` (the same
harness shape as ``test_root_cause_wiring.py``) and assert on what
``send_event`` was handed, because the contract under test is the seam:

- both failure shapes produce ``ok="false"`` — a raised exception AND a handler
  returning ``ToolOutcome.ok=False`` (the distinction the ``tool_done`` yield
  exists to make; PR #304's review caught a draft where a raise rendered as
  unconditional success);
- ``error_type`` is the exception CLASS name and never the message —
  ``str(exc)`` routinely embeds file paths and user input;
- the payload is exactly {name, ok, duration_ms, error_type}, all strings —
  no tool arguments, no result content;
- human wait (``answer_wait_s``, accumulated by ``elicit()``) is subtracted
  from the duration;
- the event name is registered in ``_POSTHOG_EVENTS``, because a name the
  collector has never heard of otherwise reaches nothing (ENG-1355/ENG-1495).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import make_mock_llm

from anton import analytics
from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall, Usage
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


def _session(workspace, dispatch_side_effect, n_tool_calls: int = 1):
    """Session whose tool dispatch runs `dispatch_side_effect`."""
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
    s = ChatSession(ChatSessionConfig(llm_client=llm, workspace=workspace,
                                      session_id="conv-tc"))
    s._max_turn_tokens = 0
    s.tool_registry.dispatch_tool = AsyncMock(side_effect=dispatch_side_effect)
    return s


async def _tool_completed_calls(session, prompt="go"):
    """Run a turn; return the kwargs of every tool_completed send_event call."""
    with patch("anton.analytics.send_event") as sent:
        async for _ in session.turn_stream(prompt):
            pass
    return [c.kwargs for c in sent.call_args_list if c.args[1] == "tool_completed"]


# ── The two failure shapes, and success ──────────────────────────────


async def test_handler_verdict_false_emits_ok_false_and_no_error_type(workspace):
    """`ToolOutcome.ok=False` with no raise → ok="false", error_type=""."""
    session = _session(workspace, [ToolOutcome(
        content="[error]\nNameError: name 'wb' is not defined",
        ok=False, reason="NameError: name 'wb' is not defined",
    )])
    events = await _tool_completed_calls(session)
    assert len(events) == 1
    assert events[0]["ok"] == "false"
    # No exception was raised, so there is no exception class to name — the
    # handler's prose `reason` must NOT be smuggled in as error_type.
    assert events[0]["error_type"] == ""
    assert events[0]["name"] == "scratchpad"


async def test_raising_tool_emits_ok_false_with_exception_class_only(workspace):
    """A raise → ok="false", error_type = the CLASS name, never the message."""
    secret = "/Users/someone/.aws/credentials leaked into the message"
    session = _session(workspace, RuntimeError(secret))
    events = await _tool_completed_calls(session)
    assert len(events) == 1
    assert events[0]["ok"] == "false"
    assert events[0]["error_type"] == "RuntimeError"
    # The security line this ticket draws: str(exc) — paths, user input —
    # must appear in NO emitted property.
    for value in events[0].values():
        assert secret not in value
        assert "credentials" not in value


async def test_success_emits_ok_true(workspace):
    session = _session(workspace, [ToolOutcome(content="fine", ok=True)])
    events = await _tool_completed_calls(session)
    assert len(events) == 1
    assert events[0]["ok"] == "true"
    assert events[0]["error_type"] == ""


async def test_unmigrated_handler_verdict_is_unknown_not_a_guess(workspace):
    """ok=None (unmigrated handler) is reported honestly as "unknown".

    The verdict must come from the exception/ToolOutcome seam, never from
    re-inferring intent out of the result text — even when the text contains
    the word "failed".
    """
    session = _session(workspace, [ToolOutcome(content="task failed maybe", ok=None)])
    events = await _tool_completed_calls(session)
    assert len(events) == 1
    assert events[0]["ok"] == "unknown"


# ── Payload contract ─────────────────────────────────────────────────


async def test_payload_is_exactly_the_six_keys_all_strings(workspace):
    """No arguments, no result content, no surprise keys — and str values only,
    because send_event's extras are wire parameters (tests/test_ask_user.py:496).
    """
    session = _session(workspace, [ToolOutcome(content="secret result body", ok=True)])
    events = await _tool_completed_calls(session)
    assert set(events[0]) == {"name", "ok", "duration_ms", "error_type",
                              "conversation_id", "turn_index"}
    assert all(isinstance(v, str) for v in events[0].values())
    assert "secret result body" not in json.dumps(events[0])
    int(events[0]["duration_ms"])  # numeric string, parseable


async def test_join_keys_match_the_same_turns_turn_completed_row(workspace):
    """The reason the two keys exist: a tool row must join to its parent turn.

    conversation_id and turn_index on tool_completed must equal the values the
    SAME run's turn_completed carries — same names, same derivation — so the
    PostHog join (and the conversation_id → Langfuse sessionId pivot) needs no
    translation table.
    """
    session = _session(workspace, [ToolOutcome(content="fine", ok=True)])
    with patch("anton.analytics.send_event") as sent:
        async for _ in session.turn_stream("go"):
            pass
    tool = [c.kwargs for c in sent.call_args_list if c.args[1] == "tool_completed"]
    turn = [c.kwargs for c in sent.call_args_list if c.args[1] == "turn_completed"]
    assert len(tool) == 1 and len(turn) == 1
    assert tool[0]["conversation_id"] == turn[0]["conversation_id"] == "conv-tc"
    assert tool[0]["turn_index"] == turn[0]["turn_index"]
    assert tool[0]["turn_index"] != ""  # a real index, not a blank join key


async def test_one_event_per_tool_call(workspace):
    outcomes = [ToolOutcome(content="a", ok=True),
                ToolOutcome(content="b", ok=False, reason="x"),
                ToolOutcome(content="c", ok=True)]
    pending = list(outcomes)
    session = _session(workspace, lambda *a, **k: pending.pop(0), n_tool_calls=3)
    events = await _tool_completed_calls(session)
    assert [e["ok"] for e in events] == ["true", "false", "true"]


# ── Duration semantics ───────────────────────────────────────────────


async def test_duration_excludes_human_wait(workspace):
    """An ask_user answered after four minutes is not a four-minute tool.

    The dispatch mock does what elicit() does, at real wall-clock scale:
    it spends 0.4s waiting and credits that same 0.4s to `answer_wait_s`.
    Only an emitted duration that actually subtracts the wait lands near
    zero — a mutation that drops the subtraction reports ~400ms and fails.
    """
    import asyncio

    session_holder = {}

    async def slow_human(*a, **k):
        await asyncio.sleep(0.4)
        session_holder["s"].answer_wait_s += 0.4
        return ToolOutcome(content="answered", ok=True)

    session = _session(workspace, slow_human)
    session_holder["s"] = session
    events = await _tool_completed_calls(session)
    assert len(events) == 1
    assert int(events[0]["duration_ms"]) < 200


async def test_wait_from_one_call_never_leaks_into_the_next(workspace):
    """Consecutive calls each get a clean wait ledger and a clamped duration.

    Guards the per-call reset plus the negative clamp — NOT a historical bug:
    before the tail emit existed, the general branch was both the only
    resetter and the only subtractor, so the old maths was self-consistent.
    The reset's earlier position is a prerequisite of the cross-branch emit.
    """
    calls = {"n": 0}

    async def first_waits(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            session_holder["s"].answer_wait_s += 240.0
        return ToolOutcome(content="ok", ok=True)

    session_holder = {}
    session = _session(workspace, first_waits, n_tool_calls=2)
    session_holder["s"] = session
    events = await _tool_completed_calls(session)
    assert len(events) == 2
    # Both near-zero: the first because its own 240s wait is subtracted, the
    # second because the first call's wait was NOT carried into its maths as
    # a negative (clamped) or its own subtraction baseline.
    assert all(int(e["duration_ms"]) < 1_000 for e in events)


async def test_slow_consumer_after_tool_done_does_not_inflate_the_duration(workspace):
    """The booked duration is the tool's runtime, not the consumer's pull rate.

    #390 review: the emit sits a few yields past the point the branch stops
    the clock, so re-reading `monotonic()` there would bill whatever the
    consumer spends pulling `tool_done` (and a scratchpad `dump`'s result) to
    the tool. Here the consumer stalls 0.4s right after `tool_done`; the
    emitted duration must stay near zero AND equal the `eta_seconds` the UI
    displayed, so the two sources can never disagree.
    """
    import asyncio

    session = _session(workspace, [ToolOutcome(content="fine", ok=True)])
    displayed: list[float] = []
    with patch("anton.analytics.send_event") as sent:
        async for ev in session.turn_stream("go"):
            if getattr(ev, "phase", None) == "tool_done":
                displayed.append(ev.eta_seconds)
                await asyncio.sleep(0.4)
    events = [c.kwargs for c in sent.call_args_list if c.args[1] == "tool_completed"]
    assert len(events) == 1 and len(displayed) == 1
    assert int(events[0]["duration_ms"]) < 200
    assert int(events[0]["duration_ms"]) == int(displayed[0] * 1000)


async def test_nonstreaming_turn_path_also_emits(workspace):
    """`turn()` has its own dispatch loop (session.py ~3315), separate from the
    streaming tail — self-review finding: without its own emit, any host on
    the non-streaming API silently undercounts. No production caller uses it
    today; this test is the seam guard for the first one that does.
    """
    session = _session(workspace, RuntimeError("boom"))
    with patch("anton.analytics.send_event") as sent:
        await session.turn("go")
    events = [c.kwargs for c in sent.call_args_list if c.args[1] == "tool_completed"]
    assert len(events) == 1
    assert events[0]["ok"] == "false"
    assert events[0]["error_type"] == "RuntimeError"
    assert events[0]["name"] == "scratchpad"


async def test_model_generated_tool_name_is_bounded(workspace):
    """`tc.name` is model output: a degenerate name must emit (it is real
    signal) but bounded, never verbatim-unbounded into a property value."""
    session = _session(workspace, [ToolOutcome(content="ok", ok=True)])
    with patch("anton.analytics.send_event") as sent:
        session._emit_tool_completed(
            name="x" * 5000, ok=False, duration_ms=1.0, error_type="",
        )
    kwargs = sent.call_args.kwargs
    assert kwargs["name"] == "x" * 200
    assert len(kwargs["name"]) == 200


# ── Analytics resilience + routing ───────────────────────────────────


async def test_turn_survives_send_event_raising(workspace):
    """Analytics must never break the tool call that just ran."""
    session = _session(workspace, [ToolOutcome(content="fine", ok=True)])
    with patch("anton.analytics.send_event", side_effect=RuntimeError("boom")):
        async for _ in session.turn_stream("go"):
            pass  # completing without raising is the assertion


def test_tool_completed_goes_to_posthog_not_the_collector(monkeypatch):
    """`tool_completed` is a new event NAME: the collector drops names it has
    never heard of (ENG-1355), so it must take the ENG-1495 direct route."""
    monkeypatch.setattr(analytics, "_cached_is_ci", None)
    for var in ("ANTON_IS_CI", "GITHUB_ACTIONS", "GITLAB_CI", "BUILDKITE",
                "CIRCLECI", "TF_BUILD", "JENKINS_URL"):
        monkeypatch.delenv(var, raising=False)

    captured: list[tuple[str, dict]] = []

    class _SyncThread:
        def __init__(self, target=None, args=(), daemon=None):
            self._target = target
            self._args = args

        def start(self):
            if self._target:
                self._target(*self._args)

    monkeypatch.setattr(analytics.threading, "Thread", _SyncThread)
    monkeypatch.setattr(
        analytics, "_fire_posthog",
        lambda url, body: captured.append((url, json.loads(body))),
    )
    monkeypatch.setattr(
        analytics, "_fire",
        lambda url: pytest.fail(f"took the collector path instead: {url}"),
    )

    class _PosthogSettings:
        analytics_enabled = True
        analytics_url = "https://example.test/collect"
        posthog_host = "https://ph.example.test"
        posthog_key = "phc_test"

    analytics.send_event(
        _PosthogSettings(), "tool_completed",
        name="scratchpad", ok="false", duration_ms="1234", error_type="TimeoutError",
    )

    assert len(captured) == 1
    url, body = captured[0]
    assert url == "https://ph.example.test/capture/"
    assert body["event"] == "tool_completed"
    assert body["properties"]["name"] == "scratchpad"
    assert body["properties"]["ok"] == "false"
