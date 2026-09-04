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
- the payload is exactly {name, ok, duration_ms, error_type, surface,
  conversation_id, turn_index, turn_attempt_id, root_cause_tier,
  root_cause_class}, all strings — no tool arguments, no result content.
  ``surface`` joined in ENG-1945, the two ``root_cause_*`` keys in ENG-2247 and
  ``turn_attempt_id`` in ENG-2243; the exact-keys assertion is widened
  deliberately each time rather than loosened, because "no surprise keys" is the
  property that keeps arguments and result prose out;
- human wait (``answer_wait_s``, accumulated by ``elicit()``) is subtracted
  from the duration — which covers ``ask_user`` and ``select_path``, the only
  tools that elicit. It does NOT cover the interactive branch, whose tools
  prompt via ``prompt_or_cancel`` and never touch that counter; see
  ``test_interactive_branch_emits_too`` for the pinned limit;
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


def _tool_call(i: int, name: str = "scratchpad") -> LLMResponse:
    # Default routes to the general dispatch branch (a scratchpad "view", not
    # an "exec"); pass connect_new_datasource to route to the interactive one.
    return LLMResponse(
        content="working",
        tool_calls=[ToolCall(id=f"tc{i}", name=name,
                             input={"action": "view", "name": "m"})],
        usage=_usage(), stop_reason="tool_use",
    )


def _text(t: str = "done") -> LLMResponse:
    return LLMResponse(content=t, tool_calls=[], usage=_usage(), stop_reason="end_turn")


def _session(workspace, dispatch_side_effect, n_tool_calls: int = 1,
             tool_name: str = "scratchpad"):
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
                response=_tool_call(seq["i"], tool_name)
                if seq["i"] <= n_tool_calls else _text()
            )
        return gen()

    async def plan(**kw):
        seq["i"] += 1
        if llm.usage_listener:
            llm.usage_listener("planning", "m", _usage())
        return _tool_call(seq["i"], tool_name) if seq["i"] <= n_tool_calls else _text()

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


async def test_payload_is_exactly_the_ten_keys_all_strings(workspace):
    """No arguments, no result content, no surprise keys — and str values only,
    because send_event's extras are wire parameters (tests/test_ask_user.py:496).
    """
    session = _session(workspace, [ToolOutcome(content="secret result body", ok=True)])
    events = await _tool_completed_calls(session)
    assert set(events[0]) == {"name", "ok", "duration_ms", "error_type",
                              "surface", "conversation_id", "turn_index",
                              "turn_attempt_id",
                              "root_cause_tier", "root_cause_class"}
    assert all(isinstance(v, str) for v in events[0].values())
    assert "secret result body" not in json.dumps(events[0])
    int(events[0]["duration_ms"])  # numeric string, parseable


async def test_surface_rides_the_tool_row_and_is_empty_when_the_host_did_not_say(workspace):
    """ENG-1945: `surface` on the tool row, same derivation as turn_completed.

    Two assertions on purpose: the unset case must be "" (never a guess —
    `_validated_surface` already dropped anything unrecognised), and a host
    that declared one must see it on every tool row, not just the turn row.
    """
    session = _session(workspace, [ToolOutcome(content="x", ok=True)])
    assert (await _tool_completed_calls(session))[0]["surface"] == ""

    session = _session(workspace, [ToolOutcome(content="x", ok=True)])
    session._surface = "desktop"
    assert (await _tool_completed_calls(session))[0]["surface"] == "desktop"


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


async def test_interactive_branch_emits_too(workspace):
    """The interactive branch (`connect_new_datasource`, `publish_or_preview`
    publish) reaches the shared emit — nothing else asserted this.

    It also pins a known limit. This branch's tools collect human input via
    `prompt_or_cancel`, which — unlike `elicit()` — does not feed
    `answer_wait_s`, so the emitted duration INCLUDES the human's typing time.
    The 0.4s stall below stands in for that and is expected in the number;
    real credential entry takes minutes. The emit's fallback subtracts
    `answer_wait_s`, but that counter is provably 0.0 here (only `elicit()`
    writes it, and its callers all route to the general branch), so the
    subtraction cannot help these two tools. Asserted as-is deliberately:
    the value is honest wall-clock, and a silent inflation would be worse
    than a documented one.
    """
    import asyncio

    async def human_types_credentials(*a, **k):
        await asyncio.sleep(0.4)
        return ToolOutcome(content="connected", ok=True)

    session = _session(workspace, human_types_credentials,
                       tool_name="connect_new_datasource")
    events = await _tool_completed_calls(session)
    assert len(events) == 1
    assert events[0]["name"] == "connect_new_datasource"
    assert events[0]["ok"] == "true"
    assert int(events[0]["duration_ms"]) >= 400  # human time is in there
    assert session.answer_wait_s == 0.0  # prompt_or_cancel never fed it


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


# ── Why it failed (ENG-2247) ─────────────────────────────────────────


async def test_a_returned_verdict_failure_carries_its_cause(workspace):
    """The whole point: `scratchpad`'s failure shape gets a groupable cause.

    A cell that errors is a normal outcome, not a raise, so `error_type` is
    empty here BY DESIGN — that is the 83% of failures that reached analytics
    with nothing to group on before this.
    """
    session = _session(workspace, [ToolOutcome(
        content="[error]\nNameError: name 'wb' is not defined",
        ok=False, reason="NameError: name 'wb' is not defined",
    )])
    ev = (await _tool_completed_calls(session))[0]
    assert ev["ok"] == "false"
    assert ev["error_type"] == ""          # no raise -> no exception class
    assert ev["root_cause_tier"] == "self_inflicted"
    assert ev["root_cause_class"] == "NameError"


async def test_an_environment_wall_is_distinguishable_from_an_agent_bug(workspace):
    """The tier is the actionable half: whose fault is it.

    `NameError` and `ModuleNotFoundError` are both `ok=false` with an empty
    `error_type`; without the tier they are the same row.
    """
    session = _session(workspace, [ToolOutcome(
        content="[error]\nModuleNotFoundError: No module named 'pyodbc'",
        ok=False, reason="ModuleNotFoundError: No module named 'pyodbc'",
    )])
    ev = (await _tool_completed_calls(session))[0]
    assert ev["root_cause_tier"] == "external_wall"
    assert ev["root_cause_class"] == "missing_dependency"


async def test_success_carries_no_cause(workspace):
    """Empty, not a placeholder — a cause on a successful call is a lie."""
    session = _session(workspace, [ToolOutcome(content="fine", ok=True)])
    ev = (await _tool_completed_calls(session))[0]
    assert ev["root_cause_tier"] == ""
    assert ev["root_cause_class"] == ""


async def test_an_unmigrated_handler_gets_no_cause(workspace):
    """`ok=None` means the handler declared nothing (ENG-2248's population).

    The event already reports `ok="unknown"`; attaching a cause would invent a
    verdict from prose the model can influence — the ENG-1276 defect one level
    up, and the reason `_record_root_cause` keeps that population
    non-trip-eligible. Text that LOOKS like a failure must not change this.
    """
    session = _session(workspace, [ToolOutcome(
        content="[error] Task failed: something broke", ok=None)])
    ev = (await _tool_completed_calls(session))[0]
    assert ev["ok"] == "unknown"
    assert ev["root_cause_tier"] == ""
    assert ev["root_cause_class"] == ""


async def test_the_cause_is_the_shared_classifier_verbatim(workspace):
    """Pins the two views together so they cannot drift.

    The turn-level `root_cause_*` tally and this per-call field must come from
    ONE vocabulary; a second taxonomy would put two spellings of the same
    failure in two fields. Asserted against `classify` itself rather than
    against hardcoded strings, so extending the vocabulary cannot silently
    desync the event.

    KNOWN LIMIT, stated so nobody reads more into a green run than is there:
    this evaluates the same expression the implementation does, so it pins the
    values but NOT the choice of inputs — were the correct call
    `classify(reason, result_text)`, this would pass anyway. That gap is a
    proven no-op rather than a risk: measured across 12 reasons x 7 result
    texts, `(tier, cls)` is identical with and without `result_text` in all 84
    combinations, because the only branch that reads it returns the constant
    `"unclassified"` class. That equivalence is what lets the cause be derived
    at the emit site, before the result is assigned.
    """
    from anton.core.root_cause import classify

    for reason in ("NameError: x", "ModuleNotFoundError: No module named 'z'",
                   "TimeoutError: slow", "scratchpad_empty_code",
                   "RuntimeError: boom"):
        session = _session(workspace, [ToolOutcome(
            content="[error]", ok=False, reason=reason)])
        ev = (await _tool_completed_calls(session))[0]
        expected = classify(reason, "")
        assert ev["root_cause_tier"] == expected.tier, reason
        assert ev["root_cause_class"] == expected.cls, reason


async def test_no_prose_from_the_reason_reaches_the_payload(workspace):
    """`reason` is prose — traceback lines, handler messages — and carries file
    paths and user input. Only the closed-vocabulary class may be emitted.
    """
    secret = "/Users/someone/.aws/credentials could not be opened"
    session = _session(workspace, [ToolOutcome(
        content=f"[error]\nPermissionError: {secret}",
        ok=False, reason=f"PermissionError: {secret}",
    )])
    ev = (await _tool_completed_calls(session))[0]
    assert ev["root_cause_class"] == "permission_denied"
    blob = json.dumps(ev)
    assert secret not in blob
    assert ".aws" not in blob
    assert "someone" not in blob


def _closed_vocabulary() -> set:
    """THE authoritative class set — `root_cause.ALL_CLASSES`, not a rebuild.

    An earlier version reassembled this from the five tables, and that was
    wrong: `classify()` returns four classes as bare LITERALS, and `timeout`
    is in no table. It is reachable on every scratchpad cell timeout
    (`backends/local.py` -> "Cell timed out after {N}s total" -> `reason`), so
    the guard reported a legitimate enumerated value as a novel class — and
    the natural reaction to that failure is to loosen the set, the one move
    this guard exists to prevent. `permission_denied` and `connection_refused`
    passed only by coincidence, being `_WALL_TYPES` values too.

    Reading the module's own export keeps the derivation honest: a new class
    must be added there, and this follows automatically.
    """
    from anton.core.root_cause import ALL_CLASSES

    return set(ALL_CLASSES)


def test_the_emitted_class_is_always_from_the_closed_vocabulary():
    """Bounded cardinality is a property of the FIELD, not of today's inputs.

    A PostHog property with unbounded values is expensive and useless as a
    breakdown. `reason` is attacker-and-model-influenced prose, so the guard
    that matters is that nothing derived from it can widen the value space.
    Adversarial reasons included: a novel exception type, a bare sentence, an
    injected-looking string, and text carrying a status code.
    """
    from anton.core.session import _tool_failure_cause

    from anton.core.root_cause import ALL_TIERS

    closed = _closed_vocabulary()
    tiers = set(ALL_TIERS)
    reasons = [
        "NameError: x", "ModuleNotFoundError: No module named 'z'",
        "TimeoutError: slow", "PermissionError: nope", "scratchpad_empty_code",
        "RuntimeError: boom", "SomeVendorSpecificError: novel type",
        "just a sentence with no exception in it", "",
        "HTTPError: 503 upstream", "KeyError: 404",
        "Error: {'sql': 'DROP TABLE users'} failed",
        "x" * 4000,
        # The four classes `classify` returns as literals rather than through
        # a table, each reaching its own branch. `timeout` is the live one —
        # every scratchpad cell timeout — and it was uncovered until review.
        #
        # NO `OSError: ` PREFIX on the last two, deliberately: `_WALL_TYPES`
        # is consulted before the lowercase-phrase branches, and its OSError
        # entry falls through to `unclassified` unless the text says "No space
        # / Disk quota / Too many open files / Cannot allocate". A first draft
        # prefixed both and they silently classified as `unclassified`, so the
        # two branches they exist to cover stayed unexercised while the test
        # still passed (both classes being in `ALL_CLASSES` either way).
        "Cell timed out after 300s total without producing any output",
        "Cell exceeded inactivity limit",
        "permission denied on /etc/shadow",
        "connection refused by db.internal:5432",
    ]
    for reason in reasons:
        tier, cls = _tool_failure_cause(False, reason)
        assert cls in closed, f"{reason[:40]!r} minted a novel class {cls!r}"
        assert tier in tiers, f"{reason[:40]!r} minted a novel tier {tier!r}"


async def test_the_tool_row_and_the_turn_tally_agree_on_the_tier(workspace):
    """The per-call field and the per-turn counter must describe one reality.

    They come from the same classifier but by different routes — this one reads
    it directly, the tally goes through `RootCauseLedger`. A divergence would
    mean two published numbers disagreeing about the same failure.
    """
    session = _session(workspace, [ToolOutcome(
        content="[error]\nNameError: name 'wb' is not defined",
        ok=False, reason="NameError: name 'wb' is not defined",
    )])
    with patch("anton.analytics.send_event") as sent:
        async for _ in session.turn_stream("go"):
            pass
    tool = [c.kwargs for c in sent.call_args_list if c.args[1] == "tool_completed"]
    turn = [c.kwargs for c in sent.call_args_list if c.args[1] == "turn_completed"]
    assert len(tool) == 1 and len(turn) == 1

    assert tool[0]["root_cause_tier"] == "self_inflicted"
    # The turn tally counted the SAME failure into the SAME tier.
    assert str(turn[0]["root_cause_self_inflicted"]) == "1"
    assert str(turn[0]["root_cause_failures"]) == "1"
    # And it stayed out of the trip-eligible rungs — the safety contract this
    # change must not disturb (ENG-1531 / ENG-836).
    assert str(turn[0]["root_cause_wall"]) == "0"
    assert turn[0]["root_cause_top_class"] == ""


async def test_a_caller_that_omits_reason_degrades_to_unclassified(workspace):
    """The reason the derivation lives inside the emit (ENG-2247 review).

    Three shapes were on the table for "a new dispatch path forgets the cause":

    | shape | forgetting yields | turn-safe |
    | -- | -- | -- |
    | cause kwargs, defaulted `""` | **"did not fail"** — wrong | yes |
    | cause kwargs, required | `TypeError` at the CALL, outside this
      method's guard — **kills the turn over telemetry** | NO |
    | derive here from `reason` (chosen) | `unclassified` — honest | yes |

    A field whose whole purpose is measuring failures must not fail toward
    "success", and must not be able to break a turn. This pins the third.
    """
    session = _session(workspace, [ToolOutcome(
        content="[error]\nNameError: name 'wb' is not defined",
        ok=False, reason="NameError: name 'wb' is not defined",
    )])
    with patch("anton.analytics.send_event") as sent:
        # Exactly what a forgetful new call site looks like: `reason` omitted.
        session._emit_tool_completed(
            name="scratchpad", ok=False, duration_ms=1.0, error_type="",
        )
    ev = sent.call_args.kwargs
    assert ev["ok"] == "false"
    assert ev["root_cause_tier"] == "unclassified", (
        "a forgotten reason must read as 'we do not know why', never as empty "
        "— empty is a legal value meaning the call did not fail"
    )
    assert ev["root_cause_class"] == "unclassified"
    # And the keys are still exactly the ten: omitting an input must not
    # change the payload SHAPE, only the honesty of one value.
    #
    # NOTE for the next person widening this event: git did NOT flag this as a
    # conflict when ENG-2243 rebased onto ENG-2247, because only one side ever
    # touched this test. A key-set assertion in a test the OTHER branch added
    # is invisible to `git merge-tree`, so measuring conflict regions
    # understates the work. Grep the key set, do not trust the conflict count.
    assert set(ev) == {"name", "ok", "duration_ms", "error_type", "surface",
                       "conversation_id", "turn_index", "turn_attempt_id",
                       "root_cause_tier", "root_cause_class"}
