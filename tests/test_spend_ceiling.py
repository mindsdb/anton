"""Per-turn spend ceiling (ENG-1286).

Drives the real ``turn_stream``/``turn`` rather than poking the predicate,
because the ceiling's whole job is wiring: which loop it is checked in, what it
does to the verifier, and whether the message the user read reaches history.

The fake LLM calls ``usage_listener`` exactly as ``LLMClient`` does — the
listener is the narrow waist ENG-1288 installs, so a stub that skips it would
leave ``total_tokens`` at 0 and every one of these tests would pass by never
reaching the gate.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import make_mock_llm

from anton.core.llm.provider import (
    LLMResponse,
    StreamComplete,
    ToolCall,
    Usage,
)
from anton.core.session import (
    ChatSession,
    ChatSessionConfig,
    _SPEND_CEILING_GRACE_FLOOR,
    _SPEND_CEILING_RESERVE,
)

CEILING = 1_000_000
# Sized so the trip needs SEVERAL rounds. The first version used 500_000, which
# breached the gate on the very first call — that made `rounds`/`continuations`
# assertions structurally unfalsifiable, so the placement guard below passed
# even with the per-round gate disabled. At 100_000 (context 75_000, reserve
# 200_000, gate 800_000) the trip lands around round 8.
PER_CALL = 100_000
# The product floor for the user-settable ceiling (cowork-server
# `UserSettings.max_turn_tokens` ge=750_000). Below one call's cost no ceiling
# can avoid overshooting, so the floor is what keeps the setting honest — these
# tests pin that the floor itself still does real work.
PRODUCT_FLOOR = 750_000


@pytest.fixture()
def workspace():
    base = Path(__file__).resolve().parents[1] / ".pytest-workspace"
    base.mkdir(parents=True, exist_ok=True)
    return MagicMock(base=base)


def _usage(n: int = PER_CALL) -> Usage:
    # Spread across the four components on purpose: the ceiling counts RAW
    # tokens, so a cache-heavy call must count the same as a fresh one. A gate
    # that read only `input_tokens` would pass every other test in this file.
    q = n // 4
    return Usage(
        input_tokens=q, output_tokens=q,
        cache_read_tokens=q, cache_creation_tokens=n - 3 * q,
    )


def _tool_call(i: int = 1) -> LLMResponse:
    return LLMResponse(
        content="working",
        tool_calls=[ToolCall(id=f"tc_{i}", name="scratchpad",
                             input={"action": "view", "name": "main"})],
        usage=_usage(), stop_reason="tool_use",
    )


def _text(text: str = "done") -> LLMResponse:
    return LLMResponse(content=text, tool_calls=[], usage=_usage(),
                       stop_reason="end_turn")


def _session(workspace, *, ceiling: int = CEILING, responses=None,
             per_call: int = PER_CALL) -> ChatSession:
    """Session whose LLM emits `responses` in order and reports usage per call.

    Falls back to a text reply once the script runs out, so a test that fails to
    trip the gate terminates instead of looping.
    """
    mock_llm = make_mock_llm()
    script = list(responses or [])

    def _plan_stream(**kwargs):
        async def _gen():
            resp = script.pop(0) if script else _text()
            # Exactly what LLMClient does on every completion (ENG-1288).
            if mock_llm.usage_listener is not None:
                mock_llm.usage_listener("planning", "test-model", _usage(per_call))
            yield StreamComplete(response=resp)
        return _gen()

    async def _plan(**kwargs):
        resp = script.pop(0) if script else _text()
        if mock_llm.usage_listener is not None:
            mock_llm.usage_listener("planning", "test-model", _usage(per_call))
        return resp

    mock_llm.usage_listener = None
    mock_llm.plan_stream = _plan_stream
    mock_llm.plan = _plan
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    # Set post-construction rather than through a settings object: ChatSession
    # resolves a real CoreSettings in __init__ (so every other knob keeps its
    # production default), and this is the only one these tests vary.
    session._max_turn_tokens = ceiling
    from anton.core.tools.registry import ToolOutcome
    session.tool_registry.dispatch_tool = AsyncMock(
        return_value=ToolOutcome(content="stubbed tool result", ok=True)
    )
    return session


def _history_text(session) -> str:
    out = []
    for m in session._history:
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
    return "\n".join(out)


async def _run(session, prompt="do the thing"):
    async for _ in session.turn_stream(prompt):
        pass


async def test_ceiling_stops_the_turn_and_marks_the_exit(workspace):
    """The turn stops on the ceiling and books `spend_ceiling`."""
    session = _session(workspace, responses=[_tool_call(i) for i in range(1, 12)])
    with patch("anton.analytics.send_event") as send:
        await _run(session)
    assert send.call_args.kwargs["ended_by"] == "spend_ceiling"


@pytest.mark.parametrize("per_call", [PER_CALL, 300_000])
async def test_total_is_bounded_by_the_ceiling(workspace, per_call):
    """The turn's spend lands at the ceiling, give or take two output budgets
    and one grace allotment.

    The reserve is what makes this hold at all: the hand-back diagnosis is
    itself an LLM call, so gating AT the ceiling would overshoot by that call.

    The slack is not slop — it is the exact residual of the design. The reserve
    is derived from `peak_context_tokens` (input + cache_read + cache_creation)
    but compared against `total_tokens`, which also counts output, so the two
    calls that land after the last passing check contribute their output on top.
    Asserting `<= CEILING` exactly passed only because the old fixture happened
    to land on 1,000,000 with zero margin; it reddened as soon as `per_call`
    moved. State the real bound instead of a knife-edge one.

    The grace term is new: this run's all-tool-calls shape trips the mid-loop
    gate, and the first trip buys one silent extension before the turn
    actually stops. Computed from the real `_grant_spend_ceiling_grace`
    formula, not a flat constant — a flat `_SPEND_CEILING_RESERVE` slack only
    held at this fixture's small `per_call`; parametrised over a second,
    bigger `per_call` so the peak-scaled branch of the formula is pinned too.
    """
    session = _session(workspace, responses=[_tool_call(i) for i in range(1, 40)],
                       per_call=per_call)
    with patch("anton.analytics.send_event") as send:
        await _run(session)
    slack = 2 * (per_call // 4)  # `_usage` puts a quarter of each call in output
    peak = per_call - (per_call // 4)  # `_usage`'s context share of one call
    grace = min(max(_SPEND_CEILING_GRACE_FLOOR, peak), CEILING // 4)
    # Analytics properties go over the wire as strings.
    assert (
        int(send.call_args.kwargs["tokens_total"])
        <= CEILING + slack + grace
    )


async def test_trips_inside_one_tool_loop_with_no_continuation(workspace):
    """The runaway shape is consecutive tool calls in ONE loop.

    A continuation-gate-only check never sees it — the measured tail is 13-26
    consecutive scratchpad calls that reach the round cap having triggered no
    continuation at all. This is the regression guard for that placement.
    """
    session = _session(workspace, responses=[_tool_call(i) for i in range(1, 40)])
    with patch("anton.analytics.send_event") as send:
        await _run(session)
    kwargs = send.call_args.kwargs
    assert kwargs["ended_by"] == "spend_ceiling"
    assert int(kwargs["continuations"]) == 0, "must trip without needing a continuation"
    assert int(kwargs["rounds"]) < 25, "must trip before the round cap, not because of it"
    # Load-bearing: without this the whole test passes on a turn that tripped on
    # its FIRST call, where "no continuation" and "under the round cap" are true
    # by construction and say nothing about where the check lives.
    assert int(kwargs["rounds"]) > 1, "must have run several rounds before tripping"


async def test_handback_asks_the_user_and_is_persisted_as_streamed(workspace):
    """ENG-1155 property: history holds what the user actually read.

    Also asserts the message asks rather than merely announcing — a ceiling the
    user cannot get past is worse than no ceiling.
    """
    session = _session(workspace, responses=[_tool_call(i) for i in range(1, 12)])
    with patch("anton.analytics.send_event"):
        await _run(session)
    text = _history_text(session)
    assert "Do NOT retry automatically" in text
    assert "ask if they'd like you to continue" in text
    # The streamed diagnosis, not the pre-stop reply, is the tail of history.
    assert session._history[-1]["role"] == "assistant"


async def test_verification_is_skipped_after_a_ceiling_trip(workspace):
    """No verdict call once the ceiling has stopped the turn.

    Verification would spend more of the budget we just declared exhausted, and
    an INCOMPLETE verdict would force a continuation straight past the ceiling.

    Asserts on CALL COUNT, not on a raising side-effect. The first version of
    this test raised `AssertionError` from the verdict call — and passed with the
    skip removed, because the verdict site catches bare `Exception` and converts
    anything raised there into a "hard verdict failure" that continues down the
    diagnosis path. A test whose failure mode is swallowed by the code under test
    proves nothing.
    """
    session = _session(workspace, responses=[_tool_call(i) for i in range(1, 12)])
    verdict = AsyncMock()
    session._llm.generate_object_code = verdict
    with patch("anton.analytics.send_event"):
        await _run(session)
    assert verdict.call_count == 0, (
        f"the completion verifier ran {verdict.call_count}x after the ceiling "
        "stopped the turn"
    )


async def test_ceiling_stop_announces_itself_to_the_host(workspace):
    """The stop emits a progress marker, like the other hand-back paths.

    Without it a ceiling stop is indistinguishable, at the host boundary, from
    the agent simply choosing to ask a question — and the ceiling is expected to
    fire on roughly 1 in 5 external turns, so "the agent gave up" is the reading
    a user would otherwise land on. The marker is ephemeral in today's renderer;
    a persistent paused-state affordance is tracked separately.
    """
    from anton.core.llm.provider import StreamTaskProgress

    session = _session(workspace, responses=[_tool_call(i) for i in range(1, 12)])
    seen = []
    with patch("anton.analytics.send_event"):
        async for ev in session.turn_stream("do the thing"):
            if isinstance(ev, StreamTaskProgress):
                seen.append(ev.message or "")
    assert any("token budget" in m for m in seen), (
        f"no ceiling marker among progress events: {seen}"
    )


async def test_turn_under_the_ceiling_is_untouched(workspace):
    """No behaviour change for a turn that never reaches the gate."""
    session = _session(workspace, responses=[_tool_call(1), _text("all done")],
                       per_call=1_000)
    with patch("anton.analytics.send_event") as send:
        await _run(session)
    kwargs = send.call_args.kwargs
    assert kwargs["ended_by"] == "completed"
    text = _history_text(session)
    assert "Do NOT retry automatically" not in text


async def test_ceiling_of_zero_disables_the_gate(workspace):
    """0 means off — a host on older settings keeps pre-ENG-1286 behaviour.

    Uses per-call usage far above any plausible ceiling, so a gate that ignored
    the 0 would certainly trip.
    """
    session = _session(workspace, ceiling=0,
                       responses=[_tool_call(i) for i in range(1, 6)] + [_text()],
                       per_call=5_000_000)
    with patch("anton.analytics.send_event") as send:
        await _run(session)
    assert send.call_args.kwargs["ended_by"] != "spend_ceiling"


async def test_reserve_is_held_back_for_the_handback(workspace):
    """The gate fires below the ceiling by the reserve, not at it."""
    from anton.core.turn_cost import TurnCost
    session = _session(workspace)
    session._turn_cost = TurnCost()  # no calls yet -> reserve is the floor
    session._turn_cost.input_tokens = CEILING - _SPEND_CEILING_RESERVE - 1
    assert not session._spend_ceiling_reached()
    session._turn_cost.input_tokens = CEILING - _SPEND_CEILING_RESERVE
    assert session._spend_ceiling_reached()


async def test_reserve_scales_with_the_turn_s_own_call_size(workspace):
    """A big-context turn reserves more, because its remaining calls cost more.

    Without this the bound is a promise the ceiling cannot keep: two calls land
    after the last passing check, and at 190k+ of context per call they exceed
    a flat 200k reserve on their own.
    """
    from anton.core.turn_cost import TurnCost
    session = _session(workspace)
    session._turn_cost = TurnCost()
    session._turn_cost.peak_context_tokens = 190_000
    assert session._spend_ceiling_gate() == CEILING - 380_000
    # A small-context turn keeps the floor rather than shrinking below it.
    session._turn_cost.peak_context_tokens = 1_000
    assert session._spend_ceiling_gate() == CEILING - _SPEND_CEILING_RESERVE


async def test_cache_reads_count_toward_the_ceiling(workspace):
    """Raw tokens, not cost-weighted — cache reads draw the user's allowance
    at full 1:1 weight, so a cache-heavy turn must trip like any other.
    """
    from anton.core.turn_cost import TurnCost
    session = _session(workspace)
    session._turn_cost = TurnCost()
    session._turn_cost.cache_read_tokens = CEILING - _SPEND_CEILING_RESERVE
    assert session._spend_ceiling_reached(), (
        "a turn made almost entirely of cache reads still exhausts the "
        "user's included-token allowance"
    )


# ── The guarantee: a ceiling stop never ends a turn that did nothing ────────


@pytest.mark.parametrize("ceiling", [PRODUCT_FLOOR, 250_000, 100_000, 1])
async def test_a_ceiling_stop_never_dispatches_zero_tools(workspace, ceiling):
    """At ANY ceiling, the turn runs at least one tool before stopping.

    The regression this pins was user-reachable: the reserve was
    `max(200_000, 2 * peak)` with no relation to the ceiling, so any ceiling at
    or below the reserve drove the gate to its `max(..., 1)` floor. The check
    runs at the TOP of a round, before dispatch, so the loop broke having run
    nothing — and still paid for two calls. Measured at a 100k ceiling with
    190k contexts: 0 tools, 400k spent, 300% over what the user asked for.

    The whole lower half of the range the Settings UI advertised sat in that
    band. Parametrised down to `1` deliberately: anton's own `max_turn_tokens`
    has no lower bound for CLI and host callers, so the guarantee cannot rest
    on the product floor alone.
    """
    session = _session(workspace, ceiling=ceiling, per_call=190_000,
                       responses=[_tool_call(i) for i in range(1, 40)])
    with patch("anton.analytics.send_event") as send:
        await _run(session)
    assert session.tool_registry.dispatch_tool.call_count > 0, (
        f"ceiling={ceiling:,} ended the turn having dispatched no tools"
    )
    assert send.call_args.kwargs["ended_by"] == "spend_ceiling"


async def test_gate_stays_proportional_to_the_ceiling(workspace):
    """The reserve can never eat the whole ceiling.

    Unit-level companion to the parametrised test above: it pins the arithmetic
    rather than the behaviour, so a future change that keeps tools running by
    some other means still can't reintroduce a gate of 1.
    """
    from anton.core.turn_cost import TurnCost

    session = _session(workspace, ceiling=300_000)
    session._turn_cost = TurnCost()
    session._turn_cost.peak_context_tokens = 5_000_000  # absurd, on purpose
    assert session._spend_ceiling_gate() >= 300_000 // 2
    # …and the cap does not disturb the shipped default.
    session._max_turn_tokens = 1_250_000
    session._turn_cost.peak_context_tokens = 190_000
    assert session._spend_ceiling_gate() == 1_250_000 - 380_000


async def test_the_setting_actually_reaches_the_session(workspace):
    """The wiring from settings to session, which nothing else covers.

    `getattr(s, "max_turn_tokens", 0)` is deliberately fail-open, so a typo in
    the key name silently disables the ceiling with no exception, no log and no
    `ended_by` — and every other test in this file pokes `_max_turn_tokens`
    directly after construction, so all of them would still pass.
    """
    from anton.core.settings import CoreSettings

    session = ChatSession(ChatSessionConfig(llm_client=make_mock_llm(),
                                            workspace=workspace))
    assert session._max_turn_tokens == CoreSettings().max_turn_tokens == 1_250_000


async def test_ceiling_trips_at_the_continuation_gate(workspace):
    """A continuation whose planning call returns TEXT never enters the tool
    loop, so only the continuation gate can stop it.

    Both non-per-round call sites were previously uncovered: replacing either
    with `if False:` left the entire 2,012-test suite green.
    """
    from anton.core.session import _VerifierVerdict

    # One tool round, then text replies — so the tool loop exits and the
    # verifier's INCOMPLETE drives a continuation that uses no tools at all.
    session = _session(workspace, per_call=300_000,
                       responses=[_tool_call(1), _text("partial"),
                                  _text("still going"), _text("more")])
    verdicts = [_VerifierVerdict(status="INCOMPLETE", reason="not done")]
    session._llm.generate_object_code = AsyncMock(
        side_effect=lambda *a, **k: verdicts.pop(0) if verdicts
        else _VerifierVerdict(status="INCOMPLETE", reason="not done"))
    with patch("anton.analytics.send_event") as send:
        await _run(session)
    assert send.call_args.kwargs["ended_by"] == "spend_ceiling"


# ── Judgment-gated grace at the ceiling (ENG-1893) ──────────────────────────
#
# The ceiling above is a deliberate "always ask" design (see
# test_handback_asks_the_user_and_is_persisted_as_streamed). But the verifier
# that runs right before this gate already carries a signal about how much
# work is left — `close_to_done` — and today that signal is thrown away: the
# ask fires the same whether the task is one tool call from finished or
# nowhere close. These tests pin a small, one-time exception: a verdict that
# says the remaining work is small lets the turn keep going seamlessly
# instead of stopping to ask, exactly once per turn.


async def test_close_to_done_verdict_skips_the_ask_and_keeps_going(workspace):
    """An INCOMPLETE verdict with close_to_done=True continues past the gate.

    Same shape as test_ceiling_trips_at_the_continuation_gate — one tool
    round, then a text reply that trips the gate on the first verifier call —
    but this verdict reports the remaining work as small.

    `per_call=240_000`, not 300_000: the text reply triggers a truncation
    retry, so 3 calls land before this gate check. That total must clear the
    pre-grace gate but stay under the smaller post-grace one now that the
    grace is sized off 1x peak, not 2x — 300_000 no longer leaves enough
    room after the resize.
    """
    from anton.core.session import _VerifierVerdict

    session = _session(workspace, per_call=240_000,
                       responses=[_tool_call(1), _text("partial")])
    verdict = _VerifierVerdict(status="INCOMPLETE", reason="almost there",
                               close_to_done=True)
    generate = AsyncMock(return_value=verdict)
    session._llm.generate_object_code = generate
    with patch("anton.analytics.send_event") as send:
        await _run(session)
    assert send.call_args.kwargs["ended_by"] != "spend_ceiling"
    assert generate.call_count == 1, "grace must not re-verify — one call, one decision"
    text = _history_text(session)
    assert "ask if they'd like you to continue" not in text


async def test_close_to_done_grace_is_granted_only_once_per_turn(workspace):
    """A second ceiling breach in the same turn asks, even if close_to_done again.

    Grace is a one-time exception, not a standing invitation to talk past the
    ceiling — otherwise a model that always claims "almost done" would never
    actually stop.
    """
    from anton.core.session import _VerifierVerdict

    session = _session(workspace, per_call=300_000,
                       responses=[_tool_call(1), _text("partial"),
                                  _tool_call(2), _text("still going")])
    verdict = _VerifierVerdict(status="INCOMPLETE", reason="almost there",
                               close_to_done=True)
    session._llm.generate_object_code = AsyncMock(return_value=verdict)
    with patch("anton.analytics.send_event") as send:
        await _run(session)
    assert send.call_args.kwargs["ended_by"] == "spend_ceiling"


# ── Mid-loop grace (ENG-1893) ────────────────────────────────────────────────
#
# Inside the tool loop there is no verdict to read close_to_done off — the
# model hasn't stopped to produce a judgeable reply, it just asked for more
# tools. So this grace is unconditional rather than judgment-gated: the first
# trip buys one small, one-time extension; whatever happens in that window is
# itself the evidence. If the loop is still going after it, that is grounds
# enough to ask — no self-report needed either way.


def test_mid_loop_grace_raises_the_gate_once(workspace):
    """`_grant_spend_ceiling_grace` lifts the gate; spending through the grace
    trips it again — the bump is not renewed."""
    from anton.core.turn_cost import TurnCost

    session = _session(workspace)
    session._turn_cost = TurnCost()
    session._turn_cost.rounds = 5  # > 1, so the never-zero-tools guard is moot here
    # peak_context_tokens is 0 here, so both floors apply: the reserve lands on
    # _SPEND_CEILING_RESERVE and the (deliberately smaller) grace lands on
    # _SPEND_CEILING_GRACE_FLOOR — the gate before grace is exactly
    # CEILING - reserve, and after grace it is CEILING - reserve + grace_floor,
    # still below CEILING since grace_floor < reserve.
    session._turn_cost.input_tokens = CEILING - _SPEND_CEILING_RESERVE
    assert session._spend_ceiling_stops_the_tool_loop()
    session._grant_spend_ceiling_grace()
    assert not session._spend_ceiling_stops_the_tool_loop()
    session._turn_cost.input_tokens = CEILING
    assert session._spend_ceiling_stops_the_tool_loop(), (
        "the one-time bump must not renew itself on a second trip"
    )


async def test_mid_loop_ceiling_grants_grace_once_then_stops(workspace):
    """A long, all-tool-calls run gets one silent extension before it asks.

    Mirrors test_trips_inside_one_tool_loop_with_no_continuation's shape
    (consecutive tool calls, no continuation involved) with more responses
    scripted, so the run survives the first, now-silent trip and reaches a
    second one that still asks.
    """
    session = _session(workspace, responses=[_tool_call(i) for i in range(1, 60)])
    with patch("anton.analytics.send_event") as send:
        await _run(session)
    kwargs = send.call_args.kwargs
    assert kwargs["ended_by"] == "spend_ceiling"
    assert session._spend_ceiling_grace_used
    assert int(kwargs["continuations"]) == 0, "still trips inside one tool loop"


@pytest.mark.parametrize("ceiling", [CEILING, 100_000, 1])
async def test_ceiling_applies_to_the_non_streaming_turn(workspace, ceiling):
    """`turn()` runs no verifier, so its per-round check is the whole gate.

    Public API with no in-tree caller, but its cost books are wired, so a
    silently unbounded path here would under-report every host that uses it.

    Parametrised down to a ceiling of 1 because this path is where the
    never-zero-tools guarantee is easiest to lose: it gated on the bare
    predicate rather than `_spend_ceiling_stops_the_tool_loop`, so a small
    ceiling broke the loop on round 1 having dispatched nothing (#344 review).
    CLI and host callers have no lower bound on `max_turn_tokens`, so the
    product floor does not protect them.
    """
    session = _session(workspace, ceiling=ceiling, per_call=190_000,
                       responses=[_tool_call(i) for i in range(1, 40)])
    with patch("anton.analytics.send_event") as send:
        await session.turn("do the thing")
    assert send.call_args.kwargs["ended_by"] == "spend_ceiling"
    assert session.tool_registry.dispatch_tool.call_count > 0, (
        f"ceiling={ceiling:,} ended turn() having dispatched no tools"
    )


async def test_empty_diagnosis_does_not_double_append_the_reply(workspace):
    """The ENG-1155 guard on this path is load-bearing and was untested.

    `_reply_persisted = True` is the SOLE writer on the ceiling path — the
    verification block's own assignment is unreachable because the verifier
    skip breaks first. Deleting it survived the whole repo.
    """
    session = _session(workspace, responses=[_tool_call(i) for i in range(1, 40)])
    with patch("anton.analytics.send_event"), \
         patch.object(ChatSession, "_stream_handback_diagnosis",
                      lambda self, **kw: _empty_stream()):
        await _run(session)
    assistants = [i for i, m in enumerate(session._history)
                  if m.get("role") == "assistant" and isinstance(m.get("content"), str)]
    tails = [session._history[i]["content"] for i in assistants[-2:]]
    assert len(tails) < 2 or tails[0] != tails[1], (
        f"the pre-stop reply was appended twice: {tails!r}"
    )


async def _empty_stream():
    """A hand-back that streams nothing — the guarded empty-diagnosis case."""
    return
    yield  # pragma: no cover - makes this an async generator
