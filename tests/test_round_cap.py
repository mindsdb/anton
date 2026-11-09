"""The round cap (`max_tool_rounds`) in isolation, and the auto-continue
count (`max_continuations`) it shares its retry behaviour with.

Drives the spend-ceiling suite's scripted LLM with `per_call` usage kept tiny
relative to the ceiling, so the ceiling never interferes, and `max_tool_rounds`
set directly on the session (not through settings) so these stay cheap to
script.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from anton.core.session import ChatSession, _VerifierVerdict

from tests.test_limit_continue import _run as _run_turn
from tests.test_spend_ceiling import (
    _session as _ceiling_session,
    _small,
    _text,
    _tool_call,
    workspace,  # noqa: F401 - fixture
)


def _session(workspace, *, max_tool_rounds: int, responses=None) -> ChatSession:
    session = _ceiling_session(workspace, responses=responses, per_call=1_000)
    session._max_tool_rounds = max_tool_rounds
    return session


async def _run(session, prompt="do the thing") -> dict:
    """Run one turn and return the `turn_completed` event's fields."""
    _, kwargs = await _run_turn(session, prompt)
    return kwargs


async def test_round_under_the_cap_is_untouched(workspace):
    """No behaviour change for a turn that never reaches the cap."""
    session = _session(workspace, max_tool_rounds=5,
                       responses=[_tool_call(1), _small(_text("done"))])
    verdict = _VerifierVerdict(status="COMPLETE", reason="done", close_to_done=False)
    session._llm.generate_object_code = AsyncMock(return_value=verdict)
    kwargs = await _run(session)
    assert kwargs["ended_by"] == "completed"


async def test_a_zero_cap_denies_the_first_round(workspace):
    """`max_tool_rounds=0` means "no tool rounds"."""
    session = _session(workspace, max_tool_rounds=0, responses=[_tool_call(1)])
    kwargs = await _run(session)
    assert kwargs["ended_by"] == "round_cap"
    assert kwargs["rounds"] == "0"
    assert kwargs["grace_granted"] == ""


async def test_the_step_limit_survives_a_mid_loop_retry(workspace):
    """A provider error the turn retries re-enters `_stream_and_handle_tools`.
    The step count must carry over: with the count restarting per attempt, a
    cap of 5 let 3 steps before the error and 5 after run with no message."""
    cap = 5
    session = _session(
        workspace, max_tool_rounds=cap,
        responses=(
            [_tool_call(i) for i in range(1, 4)]
            + [RuntimeError("provider blip")]
            + [_tool_call(i) for i in range(4, 14)]
        ),
    )
    kwargs = await _run(session)
    assert kwargs["ended_by"] == "round_cap"
    assert kwargs["rounds"] == str(cap)


async def test_the_auto_continue_count_survives_a_mid_loop_retry(workspace):
    """The same retry re-entry restarted the auto-continue count. With one
    continuation allowed and the verifier never satisfied, a provider error
    during the continuation let the verifier send the task back once more
    instead of handing it to the user."""
    attempt = [_tool_call(1), _tool_call(2), _small(_text("partial"))]
    session = _session(
        workspace, max_tool_rounds=100,
        responses=attempt + [RuntimeError("provider blip")] + attempt,
    )
    session._max_continuations = 1
    verdict = _VerifierVerdict(status="INCOMPLETE", reason="not done", close_to_done=False)
    session._llm.generate_object_code = AsyncMock(return_value=verdict)
    kwargs = await _run(session)
    assert kwargs["ended_by"] == "handback_budget"
    assert kwargs["continuations"] == "1"
    # The continuation already spent the budget, so the retried attempt is
    # handed back without a second verdict.
    assert session._llm.generate_object_code.await_count == 1


async def test_the_step_limit_does_not_multiply_across_continuations(workspace):
    """Three continuations of three steps each, against a cap of five: the
    limit lands on the second continuation's second step, not after fifteen."""
    cap = 5
    attempt = [_tool_call(i) for i in range(1, 4)] + [_small(_text("partial"))]
    session = _session(workspace, max_tool_rounds=cap, responses=attempt * 3)
    verdict = _VerifierVerdict(status="INCOMPLETE", reason="not done", close_to_done=False)
    session._llm.generate_object_code = AsyncMock(return_value=verdict)
    kwargs = await _run(session)
    assert kwargs["ended_by"] == "round_cap"
    assert kwargs["continuations"] == "1"
    assert kwargs["rounds"] == str(cap)


async def test_the_ceiling_grace_is_reported(workspace):
    """The spend ceiling keeps its one-time grace, and `grace_granted` names it."""
    session = _ceiling_session(
        workspace, ceiling=300_000,
        responses=[_tool_call(i) for i in range(1, 16)], per_call=50_000,
    )
    session._max_tool_rounds = 100
    kwargs = await _run(session)
    assert kwargs["grace_granted"] == "ceiling"
    assert session._spend_ceiling_grace_used
