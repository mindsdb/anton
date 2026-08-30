"""SpendGuard: the two-level budget check inside generate_artifact (I-20)."""
from __future__ import annotations

from types import SimpleNamespace

from anton.core.tools.generate_artifact.spend import WIND_DOWN_ROUNDS, SpendGuard


def _session(reached: bool):
    return SimpleNamespace(spend_ceiling_reached=lambda: reached)


def test_below_ceiling_is_not_winding_down():
    guard = SpendGuard(session=_session(False))
    assert guard.should_wind_down() is False
    assert guard.winding_down is False


def test_reaching_the_ceiling_starts_wind_down():
    guard = SpendGuard(session=_session(True))
    assert guard.should_wind_down() is True
    assert guard.winding_down is True


def test_wind_down_is_sticky_even_if_the_ceiling_stops_reporting():
    flag = {"reached": True}
    guard = SpendGuard(session=SimpleNamespace(spend_ceiling_reached=lambda: flag["reached"]))
    assert guard.should_wind_down() is True
    flag["reached"] = False
    assert guard.should_wind_down() is True


def test_the_guard_holds_no_round_counter():
    """The closing-round budget is PER WRITE LOOP, not per run.

    The fullstack path runs two write loops under `asyncio.gather`, both
    sharing this guard. A counter here would give the pair WIND_DOWN_ROUNDS
    between them — roughly one round each — and one round is not enough to
    both emit a final chunk and call `finish`. So the guard only latches;
    each loop counts its own rounds down from WIND_DOWN_ROUNDS.
    """
    guard = SpendGuard(session=_session(True))
    assert not hasattr(guard, "wind_down_rounds_left")
    assert not hasattr(guard, "consume_wind_down_round")
    assert WIND_DOWN_ROUNDS >= 2


def test_a_session_without_the_public_probe_never_winds_down():
    # bench_generate.py and most unit tests pass a bare stub session.
    guard = SpendGuard(session=SimpleNamespace())
    assert guard.should_wind_down() is False


def test_a_probe_that_raises_is_treated_as_below_ceiling():
    def boom() -> bool:
        raise RuntimeError("no turn in progress")

    guard = SpendGuard(session=SimpleNamespace(spend_ceiling_reached=boom))
    assert guard.should_wind_down() is False


# ── Pipeline-level placement ────────────────────────────────────────────────


class _CeilingSession:
    """A session whose spend ceiling trips after N LLM calls.

    Counting calls rather than tokens keeps these tests about the guard's
    PLACEMENT — which is what was missing — instead of about arithmetic that
    `TurnCost` already owns and already has tests for.
    """

    def __init__(self, *, trips_after: int):
        self._trips_after = trips_after
        self.calls = 0
        self.question_count = 0
        self.elicitor = None
        self._llm = SimpleNamespace()

    def spend_ceiling_reached(self) -> bool:
        return self.calls >= self._trips_after

    async def emit(self, *_a, **_kw) -> None:  # signal_thinking
        return None


def _pipeline_state(tmp_path, session, **over):
    from anton.core.tools.generate_artifact.state import GenState

    base = dict(
        session=session, artifact_type="html-app", artifact_path=tmp_path,
        slug="s", user_request="build a clock", agent_understanding="a clock",
        spend=SpendGuard(session=session),
    )
    base.update(over)
    return GenState(**base)


def _llm_response(*, tool: str | None = None, text: str = "ok"):
    from anton.core.llm.provider import LLMResponse, ToolCall, Usage

    return LLMResponse(
        content=text,
        tool_calls=[ToolCall(id="1", name=tool, input={})] if tool else [],
        usage=Usage(input_tokens=1, output_tokens=1),
    )


async def test_the_gathering_loop_stops_at_the_ceiling(tmp_path):
    """Design 7.3: wind-down in phase A ends the gathering round, and the
    state records that gathering never completed — which is also what opens
    the emergency data loop on the continuation."""
    from anton.core.tools.generate_artifact.discovery.engine import run_gathering_loop

    session = _CeilingSession(trips_after=2)

    async def _call(**kw):
        session.calls += 1
        return _llm_response(tool="scratchpad")

    session._llm.plan = _call
    session._llm.code = _call
    state = _pipeline_state(tmp_path, session)

    await run_gathering_loop(state)

    assert state.gathering_complete is False
    assert any(s.outcome == "stopped_over_budget" for s in state.trace)
    # MAX_ROUNDS is 20; the ceiling tripped after 2.
    assert session.calls <= 3


async def test_phase_e_does_not_start_when_the_ceiling_is_already_reached(tmp_path):
    """prd.md and spec.md are on disk by then, so the continuation is cheap —
    which is the whole reason the boundary is a stopping point."""
    from anton.core.tools.generate_artifact import orchestrator
    from anton.core.tools.generate_artifact.discovery import checkpoint as cp

    session = _CeilingSession(trips_after=0)
    state = _pipeline_state(tmp_path, session)

    result = await orchestrator.run(state, entry=cp.ENTRY_GENERATE)

    assert result["status"] == "stopped_over_budget"
    assert state.files_written == []


async def test_the_spec_phase_does_not_start_when_the_ceiling_is_reached(tmp_path):
    """The threshold caught in phase B or C must not still buy
    `make_tech_spec` — the single most expensive call of the run, because it
    carries the whole shared history."""
    from anton.core.artifacts.internal_files import TECH_SPEC_FILENAME
    from anton.core.tools.generate_artifact import orchestrator
    from anton.core.tools.generate_artifact.discovery import checkpoint as cp

    session = _CeilingSession(trips_after=0)
    state = _pipeline_state(tmp_path, session)

    result = await orchestrator.run(state, entry=cp.ENTRY_SPEC)

    assert result["status"] == "stopped_over_budget"
    assert not (tmp_path / TECH_SPEC_FILENAME).exists()


async def test_a_budget_stop_always_carries_the_brief(tmp_path):
    """The run can end before the user has seen a brief at all, and then this
    is the only way to show them one."""
    from anton.core.tools.generate_artifact import orchestrator
    from anton.core.tools.generate_artifact.discovery import checkpoint as cp

    session = _CeilingSession(trips_after=0)
    state = _pipeline_state(tmp_path, session, brief="## Goal\nA clock.")

    result = await orchestrator.run(state, entry=cp.ENTRY_GENERATE)
    assert result["brief_summary"] == "## Goal\nA clock."


# ── The write loop ──────────────────────────────────────────────────────────


def _stream_of(response):
    from anton.core.llm.provider import StreamComplete

    async def _gen(**kw):
        yield StreamComplete(response=response)

    return _gen


async def test_wind_down_injects_a_closing_instruction_into_the_write_loop(tmp_path):
    from anton.core.tools.generate_artifact import engine

    session = _CeilingSession(trips_after=0)
    seen: list[str] = []

    def _client(**kw):
        seen.extend(
            m["content"] for m in kw["messages"]
            if m["role"] == "user" and isinstance(m["content"], str)
        )
        session.calls += 1
        return _stream_of(_llm_response(tool="finish"))()

    session._llm.plan_stream = _client
    session._llm.code_stream = _client
    session._llm.max_tokens = 8192

    result = await engine._run_loop(
        session=session, system="s", kickoff="k", artifact_path=tmp_path,
        node_label="generate_frontend", require_files=False,
        spend=SpendGuard(session=session),
    )
    assert any("call `finish`" in s for s in seen)
    assert isinstance(result, dict)


async def test_the_write_loop_ends_when_its_closing_rounds_run_out(tmp_path):
    """The stop is a counter, not a second reading of the same threshold —
    the threshold stays true forever once crossed."""
    from anton.core.tools.generate_artifact import engine

    session = _CeilingSession(trips_after=0)

    def _client(**kw):
        session.calls += 1
        return _stream_of(_llm_response(tool="read_file"))()  # never finishes

    session._llm.plan_stream = _client
    session._llm.code_stream = _client
    session._llm.max_tokens = 8192

    result = await engine._run_loop(
        session=session, system="s", kickoff="k", artifact_path=tmp_path,
        node_label="generate_frontend", require_files=False,
        spend=SpendGuard(session=session),
    )
    assert result["over_budget"] is True
    assert session.calls == WIND_DOWN_ROUNDS


async def test_each_parallel_loop_gets_its_own_closing_rounds(tmp_path):
    """Two concurrent write loops share the latch, not the budget.

    A counter on the guard would give the pair WIND_DOWN_ROUNDS between them —
    roughly one round each — and one round is not enough to both emit a final
    chunk and call `finish`.
    """
    import asyncio

    from anton.core.tools.generate_artifact import engine

    session = _CeilingSession(trips_after=0)
    session._llm.max_tokens = 8192
    rounds = {"backend": 0, "frontend": 0}
    guard = SpendGuard(session=session)

    async def _loop(label: str):
        def _client(**kw):
            rounds[label] += 1
            return _stream_of(_llm_response(tool="read_file"))()

        local = SimpleNamespace(
            _llm=SimpleNamespace(
                plan_stream=_client, code_stream=_client, max_tokens=8192
            ),
            spend_ceiling_reached=session.spend_ceiling_reached,
        )
        return await engine._run_loop(
            session=local, system="s", kickoff="k", artifact_path=tmp_path,
            node_label=f"generate_{label}", require_files=False, spend=guard,
        )

    await asyncio.gather(_loop("backend"), _loop("frontend"))
    assert rounds["backend"] == WIND_DOWN_ROUNDS
    assert rounds["frontend"] == WIND_DOWN_ROUNDS
