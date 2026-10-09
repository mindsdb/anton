"""`ChatSession.model_calls` across a real turn.

A host keeps a quiet turn alive only while `session.model_calls.snapshot()`
reports a model call waiting on the provider. These drive a real ChatSession
and a real LLMClient over a scripted provider, hold one call at a time, and
read the snapshot while it is held. Every silent stretch a turn can sit in
while a model works must report a wait: the first call, the doubled retry
after a reply cut at its budget, a tool call whose arguments arrive in one
burst after a silence, a tool that calls the model itself, compaction, the
rule filter and the completion verifier. A tool that hangs on its own, and an
open question, must report none, so the host's idle bounds still end them.

The snapshot is read through `getattr(session, "model_calls", None)`, so a
session without the tracker fails these on behavior.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from anton.core.llm.client import LLMClient
from anton.core.llm.provider import (
    LLMProvider,
    LLMResponse,
    StreamComplete,
    StreamTextDelta,
    StreamToolUseDelta,
    StreamToolUseEnd,
    StreamToolUseStart,
    ToolCall,
    Usage,
)
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.tools.tool_defs import ToolDef

PROBE_TOOL = "probe"
GUARD_S = 5.0


@pytest.fixture(autouse=True)
def _verifier_on_single_round_turns(monkeypatch):
    # One tool round is enough to reach the completion verifier, and the LLM
    # verifier decides alone (Jev is a MindsHub-only shortcut).
    monkeypatch.setenv("ANTON_VERIFY_MIN_TOOL_ROUNDS", "1")
    monkeypatch.setenv("ANTON_VERIFIER_JEV", "off")


def _usage(*, output_tokens: int = 5, context_pressure: float = 0.0) -> Usage:
    return Usage(input_tokens=10, output_tokens=output_tokens, context_pressure=context_pressure)


def _text_response(text: str, **usage) -> LLMResponse:
    return LLMResponse(content=text, tool_calls=[], usage=_usage(**usage), stop_reason="end_turn")


def _tool_response(**usage) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id="call_1", name=PROBE_TOOL, input={})],
        usage=_usage(**usage),
        stop_reason="tool_use",
    )


def _verdict_response() -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(
            id="v1", name="_VerifierVerdict",
            input={"status": "COMPLETE", "reason": "done"},
        )],
        usage=_usage(),
        stop_reason="tool_use",
    )


class _Gate:
    """A point a scripted call stops at until the test releases it."""

    def __init__(self) -> None:
        self.reached = asyncio.Event()
        self.release = asyncio.Event()

    async def hold(self) -> None:
        self.reached.set()
        await self.release.wait()


class _ScriptedProvider(LLMProvider):
    """Plays one scripted stream per `stream` call and one scripted answer per
    `complete` call, in order. Records each call's output budget."""

    name = "scripted"

    def __init__(self, *, streams=(), completes=()) -> None:
        self.streams = list(streams)
        self.completes = list(completes)
        self.stream_budgets: list[int] = []
        self.complete_budgets: list[int] = []

    async def stream(self, *, max_tokens: int = 0, **_kwargs):
        self.stream_budgets.append(max_tokens)
        script = self.streams.pop(0)
        async for event in script():
            yield event

    async def complete(self, *, max_tokens: int = 0, **_kwargs) -> LLMResponse:
        self.complete_budgets.append(max_tokens)
        script = self.completes.pop(0)
        return await script()


def _stream_of(response: LLMResponse, *, gate: _Gate | None = None):
    async def _events():
        if gate is not None:
            await gate.hold()
        if response.content:
            yield StreamTextDelta(text=response.content)
        yield StreamComplete(response=response)

    return _events


def _answer(response: LLMResponse, *, gate: _Gate | None = None):
    async def _call():
        if gate is not None:
            await gate.hold()
        return response

    return _call


async def _hang():
    await asyncio.Event().wait()


async def _quick_tool(_session, _input):
    return "ok"


def _session(provider: _ScriptedProvider, *, tool_handler=_quick_tool, **config) -> ChatSession:
    client = LLMClient(
        planning_provider=provider, planning_model="planner",
        coding_provider=provider, coding_model="coder",
    )
    base = Path(__file__).resolve().parents[1] / ".pytest-workspace"
    base.mkdir(parents=True, exist_ok=True)
    session = ChatSession(ChatSessionConfig(
        llm_client=client, workspace=MagicMock(base=base), **config,
    ))
    session.tool_registry.register_tool(ToolDef(
        name=PROBE_TOOL,
        description="test-only tool",
        input_schema={"type": "object", "properties": {}},
        handler=tool_handler,
    ))
    return session


def _snapshot(session):
    calls = getattr(session, "model_calls", None)
    return calls.snapshot() if calls is not None else None


async def _drain(session, message: str = "do it") -> list:
    return [event async for event in session.turn_stream(message)]


async def _snapshot_at(session, gate: _Gate):
    """Run a turn until ``gate`` is reached, read the snapshot, then finish."""
    turn = asyncio.ensure_future(_drain(session))
    try:
        await asyncio.wait_for(gate.reached.wait(), timeout=GUARD_S)
        await asyncio.sleep(0.01)
        snap = _snapshot(session)
        gate.release.set()
        await asyncio.wait_for(turn, timeout=GUARD_S)
    finally:
        if not turn.done():
            turn.cancel()
            await asyncio.gather(turn, return_exceptions=True)
    return snap


async def _close(session) -> None:
    await session.close()


# --------------------------------------------------------------------------- #
# Silent stretches that must report a wait
# --------------------------------------------------------------------------- #

async def test_the_first_call_of_a_turn_reports_a_wait():
    gate = _Gate()
    session = _session(_ScriptedProvider(streams=[_stream_of(_text_response("done"), gate=gate)]))
    try:
        snap = await _snapshot_at(session, gate)
    finally:
        await _close(session)

    assert snap is not None, "a silent first call reported no wait"
    assert snap.role == "planning"
    assert snap.message.startswith("Waiting for the model (")


async def test_the_retry_after_a_cut_reply_reports_a_wait():
    """A reply that fills its budget is retried at twice the budget. That
    retry thinks longer, in silence, so it must report a wait too."""
    gate = _Gate()
    cut = _text_response("", output_tokens=8192)
    cut.stop_reason = "length"
    provider = _ScriptedProvider(streams=[
        _stream_of(cut),
        _stream_of(_text_response("done"), gate=gate),
    ])
    session = _session(provider)
    try:
        snap = await _snapshot_at(session, gate)
    finally:
        await _close(session)

    assert provider.stream_budgets == [8192, 16384]
    assert snap is not None, "the doubled retry reported no wait"


async def test_tool_arguments_arriving_after_a_silence_report_a_wait():
    """The prod shape: the call's start, then minutes of nothing, then every
    argument in one burst. The start never reaches a host's wire."""
    gate = _Gate()

    async def _burst_after_hold():
        yield StreamToolUseStart(id="call_1", name=PROBE_TOOL)
        await gate.hold()
        yield StreamToolUseDelta(id="call_1", json_delta="{}")
        yield StreamToolUseEnd(id="call_1")
        yield StreamComplete(response=_tool_response())

    provider = _ScriptedProvider(
        streams=[_burst_after_hold, _stream_of(_text_response("done"))],
        completes=[_answer(_verdict_response())],
    )
    session = _session(provider)
    try:
        snap = await _snapshot_at(session, gate)
    finally:
        await _close(session)

    assert snap is not None, "a held tool call reported no wait"
    assert snap.message.startswith("The model is still writing (")


async def test_a_tool_waiting_on_the_model_reports_a_wait():
    """generate_artifact and other tools call the model themselves."""
    gate = _Gate()

    async def _tool_that_asks_the_model(session, _input):
        response = await session._llm.plan(system="s", messages=[{"role": "user", "content": "x"}])
        return response.content

    provider = _ScriptedProvider(
        streams=[_stream_of(_tool_response()), _stream_of(_text_response("done"))],
        completes=[_answer(_text_response("inner"), gate=gate), _answer(_verdict_response())],
    )
    session = _session(provider, tool_handler=_tool_that_asks_the_model)
    try:
        snap = await _snapshot_at(session, gate)
    finally:
        await _close(session)

    assert snap is not None, "a tool's own model call reported no wait"


async def test_the_real_generate_artifact_tool_reports_a_wait(tmp_path):
    """The stand-in above pins the rule; this pins the real tool. Its model
    calls tick only because they go through the session's client, so a
    rewrite that gives it its own client or provider fails here."""
    import traceback

    from anton.core.artifacts import ArtifactStore

    artifacts = tmp_path / "artifacts"
    slug = ArtifactStore(artifacts).create(name="Clock", description="d", type="html-app").slug
    build = LLMResponse(
        content="",
        tool_calls=[ToolCall(
            id="ga1", name="generate_artifact",
            input={"slug": slug, "user_request": "build a clock", "agent_understanding": "an analog clock"},
        )],
        usage=_usage(),
        stop_reason="tool_use",
    )
    gate = _Gate()
    callers: list[str] = []

    def _held_tool_call():
        async def _events():
            callers.extend(frame.filename for frame in traceback.extract_stack())
            await gate.hold()
            yield StreamComplete(response=_text_response("gathering notes"))

        return _events()

    provider = _ScriptedProvider(
        streams=[_stream_of(build), _held_tool_call, _stream_of(_text_response("done"))],
    )
    client = LLMClient(
        planning_provider=provider, planning_model="planner",
        coding_provider=provider, coding_model="coder",
    )
    session = ChatSession(ChatSessionConfig(
        llm_client=client, workspace=MagicMock(base=tmp_path, artifacts_dir=artifacts),
    ))
    turn = asyncio.ensure_future(_drain(session, "make me a clock artifact"))
    try:
        await asyncio.wait_for(gate.reached.wait(), timeout=GUARD_S)
        await asyncio.sleep(0.01)
        snap = _snapshot(session)
    finally:
        turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
        await _close(session)

    assert any("generate_artifact" in name for name in callers), "the held call did not come from the tool"
    assert snap is not None, "the real generate_artifact model call reported no wait"


async def test_compaction_reports_a_wait():
    gate = _Gate()
    provider = _ScriptedProvider(
        streams=[
            _stream_of(_tool_response(context_pressure=0.95)),
            _stream_of(_text_response("done")),
        ],
        completes=[_answer(_text_response("summary"), gate=gate), _answer(_verdict_response())],
    )
    session = _session(provider)
    for i in range(6):
        session._history.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"})
    try:
        snap = await _snapshot_at(session, gate)
    finally:
        await _close(session)

    assert snap is not None, "a silent compaction call reported no wait"
    assert snap.role == "router"


async def test_the_rule_filter_reports_a_wait(tmp_path):
    """Enough conditional rules make the system prompt ask the coding model
    which ones apply, before the first reply token."""
    from anton.core.memory.cortex import Cortex
    from anton.core.memory.hippocampus import Engram, Hippocampus

    gate = _Gate()
    provider = _ScriptedProvider(
        streams=[_stream_of(_text_response("done"))],
        completes=[_answer(_text_response("NONE"), gate=gate)],
    )
    (tmp_path / "global").mkdir()
    (tmp_path / "project").mkdir()
    cortex = Cortex(
        global_hc=Hippocampus(tmp_path / "global"),
        project_hc=Hippocampus(tmp_path / "project"),
        mode="copilot",
    )
    rules = [
        Engram(text=f"When condition {i:03d} applies, do the thing {'x' * 500}",
               kind="when", scope="global", confidence="high")
        for i in range(12)
    ]
    cortex.global_hc.get_rules = lambda exclude_scratchpad_when=False: rules
    session = _session(provider, cortex=cortex)
    cortex._llm = session._llm
    try:
        snap = await _snapshot_at(session, gate)
    finally:
        await _close(session)

    assert provider.complete_budgets, "the rule filter never called the model"
    assert snap is not None, "a silent rule filter call reported no wait"
    assert snap.role == "coding"


async def test_the_completion_verifier_reports_a_wait():
    gate = _Gate()
    provider = _ScriptedProvider(
        streams=[_stream_of(_tool_response()), _stream_of(_text_response("done"))],
        completes=[_answer(_verdict_response(), gate=gate)],
    )
    session = _session(provider)
    try:
        snap = await _snapshot_at(session, gate)
    finally:
        await _close(session)

    assert snap is not None, "a silent verdict call reported no wait"
    assert snap.role == "coding"


# --------------------------------------------------------------------------- #
# Hangs that must report nothing
# --------------------------------------------------------------------------- #

async def test_a_tool_that_never_returns_reports_no_wait():
    """A hung tool outside any model call must still go quiet, so the host's
    300 s and 600 s bounds end the turn. The tool here is not a cell."""
    started = asyncio.Event()

    async def _never_returns(_session, _input):
        started.set()
        await asyncio.Event().wait()

    provider = _ScriptedProvider(streams=[_stream_of(_tool_response())])
    session = _session(provider, tool_handler=_never_returns)
    turn = asyncio.ensure_future(_drain(session))
    try:
        await asyncio.wait_for(started.wait(), timeout=GUARD_S)
        await asyncio.sleep(0.05)
        assert getattr(session, "model_calls", None) is not None, "no tracker on the session"
        assert _snapshot(session) is None, "a hung tool reported a model wait"
    finally:
        turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
        await _close(session)


async def test_an_open_question_reports_no_wait():
    """While a question waits on the user, its own timeout decides, even when
    a model call is open beside it."""
    from anton.core.interaction.elicit import AskOption, AskRequest, elicit
    from anton.core.interaction.emitter import TurnEmitter
    from anton.core.llm.liveness import ModelCallTracker

    seen: list = []

    class _Elicitor:
        supported_kinds = ("choice",)
        answer_hint = ""
        timeout_s = 300

        async def begin(self, question_id, request): ...

        async def ask(self, question_id, request):
            seen.append(_snapshot(session))
            from anton.core.interaction.elicit import AskAnswer

            return AskAnswer(status="answered", values=("a",))

        async def end(self, question_id): ...

    session = _session(_ScriptedProvider())
    session.elicitor = _Elicitor()
    session.emitter = TurnEmitter()
    session.model_calls = ModelCallTracker()
    session.model_calls.open(role="planning", idle_timeout_s=600.0).awaiting = True
    try:
        await elicit(session, "q1", AskRequest(
            prompt="Which?", options=(AskOption(value="a", label="A"), AskOption(value="b", label="B")),
        ))
    finally:
        await _close(session)

    assert seen == [None], "a model wait was reported while a question was open"
    assert _snapshot(session) is not None, "the gate must lift when the question closes"


async def test_nothing_reports_after_the_turn():
    provider = _ScriptedProvider(streams=[_stream_of(_text_response("done"))])
    session = _session(provider)
    try:
        await asyncio.wait_for(_drain(session), timeout=GUARD_S)
        calls = getattr(session, "model_calls", None)
        assert calls is not None, "no tracker on the session"
        assert calls.closed
        assert session._llm.call_tracker is None, "the turn left its tracker armed"
    finally:
        await _close(session)


async def test_the_session_applies_the_deadline_setting_to_a_client_the_host_built():
    """cowork-server builds its own LLMClient, so the setting reaches the
    client only through the settings it hands the session."""
    from anton.core.settings import CoreSettings

    session = _session(_ScriptedProvider(), settings=CoreSettings(model_call_idle_timeout_s=42))
    try:
        assert session._llm.model_call_idle_timeout_s == 42.0
    finally:
        await _close(session)


async def test_a_late_finalizer_of_an_abandoned_turn_leaves_the_next_turn_armed():
    """A turn abandoned at a yield is finalized later, from another task. Its
    finally must not disarm or close the turn running by then, whose calls
    must still register and latch on its own tracker."""
    tool_started, finalized = asyncio.Event(), asyncio.Event()
    seen: dict = {}

    async def _checks_after_the_finalizer(session, _input):
        tool_started.set()
        await finalized.wait()
        seen["armed"] = session._llm.call_tracker is session.model_calls
        seen["closed"] = session.model_calls.closed
        return "ok"

    provider = _ScriptedProvider(
        streams=[
            _stream_of(_text_response("abandoned")),
            _stream_of(_tool_response()),
            _stream_of(_text_response("done")),
        ],
        completes=[_answer(_verdict_response())],
    )
    session = _session(provider, tool_handler=_checks_after_the_finalizer)
    first = session.turn_stream("first")
    second = None
    try:
        await asyncio.wait_for(asyncio.ensure_future(anext(first)), timeout=GUARD_S)
        second = asyncio.ensure_future(_drain(session, "second"))
        await asyncio.wait_for(tool_started.wait(), timeout=GUARD_S)

        await asyncio.wait_for(asyncio.ensure_future(first.aclose()), timeout=GUARD_S)
        finalized.set()
        await asyncio.wait_for(second, timeout=GUARD_S)
    finally:
        if second is not None and not second.done():
            second.cancel()
            await asyncio.gather(second, return_exceptions=True)
        await _close(session)

    assert seen == {"armed": True, "closed": False}


async def test_work_a_turn_leaves_running_cannot_reach_the_next_turn():
    """The anton CLI keeps one session and one client across turns. Model work
    started after a turn ends (memory consolidation, the cerebellum flush)
    must not register on the next turn's tracker. There it would report a
    wait while that turn hangs in a tool, and its expiry would latch the next
    turn, failing that turn's later calls unsent."""
    tool_started = asyncio.Event()

    async def _hangs(_session, _input):
        tool_started.set()
        await asyncio.Event().wait()

    provider = _ScriptedProvider(
        streams=[_stream_of(_text_response("first")), _stream_of(_tool_response())],
        completes=[_hang],
    )
    session = _session(provider, tool_handler=_hangs)
    session._llm.model_call_idle_timeout_s = 0.2
    second = leftover = None
    try:
        await asyncio.wait_for(_drain(session), timeout=GUARD_S)

        async def _leftover_work():
            # Started after turn 1, issued once turn 2 is running.
            await tool_started.wait()
            return await session._llm.summarize(
                system="s", messages=[{"role": "user", "content": "x"}],
            )

        leftover = asyncio.ensure_future(_leftover_work())
        second = asyncio.ensure_future(_drain(session, "again"))
        await asyncio.wait_for(tool_started.wait(), timeout=GUARD_S)
        await asyncio.sleep(0.05)
        assert _snapshot(session) is None, "turn 1's leftover call reported a wait in turn 2"

        (outcome,) = await asyncio.wait_for(
            asyncio.gather(leftover, return_exceptions=True), timeout=GUARD_S,
        )
        assert type(outcome).__name__ == "ModelCallTimeoutError", repr(outcome)
        session.model_calls.raise_if_expired()  # turn 2 latched nothing
    finally:
        for task in (second, leftover):
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await _close(session)


# --------------------------------------------------------------------------- #
# The deadline inside a turn
# --------------------------------------------------------------------------- #

async def test_a_verifier_deadline_ends_the_turn_quietly_on_the_answer():
    """The answer already streamed. An error card under it would mislead, so
    the turn ends unverified, with no hand-back and the verifier latch left
    alone."""
    provider = _ScriptedProvider(
        streams=[_stream_of(_tool_response()), _stream_of(_text_response("done"))],
        completes=[_hang],
    )
    session = _session(provider, session_id="conv-model-calls")
    session._llm.model_call_idle_timeout_s = 0.2
    try:
        with patch("anton.analytics.send_event") as send:
            events = await asyncio.wait_for(_drain(session), timeout=GUARD_S)
    finally:
        await _close(session)

    text = "".join(e.text for e in events if isinstance(e, StreamTextDelta))
    assert text == "done"
    assert len(provider.complete_budgets) == 1, "the verdict call was retried"
    assert len(provider.stream_budgets) == 2, "a hand-back ran after the verifier deadline"
    fields = send.call_args.kwargs
    assert fields["ended_by"] == "completed"
    assert fields["verification_skipped"] == "true"
    assert fields["verifier_failure"] == "timeout"
    assert fields["verifier_error_type"] == "ModelCallTimeoutError:coding"
    assert session._verifier_latch.no_verdict_failures == 0
    assert not session._verifier_latch.latched


async def test_a_compaction_deadline_before_the_verifier_books_the_compaction_role():
    """Compaction on the final answer runs out its deadline and swallows it.
    The latch then fails the verdict call before it is sent. The books must
    name the router call that went silent, not file it as a verifier timeout."""
    provider = _ScriptedProvider(
        streams=[
            _stream_of(_tool_response()),
            _stream_of(_text_response("done", context_pressure=0.95)),
        ],
        completes=[_hang],
    )
    session = _session(provider, session_id="conv-model-calls")
    # Long enough that the summarizer has material to fold, so it calls the model.
    for i in range(6):
        session._history.append({
            "role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + "x" * 500,
        })
    session._llm.model_call_idle_timeout_s = 0.2
    try:
        with patch("anton.analytics.send_event") as send:
            await asyncio.wait_for(_drain(session), timeout=GUARD_S)
    finally:
        await _close(session)

    assert len(provider.complete_budgets) == 1, "the latched verdict call reached the provider"
    fields = send.call_args.kwargs
    assert fields["verifier_failure"] == "timeout"
    assert fields["verifier_error_type"] == "ModelCallTimeoutError:router"
