"""ask_user: value types, validation, the elicit() lifecycle, the tool
handler, and registration gating."""

from __future__ import annotations

import asyncio

import pytest

from anton.core.interaction.elicit import (
    MAX_QUESTIONS_PER_TURN,
    AskAnswer,
    AskOption,
    AskRequest,
    elicit,
    validate_request,
)
from anton.core.llm.provider import (
    StreamAskUser,
    StreamAskUserAnswered,
    StreamTaskProgress,
)


def _choice(**over) -> AskRequest:
    base = dict(
        prompt="Which database?",
        options=(AskOption(value="pg", label="postgres"), AskOption(value="my", label="mysql")),
    )
    base.update(over)
    return AskRequest(**base)


def test_defaults():
    r = _choice()
    assert (r.kind, r.select, r.allow_custom, r.timeout_s) == ("choice", "one", True, None)
    assert AskAnswer(status="answered").values == ()


def test_valid_choice_passes():
    assert validate_request(_choice()) is True
    assert validate_request(_choice(select="many")) is True


@pytest.mark.parametrize(
    "request_",
    [
        _choice(prompt=""),
        _choice(prompt="   "),
        _choice(options=()),
        _choice(options=(AskOption(value="pg", label="postgres"),)),
        _choice(options=tuple(AskOption(value=f"v{i}", label=f"l{i}") for i in range(11))),
        _choice(options=(AskOption(value="pg", label="a"), AskOption(value="pg", label="b"))),
        _choice(options=(AskOption(value="", label="a"), AskOption(value="b", label="b"))),
        _choice(select="several"),
    ],
    ids=[
        "empty-prompt", "blank-prompt", "no-options", "one-option",
        "eleven-options", "duplicate-values", "empty-value", "bad-select",
    ],
)
def test_invalid_choice_rejected(request_):
    assert validate_request(request_) is False


def test_path_request_ignores_choice_rules():
    # A path picker has no options and no select mode — it must still validate.
    assert validate_request(AskRequest(prompt="Pick a folder", kind="path")) is True
    assert validate_request(AskRequest(prompt="", kind="path")) is False


def test_budget_constant_is_three():
    assert MAX_QUESTIONS_PER_TURN == 3


# ─── TurnEmitter + session wiring ───────────────────────────────────────


async def test_turn_emitter_is_fifo_and_drains():
    from anton.core.interaction.emitter import TurnEmitter

    emitter = TurnEmitter()
    assert emitter.empty() is True
    await emitter.emit("a")
    await emitter.emit("b")
    assert emitter.empty() is False
    assert await emitter.get() == "a"
    assert emitter.get_nowait() == "b"
    assert emitter.empty() is True


async def test_turn_emitter_never_blocks_the_producer():
    """Unbounded by contract: a bounded queue would deadlock a tool that
    emits while nothing is draining."""
    from anton.core.interaction.emitter import TurnEmitter

    emitter = TurnEmitter()
    for i in range(1000):
        await asyncio.wait_for(emitter.emit(i), timeout=1)
    assert emitter.empty() is False


def test_session_exposes_the_new_public_attributes(make_session):
    session = make_session()
    assert session.emitter is None
    assert session.question_count == 0
    assert session.answer_wait_s == 0.0
    assert session.escape_watcher is None
    assert not hasattr(session, "selection_elicitor")


async def test_session_emit_is_a_noop_without_an_emitter(make_session):
    session = make_session()
    await session.emit("anything")  # must not raise


async def test_session_emit_forwards_to_the_attached_emitter(make_session):
    from anton.core.interaction.emitter import TurnEmitter

    session = make_session()
    session.emitter = TurnEmitter()
    await session.emit("ev")
    assert await session.emitter.get() == "ev"


def test_console_only_session_constructs(make_session):
    """Regression: chat.py builds the real session with console=console, so
    anything unimportable in that path breaks every CLI run at construction."""
    from rich.console import Console

    session = make_session(console=Console(quiet=True))
    assert session.elicitor is None  # host-injected only until CLIElicitor exists


# ─── elicit() lifecycle ─────────────────────────────────────────────────


class _RecordingElicitor:
    supported_kinds = ("choice", "path")
    answer_hint = "hint"
    timeout_s = 300

    def __init__(self, answer=None, raises=None) -> None:
        self.answer = answer or AskAnswer(status="answered", values=("pg",))
        self.raises = raises
        self.calls: list[str] = []

    async def begin(self, question_id, request):
        self.calls.append(f"begin:{question_id}")

    async def ask(self, question_id, request):
        self.calls.append(f"ask:{question_id}")
        if self.raises is not None:
            raise self.raises
        return self.answer

    async def end(self, question_id):
        self.calls.append(f"end:{question_id}")


class _Watcher:
    def __init__(self) -> None:
        self.events: list[str] = []

    def pause(self) -> None:
        self.events.append("pause")

    def resume(self) -> None:
        self.events.append("resume")


def _wired(make_session, elicitor):
    from anton.core.interaction.emitter import TurnEmitter

    session = make_session()
    session.elicitor = elicitor
    session.emitter = TurnEmitter()
    session.escape_watcher = _Watcher()
    return session


def _drain(emitter) -> list:
    out = []
    while not emitter.empty():
        out.append(emitter.get_nowait())
    return out


async def test_happy_path_order_begin_progress_card_ask_answered_end(make_session):
    elicitor = _RecordingElicitor()
    session = _wired(make_session, elicitor)

    answer = await elicit(session, "q1", _choice())

    assert answer.status == "answered"
    assert elicitor.calls == ["begin:q1", "ask:q1", "end:q1"]
    events = _drain(session.emitter)
    assert isinstance(events[0], StreamTaskProgress)
    assert events[0].phase == "interactive"
    assert isinstance(events[1], StreamAskUser)
    assert events[1].id == "q1"
    assert isinstance(events[2], StreamAskUserAnswered)
    assert events[2].answer.status == "answered"
    assert session.escape_watcher.events == ["pause", "resume"]


async def test_begin_happens_before_anything_is_published(make_session):
    """The answer channel must exist before the card can be clicked."""
    order: list[str] = []

    class _Elicitor(_RecordingElicitor):
        async def begin(self, question_id, request):
            order.append("begin")

    session = _wired(make_session, _Elicitor())
    original_emit = session.emitter.emit

    async def _tracking_emit(event):
        order.append(f"emit:{type(event).__name__}")
        await original_emit(event)

    session.emitter.emit = _tracking_emit
    await elicit(session, "q1", _choice())
    assert order[0] == "begin"
    assert order[1] == "emit:StreamTaskProgress"
    assert order[2] == "emit:StreamAskUser"


async def test_unavailable_when_no_elicitor(make_session):
    session = _wired(make_session, None)
    session.elicitor = None
    assert (await elicit(session, "q1", _choice())).status == "unavailable"
    assert _drain(session.emitter) == []


async def test_unavailable_when_kind_unsupported(make_session):
    class _PathOnly(_RecordingElicitor):
        supported_kinds = ("path",)

    session = _wired(make_session, _PathOnly())
    assert (await elicit(session, "q1", _choice())).status == "unavailable"
    assert _drain(session.emitter) == []


async def test_unavailable_when_request_is_invalid(make_session):
    session = _wired(make_session, _RecordingElicitor())
    bad = _choice(options=(AskOption(value="pg", label="postgres"),))
    assert (await elicit(session, "q1", bad)).status == "unavailable"
    assert _drain(session.emitter) == []


async def test_choice_needs_an_emitter_but_path_does_not(make_session):
    elicitor = _RecordingElicitor(answer=AskAnswer(status="answered", values=("/tmp",)))
    session = make_session()
    session.elicitor = elicitor
    session.emitter = None  # e.g. the non-streaming turn()

    assert (await elicit(session, "q1", _choice())).status == "unavailable"
    # A path picker renders in the terminal, so it must still work.
    path_request = AskRequest(prompt="Pick a folder", kind="path")
    assert (await elicit(session, "q2", path_request)).status == "answered"


async def test_path_questions_are_not_published(make_session):
    session = _wired(make_session, _RecordingElicitor())
    await elicit(session, "q1", AskRequest(prompt="Pick a folder", kind="path"))
    kinds = [type(e).__name__ for e in _drain(session.emitter)]
    assert "StreamAskUser" not in kinds
    assert "StreamTaskProgress" in kinds  # the spinner still has to stop


async def test_rate_limit_publishes_nothing_and_never_opens_a_channel(make_session):
    elicitor = _RecordingElicitor()
    session = _wired(make_session, elicitor)
    for _ in range(MAX_QUESTIONS_PER_TURN):
        assert (await elicit(session, "q", _choice())).status == "answered"
    _drain(session.emitter)
    elicitor.calls.clear()

    assert (await elicit(session, "over", _choice())).status == "limit"
    assert elicitor.calls == []  # no begin() — the zombie-card regression
    assert _drain(session.emitter) == []


async def test_end_runs_when_ask_raises(make_session):
    elicitor = _RecordingElicitor(raises=RuntimeError("boom"))
    session = _wired(make_session, elicitor)
    with pytest.raises(RuntimeError):
        await elicit(session, "q1", _choice())
    assert "end:q1" in elicitor.calls
    assert session.escape_watcher.events == ["pause", "resume"]


async def test_end_runs_when_ask_is_cancelled(make_session):
    class _Hangs(_RecordingElicitor):
        async def ask(self, question_id, request):
            self.calls.append(f"ask:{question_id}")
            await asyncio.sleep(3600)

    elicitor = _Hangs()
    session = _wired(make_session, elicitor)
    task = asyncio.create_task(elicit(session, "q1", _choice()))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "end:q1" in elicitor.calls


async def test_answer_wait_is_accumulated_on_the_session(make_session):
    class _Slow(_RecordingElicitor):
        async def ask(self, question_id, request):
            await asyncio.sleep(0.05)
            return AskAnswer(status="answered", values=("pg",))

    session = _wired(make_session, _Slow())
    await elicit(session, "q1", _choice())
    assert session.answer_wait_s >= 0.05
