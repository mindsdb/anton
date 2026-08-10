"""ask_user: value types, validation, the elicit() lifecycle, the tool
handler, and registration gating."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

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
from anton.core.tools.tool_defs import ASK_USER_TOOL
from anton.core.tools.tool_handlers import build_ask_request, handle_ask_user


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


def test_budget_constant_is_eight():
    """Raised from 3 to 8 to give generate_prd's phase 2 (brief confirm +
    revise cycles) room without starving the main agent's own questions in
    the same turn — see prd-design.md, "Limits and error handling"."""
    assert MAX_QUESTIONS_PER_TURN == 8


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
    # Stated directly as well as demonstrated: 1000 puts also fit through a
    # maxsize of 1001, so the loop below alone would not catch a bound.
    assert emitter._queue.maxsize == 0
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


def test_console_only_session_gets_the_cli_elicitor(make_session):
    """Regression: chat.py builds the real session with console=console, so a
    broken fallback here breaks every CLI run at construction time."""
    from rich.console import Console

    from anton.core.interaction.cli import CLIElicitor

    session = make_session(console=Console(quiet=True))
    assert isinstance(session.elicitor, CLIElicitor)
    assert "choice" in session.elicitor.supported_kinds


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
    # Symmetric: no card was published, so nothing is retired either.
    assert "StreamAskUserAnswered" not in kinds


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


async def test_a_raising_ask_still_retires_the_published_card(make_session):
    """handle_ask_user does not catch, so a raising ask() becomes a tool failure
    and the TURN CONTINUES. The frontend retires a card only on
    StreamAskUserAnswered — its expired-card fallback keys on there being no
    in-flight run, which is false here — so without a retirement the user is
    left clicking live buttons and gets a 404 for a question the agent has
    already moved past."""
    elicitor = _RecordingElicitor(raises=RuntimeError("boom"))
    session = _wired(make_session, elicitor)
    with pytest.raises(RuntimeError):
        await elicit(session, "q1", _choice())

    events = _drain(session.emitter)
    assert [type(e).__name__ for e in events] == [
        "StreamTaskProgress",
        "StreamAskUser",
        "StreamAskUserAnswered",
    ]
    assert events[-1].id == "q1"
    # "error" is one of the pinned tool-result statuses, so this is not a
    # contract change.
    assert events[-1].answer.status == "error"


async def test_a_raising_ask_retires_nothing_for_an_unpublished_path_question(
    make_session,
):
    """Symmetry with the success path: a path question publishes no card, so
    there is nothing to retire and handing the host an 'answered' event for an
    id it has never seen would be a bug."""
    session = _wired(make_session, _RecordingElicitor(raises=RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        await elicit(session, "q1", AskRequest(prompt="Pick a folder", kind="path"))
    kinds = [type(e).__name__ for e in _drain(session.emitter)]
    assert "StreamAskUser" not in kinds
    assert "StreamAskUserAnswered" not in kinds


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


async def test_answer_wait_accumulates_rather_than_overwrites(make_session):
    """A later task subtracts this from a tool's elapsed time, so a regression
    from += to = would silently under-report every question after the first."""

    class _Slow(_RecordingElicitor):
        async def ask(self, question_id, request):
            await asyncio.sleep(0.05)
            return AskAnswer(status="answered", values=("pg",))

    session = _wired(make_session, _Slow())
    session.answer_wait_s = 5.0
    await elicit(session, "q1", _choice())
    assert session.answer_wait_s >= 5.05


async def test_answer_wait_is_credited_when_ask_raises(make_session):
    session = _wired(make_session, _RecordingElicitor(raises=RuntimeError("boom")))
    session.answer_wait_s = 1.0
    with pytest.raises(RuntimeError):
        await elicit(session, "q1", _choice())
    assert session.answer_wait_s > 1.0


async def test_answer_wait_is_credited_when_ask_is_cancelled(make_session):
    class _Hangs(_RecordingElicitor):
        async def ask(self, question_id, request):
            await asyncio.sleep(3600)

    session = _wired(make_session, _Hangs())
    session.answer_wait_s = 1.0
    task = asyncio.create_task(elicit(session, "q1", _choice()))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert session.answer_wait_s > 1.0


# ─── the tool ───────────────────────────────────────────────────────────

_TC_INPUT = {
    "question": "Which database should I read from?",
    "options": [
        {"value": "pg", "label": "postgres"},
        {"value": "my", "label": "mysql", "detail": "read replica"},
    ],
}


def test_build_ask_request_maps_the_schema():
    request = build_ask_request(_TC_INPUT, timeout_s=300)
    assert request.prompt == "Which database should I read from?"
    assert request.kind == "choice"
    assert request.timeout_s == 300
    assert request.select == "one"
    assert request.allow_custom is True
    assert [o.value for o in request.options] == ["pg", "my"]
    assert request.options[1].detail == "read replica"


def test_build_ask_request_honours_select_and_allow_custom():
    request = build_ask_request(
        {**_TC_INPUT, "select": "many", "allow_custom": False}, timeout_s=None
    )
    assert (request.select, request.allow_custom) == ("many", False)


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"question": "q"},
        {"question": "q", "options": "not-a-list"},
        {"question": "q", "options": [{"label": "no value"}]},
        {**_TC_INPUT, "select": "several"},
    ],
    ids=["empty", "no-options", "options-not-a-list", "option-without-value", "bad-select"],
)
def test_build_ask_request_returns_none_on_junk(bad):
    assert build_ask_request(bad, timeout_s=None) is None


async def test_handler_serializes_each_status(make_session):
    cases = [
        (AskAnswer(status="answered", values=("pg",)), {"status": "answered", "values": ["pg"]}),
        (AskAnswer(status="answered", text="clickhouse"), {"status": "answered", "text": "clickhouse"}),
        (
            AskAnswer(status="answered", values=("pg", "my"), text="and duckdb"),
            {"status": "answered", "values": ["pg", "my"], "text": "and duckdb"},
        ),
        (AskAnswer(status="cancelled"), {"status": "cancelled"}),
        (AskAnswer(status="timeout"), {"status": "timeout"}),
    ]
    for answer, expected in cases:
        session = _wired(make_session, _RecordingElicitor(answer=answer))
        assert json.loads(await handle_ask_user(session, _TC_INPUT)) == expected


async def test_handler_maps_internal_statuses_onto_error(make_session):
    session = _wired(make_session, _RecordingElicitor(answer=AskAnswer(status="unavailable")))
    result = json.loads(await handle_ask_user(session, _TC_INPUT))
    assert result["status"] == "error"
    assert "message" in result

    session = _wired(make_session, _RecordingElicitor())
    session.question_count = MAX_QUESTIONS_PER_TURN
    limited = json.loads(await handle_ask_user(session, _TC_INPUT))
    assert limited["status"] == "error"
    assert "limit" in limited["message"]


async def test_handler_reports_an_unlisted_status_as_error_not_answered(make_session):
    """`Elicitor` is a structural Protocol implemented out of tree, so a status
    nobody here anticipated — a host-side typo, a future status, the "error" the
    retirement event now carries — is reachable without touching this repo. An
    if-chain that falls through to `_status("answered")` would tell the LLM the
    user answered and chose nothing, which is the worst failure shape a decision
    tool has."""
    session = _wired(
        make_session, _RecordingElicitor(answer=AskAnswer(status="wat", values=("pg",)))
    )
    result = json.loads(await handle_ask_user(session, _TC_INPUT))
    assert result["status"] == "error"
    assert "wat" in result["message"]
    assert "values" not in result


async def test_handler_rejects_junk_before_touching_the_elicitor(make_session):
    elicitor = _RecordingElicitor()
    session = _wired(make_session, elicitor)
    result = json.loads(await handle_ask_user(session, {"question": "q"}))
    assert result["status"] == "error"
    assert elicitor.calls == []


async def test_handler_emits_one_telemetry_event_per_outcome(make_session, monkeypatch):
    """Real `send_event(settings, action, **extra)` takes only string extras
    and a settings object — not the (name, props) shape a first draft of
    this test assumed. Verified against every existing call site
    (anton/tools.py, anton/chat.py, anton/cli.py): each does a local
    `from anton.analytics import send_event` right before calling it, so
    patching the module attribute here is picked up correctly.
    """
    sent: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "anton.analytics.send_event",
        lambda settings, action, **extra: sent.append((action, extra)),
    )
    for answer, expected in [
        (AskAnswer(status="answered", values=("pg",)), "answered"),
        (AskAnswer(status="cancelled"), "cancelled"),
        (AskAnswer(status="timeout"), "timeout"),
    ]:
        sent.clear()
        session = _wired(make_session, _RecordingElicitor(answer=answer))
        await handle_ask_user(session, _TC_INPUT)
        names = [name for name, _ in sent]
        assert names == ["ask_user_asked", f"ask_user_{expected}"]
        assert sent[0][1]["select"] == "one"
        assert sent[0][1]["options"] == "2"


async def test_handler_survives_send_event_raising(make_session, monkeypatch):
    """Analytics must never break a turn: even if send_event itself raises,
    handle_ask_user still returns a normal tool result."""

    def _raise(settings, action, **extra):
        raise RuntimeError("analytics backend is down")

    monkeypatch.setattr("anton.analytics.send_event", _raise)
    session = _wired(make_session, _RecordingElicitor())
    result = json.loads(await handle_ask_user(session, _TC_INPUT))
    assert result["status"] == "answered"


# ─── registration gating ────────────────────────────────────────────────


def _tool_names(session) -> set[str]:
    session._build_tools()
    return {t["name"] for t in session.tool_registry.dump()}


def test_ask_user_absent_without_an_elicitor_or_console(make_session):
    session = make_session()
    names = _tool_names(session)
    assert "ask_user" not in names
    assert "select_path" in names  # always registered, degrades on its own


def test_ask_user_present_when_the_elicitor_supports_choice(make_session):
    session = make_session(elicitor=_RecordingElicitor())
    assert "ask_user" in _tool_names(session)


def test_ask_user_absent_for_a_path_only_elicitor(make_session):
    class _PathOnly(_RecordingElicitor):
        supported_kinds = ("path",)

    session = make_session(elicitor=_PathOnly())
    names = _tool_names(session)
    assert "ask_user" not in names
    assert "select_path" in names


def test_registration_copies_the_tooldef_instead_of_mutating_the_singleton(make_session):
    class _A(_RecordingElicitor):
        answer_hint = "HINT-A"

    class _B(_RecordingElicitor):
        answer_hint = "HINT-B"

    pristine = ASK_USER_TOOL.description
    session_a = make_session(elicitor=_A())
    session_b = make_session(elicitor=_B())
    session_a._build_tools()
    session_b._build_tools()

    def _desc(session):
        return next(t["description"] for t in session.tool_registry.dump() if t["name"] == "ask_user")

    assert "HINT-A" in _desc(session_a)
    assert "HINT-B" in _desc(session_b)
    assert ASK_USER_TOOL.description == pristine  # module singleton untouched


# ─── CLI elicitor ───────────────────────────────────────────────────────


@pytest.fixture()
def cli_elicitor():
    from rich.console import Console

    from anton.core.interaction.cli import CLIElicitor

    return CLIElicitor(Console(quiet=True))


async def test_cli_elicitor_satisfies_the_protocol(cli_elicitor):
    assert cli_elicitor.supported_kinds == ("choice", "path")
    assert cli_elicitor.timeout_s is None
    assert isinstance(cli_elicitor.answer_hint, str) and cli_elicitor.answer_hint
    assert await cli_elicitor.begin("q1", _choice()) is None
    assert await cli_elicitor.end("q1") is None


async def test_ask_without_before_prompt_set_does_not_crash(cli_elicitor, monkeypatch):
    # Default from the fixture: before_prompt is None. A host that never
    # wires one (or a test double) must not blow up on ask().
    assert cli_elicitor.before_prompt is None
    monkeypatch.setattr(
        "anton.utils.prompt.prompt_or_cancel", AsyncMock(return_value="1")
    )
    answer = await cli_elicitor.ask("q1", _choice())
    assert answer.status == "answered"


async def test_ask_calls_before_prompt_before_reading_choice_input(
    cli_elicitor, monkeypatch
):
    # before_prompt exists to stop chat_ui's spinner (and, for a published
    # question, wait for it to be printed) before prompt_toolkit renders its
    # own line — it must run BEFORE the input read, not after, or the race
    # it exists to prevent is still there.
    order: list[str] = []

    async def _before_prompt(question_id, request):
        order.append("before_prompt")

    cli_elicitor.before_prompt = _before_prompt

    async def _fake_prompt(*args, **kwargs):
        order.append("prompt")
        return "1"

    monkeypatch.setattr("anton.utils.prompt.prompt_or_cancel", _fake_prompt)
    await cli_elicitor.ask("q1", _choice())
    assert order == ["before_prompt", "prompt"]


async def test_ask_calls_before_prompt_before_reading_path_input(
    cli_elicitor, monkeypatch
):
    order: list[str] = []

    async def _before_prompt(question_id, request):
        order.append("before_prompt")

    cli_elicitor.before_prompt = _before_prompt

    async def _fake_prompt(*args, **kwargs):
        order.append("prompt")
        return "/tmp/x"

    monkeypatch.setattr("anton.utils.prompt.prompt_or_cancel", _fake_prompt)
    await cli_elicitor.ask(
        "q1",
        AskRequest(prompt="Pick a folder", kind="path", path_mode="browse"),
    )
    assert order == ["before_prompt", "prompt"]


async def test_ask_passes_question_id_and_request_to_before_prompt(
    cli_elicitor, monkeypatch
):
    seen = {}

    async def _before_prompt(question_id, request):
        seen["question_id"] = question_id
        seen["request"] = request

    cli_elicitor.before_prompt = _before_prompt
    monkeypatch.setattr(
        "anton.utils.prompt.prompt_or_cancel", AsyncMock(return_value="1")
    )
    request = _choice()
    await cli_elicitor.ask("q-42", request)
    assert seen == {"question_id": "q-42", "request": request}


async def test_cli_choice_maps_a_number_to_the_option_value(cli_elicitor, monkeypatch):
    monkeypatch.setattr(
        "anton.utils.prompt.prompt_or_cancel", AsyncMock(return_value="2")
    )
    answer = await cli_elicitor.ask("q1", _choice())
    assert answer == AskAnswer(status="answered", values=("my",))


async def test_cli_choice_accepts_a_comma_list_for_many(cli_elicitor, monkeypatch):
    monkeypatch.setattr(
        "anton.utils.prompt.prompt_or_cancel", AsyncMock(return_value="1, 2")
    )
    answer = await cli_elicitor.ask("q1", _choice(select="many"))
    assert answer.values == ("pg", "my")


async def test_cli_choice_keeps_only_the_first_pick_when_select_is_one(
    cli_elicitor, monkeypatch
):
    monkeypatch.setattr(
        "anton.utils.prompt.prompt_or_cancel", AsyncMock(return_value="2,1")
    )
    answer = await cli_elicitor.ask("q1", _choice())
    assert answer.values == ("my",)


async def test_cli_choice_treats_free_text_as_a_custom_answer(cli_elicitor, monkeypatch):
    monkeypatch.setattr(
        "anton.utils.prompt.prompt_or_cancel", AsyncMock(return_value="clickhouse")
    )
    answer = await cli_elicitor.ask("q1", _choice())
    assert answer == AskAnswer(status="answered", text="clickhouse")


async def test_cli_choice_escape_and_empty_input_are_cancelled(cli_elicitor, monkeypatch):
    for value in (None, "", "   "):
        monkeypatch.setattr(
            "anton.utils.prompt.prompt_or_cancel", AsyncMock(return_value=value)
        )
        assert (await cli_elicitor.ask("q1", _choice())).status == "cancelled"


# ─── CLI elicitor — _ask_path (browse mode) ─────────────────────────────


def _path_request(**over) -> AskRequest:
    base = dict(prompt="Pick a folder", kind="path", path_mode="browse", root="/home/user")
    base.update(over)
    return AskRequest(**base)


async def test_cli_browse_prints_prompt_and_root_and_returns_the_typed_path(monkeypatch):
    from rich.console import Console

    from anton.core.interaction.cli import CLIElicitor

    console = Console(record=True, width=100)
    elicitor = CLIElicitor(console)
    monkeypatch.setattr(
        "anton.utils.prompt.prompt_or_cancel", AsyncMock(return_value="/home/user/docs")
    )

    answer = await elicitor.ask("q1", _path_request())

    assert answer == AskAnswer(status="answered", values=("/home/user/docs",))
    text = console.export_text()
    assert "Pick a folder" in text
    assert "/home/user" in text


async def test_cli_browse_with_no_root_prints_no_starting_at_line(monkeypatch):
    from rich.console import Console

    from anton.core.interaction.cli import CLIElicitor

    console = Console(record=True, width=100)
    elicitor = CLIElicitor(console)
    monkeypatch.setattr(
        "anton.utils.prompt.prompt_or_cancel", AsyncMock(return_value="/tmp/x")
    )

    await elicitor.ask("q1", _path_request(root=""))

    assert "starting at" not in console.export_text()


async def test_cli_browse_escapes_rich_markup_in_the_prompt_and_root(monkeypatch):
    """`root` is a filesystem path, so a directory named "[dim]" would be
    swallowed and one containing "[/]" would raise MarkupError out of ask()."""
    from rich.console import Console

    from anton.core.interaction.cli import CLIElicitor

    console = Console(record=True, width=120)
    monkeypatch.setattr(
        "anton.utils.prompt.prompt_or_cancel", AsyncMock(return_value="/tmp/x")
    )
    await CLIElicitor(console).ask(
        "q1", _path_request(prompt="Pick [bold]a[/] folder", root="/tmp/[dim]/[/]")
    )
    text = console.export_text()
    assert "Pick [bold]a[/] folder" in text
    assert "/tmp/[dim]/[/]" in text


async def test_cli_browse_escape_is_cancelled(cli_elicitor, monkeypatch):
    monkeypatch.setattr(
        "anton.utils.prompt.prompt_or_cancel", AsyncMock(return_value=None)
    )
    answer = await cli_elicitor.ask("q1", _path_request())
    assert answer.status == "cancelled"


async def test_cli_browse_empty_or_whitespace_answer_is_cancelled(cli_elicitor, monkeypatch):
    for value in ("", "   "):
        monkeypatch.setattr(
            "anton.utils.prompt.prompt_or_cancel", AsyncMock(return_value=value)
        )
        answer = await cli_elicitor.ask("q1", _path_request())
        assert answer.status == "cancelled"


# ─── CLI elicitor — _ask_path (pick mode) ───────────────────────────────


def _pick_request(**over) -> AskRequest:
    base = dict(
        prompt="Which file?",
        kind="path",
        path_mode="pick",
        options=(
            AskOption(value="/a", label="a.csv", kind="file"),
            AskOption(value="/b", label="b", kind="folder", detail="12 files"),
        ),
    )
    base.update(over)
    return AskRequest(**base)


async def test_cli_pick_renders_numbered_options_with_icons_and_detail(monkeypatch):
    from rich.console import Console

    from anton.core.interaction.cli import CLIElicitor

    console = Console(record=True, width=100)
    elicitor = CLIElicitor(console)
    monkeypatch.setattr(
        "anton.utils.prompt.prompt_or_cancel", AsyncMock(return_value="1")
    )

    await elicitor.ask("q1", _pick_request())

    text = console.export_text()
    lines = text.splitlines()
    file_line = next(line for line in lines if "a.csv" in line)
    folder_line = next(line for line in lines if line.strip().startswith("2."))
    assert "Which file?" in text
    assert "1." in file_line and "📄" in file_line
    assert "b" in folder_line and "📁" in folder_line
    assert "12 files" in folder_line


async def test_cli_pick_escapes_rich_markup_in_filesystem_controlled_text(monkeypatch):
    """Labels and details here come from the filesystem: a directory named
    "[dim]" would be swallowed as a Rich tag, and one containing "[/]" would
    raise MarkupError out of ask() while the tool is mid-dispatch."""
    from rich.console import Console

    from anton.core.interaction.cli import CLIElicitor

    console = Console(record=True, width=120)
    monkeypatch.setattr(
        "anton.utils.prompt.prompt_or_cancel", AsyncMock(return_value="1")
    )
    await CLIElicitor(console).ask(
        "q1",
        _pick_request(
            prompt="Which [bold]file[/]?",
            options=(
                AskOption(value="/a", label="[dim]a.csv", kind="file", detail="[/]"),
                AskOption(value="/b", label="b [/]", kind="folder"),
            ),
        ),
    )
    text = console.export_text()
    assert "Which [bold]file[/]?" in text
    assert "[dim]a.csv" in text
    assert "b [/]" in text


async def test_cli_pick_valid_number_selects_the_matching_option_value(cli_elicitor, monkeypatch):
    monkeypatch.setattr(
        "anton.utils.prompt.prompt_or_cancel", AsyncMock(return_value="2")
    )
    answer = await cli_elicitor.ask("q1", _pick_request())
    assert answer == AskAnswer(status="answered", values=("/b",))


async def test_cli_pick_non_numeric_reply_is_cancelled(cli_elicitor, monkeypatch):
    monkeypatch.setattr(
        "anton.utils.prompt.prompt_or_cancel", AsyncMock(return_value="foo")
    )
    answer = await cli_elicitor.ask("q1", _pick_request())
    assert answer.status == "cancelled"


async def test_cli_pick_out_of_range_number_is_cancelled(cli_elicitor, monkeypatch):
    monkeypatch.setattr(
        "anton.utils.prompt.prompt_or_cancel", AsyncMock(return_value="99")
    )
    answer = await cli_elicitor.ask("q1", _pick_request())
    assert answer.status == "cancelled"


async def test_cli_pick_escape_is_cancelled(cli_elicitor, monkeypatch):
    monkeypatch.setattr(
        "anton.utils.prompt.prompt_or_cancel", AsyncMock(return_value=None)
    )
    answer = await cli_elicitor.ask("q1", _pick_request())
    assert answer.status == "cancelled"


async def test_cli_pick_with_no_options_is_cancelled(cli_elicitor):
    answer = await cli_elicitor.ask("q1", _pick_request(options=()))
    assert answer.status == "cancelled"


def test_show_question_renders_numbered_options_with_icons():
    from rich.console import Console

    from anton.chat_ui import StreamDisplay

    console = Console(record=True, width=100)
    display = StreamDisplay(console)
    display.show_question(
        AskRequest(
            prompt="Which one?",
            options=(
                AskOption(value="/a", label="a.csv", kind="file"),
                AskOption(value="/b", label="b", kind="folder", detail="12 files"),
            ),
        )
    )
    text = console.export_text()
    assert "Which one?" in text
    assert "1." in text and "a.csv" in text
    assert "2." in text and "b" in text
    assert "12 files" in text
    assert "📄" in text and "📁" in text


def test_show_question_escapes_rich_markup_in_model_controlled_text():
    """Labels and prompts are model-controlled. Unescaped, "[recommended] pg"
    renders as " pg" — the user picks from a mangled list — and a "[/]" anywhere
    raises MarkupError out of show_question, killing the turn while the tool is
    mid-dispatch."""
    from rich.console import Console

    from anton.chat_ui import StreamDisplay

    console = Console(record=True, width=120)
    StreamDisplay(console).show_question(
        AskRequest(
            prompt="Which [bold]one[/] to use?",
            options=(
                AskOption(value="pg", label="[recommended] postgres", detail="[dim]fast[/]"),
                AskOption(value="my", label="mysql [/]"),
            ),
        )
    )
    text = console.export_text()
    assert "[recommended] postgres" in text
    assert "Which [bold]one[/] to use?" in text
    assert "[dim]fast[/]" in text
    assert "mysql [/]" in text


# ─── goal mode: without_ask_user ────────────────────────────────────────


def test_without_ask_user_hides_it_then_restores_it(make_session):
    """An autonomous loop must never block on a human. run_goal_loop gets an
    already-built session, so the registry is the only lever — and the tool has
    to come back, because the interactive chat owns that same session."""
    from anton.commands.goal import without_ask_user

    session = make_session(elicitor=_RecordingElicitor())
    session._build_tools()
    registry = session.tool_registry
    original = next(t for t in registry.get_tool_defs() if t.name == "ask_user")

    with without_ask_user(registry) as stashed:
        assert stashed is original
        assert "ask_user" not in {t["name"] for t in registry.dump()}
        # select_path is untouched — it degrades on its own and cannot block.
        assert "select_path" in {t["name"] for t in registry.dump()}

    restored = next(t for t in registry.dump() if t["name"] == "ask_user")
    assert restored["description"] == original.description


def test_without_ask_user_restores_on_an_exception(make_session):
    from anton.commands.goal import without_ask_user

    session = make_session(elicitor=_RecordingElicitor())
    session._build_tools()
    registry = session.tool_registry
    with pytest.raises(RuntimeError):
        with without_ask_user(registry):
            raise RuntimeError("goal loop died")
    assert "ask_user" in {t["name"] for t in registry.dump()}


def test_without_ask_user_is_a_noop_when_the_tool_is_absent(make_session):
    """Headless sessions never registered it in the first place."""
    from anton.commands.goal import without_ask_user

    session = make_session()  # no elicitor, no console
    session._build_tools()
    before = {t["name"] for t in session.tool_registry.dump()}
    with without_ask_user(session.tool_registry) as stashed:
        assert stashed is None
    assert {t["name"] for t in session.tool_registry.dump()} == before


# ─── prompt discipline ──────────────────────────────────────────────────


def test_neither_discipline_commands_a_tool_that_may_be_absent():
    """The discipline text is injected unconditionally (see prompt_builder), but
    `ask_user` is registered only when an elicitor advertises "choice" — goal
    mode explicitly unregisters it, and headless/telegram runs never had it. So
    the carve-out lives in `ASK_USER_TOOL.prompt`, which is emitted only for
    registered tools, and the discipline text keeps only the unconditional
    STOP-on-text rule."""
    from anton.core.llm.prompts import (
        CONVERSATION_DISCIPLINE_ACT_FIRST,
        CONVERSATION_DISCIPLINE_ASK_FIRST,
    )

    for discipline in (CONVERSATION_DISCIPLINE_ACT_FIRST, CONVERSATION_DISCIPLINE_ASK_FIRST):
        assert "ask_user" not in discipline
        assert "STOP" in discipline
    assert "ask_user" in ASK_USER_TOOL.prompt
    assert "SAME turn" in ASK_USER_TOOL.prompt


def test_ask_first_discipline_scopes_its_later_turn_rule_to_text_questions():
    """The unqualified "ask first, then act in a LATER turn" bullet re-imposed
    exactly the stop that `ask_user` removes, and was the more categorical of
    the two rules. It has to name text as the thing that defers."""
    from anton.core.llm.prompts import CONVERSATION_DISCIPLINE_ASK_FIRST

    later_turn = [
        line
        for line in CONVERSATION_DISCIPLINE_ASK_FIRST.splitlines()
        if "LATER turn" in line
    ]
    assert len(later_turn) == 1
    assert "text" in later_turn[0]


def test_the_ask_user_carve_out_reaches_the_prompt_only_when_the_tool_is_there():
    """End to end through the real builder: with the tool registered the
    carve-out is present; without it, the built prompt never mentions
    `ask_user`."""
    from anton.core.llm.prompt_builder import ChatSystemPromptBuilder, SystemPromptContext

    def _build(tool_defs):
        return ChatSystemPromptBuilder().build(
            conversation_started="2026-07-31T12:00:00+00:00",
            current_datetime="2026-07-31T12:00:00+00:00",
            system_prompt_context=SystemPromptContext(runtime_context="test"),
            proactive_dashboards=False,
            output_dir="",
            tool_defs=tool_defs,
        )

    with_tool = _build([ASK_USER_TOOL])
    assert "ask_user" in with_tool
    assert "SAME turn" in with_tool

    without_tool = _build([])
    assert "ask_user" not in without_tool
