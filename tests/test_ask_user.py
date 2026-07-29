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
    validate_request,
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
