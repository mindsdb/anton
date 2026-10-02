"""CloudElicitor: questions answered by lines the controller writes to stdin."""

from __future__ import annotations

import asyncio
import time

import pytest

from anton.cloud_turn.elicitor import (
    DEFAULT_TIMEOUT_S,
    TIMEOUT_ENV,
    CloudElicitor,
    ask_user_answered_event,
    ask_user_event,
)
from anton.core.interaction.elicit import AskAnswer, AskOption, AskRequest
from anton.core.llm.provider import StreamAskUser, StreamAskUserAnswered


def _choice(**over) -> AskRequest:
    base = dict(
        prompt="Which database?",
        options=(AskOption(value="pg", label="postgres"), AskOption(value="my", label="mysql")),
    )
    base.update(over)
    return AskRequest(**base)


def _answer(question_id="q1", answer_id="a1", values=(), text="", skipped=False) -> dict:
    return {"question_id": question_id, "answer_id": answer_id,
            "values": list(values), "text": text, "skipped": skipped}


async def _open(elicitor, question_id="q1", request=None):
    request = request or _choice()
    await elicitor.begin(question_id, request)
    task = asyncio.create_task(elicitor.ask(question_id, request))
    await asyncio.sleep(0)
    return task


async def test_a_valid_answer_resolves_the_question():
    emitted = []
    elicitor = CloudElicitor(emitted.append)
    task = await _open(elicitor)
    elicitor.deliver(_answer(values=["pg"]))
    assert await task == AskAnswer(status="answered", values=("pg",), text="")
    assert elicitor.accepted_answer_id("q1") == "a1"
    assert emitted == []


async def test_free_text_answer():
    elicitor = CloudElicitor(lambda e: None)
    task = await _open(elicitor)
    elicitor.deliver(_answer(text="duckdb"))
    assert await task == AskAnswer(status="answered", values=(), text="duckdb")


async def test_skip_cancels_and_records_the_answer_id():
    elicitor = CloudElicitor(lambda e: None)
    task = await _open(elicitor)
    elicitor.deliver(_answer(skipped=True))
    assert await task == AskAnswer(status="cancelled")
    assert elicitor.accepted_answer_id("q1") == "a1"


async def test_invalid_option_is_rejected_and_the_question_stays_open():
    emitted = []
    elicitor = CloudElicitor(emitted.append)
    task = await _open(elicitor)
    elicitor.deliver(_answer(answer_id="bad", values=["sqlite"]))
    assert emitted == [{"kind": "ask_user_answer_rejected", "id": "q1",
                        "answer_id": "bad", "reason": "invalid_option"}]
    assert not task.done()
    elicitor.deliver(_answer(answer_id="good", values=["my"]))
    assert (await task).values == ("my",)
    assert elicitor.accepted_answer_id("q1") == "good"


async def test_unknown_question_is_not_found():
    emitted = []
    CloudElicitor(emitted.append).deliver(_answer(question_id="nope"))
    assert emitted[0]["reason"] == "not_found"
    assert emitted[0]["id"] == "nope"


async def test_second_answer_is_already_answered():
    emitted = []
    elicitor = CloudElicitor(emitted.append)
    task = await _open(elicitor)
    elicitor.deliver(_answer(answer_id="a1", values=["pg"]))
    elicitor.deliver(_answer(answer_id="a2", values=["my"]))
    assert (await task).values == ("pg",)
    assert emitted == [{"kind": "ask_user_answer_rejected", "id": "q1",
                        "answer_id": "a2", "reason": "already_answered"}]
    assert elicitor.accepted_answer_id("q1") == "a1"


async def test_timeout_then_a_late_answer_is_not_found():
    emitted = []
    elicitor = CloudElicitor(emitted.append)
    elicitor.timeout_s = 0.05
    task = await _open(elicitor)
    assert await task == AskAnswer(status="timeout")
    elicitor.deliver(_answer(values=["pg"]))
    assert emitted[0]["reason"] == "not_found"
    assert elicitor.accepted_answer_id("q1") is None


async def test_late_delivery_near_timeout_is_not_lost():
    """deliver() resolves the future just before the deadline, then the loop
    stalls (a blocking call elsewhere) past it. asyncio.wait_for's cancellation
    can still fire on the shielded future even though it is already done —
    ask() must return the delivered answer instead of reporting a lost
    timeout, and the accepted answer_id must match."""
    emitted = []
    elicitor = CloudElicitor(emitted.append)
    elicitor.timeout_s = 0.2
    task = await _open(elicitor)
    await asyncio.sleep(0.18)
    elicitor.deliver(_answer(values=["pg"]))
    time.sleep(0.05)  # block the loop past the 0.2s deadline
    assert await task == AskAnswer(status="answered", values=("pg",), text="")
    assert elicitor.accepted_answer_id("q1") == "a1"


async def test_an_answer_after_end_is_not_found():
    emitted = []
    elicitor = CloudElicitor(emitted.append)
    task = await _open(elicitor)
    elicitor.deliver(_answer(values=["pg"]))
    await task
    await elicitor.end("q1")
    elicitor.deliver(_answer(answer_id="late", values=["pg"]))
    assert emitted[-1]["reason"] == "not_found"
    assert elicitor.accepted_answer_id("q1") == "a1"  # kept for the answered event


def test_timeout_from_env(monkeypatch):
    monkeypatch.setenv(TIMEOUT_ENV, "42")
    assert CloudElicitor(lambda e: None).timeout_s == 42


@pytest.mark.parametrize("raw", ["", "junk", "0", "-5"])
def test_bad_timeout_env_falls_back(monkeypatch, raw):
    monkeypatch.setenv(TIMEOUT_ENV, raw)
    assert CloudElicitor(lambda e: None).timeout_s == DEFAULT_TIMEOUT_S


def test_protocol_shape():
    elicitor = CloudElicitor(lambda e: None)
    assert elicitor.supported_kinds == ("choice",)
    assert "buttons" in elicitor.answer_hint


def test_ask_user_event_shape():
    request = _choice(timeout_s=300, select="many", allow_custom=False)
    assert ask_user_event(StreamAskUser(id="q1", request=request)) == {
        "kind": "ask_user", "id": "q1", "prompt": "Which database?",
        "options": [{"value": "pg", "label": "postgres", "detail": ""},
                    {"value": "my", "label": "mysql", "detail": ""}],
        "select": "many", "allow_custom": False, "timeout_s": 300,
    }


async def test_ask_user_answered_event_carries_the_accepted_answer_id():
    elicitor = CloudElicitor(lambda e: None)
    task = await _open(elicitor)
    elicitor.deliver(_answer(values=["pg"]))
    answer = await task
    assert ask_user_answered_event(StreamAskUserAnswered(id="q1", answer=answer), elicitor) == {
        "kind": "ask_user_answered", "id": "q1", "status": "answered",
        "values": ["pg"], "text": "", "answer_id": "a1",
    }


def test_ask_user_answered_event_without_an_elicitor_has_no_answer_id():
    event = StreamAskUserAnswered(id="q1", answer=AskAnswer(status="timeout"))
    assert ask_user_answered_event(event, None)["answer_id"] is None
