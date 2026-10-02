"""Elicitor for cloud turns.

A question leaves the pod as an ``ask_user`` event on the protocol stdout (via
``elicit()`` → ``StreamAskUser`` → ``stream_turn``). The user's answer comes
back as a line on stdin: cowork-server queues it in Redis, scratchpad-controller
writes it to the exec session's stdin, and ``cloud_turn/stdin.py`` hands it to
``CloudElicitor.deliver``. The pod is the only place that decides whether an
answer closes its question, so every rejection is reported back on the wire
with the answer's id: cowork-server's ``/answer`` waits for exactly that.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

from anton.core.interaction.elicit import (
    GUI_ANSWER_HINT,
    AskAnswer,
    AskRequest,
    validate_answer,
)
from anton.core.llm.provider import StreamAskUser, StreamAskUserAnswered

logger = logging.getLogger(__name__)

TIMEOUT_ENV = "ANTON_CLOUD_ASK_USER_TIMEOUT_SECONDS"
#: Set on the pod by scratchpad-controller (`ask_user_timeout_seconds`). Must
#: stay below both cowork-server idle timers, COWORK_TURN_REPLY_IDLE_TIMEOUT_SECONDS
#: and COWORK_MAX_TURN_IDLE_SECONDS (600 s each): while a question is open the
#: turn produces no reply and no buffer record, so both count down.
DEFAULT_TIMEOUT_S = 300


def _timeout_from_env() -> int:
    try:
        value = int(os.environ.get(TIMEOUT_ENV, ""))
    except ValueError:
        return DEFAULT_TIMEOUT_S
    return value if value > 0 else DEFAULT_TIMEOUT_S


@dataclass
class _Question:
    request: AskRequest
    future: asyncio.Future
    # Set once `ask()` gave up waiting: the question no longer takes answers
    # even though `end()` has not removed it yet.
    closed: bool = False


class CloudElicitor:
    supported_kinds = ("choice",)
    answer_hint = GUI_ANSWER_HINT

    def __init__(self, emit: Callable[[dict], None]) -> None:
        self._emit = emit
        self.timeout_s = _timeout_from_env()
        self._questions: dict[str, _Question] = {}
        # Outlives `end()`: the answered event that reads it is drained from
        # the turn's event queue after `elicit()` has already ended the question.
        self._accepted: dict[str, str] = {}

    async def begin(self, question_id: str, request: AskRequest) -> None:
        self._questions[question_id] = _Question(
            request, asyncio.get_running_loop().create_future()
        )

    async def ask(self, question_id: str, request: AskRequest) -> AskAnswer:
        question = self._questions[question_id]
        timeout = request.timeout_s or self.timeout_s
        try:
            return await asyncio.wait_for(asyncio.shield(question.future), timeout)
        except TimeoutError:
            if question.future.done() and not question.future.cancelled():
                return question.future.result()  # the answer landed as the timer fired
            question.closed = True
            return AskAnswer(status="timeout")

    async def end(self, question_id: str) -> None:
        question = self._questions.pop(question_id, None)
        if question is not None and not question.future.done():
            question.future.cancel()  # timed out: nothing awaits it any more

    def accepted_answer_id(self, question_id: str) -> str | None:
        return self._accepted.get(question_id)

    def deliver(self, answer: dict) -> None:
        """Apply one answer line. Runs on the event loop thread."""
        question_id = answer["question_id"]
        answer_id = answer["answer_id"]
        question = self._questions.get(question_id)
        if question is None or question.closed:
            self._reject(question_id, answer_id, "not_found")
            return
        if question.future.done():
            self._reject(question_id, answer_id, "already_answered")
            return
        if answer["skipped"]:
            result = AskAnswer(status="cancelled")
        else:
            values = tuple(answer["values"])
            text = answer["text"].strip()
            if validate_answer(question.request, values, text) is not None:
                self._reject(question_id, answer_id, "invalid_option")
                return
            result = AskAnswer(status="answered", values=values, text=text)
        self._accepted[question_id] = answer_id
        question.future.set_result(result)

    def _reject(self, question_id: str, answer_id: str, reason: str) -> None:
        logger.info(
            "ask_user answer rejected question_id=%s answer_id=%s reason=%s",
            question_id, answer_id, reason,
        )
        self._emit({
            "kind": "ask_user_answer_rejected",
            "id": question_id,
            "answer_id": answer_id,
            "reason": reason,
        })


def ask_user_event(event: StreamAskUser) -> dict:
    request = event.request
    return {
        "kind": "ask_user",
        "id": event.id,
        "prompt": request.prompt,
        "options": [
            {"value": option.value, "label": option.label, "detail": option.detail}
            for option in request.options
        ],
        "select": request.select,
        "allow_custom": request.allow_custom,
        "timeout_s": request.timeout_s,
    }


def ask_user_answered_event(
    event: StreamAskUserAnswered, elicitor: CloudElicitor | None
) -> dict:
    answer = event.answer
    return {
        "kind": "ask_user_answered",
        "id": event.id,
        "status": answer.status,
        "values": list(answer.values),
        "text": answer.text,
        "answer_id": elicitor.accepted_answer_id(event.id) if elicitor else None,
    }
