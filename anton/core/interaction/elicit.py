"""Mid-turn human interaction: value types, the elicitor strategy, and the
single code path every question takes.

Replaces the narrower ``selection.py``. Two question kinds share one
contract: ``kind="choice"`` (the ``ask_user`` tool — published to the host
so a GUI can render buttons) and ``kind="path"`` (the ``select_path`` tool
— rendered by the elicitor itself, since cowork has no file-browser
widget yet).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from anton.core.llm.provider import (
    StreamAskUser,
    StreamAskUserAnswered,
    StreamTaskProgress,
)

__all__ = [
    "MAX_QUESTIONS_PER_TURN",
    "AskAnswer",
    "AskOption",
    "AskRequest",
    "Elicitor",
    "elicit",
    "validate_request",
]

# Shared across the outer agent and any sub-agent: the user does not care
# which layer is interrupting them. Every question kind draws on this same
# budget — a `select_path` picker spends it too — so "8 questions per turn"
# is not "8 `ask_user` calls per turn". Raised from 3 to 8 alongside
# generate_prd (ENG-969): its phase 2 (show the brief, then up to two
# revise cycles) reserves 3 of these for itself so the confirm step is not
# a one-shot — see generate_prd/state.py's PHASE2_RESERVED_QUESTIONS.
MAX_QUESTIONS_PER_TURN = 8

# One option is nothing to choose; thirty buttons is a UI failure and the
# model should narrow the question instead.
MIN_OPTIONS = 2
MAX_OPTIONS = 10


@dataclass(frozen=True, slots=True)
class AskOption:
    """One selectable answer.

    ``value`` is what goes back to the LLM, ``label`` is what the user sees.
    ``kind`` ("file" | "folder" | "") drives the picker icon for path
    questions — a mixed listing would otherwise be indistinguishable.
    """

    value: str
    label: str
    detail: str = ""
    kind: str = ""
    style: str = "default"  # default | primary | destructive


@dataclass(frozen=True, slots=True)
class AskRequest:
    """A question awaiting a human answer.

    ``timeout_s`` is a deployment concern: the elicitor declares it and the
    caller stamps it here, so the published event can carry it.
    """

    prompt: str
    kind: str = "choice"  # "choice" | "path"
    timeout_s: int | None = None
    # kind="choice"
    options: tuple[AskOption, ...] = ()
    select: str = "one"  # "one" | "many"
    allow_custom: bool = True
    # The value of one of `options`, chosen when the user presses Enter
    # with no input instead of picking. Empty string (default) means "no
    # default" — a blank answer is `cancelled`, as it always was. Only
    # meaningful for orchestrator code that builds `AskRequest` directly
    # (`ask_user`'s own JSON schema has no way to set this — the LLM never
    # picks a default on the human's behalf).
    default_value: str = ""
    # kind="path"
    path_mode: str = "pick"  # "pick" | "browse"
    path_kind: str = "any"  # "file" | "folder" | "any"
    root: str = ""


@dataclass(frozen=True, slots=True)
class AskAnswer:
    """The outcome of one question.

    ``unavailable`` and ``limit`` are internal: callers map them onto their
    own error shape and they never reach the user. ``values`` and ``text``
    may both be set — picking two options and adding a third by hand is a
    legitimate answer to a multi-select question.
    """

    # `error` appears only on the retirement event emitted when the elicitor's
    # ask() raises; elicit() re-raises rather than returning it.
    status: str  # answered | cancelled | timeout | unavailable | limit | error
    values: tuple[str, ...] = ()
    text: str = ""


class Elicitor(Protocol):
    """Strategy for surfacing a question and awaiting the answer.

    A Protocol on purpose: hosts satisfy it by shape. ``begin`` opens the
    answer channel, ``ask`` waits, ``end`` closes it — the three are paired
    by ``elicit()``, never by the implementation.
    """

    supported_kinds: tuple[str, ...]
    answer_hint: str  # appended to ask_user's description
    timeout_s: int | None

    async def begin(self, question_id: str, request: AskRequest) -> None: ...

    async def ask(self, question_id: str, request: AskRequest) -> AskAnswer: ...

    async def end(self, question_id: str) -> None: ...


def validate_request(request: AskRequest) -> bool:
    """Whether *request* is well-formed enough to show to a human.

    Lives here rather than in the tool handler because the orchestrator
    builds requests in Python and calls ``elicit()`` directly: a check that
    sits in one caller is a check the other caller does not get.
    """
    if not (request.prompt or "").strip():
        return False
    if request.kind != "choice":
        return True
    if not MIN_OPTIONS <= len(request.options) <= MAX_OPTIONS:
        return False
    if request.select not in ("one", "many"):
        return False
    values = [option.value for option in request.options]
    if any(not value for value in values):
        return False
    if request.default_value and request.default_value not in values:
        return False
    return len(set(values)) == len(values)


async def elicit(session, question_id: str, request: AskRequest) -> AskAnswer:
    """Ask *request* and wait for the answer. The only path a question takes.

    Called by the ``ask_user`` / ``select_path`` handlers and directly by
    orchestrator code in ``generate_artifact``. Everything that decides
    "there will be no question" happens before ``begin()``, so a card is
    never published for a question that cannot be answered.
    """
    elicitor = getattr(session, "elicitor", None)
    if elicitor is None or request.kind not in elicitor.supported_kinds:
        return AskAnswer(status="unavailable")
    if not validate_request(request):
        return AskAnswer(status="unavailable")
    # Only published kinds need a live stream. A path picker renders in the
    # terminal, so gating it on the emitter would break select_path in every
    # host that has none — including the non-streaming turn().
    if request.kind == "choice" and session.emitter is None:
        return AskAnswer(status="unavailable")
    # Checked before begin(): if this ran after, a question over budget would
    # still open a channel and publish a card that can never be answered —
    # nothing emits StreamAskUserAnswered to retire it, and the "expired"
    # state does not apply while the run is still alive.
    if session.question_count >= MAX_QUESTIONS_PER_TURN:
        return AskAnswer(status="limit")
    session.question_count += 1

    # begin() sits outside the try/finally on purpose: if it fails there is
    # no channel to end, so end() must not run. The budget above is
    # intentionally consumed even if this then fails — a future editor
    # should not "fix" that by decrementing it back.
    #
    # It also has to run before anything below it: if an answer arrived
    # before the channel existed, the host would 404 it, the frontend would
    # leave answer mode, and the turn would hang until the timeout.
    await elicitor.begin(question_id, request)  # open the channel FIRST
    watcher = getattr(session, "escape_watcher", None)
    started = time.monotonic()
    try:
        if watcher is not None:
            # Paused before any emit so an emit() raising can't leave a
            # pause() unmatched by a resume().
            watcher.pause()
        # Stops the CLI spinner before anything is printed into it: Rich
        # Live is still running, so printed options would collide with the
        # spinner otherwise.
        await session.emit(StreamTaskProgress(phase="interactive", message=""))
        # Symmetric on purpose: retiring a card that was never published
        # would hand the host an "answered" event for an id it has never
        # seen. Path questions render in the terminal and publish nothing,
        # so they stay silent on both halves.
        published = request.kind == "choice"
        if published:
            await session.emit(StreamAskUser(id=question_id, request=request))
        try:
            answer = await elicitor.ask(question_id, request)
        except Exception:
            # A raising ask() still leaves a published card with live buttons,
            # and the turn continues (handle_ask_user does not catch, so this
            # only becomes a tool failure). The frontend retires a card on
            # StreamAskUserAnswered alone — its "expired" fallback keys on
            # there being no in-flight run, which is false here — so without
            # this the user clicks and gets a 404 for a question the agent has
            # already moved past. "error" is a pinned tool-result status, so
            # this needs no contract change.
            #
            # `Exception`, not `BaseException`: a cancellation unwinds the whole
            # run, and the frontend's no-in-flight-run fallback already retires
            # the card in that case.
            if published:
                await session.emit(
                    StreamAskUserAnswered(
                        id=question_id, answer=AskAnswer(status="error")
                    )
                )
            raise
        if published:
            await session.emit(StreamAskUserAnswered(id=question_id, answer=answer))
        return answer
    finally:
        session.answer_wait_s += time.monotonic() - started
        if watcher is not None:
            watcher.resume()
        await elicitor.end(question_id)
