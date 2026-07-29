"""Mid-turn human interaction: value types, the elicitor strategy, and the
single code path every question takes.

Replaces the narrower ``selection.py``. Two question kinds share one
contract: ``kind="choice"`` (the ``ask_user`` tool — published to the host
so a GUI can render buttons) and ``kind="path"`` (the ``select_path`` tool
— rendered by the elicitor itself, since cowork has no file-browser
widget yet).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

__all__ = [
    "MAX_QUESTIONS_PER_TURN",
    "AskAnswer",
    "AskOption",
    "AskRequest",
    "Elicitor",
    "validate_request",
]

# Shared across the outer agent and any sub-agent: the user does not care
# which layer is interrupting them.
MAX_QUESTIONS_PER_TURN = 3

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

    status: str  # answered | cancelled | timeout | unavailable | limit
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
    return len(set(values)) == len(values)
