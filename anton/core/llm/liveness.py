"""Which model calls a turn has open, and whether one has gone quiet.

A model can think for minutes before it sends its first token, and a gateway
that hides thinking from a chat-completions client sends nothing at all in that
time. A host that ends a turn after a stretch of silence on its wire cannot
tell that wait from a hung turn. ``ModelCallTracker`` lets it tell them apart.

The session arms one tracker per turn (``ChatSession.model_calls``) and hands
it to ``LLMClient.call_tracker``. Every call the client issues while armed
registers here, marks when it is awaiting the provider, and stamps each event
the provider sends. A host polls ``snapshot()`` and, when it reports a waiting
call and its own wire has been quiet for ``MODEL_WAIT_TICK_S``, writes one
progress line with phase ``MODEL_WAIT_PHASE``. A hung tool, a hung cell or an
open question registers nothing, so silence there still reaches every idle
bound the host keeps.

The tracker also latches the first model-call deadline of the turn
(``record_expiry``). Side paths such as compaction and tool dispatch swallow
exceptions, so without the latch every later call in the turn would wait out a
full deadline of its own. With it, the next call fails as soon as it is issued.

Each call captures the tracker when it is issued. Model work started
fire-and-forget during a turn therefore registers on that turn's tracker and
can keep it reporting a wait. Start such work after the turn ends: the turn's
``finally`` closes the tracker, and a closed tracker reports nothing, opens
nothing and latches nothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from anton.core.llm.provider import ModelCallTimeoutError

#: Progress phase of the host's still-working line. Hosts, the cloud pod and
#: the UI all match this string.
MODEL_WAIT_PHASE = "model_wait"

#: How long a host's wire must be quiet before it writes a still-working line.
#: The same window decides whether a call "is still writing": output inside it
#: that never reached the wire (tool-argument deltas, generate_artifact text).
MODEL_WAIT_TICK_S = 20.0


class ModelCallSnapshot(BaseModel):
    """The oldest model call a turn is waiting on, as a host reports it."""

    model_config = ConfigDict(frozen=True)

    role: str
    open_for_s: float
    quiet_for_s: float
    message: str


@dataclass(eq=False)
class OpenModelCall:
    """One model call the tracker knows about. Internal to the tracker and
    ``LLMClient``; hosts read ``ModelCallSnapshot`` instead. Compared by
    identity, so two calls issued in the same instant stay distinct."""

    role: str
    issued_at: float
    idle_timeout_s: float | None
    last_output_at: float
    produced_output: bool = False
    awaiting: bool = False


def _format_wait(*, seconds: float) -> str:
    # Rounded, not floored: a host's quiet clock starts a moment before the
    # call is issued, so its first tick reads the call at 19.9 s, and a
    # floored "(19s)" would look like it fired early.
    total = max(0, round(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"


class ModelCallTracker:
    """Per-turn record of open model calls, the question gate and the latch."""

    def __init__(self) -> None:
        self._calls: list[OpenModelCall] = []
        self._questions_open = 0
        self._expiry: ModelCallTimeoutError | None = None
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def open(self, *, role: str, idle_timeout_s: float | None) -> OpenModelCall | None:
        """Register a call as it is issued. Returns None once the tracker is closed."""
        if self._closed:
            return None
        now = time.monotonic()
        call = OpenModelCall(
            role=role,
            issued_at=now,
            idle_timeout_s=idle_timeout_s,
            last_output_at=now,
        )
        self._calls.append(call)
        return call

    def output(self, *, call: OpenModelCall) -> None:
        """Stamp an event the provider sent on ``call``."""
        call.last_output_at = time.monotonic()
        call.produced_output = True

    def close_call(self, *, call: OpenModelCall) -> None:
        """Forget ``call``. Safe to call twice."""
        try:
            self._calls.remove(call)
        except ValueError:
            pass

    def snapshot(self) -> ModelCallSnapshot | None:
        """The oldest call waiting on the provider, or None.

        None when the tracker is closed, a question is open, or no call is
        awaiting the provider inside its idle timeout. A call parked between
        events, waiting for its consumer, is not waiting on the provider.
        """
        if self._closed or self._questions_open:
            return None
        now = time.monotonic()
        waiting = [
            c
            for c in self._calls
            if c.awaiting
            and (c.idle_timeout_s is None or now - c.last_output_at < c.idle_timeout_s)
        ]
        if not waiting:
            return None
        oldest = min(waiting, key=lambda c: c.issued_at)
        open_for = now - oldest.issued_at
        quiet_for = now - oldest.last_output_at
        if oldest.produced_output and quiet_for < MODEL_WAIT_TICK_S:
            message = f"The model is still writing ({_format_wait(seconds=open_for)})"
        else:
            message = f"Waiting for the model ({_format_wait(seconds=open_for)})"
        return ModelCallSnapshot(
            role=oldest.role,
            open_for_s=open_for,
            quiet_for_s=quiet_for,
            message=message,
        )

    def record_expiry(self, *, exc: ModelCallTimeoutError) -> None:
        """Latch the turn's first deadline so later calls fail at once."""
        if self._closed or self._expiry is not None:
            return
        self._expiry = exc

    def raise_if_expired(self) -> None:
        """Raise a fresh ``ModelCallTimeoutError`` when a call in this turn
        already ran out its deadline."""
        if self._closed or self._expiry is None:
            return
        first = self._expiry
        raise ModelCallTimeoutError(
            str(first),
            role=first.role,
            model=first.model,
            idle_timeout_s=first.idle_timeout_s,
        )

    def question_opened(self) -> None:
        """An ask_user question is waiting on the user: report no wait."""
        self._questions_open += 1

    def question_closed(self) -> None:
        self._questions_open = max(0, self._questions_open - 1)

    def close(self) -> None:
        """End of turn. Nothing registers, reports or latches after this."""
        self._closed = True
        self._calls.clear()
