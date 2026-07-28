"""Drive one buffered anton turn to completion and emit terminal events.

Buffered: consume ``turn_stream()`` (not ``turn()`` — only the streaming path
persists history) to the end, then derive the result from ``session.history``.
No live token streaming yet; that's a later protocol extension.

Event contract: every VALID request emits ``turn.started`` (sequence 1) then
exactly ONE terminal event (sequence 2). An invalid request is rejected upstream
(``__main__``) with a single ``turn.failed`` and no ``turn.started``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any, Callable

from anton.cloud_turn.errors import DeadlineExceededError, classify_error
from anton.cloud_turn.messages import final_assistant_text, turn_output_messages
from anton.cloud_turn.protocol import (
    ErrorCodeV1,
    MessageV1,
    TurnCompletedV1,
    TurnErrorV1,
    TurnFailedV1,
    TurnRequestV1,
    TurnStartedV1,
)
from anton.cloud_turn.session import build_cloud_chat_session

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 3

Emit = Callable[[object], None]
SessionBuilder = Callable[[TurnRequestV1], Any]


class _Sequencer:
    """Hands out 1-based monotonic event sequence numbers."""

    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        self._n += 1
        return self._n


def _now_ms() -> int:
    return int(time.time() * 1000)


def _remaining_seconds(request: TurnRequestV1, now_ms: int) -> float | None:
    """Soft timeout from the absolute deadline. None = no inner deadline.

    Raises :class:`DeadlineExceededError` immediately if the deadline has
    already passed."""
    if request.deadline_unix_ms is None:
        return None
    remaining_ms = request.deadline_unix_ms - now_ms
    if remaining_ms <= 0:
        raise DeadlineExceededError(
            f"deadline passed {abs(remaining_ms)}ms before turn start"
        )
    return remaining_ms / 1000


def _trace_metadata(request: TurnRequestV1) -> dict[str, str]:
    return {
        k: v
        for k, v in {
            "run_id": request.run_id,
            "attempt_id": request.attempt_id,
            "conversation_id": request.conversation_id,
            "organization_id": request.organization_id,
            "user_id": request.user_id,
            "workspace_id": request.workspace_id,
        }.items()
        if v is not None
    }


async def _drive_turn(
    request: TurnRequestV1, session: Any
) -> tuple[str, list[MessageV1]]:
    """Run the turn to completion; return (final_text, output_messages).

    ``output_messages`` are only the messages generated this turn (assistant
    text, tool calls, tool results, final assistant message) — never the input
    history. ``final_text`` is the last assistant message's text.
    """
    # Identity anchor (NOT a length) — turn_stream may compact/rewrite history
    # mid-turn. We hold the pre-turn message objects alive across the whole turn
    # so their ids can't be recycled by messages created during compaction.
    pre_turn_messages = list(session.history)
    async for _event in session.turn_stream(
        request.input, trace_metadata=_trace_metadata(request)
    ):
        # Buffered: the generator exhausting is the turn-ended signal. We derive
        # the result from history, so intermediate stream events are just drive.
        pass
    output_messages = turn_output_messages(session.history, pre_turn_messages)
    return final_assistant_text(output_messages), output_messages


async def _close(session: Any) -> None:
    close = getattr(session, "close", None)
    if close is None:
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.warning("cloud session close failed (non-fatal)", exc_info=True)


async def run_turn(
    request: TurnRequestV1,
    emit: Emit,
    session_builder: SessionBuilder | None = None,
    now_ms: int | None = None,
) -> int:
    """Run one turn, emitting ``turn.started`` then exactly one terminal event.

    Returns an exit code (``EXIT_OK`` / ``EXIT_FAILED``).
    """
    builder = session_builder or build_cloud_chat_session
    seq = _Sequencer()
    emit(
        TurnStartedV1(
            run_id=request.run_id, attempt_id=request.attempt_id, sequence=seq.next()
        )
    )
    session: Any = None
    try:
        # Deadline first: fail immediately if it already passed (before any work).
        remaining_s = _remaining_seconds(request, now_ms if now_ms is not None else _now_ms())
        session = builder(request)
        if remaining_s is not None:
            final_text, output_messages = await asyncio.wait_for(
                _drive_turn(request, session), timeout=remaining_s
            )
        else:
            final_text, output_messages = await _drive_turn(request, session)
        emit(
            TurnCompletedV1(
                run_id=request.run_id,
                attempt_id=request.attempt_id,
                sequence=seq.next(),
                final_text=final_text,
                output_messages=output_messages,
            )
        )
        return EXIT_OK
    except asyncio.TimeoutError:
        logger.warning("cloud turn timed out run_id=%s", request.run_id)
        emit(
            TurnFailedV1(
                run_id=request.run_id,
                attempt_id=request.attempt_id,
                sequence=seq.next(),
                error=TurnErrorV1(
                    code=ErrorCodeV1.DEADLINE_EXCEEDED,
                    message="turn exceeded its deadline",
                    retryable=True,
                ),
            )
        )
        return EXIT_FAILED
    except Exception as exc:
        # Full traceback → stderr only (may contain secrets). The wire carries a
        # short, credential-scrubbed, structured error.
        logger.exception("cloud turn failed run_id=%s", request.run_id)
        emit(
            TurnFailedV1(
                run_id=request.run_id,
                attempt_id=request.attempt_id,
                sequence=seq.next(),
                error=classify_error(exc),
            )
        )
        return EXIT_FAILED
    finally:
        if session is not None:
            await _close(session)
