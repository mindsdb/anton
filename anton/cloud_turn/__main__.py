"""`python -m anton.cloud_turn` - the sandbox-pod turn entrypoint.

Contract (matches scratchpad-controller + cowork-server):
  stdin : ONE newline-terminated TurnRequestV1 JSON line (controller closes stdin)
  stdout: JSONL events - `delta` / `turn_completed` / `turn_failed`, nothing else
  stderr: diagnostic logs + full tracebacks
  exit  : 0 (the controller detects the terminal from the event, not the code)

stdout is isolated at the OS file-descriptor level: the real FD 1 is duplicated
to a private, non-inheritable descriptor used only for protocol events, and
process FD 1 is redirected to stderr. So `print()`, `os.write(1, ...)`,
native-library writes, and child-process stdout can never corrupt the stream.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import os
import sys
import time

from anton.cloud_turn.contract import TurnRequestV1
from anton.cloud_turn.session import build_cloud_chat_session, drain_pending_memory

logger = logging.getLogger(__name__)

#: Bound step-event payloads (tool args / results) on the wire. Matches the
#: cap cowork's SSE formatter applies to the same content.
MAX_STEP_CHARS = 65536
MAX_PROGRESS_CHARS = 2000
#: Per-string-field cap when shrinking a cell-result JSON to fit the wire.
RESULT_FIELD_CAP = 16384
#: Rate limit for repeat progress events on the wire — agent code can call
#: progress() in a tight loop; step-creating/closing phases are never dropped.
#: Matches cowork's SSE-side PROGRESS_THROTTLE.
PROGRESS_WIRE_INTERVAL = 0.25

_TRUNCATION_MARKER = "\n[… truncated …]\n"


def _clip_result_content(content: str, cap: int = MAX_STEP_CHARS) -> str:
    """Bound a tool result without losing its verdict.

    Exec results are cell JSON whose `error` field decides ok/timeout/error
    downstream (cowork's classify_cell_status parses it structurally), so
    shrink the bulky fields and keep the JSON — and `error` — intact. A blind
    prefix cut made long failed cells render as successful. Non-JSON content
    falls back to a middle clip so trailing error text still survives."""
    if len(content) <= cap:
        return content
    try:
        cell = json.loads(content)
    except (ValueError, TypeError):
        cell = None
    if isinstance(cell, dict):
        for key, val in cell.items():
            if key != "error" and isinstance(val, str) and len(val) > RESULT_FIELD_CAP:
                keep = RESULT_FIELD_CAP // 2
                cell[key] = val[:keep] + _TRUNCATION_MARKER + val[-keep:]
        clipped = json.dumps(cell)
        if len(clipped) <= cap:
            return clipped
        content = clipped
    head = cap * 3 // 4
    tail = cap - head - len(_TRUNCATION_MARKER)
    return content[:head] + _TRUNCATION_MARKER + content[-tail:]

#: Bound the single request line so a malformed/huge stdin can't exhaust memory.
MAX_REQUEST_BYTES = 10 * 1024 * 1024
#: Keep wire error strings short and log-safe.
MAX_ERROR_MESSAGE_CHARS = 300


def _scrub(exc: Exception) -> str:
    """Short, credential-scrubbed error string for the wire. Full traceback
    stays on stderr (logged by the caller)."""
    from anton.utils.datasources import scrub_credentials

    text = scrub_credentials(f"{type(exc).__name__}: {exc}")
    if len(text) > MAX_ERROR_MESSAGE_CHARS:
        text = text[: MAX_ERROR_MESSAGE_CHARS - 1] + "…"
    return text


@contextlib.contextmanager
def _isolated_protocol_stdout():
    """OS-level stdout isolation. Yields ``emit(event: dict)`` writing JSONL to
    the saved protocol descriptor; everything else (FD 1) goes to stderr."""
    sys.stdout.flush()
    sys.stderr.flush()
    stderr_fd = sys.stderr.fileno()

    protocol_fd = os.dup(1)
    os.set_inheritable(protocol_fd, False)  # children never inherit the protocol channel
    os.dup2(stderr_fd, 1)                   # any write to fd 1 now lands on stderr
    saved_sys_stdout = sys.stdout
    sys.stdout = sys.stderr
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    def emit(event: dict) -> None:
        data = (json.dumps(event) + "\n").encode("utf-8")
        view = memoryview(data)
        while view:  # os.write may partial-write; loop until fully flushed
            n = os.write(protocol_fd, view)
            view = view[n:]

    try:
        yield emit
    finally:
        with contextlib.suppress(Exception):
            sys.stderr.flush()
        sys.stdout = saved_sys_stdout
        os.close(protocol_fd)


async def _close(session) -> None:
    close = getattr(session, "close", None)
    if close is None:
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.warning("cloud session close failed (non-fatal)", exc_info=True)


async def _settle_memory(session) -> None:
    """Await memory encoding the session registered. Optional like ``close``, and
    best-effort: a lost memory must never cost the turn its reply."""
    settle = getattr(session, "settle_memory_writes", None)
    if settle is None:
        return
    try:
        await settle()
    except Exception:
        logger.warning("memory settle failed (non-fatal)", exc_info=True)


async def stream_turn(raw_line: str, emit, session_builder=None) -> None:
    """Parse the request, run one turn, and emit exactly one terminal event.

    Streaming: assistant text is emitted as ``delta`` events as it arrives, then
    a bare ``turn_completed``. Any failure (parse or turn) -> one ``turn_failed``
    with a scrubbed error string.

    A background ticker emits a bare ``heartbeat`` event every
    ``ANTON_CLOUD_TURN_HEARTBEAT_SECONDS`` (default 5) so long-but-alive turns
    keep the controller's stall timer reset. It is cancelled once the turn
    ends. ``emit`` is a synchronous ``os.write`` with no ``await`` inside, so
    the ticker and the delta loop cannot interleave mid-line - no lock needed.
    """
    from anton.core.llm.provider import (
        StreamComplete,
        StreamContextCompacted,
        StreamTaskProgress,
        StreamTextDelta,
        StreamToolResult,
        StreamToolUseDelta,
        StreamToolUseEnd,
        StreamToolUseStart,
    )

    builder = session_builder or build_cloud_chat_session
    session = None
    interval = float(os.environ.get("ANTON_CLOUD_TURN_HEARTBEAT_SECONDS", "5"))

    async def _heartbeat() -> None:
        try:
            while True:
                await asyncio.sleep(interval)
                emit({"kind": "heartbeat"})
        except asyncio.CancelledError:
            return

    hb = asyncio.create_task(_heartbeat())
    try:
        req = TurnRequestV1.from_json(raw_line)
        session = builder(req)
        # Tool-call args accumulate here and ship once on tool_end, so the
        # wire carries one event per call instead of one per args token.
        # Accumulation stops at the wire cap — a runaway call can't grow
        # pod memory with args that would be clipped anyway.
        tool_args: dict[str, list[str]] = {}
        tool_args_len: dict[str, int] = {}
        seen_tool_progress: set[str] = set()
        last_progress_wire = 0.0
        async for event in session.turn_stream(req.input):
            if isinstance(event, StreamTextDelta):
                emit({"kind": "delta", "text": event.text or ""})
            # Step events go on the wire for cowork's thinking/steps UI;
            # stderr keeps the controller-log narration.
            elif isinstance(event, StreamToolUseStart):
                logger.info("tool call: %s", event.name)
                tool_args[event.id] = []
                emit({"kind": "tool_start", "id": event.id, "name": event.name})
            elif isinstance(event, StreamToolUseDelta):
                parts = tool_args.get(event.id)
                if parts is not None and tool_args_len.get(event.id, 0) < MAX_STEP_CHARS:
                    parts.append(event.json_delta)
                    tool_args_len[event.id] = (
                        tool_args_len.get(event.id, 0) + len(event.json_delta)
                    )
            elif isinstance(event, StreamToolUseEnd):
                args = "".join(tool_args.pop(event.id, []))
                tool_args_len.pop(event.id, None)
                emit({"kind": "tool_end", "id": event.id,
                      "args": args[:MAX_STEP_CHARS]})
            elif isinstance(event, StreamTaskProgress):
                logger.info("progress [%s]: %s", event.phase, event.message)
                phase = event.phase or ""
                first_progress = (phase == "tool_progress" and event.id
                                  and event.id not in seen_tool_progress)
                if first_progress:
                    seen_tool_progress.add(event.id)
                # Step-creating/closing phases and the first tool_progress per
                # id must never be dropped (they open/close renderer steps);
                # the rest is rate-limited.
                always = bool(first_progress) or phase in (
                    "scratchpad_start", "scratchpad_done", "tool_done")
                now = time.monotonic()
                if always or now - last_progress_wire >= PROGRESS_WIRE_INTERVAL:
                    if not always:
                        last_progress_wire = now
                    emit({"kind": "progress", "phase": phase,
                          "message": (event.message or "")[:MAX_PROGRESS_CHARS],
                          "eta_seconds": event.eta_seconds,
                          "id": event.id, "ok": event.ok})
            elif isinstance(event, StreamToolResult):
                action = f" action={event.action}" if event.action else ""
                logger.info("tool result: %s%s (%d chars)",
                            event.name, action, len(event.content or ""))
                emit({"kind": "tool_result", "id": event.id, "name": event.name,
                      "action": event.action,
                      "content": _clip_result_content(event.content or "")})
            elif isinstance(event, StreamContextCompacted):
                logger.info("context compacted: %s", event.message)
                emit({"kind": "compacted", "message": event.message})
            elif isinstance(event, StreamComplete):
                logger.info("model response complete")
                # Round boundary: cowork's formatter separates rounds with a
                # paragraph break unless the round was truncated mid-sentence.
                emit({"kind": "round_end",
                      "stop_reason": event.response.stop_reason,
                      "had_tool_calls": bool(event.response.tool_calls)})
        await _settle_memory(session)
        entries = drain_pending_memory(session)
        if entries:
            # Before the terminal event: cowork persists on this, then stops reading.
            logger.info("emitting %d memory entr(ies)", len(entries))
            emit({"kind": "memory", "entries": entries})
        logger.info("cloud turn completed")
        emit({"kind": "turn_completed"})
    except Exception as exc:
        # Full traceback -> stderr only; wire carries a short scrubbed string.
        logger.exception("cloud turn failed")
        emit({"kind": "turn_failed", "error": _scrub(exc)})
    finally:
        hb.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await hb
        if session is not None:
            await _close(session)


def main(argv: list[str] | None = None) -> int:
    with _isolated_protocol_stdout() as emit:
        # One bounded line (the controller writes a single JSON line + \n, then
        # closes stdin). ``readline`` returns on the newline without blocking.
        raw_line = sys.stdin.readline(MAX_REQUEST_BYTES + 1)
        asyncio.run(stream_turn(raw_line, emit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
