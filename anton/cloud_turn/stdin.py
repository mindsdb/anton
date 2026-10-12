"""Answer lines on the cloud turn's stdin.

The controller writes the TurnRequestV1 line first and, for an interactive
turn, one line per answer the user gives (format in contract.py). The request
line and the answer lines must come through the SAME buffered reader: a second
reader opened later would miss whatever bytes the first one had already
buffered past the request line.

That shared reader must be a private one over a dup of fd 0, never
``sys.stdin``/``sys.stdin.buffer`` itself: the reader here runs in a daemon
thread and blocks in ``readline()`` for the life of the turn (the live-pod
exec never closes stdin), so at interpreter shutdown ``sys.stdin``'s own
buffer would still be busy. Finalizing ``sys.stdin`` then needs that buffer's
lock, the daemon thread already holds it, and the process aborts ("Fatal
Python error: _enter_buffered_busy ... at interpreter shutdown"). A reader
``sys`` never references sidesteps that - finalization has nothing of its own
to close here.

A thread rather than ``loop.connect_read_pipe``: it works with any stdin (a
pipe in the pod, a file or BytesIO in tests) and survives EOF without special
handling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Callable, Iterator
from typing import BinaryIO

logger = logging.getLogger(__name__)

MAX_ANSWER_LINE_BYTES = 64 * 1024


def iter_lines(stream: BinaryIO, limit: int) -> Iterator[bytes]:
    """Complete lines of at most *limit* bytes, without the newline.

    ``readline(n)`` hands an over-long line back in n-byte pieces, each of
    which would then fail as its own "invalid JSON" line. Such a line is
    skipped whole instead, and logged once.
    """
    while True:
        chunk = stream.readline(limit + 1)
        if not chunk:
            return
        if chunk.endswith(b"\n"):
            yield chunk[:-1]
            continue
        if len(chunk) <= limit:
            yield chunk  # last line, no trailing newline
            return
        logger.warning("cloud turn: dropped an over-long stdin line (> %d bytes)", limit)
        while chunk and not chunk.endswith(b"\n"):
            chunk = stream.readline(limit + 1)


def parse_answer_line(raw: bytes) -> dict | None:
    """The answer in *raw*, normalized, or None when it is not a valid answer line."""
    try:
        body = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(body, dict) or body.get("kind") != "answer":
        return None
    question_id = body.get("question_id")
    answer_id = body.get("answer_id")
    values = body.get("values", [])
    text = body.get("text", "")
    skipped = body.get("skipped", False)
    if not (isinstance(question_id, str) and question_id):
        return None
    if not (isinstance(answer_id, str) and answer_id):
        return None
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        return None
    if not isinstance(text, str) or not isinstance(skipped, bool):
        return None
    return {"question_id": question_id, "answer_id": answer_id,
            "values": values, "text": text, "skipped": skipped}


def start_answer_reader(
    stream: BinaryIO,
    loop: asyncio.AbstractEventLoop,
    deliver: Callable[[dict], None],
) -> threading.Thread:
    """Read answer lines until EOF and hand each valid one to *deliver* on *loop*."""

    def _run() -> None:
        for raw in iter_lines(stream, MAX_ANSWER_LINE_BYTES):
            answer = parse_answer_line(raw)
            if answer is None:
                # No content in the log: the line may carry the user's answer.
                logger.warning("cloud turn: ignored a malformed stdin line (%d bytes)", len(raw))
                continue
            try:
                loop.call_soon_threadsafe(deliver, answer)
            except RuntimeError:
                return  # the loop is closed: the turn is over
        logger.info("cloud turn: stdin closed")

    thread = threading.Thread(target=_run, name="cloud-turn-stdin", daemon=True)
    thread.start()
    return thread
