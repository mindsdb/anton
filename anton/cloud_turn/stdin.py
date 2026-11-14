"""Answer, addon and check_result lines on the cloud turn's stdin.

The controller writes the TurnRequestV1 line first, then one line per answer
the user gives to an interactive turn's question, and one line per message the
user adds to the running turn or takes back (formats in contract.py). The
request line and the later lines must come through the SAME buffered reader: a second
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


def parse_addon_line(raw: bytes) -> dict | None:
    """The addon or addon_withdraw line in *raw*, normalized, or None.

    ``{"kind": "addon", "addon_id", "text", "decided_by"}`` hands the running
    turn a message; ``{"kind": "addon_withdraw", "addon_id", "request_id"}``
    takes one back. The result keeps ``kind`` so the caller tells them apart.
    """
    try:
        body = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(body, dict) or body.get("kind") not in ("addon", "addon_withdraw"):
        return None
    kind = body["kind"]
    addon_id = body.get("addon_id")
    if not (isinstance(addon_id, str) and addon_id):
        return None
    if kind == "addon":
        text = body.get("text")
        decided_by = body.get("decided_by")
        if not (isinstance(text, str) and text.strip()):
            return None
        if decided_by is not None and not isinstance(decided_by, str):
            return None
        return {"kind": kind, "addon_id": addon_id, "text": text, "decided_by": decided_by}
    request_id = body.get("request_id")
    if request_id is not None and not isinstance(request_id, str):
        return None
    return {"kind": kind, "addon_id": addon_id, "request_id": request_id}


CHECK_STATUSES = ("passed", "failed", "unavailable")
MAX_CHECK_ID_CHARS = 128


def parse_check_result_line(raw: bytes) -> dict | None:
    """The check_result line in *raw*, normalized, or None (ENG-4227, ``cloud_turn/preview_check.py``).

    ``{"kind": "check_result", "check_id": "acheck:<hex>", "body": {...} | null}``.
    A body is kept only with a known ``status``; its ``errors`` keep only
    objects, and ``ms`` only a whole number. ``null`` means nobody answered.
    """
    try:
        body = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(body, dict) or body.get("kind") != "check_result":
        return None
    check_id = body.get("check_id")
    if not (isinstance(check_id, str) and check_id.startswith("acheck:") and len(check_id) <= MAX_CHECK_ID_CHARS):
        return None
    answer = body.get("body")
    if answer is None:
        return {"check_id": check_id, "body": None}
    if not isinstance(answer, dict) or answer.get("status") not in CHECK_STATUSES:
        return None
    errors = answer.get("errors") or []
    if not isinstance(errors, list):
        return None
    ms = answer.get("ms")
    clean = {
        "status": answer["status"],
        "errors": [e for e in errors if isinstance(e, dict)],
        "ms": ms if isinstance(ms, int) and not isinstance(ms, bool) and ms >= 0 else 0,
    }
    return {"check_id": check_id, "body": clean}


def start_answer_reader(
    stream: BinaryIO,
    loop: asyncio.AbstractEventLoop,
    deliver: Callable[[dict], None] | None,
    deliver_addon: Callable[[dict], None] | None = None,
    deliver_check: Callable[[dict], None] | None = None,
    on_close: Callable[[], None] | None = None,
) -> threading.Thread:
    """Read stdin lines until EOF and hand each valid one to its handler on *loop*.

    Answer lines go to *deliver*, addon and addon_withdraw lines to
    *deliver_addon*, check_result lines to *deliver_check*. A malformed line,
    a line of any other kind, and a line whose handler is None are logged and
    skipped. *on_close* runs on *loop* once stdin reaches EOF.
    """

    def _run() -> None:
        for raw in iter_lines(stream, MAX_ANSWER_LINE_BYTES):
            line = parse_answer_line(raw)
            handler = deliver
            if line is None:
                line = parse_addon_line(raw)
                handler = deliver_addon
            if line is None:
                line = parse_check_result_line(raw)
                handler = deliver_check
            if line is None or handler is None:
                # No content in the log: the line may carry what the user typed.
                logger.warning("cloud turn: ignored a stdin line it does not take (%d bytes)", len(raw))
                continue
            try:
                loop.call_soon_threadsafe(handler, line)
            except RuntimeError:
                return  # the loop is closed: the turn is over
        logger.info("cloud turn: stdin closed")
        if on_close is not None:
            try:
                loop.call_soon_threadsafe(on_close)
            except RuntimeError:
                pass  # the loop is closed: the turn is over

    thread = threading.Thread(target=_run, name="cloud-turn-stdin", daemon=True)
    thread.start()
    return thread
