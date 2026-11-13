"""The user's open preview, asked from a cloud turn (ENG-4227).

An in-process turn gets ``ChatSessionConfig.preview_checker`` from its host,
which reloads the artifact's preview in the user's browser and reports its
errors. A cloud turn runs in a pod with no such host in process, so the same
question travels the way ``ask_user`` answers do:

* pod → host: one stdout event,
  ``{"kind": "artifact_check", "type": "response.artifact_check",
  "check_id": "acheck:<hex>", "artifact_slug": str, "timeout_s": int}``;
* host → pod: one stdin line,
  ``{"kind": "check_result", "check_id": "acheck:<hex>",
  "body": {"status": "passed" | "failed" | "unavailable", "errors": [...],
  "ms": int} | null}``, read by ``cloud_turn/stdin.py`` and handed to
  :meth:`CloudPreviewChecks.deliver`.

No answer within the check's timeout, ``null``, stdin closing, or the turn
ending all come back as None, which the editor's :class:`PreviewChecker`
reports as unverified: never as passed.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import Callable

logger = logging.getLogger(__name__)

EVENT_KIND = "artifact_check"
EVENT_TYPE = "response.artifact_check"
LINE_KIND = "check_result"
ID_PREFIX = "acheck:"
MAX_TIMEOUT_S = 120


class CloudPreviewChecks:
    """``preview_checker`` for a cloud turn: an event out, a stdin line back."""

    def __init__(self, emit: Callable[[dict], None]) -> None:
        self._emit = emit
        self._pending: dict[str, asyncio.Future] = {}
        self._closed = False

    async def __call__(self, artifact_slug: str, timeout_s: float) -> dict | None:
        """Ask the host to check ARTIFACT_SLUG in the user's open preview; its answer body, or None."""
        if self._closed:
            return None
        timeout = max(1.0, min(float(timeout_s or 0) or 20.0, MAX_TIMEOUT_S))
        check_id = ID_PREFIX + secrets.token_hex(12)
        future = asyncio.get_running_loop().create_future()
        self._pending[check_id] = future
        try:
            self._emit({"kind": EVENT_KIND, "type": EVENT_TYPE, "check_id": check_id,
                        "artifact_slug": str(artifact_slug), "timeout_s": int(round(timeout))})
            return await asyncio.wait_for(asyncio.shield(future), timeout)
        except TimeoutError:
            if future.done() and not future.cancelled():
                return future.result()  # the answer landed as the timer fired
            return None
        except Exception:  # noqa: BLE001 — the bridge failing is "nobody answered"
            logger.warning("cloud turn: the preview check could not be asked", exc_info=True)
            return None
        finally:
            self._pending.pop(check_id, None)
            if not future.done():
                future.cancel()

    def deliver(self, line: dict) -> None:
        """Apply one ``check_result`` line. Runs on the event loop thread."""
        future = self._pending.get(line["check_id"])
        if future is None or future.done():
            logger.info("cloud turn: a check_result for no pending check (%s)", line["check_id"])
            return
        future.set_result(line["body"])

    def close(self) -> None:
        """Resolve every pending check as unanswered (stdin closed, or the turn ended)."""
        self._closed = True
        for future in list(self._pending.values()):
            if not future.done():
                future.set_result(None)

    @property
    def pending(self) -> int:
        return sum(not f.done() for f in self._pending.values())
