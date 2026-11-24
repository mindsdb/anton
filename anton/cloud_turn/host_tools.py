"""Tools the host runs, called from a cloud turn.

A request may name tools that the host, not the pod, carries out
(``TurnRequestV1.host_tools``). The pod offers one proxy tool per entry, and
a call travels the way a preview check does:

* pod -> host: one stdout event,
  ``{"kind": "host_tool_call", "call_id": "htool:<32 hex>", "tool": str,
  "input": {...}}``;
* host -> pod: stdin lines read by ``cloud_turn/stdin.py`` and handed to
  :meth:`CloudHostTools.deliver`: any number of
  ``{"kind": "host_tool_progress", "call_id", "text"}``, each relayed as a
  ``ToolProgress`` step on the tool's line, then the result as one or more
  ``{"kind": "host_tool_result", "call_id", "seq", "last", "ok", "text"}``
  parts, joined in ``seq`` order once the ``last`` part and every part
  before it arrived.

No complete result within the tool's ``timeout_s``, stdin closing, or the
turn ending all return :data:`NO_ANSWER_TEXT` with ``ok=False``. The host may
still have applied the call, so the text tells the model to check first.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import AsyncIterator, Callable, Collection, Iterable
from typing import TYPE_CHECKING, Any

from anton.core.tools.progress import ToolProgress
from anton.core.tools.registry import ToolOutcome

if TYPE_CHECKING:
    from anton.cloud_turn.contract import HostToolSpecV1
    from anton.core.tools.tool_defs import ToolDef

logger = logging.getLogger(__name__)

EVENT_KIND = "host_tool_call"
PROGRESS_KIND = "host_tool_progress"
RESULT_KIND = "host_tool_result"
CALL_ID_PREFIX = "htool:"
#: The contract's text for a call nobody answered; the host may still have run it.
NO_ANSWER_TEXT = "The host did not answer. The call may have been applied; check its effect before retrying."
NO_ANSWER_REASON = "host_no_answer"
FAILED_REASON = "host_tool_failed"

_CLOSED = object()


def _no_answer() -> ToolOutcome:
    return ToolOutcome(content=NO_ANSWER_TEXT, ok=False, reason=NO_ANSWER_REASON)


class _ResultParts:
    """The parts of one call's result, joined in ``seq`` order."""

    def __init__(self) -> None:
        self._parts: dict[int, tuple[str, bool]] = {}
        self._last: int | None = None

    def add(self, line: dict) -> ToolOutcome | None:
        """Keep one part; the joined outcome once every part up to the last one is in, else None.

        A repeated part, a part past the last one, and a second last part are
        dropped: the host sends exactly one of each.
        """
        seq = line["seq"]
        if seq in self._parts or (self._last is not None and (seq > self._last or line["last"])):
            logger.info("cloud turn: dropped a repeated or out-of-range host tool result part")
            return None
        self._parts[seq] = (line["text"], line["ok"])
        if line["last"]:
            self._last = seq
            self._parts = {k: v for k, v in self._parts.items() if k <= seq}
        if self._last is None or len(self._parts) != self._last + 1:
            return None
        ordered = [self._parts[i] for i in range(self._last + 1)]
        ok = all(part_ok for _, part_ok in ordered)
        return ToolOutcome(content="".join(text for text, _ in ordered), ok=ok,
                           reason="" if ok else FAILED_REASON)


class CloudHostTools:
    """Calls to host-run tools for one cloud turn: an event out, stdin lines back."""

    def __init__(self, emit: Callable[[dict], None]) -> None:
        self._emit = emit
        self._pending: dict[str, asyncio.Queue] = {}
        self._closed = False

    async def call(
        self, tool: str, tool_input: dict, timeout_s: float,
    ) -> AsyncIterator[ToolProgress | ToolOutcome]:
        """Run TOOL on the host: its steps as ``ToolProgress``, then exactly one ``ToolOutcome``."""
        if self._closed:
            yield _no_answer()
            return
        loop = asyncio.get_running_loop()
        call_id = CALL_ID_PREFIX + secrets.token_hex(16)
        queue: asyncio.Queue = asyncio.Queue()
        self._pending[call_id] = queue
        parts = _ResultParts()
        deadline = loop.time() + timeout_s
        try:
            try:
                self._emit({"kind": EVENT_KIND, "call_id": call_id, "tool": tool, "input": tool_input})
            except Exception:  # noqa: BLE001 - a call that never left reads as unanswered
                logger.warning("cloud turn: a host tool call could not be sent", exc_info=True)
                yield _no_answer()
                return
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    lines = self._drain(queue)
                    timed_out = True
                else:
                    try:
                        lines = [await asyncio.wait_for(queue.get(), remaining)]
                        timed_out = False
                    except TimeoutError:
                        # A line that landed as the timer fired still counts.
                        lines = self._drain(queue)
                        timed_out = True
                for line in lines:
                    if line is _CLOSED:
                        break
                    if line["kind"] == PROGRESS_KIND:
                        yield ToolProgress(text=line["text"])
                        continue
                    outcome = parts.add(line)
                    if outcome is not None:
                        yield outcome
                        return
                else:
                    if timed_out:
                        break
                    continue
                break
            # The name passed the contract's name rule, so the formatter may keep it.
            logger.info("cloud turn: host tool %s got no complete answer", tool,
                        extra={"event": "host_tool_unanswered", "reason": NO_ANSWER_REASON,
                               "_diagnostic_safe_template": True})
            yield _no_answer()
        finally:
            self._pending.pop(call_id, None)

    @staticmethod
    def _drain(queue: asyncio.Queue) -> list:
        lines = []
        while True:
            try:
                lines.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                return lines

    def deliver(self, line: dict) -> None:
        """Hand one parsed host_tool_progress or host_tool_result line to its call. Runs on the event loop thread."""
        queue = self._pending.get(line["call_id"])
        if queue is None:
            logger.info("cloud turn: a %s line for no pending host tool call", line["kind"])
            return
        queue.put_nowait(line)

    def close(self) -> None:
        """Answer every pending call as unanswered (stdin closed, or the turn ended); later calls fail at once."""
        self._closed = True
        for queue in list(self._pending.values()):
            queue.put_nowait(_CLOSED)

    @property
    def pending(self) -> int:
        return len(self._pending)


def _proxy_handler(bridge: CloudHostTools, name: str, timeout_s: int) -> Callable[[Any, dict], Any]:
    def handle(session: Any, tc_input: dict) -> AsyncIterator[ToolProgress | ToolOutcome]:
        return bridge.call(name, tc_input, timeout_s)

    return handle


def proxy_tool_defs(
    bridge: CloudHostTools, specs: Iterable["HostToolSpecV1"], taken: Collection[str],
) -> list["ToolDef"]:
    """One proxy ``ToolDef`` per spec whose name is not in *taken*; each skipped name logs a warning."""
    from anton.core.tools.tool_defs import ToolDef

    tools: list[ToolDef] = []
    for spec in specs:
        if spec.name in taken:
            logger.warning("cloud turn: skipped host tool %s, another tool of this turn has its name",
                           spec.name, extra={"event": "host_tool_skipped", "reason": "name_taken",
                                             "_diagnostic_safe_template": True})
            continue
        tools.append(ToolDef(
            name=spec.name,
            description=spec.description,
            input_schema=spec.input_schema,
            handler=_proxy_handler(bridge, spec.name, spec.timeout_s),
            prompt=spec.prompt,
            unlock_skill=spec.unlock_skill,
        ))
    return tools
