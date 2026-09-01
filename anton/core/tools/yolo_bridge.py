"""Turning a yolo run into a streaming tool call.

Two impedance mismatches to solve, and they are the whole module.

The first is direction. `YoloEditor` reports what it is doing by *calling*
a `Progress` object, synchronously, from inside the coroutine doing the
work. A streaming tool handler reports by *yielding* `ToolProgress` from
an async generator. One pushes, the other pulls, so something has to sit
between them: a queue the editor writes into and the generator drains.

The second is that the work and the reporting have to proceed together.
Awaiting the edit first and replaying its log afterwards would produce
exactly the silent pause the progress exists to prevent — the lines would
all arrive at the end, after the thing they describe.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from anton.core.tools.progress import ToolProgress

__all__ = ["QueueProgress", "run_with_progress"]


class QueueProgress:
    """A yolo `Progress` that puts its lines on a queue.

    `put_nowait` because the editor calls these from synchronous code and
    cannot await. The queue is unbounded, which is safe here: a run emits
    on the order of ten lines, and dropping progress to apply
    backpressure to real work would be the wrong trade anyway.
    """

    def __init__(self, queue: asyncio.Queue[str]) -> None:
        self._queue = queue

    def status(self, message: str) -> None:
        self._queue.put_nowait(message)

    def log(self, message: str) -> None:
        # Yolo indents its detail lines to sit under a status. That reads
        # in a terminal; in a chat stream it is just leading whitespace.
        self._queue.put_nowait(message.strip())


async def run_with_progress(
    work: asyncio.Future, queue: asyncio.Queue[str]
) -> AsyncIterator[ToolProgress | object]:
    """Yield progress as it arrives, then the result, in order.

    Waits on the queue and the work at once so a line appears the moment
    it is written rather than on a polling interval. When the work
    finishes, whatever is still queued is drained before the result goes
    out — those lines describe steps that really happened, and dropping
    them would make the last thing the user sees be from the middle of
    the run.

    Cancellation is handled explicitly: the pending queue read is a task,
    and leaving it dangling would log "Task was destroyed but it is
    pending" every time a tool call ends.
    """
    pending_read: asyncio.Task | None = asyncio.ensure_future(queue.get())
    try:
        while True:
            done, _ = await asyncio.wait(
                {pending_read, work}, return_when=asyncio.FIRST_COMPLETED
            )
            if pending_read in done:
                yield ToolProgress(text=pending_read.result())
                pending_read = asyncio.ensure_future(queue.get())
            if work in done:
                break
    finally:
        if pending_read is not None and not pending_read.done():
            pending_read.cancel()

    while not queue.empty():
        yield ToolProgress(text=queue.get_nowait())
    # Re-raises here if the work failed, which is what the caller wants:
    # a crash inside the editor should surface as a failed tool call, not
    # as a silently empty result.
    yield work.result()
