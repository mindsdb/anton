"""The `turn_failed` shape `stream_turn` emits when it ends without a
terminal event — this must classify downstream (cowork-server's
`remote_turn_error` keys on a `TypeName: message` prefix; a bare message
with no colon falls through to the fully generic "An unexpected error
occurred.", discarding this curated copy).

Scope, precisely: `stream_turn`'s `finally` only reaches this fallback when
a `BaseException` that is not an `Exception` escapes — in practice a
`CancelledError` (or `SystemExit`/`KeyboardInterrupt`). Today's deployed k8s
path never raises one from inside the pod: `main()` does a bare
`asyncio.run(...)` with no signal handler, and anton's own internal cancels
are caught locally. This test proves the fallback classifies correctly when
reached, NOT that it fires on a real pod OOM-kill or a dropped exec
channel — those are detected and reported by scratchpad-controller instead
(its own "pod stream ended without a terminal event" literal), never from
inside the pod itself."""

from __future__ import annotations

import asyncio

import pytest

import anton.cloud_turn.__main__ as m

RAW = '{"protocol_version":1,"conversation_id":"c","input":"hi"}'


class HangingSession:
    """A session whose turn never resolves on its own — only an external
    `task.cancel()` ends it. Simulates the one path that actually reaches
    `stream_turn`'s no-terminal-event fallback today (see module docstring)."""

    async def turn_stream(self, user_input, **kwargs):
        await asyncio.Event().wait()
        if False:  # pragma: no cover - makes this an async generator
            yield

    def close(self): ...


@pytest.mark.asyncio
async def test_a_cancelled_turn_emits_a_classifiable_turn_failed():
    events = []
    task = asyncio.ensure_future(
        m.stream_turn(RAW, emit=events.append, session_builder=lambda req: HangingSession())
    )
    await asyncio.sleep(0)  # let it reach the hanging await
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(events) == 1
    assert events[0]["kind"] == "turn_failed"
    # No colon prefix here previously meant cowork-server's classifier read
    # the whole string as an unrecognized "type name" and discarded it.
    assert events[0]["error"].startswith("TurnInterrupted:")
