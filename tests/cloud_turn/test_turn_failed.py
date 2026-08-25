"""`turn_failed` error shapes `stream_turn` emits — both must classify
downstream (cowork-server's `remote_turn_error` keys on a `TypeName: message`
prefix; a bare message with no colon falls through to the fully generic
"An unexpected error occurred.", discarding this curated copy)."""

from __future__ import annotations

import asyncio

import pytest

import anton.cloud_turn.__main__ as m

RAW = '{"protocol_version":1,"conversation_id":"c","input":"hi"}'


class HangingSession:
    """A session whose turn never resolves on its own — only cancellation
    ends it, mirroring a pod torn down mid-turn."""

    async def turn_stream(self, user_input, **kwargs):
        await asyncio.Event().wait()
        if False:  # pragma: no cover - makes this an async generator
            yield

    def close(self): ...


@pytest.mark.asyncio
async def test_a_turn_cancelled_before_completion_emits_a_classifiable_turn_failed():
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
