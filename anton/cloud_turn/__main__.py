from __future__ import annotations
import asyncio
import json
import sys
from typing import AsyncIterator

from anton.core.runtime import build_chat_session
from anton.core.llm.provider import StreamTextDelta
from anton.cloud_turn.contract import TurnRequestV1


async def stream_turn(req: TurnRequestV1) -> AsyncIterator[dict]:
    try:
        session = await build_chat_session(
            session_id=req.conversation_id,
            workspace_path=req.workspace_path,
            model=req.model,
        )
        async for event in session.turn_stream(req.input):
            if isinstance(event, StreamTextDelta):
                yield {"kind": "delta", "text": event.text}
        yield {"kind": "turn_completed"}
    except Exception as exc:
        yield {"kind": "turn_failed", "error": repr(exc)}


async def _run() -> None:
    line = sys.stdin.readline()
    req = TurnRequestV1.from_json(line)
    async for ev in stream_turn(req):
        sys.stdout.write(json.dumps(ev) + "\n")
        sys.stdout.flush()


def main() -> int:
    asyncio.run(_run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
