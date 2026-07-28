from __future__ import annotations
import asyncio
import sys

from anton.core.runtime import build_chat_session
from anton.cloud_turn.contract import TurnRequestV1, TurnResultV1


async def run_turn(req: TurnRequestV1) -> TurnResultV1:
    try:
        session = await build_chat_session(
            session_id=req.conversation_id,
            workspace_path=req.workspace_path,
            model=req.model,
        )
        final_text = await session.turn(req.input)
        return TurnResultV1(protocol_version=1, kind="turn_completed", final_text=final_text)
    except Exception as exc:  # dev skeleton: surface as terminal failure
        return TurnResultV1(protocol_version=1, kind="turn_failed", error=repr(exc))


def main() -> int:
    raw = sys.stdin.read()
    req = TurnRequestV1.from_json(raw)
    result = asyncio.run(run_turn(req))
    sys.stdout.write(result.to_json() + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
