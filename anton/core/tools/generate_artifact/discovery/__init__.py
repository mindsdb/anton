"""Discovery phases of the artifact pipeline: gather -> brief -> PRD.

Formerly the standalone `generate_prd` tool. The phases still share one
message list, but the boundary they hand over is no longer a file the next
tool re-reads — it is `GenState` in memory on the hot path, and
`discovery.json` on a cold start (see the design document, sections 2.4-2.6).

Public entry point: ``generate(session, slug, artifact_path, artifact_type,
user_request, agent_understanding, known_data, user_preferences)``. Phase A
(``engine.run_gathering_loop``) is a bounded ReAct loop that determines the
artifact type, gathers/verifies data, and asks clarifying questions. Phases B
and C (``orchestrator``'s step functions) draft a short brief, show it to the
user for accept/cancel/revise, and on acceptance write the full ``prd.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .orchestrator import run
from .state import PrdState

if TYPE_CHECKING:
    from anton.chat_session import ChatSession


async def generate(
    *,
    session: "ChatSession | Any",
    slug: str,
    artifact_path: Path,
    artifact_type: str,
    user_request: str,
    agent_understanding: str,
    known_data: str,
    user_preferences: str,
) -> dict:
    from ..debug_trace import make_trace

    trace = make_trace()
    trace.run_start(
        slug=slug,
        artifact_type=artifact_type,
        user_request=user_request,
        agent_understanding=agent_understanding,
        known_data=known_data,
        user_preferences=user_preferences,
    )
    state = PrdState(
        session=session,
        slug=slug,
        artifact_path=artifact_path,
        artifact_type=artifact_type,
        is_fullstack=artifact_type != "html-app",
        brief="",
        user_request=user_request,
        agent_understanding=agent_understanding,
        known_data=known_data,
        user_preferences=user_preferences,
        trace_log=trace,
    )
    try:
        result = await run(state)
    except Exception as exc:
        trace.run_result(ok=False, error=str(exc))
        raise
    trace.run_result(ok=True, result=result)
    return result


__all__ = ["generate", "PrdState"]
