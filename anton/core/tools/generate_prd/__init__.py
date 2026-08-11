"""Two-phase PRD-drafting FSM behind the ``generate_prd`` tool.

Public entry point: ``generate(session, slug, artifact_path, artifact_type,
user_request, agent_understanding, known_data, user_preferences)``. Phase 1
(``engine.run_gathering_loop``) is a bounded ReAct loop that determines the
artifact type, gathers/verifies data, and asks clarifying questions. Phase 2
(``orchestrator``'s step functions) drafts a short brief, shows it to the
user for accept/cancel/revise, and on acceptance writes the full ``prd.md``.
See prd-design.md (docs/ENG-969) for the full design.
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
    state = PrdState(
        session=session,
        slug=slug,
        artifact_path=artifact_path,
        artifact_type=artifact_type,
        user_request=user_request,
        agent_understanding=agent_understanding,
        known_data=known_data,
        user_preferences=user_preferences,
    )
    return await run(state)


__all__ = ["generate", "PrdState"]
