"""Sequencer for the discovery phases: gather -> draft -> confirm -> write.

The phase steps themselves live in `brief.py` (phase B) and `prd.py`
(phase C); this file owns only the order they run in and the revise loop.
"""

from __future__ import annotations

from . import engine
from .brief import classify_feedback, draft_brief, show_and_confirm
from .prd import write_prd
from .state import PrdState


async def _run_gathering(state: PrdState) -> None:
    """Run phase 1 once, falling back to the originally registered type
    when the model never called `finish_gathering` (round budget
    exhausted, or it stopped with plain text)."""
    await engine.run_gathering_loop(state)
    if not state.final_artifact_type:
        state.final_artifact_type = state.artifact_type


# Not the primary stopping mechanism — the shared question budget is (every
# `show_and_confirm` call spends one of the turn's slots, and `elicit()`
# returning "limit" already routes to "unconfirmed" below). This is a
# defensive backstop for the case that mechanism doesn't fire: a test
# double, a future `elicit()` change, a custom elicitor that never
# increments `session.question_count`. Without it, that case is an
# unbounded loop burning two LLM calls (and possibly a full phase-1
# re-run) per turn, forever, instead of a turn that ends in
# `prd_written_unconfirmed`.
MAX_REVISE_CYCLES = 10


async def run(state: PrdState) -> dict:
    """The full generate_prd sequence: gather → draft → confirm → write."""
    await _run_gathering(state)
    await draft_brief(state)

    for _ in range(MAX_REVISE_CYCLES):
        outcome = await show_and_confirm(state)

        if outcome == "accepted":
            await write_prd(state)
            return {
                "status": "prd_written",
                "prd_path": str(state.artifact_path / "prd.md"),
                "artifact_type": state.final_artifact_type,
                "brief_summary": state.brief_markdown,
                "qa_log": state.qa_log_markdown(),
            }

        if outcome == "cancelled":
            return {
                "status": "cancelled",
                "reason": "user declined the PRD brief",
                "qa_log": state.qa_log_markdown(),
            }

        if outcome == "unconfirmed":
            await write_prd(state)
            return {
                "status": "prd_written_unconfirmed",
                "prd_path": str(state.artifact_path / "prd.md"),
                "artifact_type": state.final_artifact_type,
                "brief_summary": state.brief_markdown,
                "qa_log": state.qa_log_markdown(),
            }

        # outcome == "revise"
        route = await classify_feedback(state)
        if route == "back_to_gathering":
            await _run_gathering(state)
        await draft_brief(state)

    # MAX_REVISE_CYCLES exhausted without the budget ever saying "limit" —
    # should not happen in practice (see the constant's docstring above),
    # but write best-effort rather than dropping the turn's work on the floor.
    await write_prd(state)
    return {
        "status": "prd_written_unconfirmed",
        "prd_path": str(state.artifact_path / "prd.md"),
        "artifact_type": state.final_artifact_type,
        "brief_summary": state.brief_markdown,
        "qa_log": state.qa_log_markdown(),
    }
