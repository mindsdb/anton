"""Sequencer for the discovery phases: gather -> draft -> confirm -> write.

The phase steps themselves live in `brief.py` (phase B) and `prd.py`
(phase C); this file owns only the order they run in, the revise loop, and
where a repeat call re-enters.

It returns a pipeline STAGE rather than a result dict. The stage is what
lands in `discovery.json`, and it is the single fact a later call reads to
decide where to resume — deliberately not a `confirmed` flag alongside it,
because two fields describing one thing drift apart silently.
"""

from __future__ import annotations

from . import checkpoint as cp
from . import engine
from .brief import classify_feedback, draft_brief, redraw_brief, show_and_confirm
from .prd import write_prd
from .state import PrdState

# Not a stage: cancelling is not "how far we got", it is "we are not going".
# Nothing is written, so there is no checkpoint to leave behind either.
CANCELLED = "cancelled"


async def _run_gathering(state: PrdState) -> None:
    """Run phase A once, falling back to the originally registered type
    when the model never called `finish_gathering` (round budget
    exhausted, or it stopped with plain text)."""
    await engine.run_gathering_loop(state)
    if not state.final_artifact_type:
        state.final_artifact_type = state.artifact_type


def _elicit_available(session) -> bool:
    """Whether the user can be asked at all right now.

    Not a guess about the answer — only about the channel. `needs_confirmation`
    exists precisely because this can be False, and the re-show on a
    correction is conditioned on it so that an environment without a user
    never enters a confirm/return cycle.

    A heuristic, and knowingly so: a host can hold an elicitor that never
    renders the question (I-12). Being wrong here is safe in the direction
    that matters — the question comes back `unavailable`, the run returns
    `needs_confirmation`, and the repeat call resumes at
    `awaiting_confirmation`. The path still converges; it just spends one
    extra call getting there.
    """
    return getattr(session, "elicitor", None) is not None


# Not the primary stopping mechanism — the shared question budget is (every
# `show_and_confirm` call spends one of the turn's slots, and `elicit()`
# returning "limit" already routes to "unconfirmed" below). This is a
# defensive backstop for the case that mechanism doesn't fire: a test
# double, a future `elicit()` change, a custom elicitor that never
# increments `session.question_count`. Without it, that case is an
# unbounded loop burning two LLM calls (and possibly a full phase-A
# re-run) per turn, forever.
MAX_REVISE_CYCLES = 10


async def _confirm_loop(state: PrdState) -> str:
    """Show the brief, route the answer, and write the PRD on the way out.

    Returns `STAGE_PRD_WRITTEN`, `STAGE_AWAITING_CONFIRMATION` or `CANCELLED`.
    """
    for _ in range(MAX_REVISE_CYCLES):
        outcome = await show_and_confirm(state)

        if outcome == "accepted":
            await write_prd(state)
            return cp.STAGE_PRD_WRITTEN

        if outcome == "cancelled":
            return CANCELLED

        if outcome == "unconfirmed":
            # A PRD exists but nobody agreed to it and nobody could be asked.
            # The stage says exactly that, and a repeat call treats itself as
            # the agreement — see `run_discovery`'s ENTRY_CONFIRM branch.
            await write_prd(state)
            return cp.STAGE_AWAITING_CONFIRMATION

        # outcome == "revise"
        route = await classify_feedback(state)
        if route == "back_to_gathering":
            await _run_gathering(state)
        await draft_brief(state)

    # MAX_REVISE_CYCLES exhausted without the budget ever saying "limit" —
    # should not happen in practice (see the constant's docstring above),
    # but write best-effort rather than dropping the turn's work on the floor.
    await write_prd(state)
    return cp.STAGE_AWAITING_CONFIRMATION


async def run_discovery(state: PrdState, *, entry: str) -> str:
    """Phases A-C. Returns the pipeline stage reached, or `CANCELLED`.

    Three ways in, and they differ only in how much of phase B runs:

    - `ENTRY_FULL` — nothing on disk for this request. Gather, draft, confirm.
    - `ENTRY_CONFIRM` — a brief was drafted and never agreed to. The repeat
      call IS the agreement: the contract already told the outer agent to get
      it and call again, so asking a second time would be asking the same
      question twice. When the caller brought a correction AND the user can
      be reached, the redrawn brief is shown once — that brief has not been
      seen by anyone, and on the wind-down path the original never was
      either. Conditioned on reachability, not on the answer, so an
      environment without a user cannot loop here.
    - `ENTRY_NEW_ITERATION` — the previous run finished. This is new work on
      the same request, so the brief goes through normal confirmation.

    Phase C runs on every path that accepts a brief. Reusing the PRD on disk
    is not an option: `prd_section` declares it the authoritative requirements
    source, so a correction agreed in the brief would be overwritten by the
    un-corrected document at the spec step.
    """
    if entry == cp.ENTRY_FULL:
        await _run_gathering(state)
        await draft_brief(state)
        return await _confirm_loop(state)

    if entry == cp.ENTRY_CONFIRM:
        if state.call_changed:
            await redraw_brief(state)
            if _elicit_available(state.session):
                return await _confirm_loop(state)
        await write_prd(state)
        return cp.STAGE_PRD_WRITTEN

    if entry == cp.ENTRY_NEW_ITERATION:
        await draft_brief(state)
        return await _confirm_loop(state)

    raise ValueError(
        f"run_discovery called with an entry it does not own: {entry!r}"
    )
