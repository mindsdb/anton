"""Phase 2 of generate_prd: draft a short brief, show it to the user for
accept/cancel/revise, and on acceptance write the full prd.md.

`run()` (added alongside phase 1 wiring in the next task) ties phase 1's
gathering loop and these steps into the full two-phase sequence.
"""

from __future__ import annotations

from pydantic import BaseModel

from . import sub_tools
from .prompts import build_phase2_system_prompt
from .state import PrdState


class FeedbackVerdict(BaseModel):
    """`classify_feedback`'s decision when the user leaves a free-text
    comment instead of accepting the brief."""

    route: str  # "revise_brief" | "back_to_gathering"
    reasoning: str


_DRAFT_BRIEF_INSTRUCTION = (
    "Now draft a SHORT PRD brief to show the user for confirmation.\n\n"
    "Start with one short lead-in sentence introducing what follows (e.g. "
    "\"Here's a short PRD (draft requirements) for the artifact — please "
    "review it:\") — in the same language as the rest of your reply.\n\n"
    "Then the sections below, in this exact order. The user is "
    "non-technical and will never see any code — write every section in "
    "plain language, describing the artifact itself, not the process that "
    "produced it. Do NOT mention this generation tool, the PRD workflow, "
    "prior attempts, or that this is a redo/regeneration of something — "
    "describe only what the artifact IS and does, as if it were the "
    "first and only version:\n"
    "- Goal — 1-2 plain sentences: what the artifact is and what it does "
    "for the user.\n"
    "- Artifact type — the artifact type, one short line.\n"
    "- Data model — the data source(s) in plain language: a single "
    "source in one short line (e.g. \"the device's system clock\"); "
    "several sources or tables each get their own short line (e.g. "
    "\"orders table\", \"customers table\") so the user can see what's "
    "actually being read. State only what IS used — do not list "
    "protocols, APIs, schemas, or technologies that are NOT used, and "
    "skip connection details (those belong only in the full PRD, after "
    "acceptance).\n"
    "- Functional requirements — omit this section entirely for static "
    "artifacts with no backend or interactivity; otherwise, plain-language "
    "user-facing behavior only.\n"
    "- UI/UX requirements — plain-language look and feel (e.g. \"dark "
    "background\", \"large centered clock face\") — NO code, CSS/JS "
    "syntax, hex colors, exact pixel/unit sizes, or library/framework "
    "names; that level of detail belongs only in the full PRD, after "
    "acceptance.\n\n"
    "End with one short closing line asking the user to continue or say "
    "what to change — in the same language as the rest of your reply — "
    "and make clear that just pressing Enter means \"continue\" (e.g. "
    "\"Continue, or any changes? (press Enter to continue)\"). Do not "
    "invent your own accept/cancel option labels; the actual choices are "
    "handled separately, this line is only framing.\n\n"
    "Keep every section brief — this is a summary shown before the user "
    "confirms, not the final document. Reply with the lead-in sentence, "
    "the brief, and the closing line — no other text."
)

_WRITE_PRD_INSTRUCTION = (
    "Write the FULL PRD now, expanding the same five sections with full "
    "detail. Describe only what the artifact IS and does — do NOT mention "
    "this generation tool, the PRD workflow, prior attempts, or that this "
    "is a redo/regeneration of something.\n\n"
    "Goal, Artifact type (with justification for the choice), "
    "Data model (sources, schema/fields, sample rows, PLUS connection code "
    "examples for anything fetched via scratchpad or web_fetch — cite the "
    "scratchpad name and cell), Functional requirements (omit for static "
    "artifacts), UI/UX requirements (layout, components, style, and any "
    "known user preferences). Reply with the PRD document only, as "
    "markdown, no other text."
)


def _phase2_tools() -> list[dict]:
    """A non-empty tool list, WITHOUT `ask_user` or `finish_gathering` —
    passed to phase 2's `plan()` calls purely so the Anthropic API accepts a
    `messages` list that still contains phase 1's `tool_use`/`tool_result`
    blocks (the API rejects any request with those blocks present unless
    `tools` is also non-empty — see `anthropic.py`'s
    `if merged_tools: kwargs["tools"] = ...`). `finish_gathering` is
    deliberately excluded even though `sub_tools.tool_schemas` would
    normally include it: of the four schemas, it is the one whose semantics
    ("gathering is done, call this") superficially fit phase 2's "you have
    finished gathering information" framing, making it the tool a model is
    likeliest to reach for despite the system prompt's instruction not to.
    `scratchpad`/`web_search`/`web_fetch` stay — they are hard to imagine
    calling as a reply to "reply with plain text only".

    This is defense in depth, not the actual guard against an empty reply:
    if the model calls a tool anyway (any of the three still offered, or
    something else entirely), `response.content` comes back empty and
    `draft_brief`/`write_prd` raise on it — see the guard in both.
    """
    return [t for t in sub_tools.tool_schemas(include_ask_user=False) if t["name"] != "finish_gathering"]


async def draft_brief(state: PrdState) -> None:
    """Phase 2 step 1: draft the short brief. Continues `state.messages` —
    this is NOT a fresh conversation, so the model sees everything phase 1
    found."""
    state.messages.append({"role": "user", "content": _DRAFT_BRIEF_INSTRUCTION})
    response = await state.session._llm.plan(
        system=build_phase2_system_prompt(state),
        messages=state.messages,
        tools=_phase2_tools(),
    )
    brief = (response.content or "").strip()
    if not brief:
        # The model called a tool instead of replying with text (tools stay
        # defined in this call for the Anthropic API's sake — see
        # `_phase2_tools`), despite the system prompt's explicit
        # instruction not to. Raising here — rather than silently
        # continuing with an empty brief — means `orchestrator.run`'s
        # caller chain surfaces it as a normal generator crash: it
        # propagates up through `generate_prd.generate()` into
        # `handle_generate_prd`'s `except Exception`, which already wraps
        # it with `_prd_generation_failed` (see Task 7). No new
        # error-reporting path needed.
        raise RuntimeError(
            "draft_brief: the model replied with no text — it may have "
            "called a tool instead of drafting the brief."
        )
    state.brief_markdown = brief
    state.messages.append({"role": "assistant", "content": state.brief_markdown})


async def show_and_confirm(state: PrdState) -> str:
    """Phase 2 step 2: show the brief and ask the user to accept, cancel,
    or comment. Returns one of "accepted", "cancelled", "revise",
    "unconfirmed".

    Builds the `AskRequest` directly in Python (the same pattern
    `select_path` uses) instead of going through the `ask_user` tool/LLM
    call — the brief's full text goes in `prompt`, which `validate_request`
    does not length-limit (see prd-design.md's rendering caveat: verify
    this actually looks reasonable in the CLI / cowork before shipping).
    """
    from anton.core.interaction.elicit import AskOption, AskRequest

    request = AskRequest(
        prompt=state.brief_markdown,
        kind="choice",
        options=(
            AskOption(value="accept", label="Accept"),
            AskOption(value="cancel", label="Cancel"),
        ),
        allow_custom=True,
        # A bare Enter means "looks good, continue" — the far more common
        # case than cancelling — so it resolves to "accept" instead of the
        # `AskRequest` default of "cancelled".
        default_value="accept",
    )
    answer = await sub_tools.ask_via_elicit(state.session, request)

    if answer.status in ("unavailable", "limit", "error"):
        return "unconfirmed"

    if answer.status in ("cancelled", "timeout"):
        state.record_qa("Show PRD brief for confirmation", f"user did not respond ({answer.status})")
        return "cancelled"

    # answered
    if answer.text:
        state.record_qa("Show PRD brief for confirmation", answer.text)
        state.messages.append(
            {"role": "user", "content": f"User feedback on the brief: {answer.text}"}
        )
        return "revise"
    if "cancel" in answer.values:
        state.record_qa("Show PRD brief for confirmation", "cancel")
        return "cancelled"
    state.record_qa("Show PRD brief for confirmation", "accept")
    return "accepted"


async def classify_feedback(state: PrdState) -> str:
    """Phase 2 step 3 (only reached after a `revise` outcome): decide
    whether the user's comment needs more data/questions
    (`back_to_gathering`) or the brief can just be reworded with what we
    already know (`revise_brief`)."""
    system = build_phase2_system_prompt(state)
    instruction = (
        "The user just commented on the PRD brief instead of accepting "
        "it (see the last user message). Decide: does addressing it need "
        "MORE data or MORE questions to the user (`back_to_gathering`), or "
        "can the brief just be reworded/adjusted with what we already "
        "know (`revise_brief`)?"
    )
    result = await state.session._llm.generate_object(
        FeedbackVerdict,
        system=system,
        messages=state.messages + [{"role": "user", "content": instruction}],
    )
    return result.route if result.route in ("revise_brief", "back_to_gathering") else "revise_brief"


async def write_prd(state: PrdState) -> str:
    """Phase 2 step 5 (or the best-effort path from an unconfirmed budget):
    expand the brief into the full PRD, save it, and update the artifact's
    `type` in metadata.json if it changed. Returns the full PRD markdown."""
    state.messages.append({"role": "user", "content": _WRITE_PRD_INSTRUCTION})
    response = await state.session._llm.plan(
        system=build_phase2_system_prompt(state),
        messages=state.messages,
        tools=_phase2_tools(),
    )
    full_prd = (response.content or "").strip()
    if not full_prd:
        # Same failure shape as draft_brief's guard, and the same reason:
        # writing an empty prd.md and reporting `prd_written` would be a
        # silent lie about what happened. Raising here surfaces it as a
        # generator crash, wrapped by `_prd_generation_failed` — never an
        # empty file on disk with a success status.
        raise RuntimeError(
            "write_prd: the model replied with no text — it may have "
            "called a tool instead of writing the PRD."
        )
    state.messages.append({"role": "assistant", "content": full_prd})

    (state.artifact_path / "prd.md").write_text(full_prd, encoding="utf-8")

    final_type = state.final_artifact_type or state.artifact_type
    if final_type != state.artifact_type:
        # Reuse the exact same store-construction helper the handler used
        # (`tool_handlers._artifact_store`, keyed off `session._workspace`)
        # instead of guessing the artifacts root back out of `artifact_path`
        # — that guess (`artifact_path.parent`) only holds while
        # `artifact_path == <artifacts_root>/<slug>`, which is true today
        # but is exactly the kind of assumption that breaks silently later.
        from anton.core.tools.tool_handlers import _artifact_store

        store = _artifact_store(state.session)
        if store is not None:
            store.update(state.slug, type=final_type)

    return full_prd


from . import engine  # noqa: E402  (after the phase-2 functions on purpose — keeps the file's public order: phase 2 steps first, then the sequencer that ties both phases together)


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
