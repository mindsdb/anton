"""Phase B: draft a short brief, show it, route the user's feedback.

Split out of the old two-phase orchestrator so phase C (writing the PRD) and
the sequencer that ties the phases together each live on their own. The
functions themselves are unchanged — they still continue `state.messages`
rather than starting a fresh conversation, which is what lets the brief be
drafted from what the gathering loop actually saw.
"""

from __future__ import annotations

from pydantic import BaseModel

from . import sub_tools
from . import prompts
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
    "actually being read. If the app keeps data of its own between "
    "visits (notes, settings, counters the app itself creates), say so "
    "in one plain line per kind of thing stored (e.g. \"saved notes — "
    "kept by the app itself\") — without naming any storage technology. "
    "State only what IS used — do not list protocols, APIs, schemas, or "
    "technologies that are NOT used, and skip connection details (those "
    "belong only in the full PRD, after acceptance).\n"
    "- Functional requirements — omit this section entirely for static "
    "artifacts with no backend or interactivity; otherwise, plain-language "
    "user-facing behavior only.\n"
    "- UI/UX requirements — plain-language look and feel (e.g. \"dark "
    "background\", \"large centered clock face\") — NO code, CSS/JS "
    "syntax, hex colors, exact pixel/unit sizes, or library/framework "
    "names; that level of detail belongs only in the full PRD, after "
    "acceptance.\n\n"
    "End with one short closing line asking the user to continue or say "
    "what to change — in the same language as the rest of your reply (e.g. "
    "\"Continue, or any changes?\"). Do NOT describe how to answer: no key "
    "names, no button names, no option numbers, and do not invent your own "
    "accept/cancel option labels. The host renders the actual choices and "
    "their input hint (a terminal shows a default, a GUI shows buttons) — "
    "this line is only framing.\n\n"
    "Keep every section brief — this is a summary shown before the user "
    "confirms, not the final document. Reply with the lead-in sentence, "
    "the brief, and the closing line — no other text."
)

async def draft_brief(state: PrdState) -> None:
    """Phase 2 step 1: draft the short brief. Continues `state.messages` —
    this is NOT a fresh conversation, so the model sees everything phase 1
    found."""
    state.step_started(sub_tools.STEP_DRAFT_BRIEF)
    state.messages.append({
        "role": "user",
        "content": prompts.step_message(sub_tools.STEP_DRAFT_BRIEF, state),
    })
    await sub_tools.signal_thinking(state.session)
    system = state.pipeline_system
    response = await state.session._llm.plan(
        system=system,
        messages=state.messages,
        tools=state.pipeline_tools,
    )
    state.trace_log.llm_call(
        node="draft_brief", method="plan", system=system,
        messages=state.messages, response=response,
    )
    brief = (response.content or "").strip()
    if not brief:
        state.trace_log.node("draft_brief", "fail", detail="model replied with no text")
        # The model called a tool instead of replying with text (tools stay
        # defined in this call for the Anthropic API's sake — see
        # `_phase2_tools`), despite the system prompt's explicit
        # instruction not to. Raising here — rather than silently
        # continuing with an empty brief — means `orchestrator.run`'s
        # caller chain surfaces it as a normal generator crash: it
        # propagates up through `discovery.generate()` into
        # `handle_generate_artifact`'s `except Exception`, which already
        # wraps it with `_generation_failed`. No new error-reporting path
        # needed.
        raise RuntimeError(
            "draft_brief: the model replied with no text — it may have "
            "called a tool instead of drafting the brief."
        )
    state.brief = brief
    state.messages.append({"role": "assistant", "content": state.brief})
    state.trace_log.node("draft_brief", "done", detail=state.brief[:200])


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
        prompt=state.brief,
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
        # The brief itself ends with its own "continue, or changes?"
        # sentence (see _DRAFT_BRIEF_INSTRUCTION) — a numbered "1. Accept /
        # 2. Cancel" list underneath, plus the elicitor's own descriptive
        # caption, would just repeat what the sentence already said.
        compact=True,
    )
    # Wrapped here too, not only inside `ask_via_elicit`: this is the longest
    # question of the run and the one that always happens. The sentinels nest
    # — the drain counts depth for exactly this reason.
    async with sub_tools.progress_muted(state.session):
        answer = await sub_tools.ask_via_elicit(state.session, request)

    if answer.status in ("unavailable", "limit", "error"):
        state.trace_log.node("show_and_confirm", "unconfirmed", detail=f"answer status: {answer.status}")
        return "unconfirmed"

    if answer.status in ("cancelled", "timeout"):
        state.record_qa("Show PRD brief for confirmation", f"user did not respond ({answer.status})")
        state.trace_log.node("show_and_confirm", "cancelled", detail=f"answer status: {answer.status}")
        return "cancelled"

    # answered
    if answer.text:
        state.record_qa("Show PRD brief for confirmation", answer.text)
        state.messages.append(
            {"role": "user", "content": f"User feedback on the brief: {answer.text}"}
        )
        state.trace_log.node("show_and_confirm", "revise", detail=answer.text)
        return "revise"
    if "cancel" in answer.values:
        state.record_qa("Show PRD brief for confirmation", "cancel")
        state.trace_log.node("show_and_confirm", "cancelled", detail="user picked cancel")
        return "cancelled"
    state.record_qa("Show PRD brief for confirmation", "accept")
    state.trace_log.node("show_and_confirm", "accepted", detail="user picked accept")
    return "accepted"


async def classify_feedback(state: PrdState) -> str:
    """Phase 2 step 3 (only reached after a `revise` outcome): decide
    whether the user's comment needs more data/questions
    (`back_to_gathering`) or the brief can just be reworded with what we
    already know (`revise_brief`)."""
    system = state.pipeline_system
    instruction = (
        "The user just commented on the PRD brief instead of accepting "
        "it (see the last user message). Decide: does addressing it need "
        "MORE data or MORE questions to the user (`back_to_gathering`), or "
        "can the brief just be reworded/adjusted with what we already "
        "know (`revise_brief`)?"
    )
    await sub_tools.signal_thinking(state.session)
    messages = state.messages + [{"role": "user", "content": instruction}]
    result = await state.session._llm.generate_object(
        FeedbackVerdict,
        system=system,
        messages=messages,
    )
    state.trace_log.llm_call(
        node="classify_feedback", method="generate_object", system=system,
        messages=messages, value=result.model_dump(),
    )
    state.trace_log.verdict(node="classify_feedback", schema="FeedbackVerdict", value=result.model_dump())
    return result.route if result.route in ("revise_brief", "back_to_gathering") else "revise_brief"



async def redraw_brief(state: PrdState) -> None:
    """Redraw the brief after a user correction, and re-declare the sources.

    Separate from `draft_brief` for one reason: `finish_gathering` is allowed
    here. On the hot path it is denied on the brief step, because a model
    reaches for it whenever a prompt mentions having finished gathering. Here
    it is the point — a correction that introduces a data source nobody has
    fetched is only discoverable if the model re-states the source list.

    Not calling it is tolerated rather than fatal. A hard failure would turn
    every correction into a possible lost run, while the cost of the miss is
    bounded: the artifact is rebuilt from the sources already known, which is
    the same result the correction would have had without the new source.
    """
    from anton.core.artifacts.models import ARTIFACT_TYPES

    state.step_started(sub_tools.STEP_REDRAW_BRIEF)
    state.messages.append({
        "role": "user",
        "content": prompts.step_message(sub_tools.STEP_REDRAW_BRIEF, state),
    })
    await sub_tools.signal_thinking(state.session)
    system = state.pipeline_system
    response = await state.session._llm.plan(
        system=system,
        messages=state.messages,
        tools=state.pipeline_tools,
    )
    state.trace_log.llm_call(
        node="redraw_brief", method="plan", system=system,
        messages=state.messages, response=response,
    )

    declared = None
    new_type = ""
    for tc in getattr(response, "tool_calls", None) or []:
        if tc.name != "finish_gathering":
            continue
        inp = tc.input or {}
        new_type = str(inp.get("artifact_type") or "")
        raw = inp.get("data_sources")
        declared = [str(s) for s in raw] if isinstance(raw, list) else []

    if declared is not None:
        previous = set(state.declared_sources)
        if new_type in ARTIFACT_TYPES:
            state.final_artifact_type = new_type
        state.declared_sources = declared
        # Anything the correction introduced has nothing executed against it.
        # This — not "data_notes is empty" — is what opens the emergency data
        # loop: after a correction the notes are full of the PREVIOUS
        # gathering's cells, and an emptiness check would read that as
        # "everything is covered".
        state.unverified_sources = [s for s in declared if s not in previous]
    else:
        state.trace_log.node(
            "redraw_brief", "no_finish_gathering",
            detail="artifact type and declared sources kept from the previous call",
        )

    brief = (response.content or "").strip()
    if not brief:
        state.trace_log.node("redraw_brief", "fail", detail="model replied with no text")
        raise RuntimeError(
            "redraw_brief: the model replied with no text — it may have "
            "called a tool instead of redrawing the brief."
        )
    state.brief = brief
    state.messages.append({"role": "assistant", "content": state.brief})
    state.trace_log.node("redraw_brief", "done", detail=state.brief[:200])
