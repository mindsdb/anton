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
    await sub_tools.signal_thinking(state.session)
    system = build_phase2_system_prompt(state)
    response = await state.session._llm.plan(
        system=system,
        messages=state.messages,
        tools=_phase2_tools(),
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
        # `handle_generate_prd`'s `except Exception`, which already wraps
        # it with `_prd_generation_failed` (see Task 7). No new
        # error-reporting path needed.
        raise RuntimeError(
            "draft_brief: the model replied with no text — it may have "
            "called a tool instead of drafting the brief."
        )
    state.brief_markdown = brief
    state.messages.append({"role": "assistant", "content": state.brief_markdown})
    state.trace_log.node("draft_brief", "done", detail=state.brief_markdown[:200])


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
        # The brief itself ends with its own "continue, or changes?"
        # sentence (see _DRAFT_BRIEF_INSTRUCTION) — a numbered "1. Accept /
        # 2. Cancel" list underneath, plus the elicitor's own descriptive
        # caption, would just repeat what the sentence already said.
        compact=True,
    )
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
    system = build_phase2_system_prompt(state)
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

