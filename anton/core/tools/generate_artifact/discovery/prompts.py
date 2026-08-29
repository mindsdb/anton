"""Prompt builders for generate_prd's two phases.

Both phases share the same preamble (the input the tool was called with) so
neither has to re-derive it — only the task-specific instructions differ.
"""

from __future__ import annotations

from anton.core.artifacts.models import ARTIFACT_TYPES

from .state import PrdState


def build_pipeline_system_prompt(state: PrdState) -> str:
    """The ONE system prompt for phases A-D.

    Immutable content only. Everything that changes during the run — the
    current artifact type, the remaining question budget, the instruction for
    the step being executed — arrives as a user message instead, because
    rewriting this string at a phase switch would discard the prefix cache
    for the whole shared history.
    """
    types_list = ", ".join(f"`{t}`" for t in ARTIFACT_TYPES)
    return (
        "You are producing a web artifact end to end, in one pipeline: "
        "gather what is needed, agree a short brief with the user, write the "
        "full PRD, then the technical specification.\n\n"
        f"Artifact slug: {state.slug}\n"
        f"Artifact folder: {state.artifact_path}\n"
        f"Valid artifact types: {types_list}\n\n"
        "HOW THIS CONVERSATION WORKS. Each step's instruction arrives as a "
        "user message. Do exactly what the CURRENT instruction says and "
        "ignore instructions from earlier steps — they are history, not "
        "standing orders. The tool list stays the same for the whole "
        "pipeline; a tool being listed does NOT mean it is available on the "
        "current step. Calling one that is not available wastes a round and "
        "is answered with a refusal naming the step.\n\n"
        "Choosing between the two fullstack types: pick "
        "`fullstack-stateful-app` ONLY when the app must own durable state "
        "of its own — data the app itself creates and keeps between "
        "requests (user-created notes/todos, counters, settings, sessions). "
        "That state lives in the platform STATE store, a document/key-value "
        "store suited to LIGHT data keyed by id. An app that only reads "
        "from or writes to EXTERNAL data sources (databases, APIs) is "
        "`fullstack-stateless-app` — an external DB is a data source, not "
        "app-owned state. Heavy or relational data (joins, transactions, "
        "analytics) always belongs in an external database, which usually "
        "means stateless. Prefer stateless when in doubt."
    )


def build_call_kickoff(state: PrdState) -> str:
    """The first user message of a run: what the tool was actually asked for.

    These four fields used to live in the phase-1 system prompt. They cannot
    stay there any more: the system prompt is now shared by every step and
    built once, while `agent_understanding` legitimately differs between
    repeat calls for the same request. A model told to "gather what the
    artifact needs" without them gathers for a request it cannot see.
    """
    return (
        f"## User request\n{state.user_request}\n\n"
        f"## Agent's understanding\n{state.agent_understanding}\n\n"
        f"## Known data\n{state.known_data or '(none provided)'}\n\n"
        f"## User preferences\n{state.user_preferences or '(none known)'}\n"
    )


def restored_context(state: PrdState) -> str:
    """Everything a step needs when there is no message history to continue.

    The cold-start path: the gathering conversation happened in another
    process, so a step that normally reads it off the shared list has to be
    handed the same material explicitly. Whatever `discovery.json` restored
    goes in — never a reconstructed fake conversation, which would put words
    in the model's mouth that it never said.
    """
    parts = [build_call_kickoff(state)]
    if state.brief.strip():
        parts.append(f"## Brief agreed earlier\n{state.brief.strip()}")
    if state.declared_sources:
        parts.append(
            "## Data sources declared earlier\n"
            + "\n".join(f"- {s}" for s in state.declared_sources)
        )
    if state.data_notes.strip():
        parts.append(f"## Data gathered earlier\n{state.data_notes.strip()}")
    if state.web_notes.strip():
        parts.append(state.web_notes.strip())
    return "\n\n".join(parts)


GATHERING_CONTINUE = (
    "The user was not satisfied with the PRD brief and more data or "
    "clarifying questions are needed before it can be redrafted. "
    "Continue gathering: call `finish_gathering` again once ready."
)

_GATHERING_INSTRUCTION = (
    "Make sure the artifact type is unambiguous, gather and verify any data "
    "needed (fetch samples via scratchpad, web_search, web_fetch), and ask "
    "the user clarifying or open questions ONLY when truly necessary — "
    "interactive questions are scarce this turn, so prefer working from what "
    "you already know.\n\n"
    "Call `finish_gathering` once you are confident about the artifact type "
    "and have enough data (and samples, where relevant) to draft a PRD. Put "
    "everything the brief needs — goal, data sources found, sample rows, "
    "open assumptions, UI/UX hints, and (for a stateful app) what the app "
    "stores of its own — into its `notes`, and name the data sources the "
    "artifact will read in `data_sources`."
)

_REDRAW_SUFFIX = (
    "\n\nThe user has already seen a brief and asked for a change; the "
    "correction is in this call's updated understanding. Redraw the brief "
    "with the correction applied.\n\n"
    "Then call `finish_gathering` to re-state the artifact type and the data "
    "sources the corrected artifact needs. This is REQUIRED: if the "
    "correction introduces a source nobody has fetched yet, that call is the "
    "only thing that will cause it to be fetched."
)


def _step_instructions() -> dict[str, str]:
    """Built lazily so the instruction texts stay in the modules that own
    them — brief.py and prd.py — rather than being copied here."""
    from .brief import _DRAFT_BRIEF_INSTRUCTION
    from .prd import _WRITE_PRD_INSTRUCTION
    from . import sub_tools

    return {
        sub_tools.STEP_GATHERING: _GATHERING_INSTRUCTION,
        sub_tools.STEP_DRAFT_BRIEF: _DRAFT_BRIEF_INSTRUCTION,
        sub_tools.STEP_REDRAW_BRIEF: _DRAFT_BRIEF_INSTRUCTION + _REDRAW_SUFFIX,
        sub_tools.STEP_WRITE_PRD: _WRITE_PRD_INSTRUCTION,
    }


def step_message(step: str, state: PrdState, *, extra: str = "") -> str:
    """The user message that switches the pipeline to `step`.

    Carries every mutable value the step needs — current artifact type,
    remaining question budget — because the system prompt cannot.
    """
    from ..state import gathering_question_budget
    from . import sub_tools

    header = (
        f"STEP: {step}\n"
        f"Current artifact type: {state.final_artifact_type or state.artifact_type}\n"
    )
    if step == sub_tools.STEP_GATHERING:
        header += (
            "Questions you may still ask the user: "
            f"{gathering_question_budget(state.session)}\n"
        )
    body = _step_instructions()[step]
    # No history to continue → the step is handed the restored material
    # instead. The redraw suffix in particular refers to a correction that is
    # only "above" when there is a conversation above.
    #
    # Gathering is excluded, and not as an optimization: it never runs cold.
    # A repeat call restores phase A from disk rather than re-running it, and
    # the one path that DOES re-enter gathering always has a history by then.
    # Its caller prepends `build_call_kickoff` explicitly, and
    # `restored_context` opens with the very same block — so without this
    # exclusion the four call fields would be duplicated inside the first
    # message, i.e. inside the cached prefix of every A-D call.
    needs_restore = not state.messages and step != sub_tools.STEP_GATHERING
    tail = "\n\n" + restored_context(state) if needs_restore else ""
    return header + "\n" + body + tail + (("\n\n" + extra) if extra else "")
