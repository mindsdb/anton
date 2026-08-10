"""Prompt builders for generate_prd's two phases.

Both phases share the same preamble (the input the tool was called with) so
neither has to re-derive it — only the task-specific instructions differ.
"""

from __future__ import annotations

from anton.core.artifacts.models import ARTIFACT_TYPES

from .state import PrdState


def _shared_preamble(state: PrdState) -> str:
    return (
        "You are drafting a PRD (Product Requirements Document) for a web "
        "artifact BEFORE any code is written. The user has not seen any "
        "code yet — you are producing a specification and confirming it "
        "with them, not writing code.\n\n"
        f"User's request (verbatim): {state.user_request}\n"
        f"Agent's understanding of the task: {state.agent_understanding}\n"
        f"Known data: {state.known_data or '(none provided)'}\n"
        f"Known user preferences: {state.user_preferences or '(none known)'}\n"
        f"Initially registered artifact type: {state.artifact_type}\n"
    )


def build_gathering_system_prompt(state: PrdState) -> str:
    types_list = ", ".join(f"`{t}`" for t in ARTIFACT_TYPES)
    return _shared_preamble(state) + (
        "\nYour job right now: make sure the artifact type is unambiguous, "
        "gather and verify any data needed (fetch samples via scratchpad, "
        "web_search, web_fetch), and ask the user clarifying or open "
        "questions ONLY when truly necessary — interactive questions are "
        "scarce this turn, so prefer working from what you already know.\n\n"
        "Call `finish_gathering` once you are confident about the artifact "
        f"type — it MUST be exactly one of: {types_list} — and have enough "
        "data (and samples, where relevant) to draft a PRD. Put everything "
        "phase 2 needs — goal, data sources found, sample rows, open "
        "assumptions, UI/UX hints — into its `notes`."
    )


def build_gathering_kickoff(state: PrdState) -> str:
    return (
        f"## User request\n{state.user_request}\n\n"
        f"## Agent's understanding\n{state.agent_understanding}\n\n"
        f"## Known data\n{state.known_data or '(none provided)'}\n\n"
        f"## User preferences\n{state.user_preferences or '(none known)'}\n"
    )


def build_gathering_continue_message() -> str:
    """Appended to the shared message history when phase 2's feedback
    routes back to phase 1 (`classify_feedback` → `back_to_gathering`)."""
    return (
        "The user was not satisfied with the PRD brief and more data or "
        "clarifying questions are needed before it can be redrafted. "
        "Continue gathering: call `finish_gathering` again once ready."
    )


def build_phase2_system_prompt(state: PrdState) -> str:
    return _shared_preamble(state) + (
        "\nYou have finished gathering information for this artifact. Now "
        "follow the instructions in each user message to draft, and then "
        "finalize, the PRD. Tools stay defined in this call only because "
        "the message history you are continuing already contains earlier "
        "tool calls (the Anthropic API requires `tools` whenever `messages` "
        "contains a `tool_use`/`tool_result` block) — do NOT call any of "
        "them; reply with plain text/markdown only."
    )
