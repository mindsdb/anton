"""The `recall_skill` tool — retrieve a procedural skill into working memory.

Brain analogue: prefrontal cortex pulls a stored procedure from
long-term memory into the working buffer when it recognizes a familiar
pattern in the current task. The tool is the *retrieval* operation; the
LLM still has agency about whether (and how literally) to follow the
recalled procedure.

The classifier signal lives in this tool: every successful invocation
bumps the skill's `recommended` counter, giving us a precise,
mechanical signal of "the system thought this skill applied" without
relying on the LLM to emit a marker or follow any convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from anton.core.tools.tool_defs import ToolDef

if TYPE_CHECKING:
    from anton.core.session import ChatSession


_DESCRIPTION = (
    "Retrieve a procedural skill from long-term memory into your working "
    "context. Call this when you recognize that one of the skills listed in "
    "the '## Procedural memory' section of your system prompt applies to the "
    "user's current request. The tool returns the full step-by-step procedure "
    "for that skill — follow it as a guide, adapting to the specifics of the "
    "current task. You may recall multiple skills if the task spans several "
    "procedures.\n\n"
    "If you pass a label that doesn't exist, the tool will return the closest "
    "match (if any) with a warning, or list the available labels if nothing "
    "is close."
)


_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "description": (
                "The skill label to recall, e.g. 'csv-summary'. Must come from "
                "the procedural memory list in your system prompt."
            ),
        },
    },
    "required": ["label"],
}


def _recall_marker(label: str) -> str:
    """Stable marker embedded in every recall payload.

    Used to detect whether a skill's body is still present in the current
    history: repeat recalls return a short stub instead of re-appending the
    full body (which would be re-sent on every subsequent call), but if
    compaction summarized the payload away the marker disappears with it and
    the next recall re-sends the full procedure.
    """
    return f"# Skill recalled: `{label}`"


def _format_skill_response(skill, *, warning: str = "") -> str:
    """Render the recall payload sent back to the LLM as a tool result."""
    parts: list[str] = []
    if warning:
        parts.append(warning.strip())
        parts.append("")  # blank line before the procedure
    parts.append(_recall_marker(skill.label))
    parts.append(f"# Skill: {skill.name}")
    parts.append("")
    if skill.description:
        parts.append(skill.description)
        parts.append("")
    parts.append("## Procedure (Stage 1 — declarative)")
    parts.append("")
    parts.append(skill.declarative_md.strip())
    return "\n".join(parts)


def _already_in_history(session, label: str) -> bool:
    """True if a prior recall of `label` is still visible in the history."""
    history = getattr(session, "history", None)
    if not isinstance(history, list):
        return False
    marker = _recall_marker(label)
    try:
        import json as _json

        return any(
            marker in (m if isinstance(m, str) else _json.dumps(m, default=str))
            for m in history
        )
    except Exception:  # noqa: BLE001 - never let the guard break a recall
        return False


async def handle_recall_skill(session: "ChatSession", tc_input: dict) -> str:
    """Look up a skill by label and return its declarative procedure."""
    label_in = (tc_input.get("label") or "").strip()
    if not label_in:
        return (
            "ERROR: recall_skill requires a non-empty 'label' parameter. "
            "Pick one from the procedural memory list in your system prompt."
        )

    store = getattr(session, "_skill_store", None)
    if store is None:
        return (
            "ERROR: no skill store is wired into this session. "
            "Procedural memory is unavailable right now."
        )

    skill = store.load(label_in)
    warning = ""
    if skill is None:
        closest = store.closest_match(label_in)
        if closest is None:
            available = [s["label"] for s in store.list_summaries()]
            if not available:
                return (
                    f"NO MATCH: no skill named '{label_in}', and the procedural "
                    f"memory is empty. Proceed without a recalled procedure."
                )
            return (
                f"NO MATCH: no skill named '{label_in}'. Available skills: "
                f"{', '.join(available)}."
            )
        skill = store.load(closest)
        if skill is None:
            # Race or filesystem flake — be defensive
            return (
                f"NO MATCH: '{label_in}' was not found and the closest "
                f"candidate '{closest}' could not be loaded."
            )
        warning = (
            f"⚠ No skill named '{label_in}'. Returning the closest match: "
            f"'{skill.label}'. If that's not what you wanted, ignore the "
            f"procedure below and proceed without a recalled skill."
        )

    if _already_in_history(session, skill.label):
        return (
            f"Skill '{skill.label}' was already recalled in this conversation "
            "— its full procedure is in your context above (look for "
            f"'{_recall_marker(skill.label)}') and still applies. Not "
            "re-sending the body."
        )

    # Increment the recommended counter for the *resolved* label, not the
    # input. If the LLM typo'd 'csv-sumary', we credit 'csv-summary'.
    store.increment_recommended(skill.label, stage=1)

    return _format_skill_response(skill, warning=warning)


RECALL_SKILL_TOOL = ToolDef(
    name="recall_skill",
    description=_DESCRIPTION,
    input_schema=_INPUT_SCHEMA,
    handler=handle_recall_skill,
)


__all__ = ["RECALL_SKILL_TOOL", "handle_recall_skill"]
