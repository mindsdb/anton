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

from anton.core.tools.registry import ToolOutcome
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


_PROCEDURE_HEADER = "## Procedure (Stage 1 — declarative)"


def _already_in_history(session, label: str) -> bool:
    """True if a prior FULL recall of `label` is still visible in the history.

    Requires the marker AND the procedure header in the same message: only
    the full payload carries both. The stub deliberately contains neither
    (see below) — a stub that survives compaction while the body is evicted
    must not keep suppressing re-sends, and a summary that merely quotes the
    marker doesn't count as the contract being present.
    """
    history = getattr(session, "history", None)
    if not isinstance(history, list):
        return False
    marker = _recall_marker(label)
    try:
        import json as _json

        for m in history:
            # ensure_ascii=False: the header contains an em-dash, which
            # default json.dumps would escape to — and never match.
            text = (
                m
                if isinstance(m, str)
                else _json.dumps(m, default=str, ensure_ascii=False)
            )
            if marker in text and _PROCEDURE_HEADER in text:
                return True
        return False
    except Exception:  # noqa: BLE001 - never let the guard break a recall
        return False


async def handle_recall_skill(
    session: "ChatSession", tc_input: dict
) -> "str | ToolOutcome":
    """Look up a skill by label and return its declarative procedure.

    Verdicts (ENG-2248). `ok` drives the per-tool error streak, so it fires the
    resilience nudge at 2 consecutive failures and the circuit breaker at 5 —
    a verdict here is a behaviour decision, not a label. This tool ran 1,150
    times across 452 installs in 30 days, so a wrong `ok=False` reaches
    everyone.

    * `ok=True`  — a procedure was returned, INCLUDING the already-recalled
      stub: that is a success with a deliberately short body, not a failure.
    * `ok=False` — the call could not be served at all: no label, or no store.
      Repeating either cannot help, so it SHOULD reach the streak. This is the
      intended, accepted behaviour change.
    * `ok=None`  — the NO MATCH family, left unmigrated ON PURPOSE. See the
      comments at those returns.
    """
    label_in = (tc_input.get("label") or "").strip()
    if not label_in:
        # Tier 2 (ENG-2248): a malformed call. Retrying it unchanged cannot
        # work, so repetition IS thrash and belongs in the streak.
        return ToolOutcome(
            content=(
                "ERROR: recall_skill requires a non-empty 'label' parameter. "
                "Pick one from the procedural memory list in your system prompt."
            ),
            ok=False,
            reason="missing_name",
        )

    store = getattr(session, "_skill_store", None)
    if store is None:
        # Tier 2: the host wired no store, so no label can ever work.
        # `store_unavailable` is an existing `_SENTINEL_REASONS` key mapping to
        # external_wall/service_unavailable — reused, not invented.
        return ToolOutcome(
            content=(
                "ERROR: no skill store is wired into this session. "
                "Procedural memory is unavailable right now."
            ),
            ok=False,
            reason="store_unavailable",
        )

    skill = store.load(label_in)
    warning = ""
    if skill is None:
        closest = store.closest_match(label_in)
        if closest is None:
            available = [s["label"] for s in store.list_summaries()]
            # Tier 3 (ENG-2248): deliberately ok=None, NOT a failure. The tool
            # worked — it looked, found nothing, and told the model to proceed.
            # This is the most common non-success on the highest-volume tool, so
            # ok=False here would push normal exploration into the error streak
            # and trip the breaker on correct behaviour. If it should ever
            # count, that needs its own ticket and its own before/after on the
            # nudge rate.
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
            # Race or filesystem flake — be defensive.
            # Tier 3: left ok=None with the rest of the NO MATCH family.
            # Arguably a real failure (a load that should have worked), but it
            # is indistinguishable from a plain miss without a store-level
            # signal, and guessing wrong costs a breaker trip.
            return (
                f"NO MATCH: '{label_in}' was not found and the closest "
                f"candidate '{closest}' could not be loaded."
            )
        warning = (
            f"⚠ No skill named '{label_in}'. Returning the closest match: "
            f"'{skill.label}'. If that's not what you wanted, ignore the "
            f"procedure below and proceed without a recalled skill."
        )

    # Unlock gated tools before the early returns below, so the
    # bundle registers regardless of which path we take. Duck-typed session:
    # guard for minimal sessions lacking the hook.
    register_bundle = getattr(session, "_register_tool_bundle", None)
    if callable(register_bundle):
        register_bundle(skill.label)

    if _already_in_history(session, skill.label):
        # NOTE: this stub must never contain _recall_marker() or the
        # procedure header — otherwise a stub surviving compaction would
        # satisfy _already_in_history forever and the full contract would
        # never be re-sent.
        # Tier 1 (ENG-2248): a SUCCESS with a deliberately short body. The
        # procedure is already in context and still applies, so the tool did
        # its job. Behaviourally identical to today — a bare-string return
        # already resets the streak, since the legacy matcher finds none of its
        # five markers in this text.
        return ToolOutcome(
            content=(
                f"Skill '{skill.label}' was already recalled in this conversation "
                "— its full procedure is in your context above, under the "
                f"'# Skill: {skill.name}' heading, and still applies. Not "
                "re-sending the body."
            ),
            ok=True,
        )

    # Increment the recommended counter for the *resolved* label, not the
    # input. If the LLM typo'd 'csv-sumary', we credit 'csv-summary'.
    store.increment_recommended(skill.label, stage=1)

    # Tier 1: the procedure was returned. Also covers the closest-match path,
    # where `warning` explains the substitution — a substitution is still a
    # served request.
    return ToolOutcome(content=_format_skill_response(skill, warning=warning), ok=True)


RECALL_SKILL_TOOL = ToolDef(
    name="recall_skill",
    description=_DESCRIPTION,
    input_schema=_INPUT_SCHEMA,
    handler=handle_recall_skill,
)


__all__ = ["RECALL_SKILL_TOOL", "handle_recall_skill"]
