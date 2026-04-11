"""Slash-command handlers for the skills system.

Commands:

- `/skill save [optional name hint]` — LLM reads recent scratchpad work +
  conversation history and drafts a Skill (label, name, when_to_use,
  declarative procedure). Saved automatically; no interactive editing.
- `/skills list` — show all saved skills with usage counters.
- `/skill show <label>` — print a single skill's full procedure + stats.
- `/skill remove <label>` — delete a skill from disk.

Brain analogue: this is the experience-to-procedure consolidation step.
The user explicitly marks a successful piece of work as "worth
remembering as a procedure." The LLM does the synthesis (prefrontal
cortex deciding what was structural vs. contextual), and the result
gets written to long-term procedural memory. Future invocations
retrieve via the recall_skill tool.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from anton.core.memory.skills import (
    Skill,
    SkillStats,
    SkillStore,
    make_unique_label,
    slugify,
)

if TYPE_CHECKING:
    from anton.core.session import ChatSession


# ─────────────────────────────────────────────────────────────────────────────
# LLM-facing schema (Pydantic) — used by LLMClient.generate_object
# ─────────────────────────────────────────────────────────────────────────────


class _SkillDraft(BaseModel):
    """Structured output of the /skill save LLM call.

    The LLM is forced to call a tool whose input matches this schema,
    so the call site never has to parse JSON or strip fences.
    """

    label: str = Field(
        ...,
        description=(
            "snake_case identifier for the skill. Short (2-4 words), "
            "captures the essence. Examples: 'csv_summary', "
            "'web_scraping', 'api_paginated_fetch'."
        ),
    )
    name: str = Field(
        ...,
        description="Human-readable display name (e.g. 'CSV Summary').",
    )
    description: str = Field(
        default="",
        description="One-sentence description of what the skill does.",
    )
    when_to_use: str = Field(
        ...,
        description=(
            "One sentence describing when this skill applies — what the "
            "user has to ask for. This is the most important field — "
            "it's what the classifier shows to the LLM next time. "
            "Specific enough that the LLM can recognize matches "
            "without being too narrow."
        ),
    )
    declarative_md: str = Field(
        ...,
        description=(
            "Step-by-step procedure as markdown. Numbered steps. Be "
            "specific about decisions made (which library, why), "
            "reference the actual approach taken — not generic advice. "
            "A future agent will read this and follow it on a similar "
            "but not identical task. Write as instructions for a "
            "future agent, not a retrospective. Do not invent steps "
            "that didn't happen."
        ),
    )


_DRAFT_SYSTEM_PROMPT = (
    "You are helping a user save a reusable procedure (a 'skill') based on "
    "work they just completed. You will be given the recent scratchpad "
    "execution history and the relevant conversation turns. Your job is to "
    "synthesize them into a step-by-step procedure that a future agent (you) "
    "can follow when faced with a similar task."
)


_DRAFT_USER_PROMPT_TEMPLATE = """\
The user just ran the following command and wants to save the underlying procedure as a reusable skill.

{name_hint_section}

## Conversation context (most recent turns)

{conversation}

## Scratchpad execution history

{scratchpad}

---

Synthesize this into a reusable skill following the schema you've been given.
"""


def _format_scratchpad_cells(cells: list) -> str:
    """Render scratchpad cells as a compact text block for the LLM."""
    if not cells:
        return "(no scratchpad work in this session)"
    chunks: list[str] = []
    for i, cell in enumerate(cells, 1):
        code = (getattr(cell, "code", "") or "").strip()
        stdout = (getattr(cell, "stdout", "") or "").strip()
        stderr = (getattr(cell, "stderr", "") or "").strip()
        error = getattr(cell, "error", None)
        chunks.append(f"### Cell {i}")
        if code:
            # Truncate very long cells to keep the prompt manageable
            code_excerpt = code if len(code) <= 2000 else code[:2000] + "\n... [truncated]"
            chunks.append("```python")
            chunks.append(code_excerpt)
            chunks.append("```")
        if stdout:
            stdout_excerpt = stdout if len(stdout) <= 800 else stdout[:800] + "\n... [truncated]"
            chunks.append("stdout:")
            chunks.append(stdout_excerpt)
        if stderr:
            chunks.append("stderr:")
            chunks.append(stderr[:400])
        if error:
            chunks.append(f"error: {str(error)[:400]}")
        chunks.append("")
    return "\n".join(chunks)


def _format_history_turns(history: list[dict], *, max_turns: int = 8) -> str:
    """Render recent conversation history as plain text, skipping tool blocks."""
    if not history:
        return "(no conversation history yet)"
    lines: list[str] = []
    # Walk backwards collecting up to max_turns user/assistant turns with text
    collected: list[tuple[str, str]] = []
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        role = entry.get("role", "")
        content = entry.get("content", "")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            # Extract text blocks from structured content
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
            text = "\n".join(text_parts).strip()
        text = text.strip()
        if not text:
            continue
        if role not in ("user", "assistant"):
            continue
        collected.append((role, text))
        if len(collected) >= max_turns:
            break
    if not collected:
        return "(no readable conversation turns)"
    # Reverse back to chronological
    for role, text in reversed(collected):
        excerpt = text if len(text) <= 1000 else text[:1000] + "\n... [truncated]"
        lines.append(f"**{role}:** {excerpt}")
        lines.append("")
    return "\n".join(lines).strip()


def _gather_session_scratchpad_cells(session: "ChatSession") -> list:
    """Collect cells from every scratchpad in the session."""
    pads = getattr(session._scratchpads, "_pads", {})
    out: list = []
    for pad in pads.values():
        cells = getattr(pad, "cells", None) or []
        out.extend(cells)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# /skill save
# ─────────────────────────────────────────────────────────────────────────────


async def handle_skill_save(
    console: Console,
    session: "ChatSession",
    *,
    name_hint: str = "",
    store: SkillStore | None = None,
) -> None:
    """Draft a skill from recent work and save it to the procedural memory store."""
    store = store or getattr(session, "_skill_store", None) or SkillStore()

    cells = _gather_session_scratchpad_cells(session)
    history = getattr(session, "_history", []) or []

    if not cells and not history:
        console.print()
        console.print(
            "[anton.warning](anton)[/] Nothing to save yet — there's no scratchpad work "
            "or conversation history in this session."
        )
        console.print()
        return

    name_hint_section = (
        f"The user suggested the name: {name_hint!r}. "
        "Use it as the basis for `name` and `label`, but you may refine the label "
        "to be snake_case and short.\n"
        if name_hint.strip()
        else ""
    )

    user_prompt = _DRAFT_USER_PROMPT_TEMPLATE.format(
        name_hint_section=name_hint_section,
        conversation=_format_history_turns(history),
        scratchpad=_format_scratchpad_cells(cells),
    )

    console.print()
    console.print("[anton.cyan](anton)[/] Drafting a skill from recent work…")

    try:
        draft: _SkillDraft = await session._llm.generate_object(
            _SkillDraft,
            system=_DRAFT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=1500,
        )
    except Exception as exc:
        console.print()
        console.print(
            f"[anton.warning](anton)[/] Couldn't draft the skill: {exc}"
        )
        console.print()
        return

    raw_label = draft.label.strip() or slugify(name_hint or draft.name)
    name = draft.name.strip() or raw_label.replace("_", " ").title()
    description = draft.description.strip()
    when_to_use = draft.when_to_use.strip()
    declarative_md = draft.declarative_md.strip()

    if not declarative_md:
        console.print()
        console.print(
            "[anton.warning](anton)[/] The drafted skill has no procedure — refusing to save."
        )
        console.print()
        return

    label = make_unique_label(raw_label, store)

    skill = Skill(
        label=label,
        name=name,
        description=description,
        when_to_use=when_to_use,
        declarative_md=declarative_md,
        created_at=datetime.now(timezone.utc).isoformat(),
        provenance="manual",
    )
    path = store.save(skill)

    console.print()
    console.print(
        f"[anton.success](anton)[/] Saved skill [bold]{label}[/bold] → {path}"
    )
    console.print(f"        [anton.muted]Name:[/] {name}")
    if when_to_use:
        console.print(f"        [anton.muted]When to use:[/] {when_to_use}")
    console.print(
        "        [anton.muted]Available next session — and via `recall_skill` this turn.[/]"
    )
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# /skills list
# ─────────────────────────────────────────────────────────────────────────────


def handle_skills_list(console: Console, store: SkillStore | None = None) -> None:
    store = store or SkillStore()
    skills = store.list_all()
    console.print()
    if not skills:
        console.print(
            "[anton.muted]No skills saved yet. Use [bold]/skill save[/bold] "
            "after a successful task to create one.[/]"
        )
        console.print()
        return

    table = Table(title="Procedural memory — saved skills", show_lines=False)
    table.add_column("Label", style="bold")
    table.add_column("Name")
    table.add_column("When to use")
    table.add_column("Recalls", justify="right")
    table.add_column("Stages")

    for s in skills:
        stages = []
        if s.stage_1_present:
            stages.append("1")
        if s.stage_2_present:
            stages.append("2")
        if s.stage_3_present:
            stages.append("3")
        when = s.when_to_use if len(s.when_to_use) <= 60 else s.when_to_use[:57] + "..."
        table.add_row(
            s.label,
            s.name,
            when,
            str(s.stats.total_recalls),
            ",".join(stages) or "-",
        )

    console.print(table)
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# /skill show
# ─────────────────────────────────────────────────────────────────────────────


def handle_skill_show(
    console: Console, label: str, store: SkillStore | None = None
) -> None:
    store = store or SkillStore()
    if not label:
        console.print()
        console.print("[anton.warning]Usage: /skill show <label>[/]")
        console.print()
        return
    skill = store.load(label)
    if skill is None:
        closest = store.closest_match(label)
        console.print()
        if closest:
            console.print(
                f"[anton.warning]No skill '{label}'. Did you mean '{closest}'?[/]"
            )
        else:
            console.print(f"[anton.warning]No skill named '{label}'.[/]")
        console.print()
        return

    console.print()
    console.print(f"[anton.cyan](anton)[/] [bold]{skill.name}[/]  ([dim]{skill.label}[/])")
    if skill.description:
        console.print(f"        {skill.description}")
    if skill.when_to_use:
        console.print(f"        [dim]when to use:[/] {skill.when_to_use}")
    console.print()
    console.print(
        f"        [dim]recalls:[/] {skill.stats.total_recalls}  "
        f"[dim]stage 1:[/] {skill.stats.stage_1.recommended}  "
        f"[dim]stage 2:[/] {skill.stats.stage_2.recommended}  "
        f"[dim]stage 3 used:[/] {skill.stats.stage_3.used}"
    )
    console.print()
    console.print(Markdown(skill.declarative_md))
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# /skill remove
# ─────────────────────────────────────────────────────────────────────────────


def handle_skill_remove(
    console: Console, label: str, store: SkillStore | None = None
) -> None:
    store = store or SkillStore()
    if not label:
        console.print()
        console.print("[anton.warning]Usage: /skill remove <label>[/]")
        console.print()
        return
    if store.delete(label):
        console.print()
        console.print(
            f"[anton.success](anton)[/] Removed skill [bold]{label}[/bold]."
        )
        console.print()
    else:
        console.print()
        console.print(f"[anton.warning]No skill named '{label}'.[/]")
        console.print()


__all__ = [
    "handle_skill_save",
    "handle_skill_show",
    "handle_skill_remove",
    "handle_skills_list",
]
