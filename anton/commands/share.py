"""Slash-command handlers for /share."""
from __future__ import annotations

import getpass
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from rich.console import Console

if TYPE_CHECKING:
    from anton.core.llm.client import LLMClient
    from anton.core.memory.episodes import EpisodicMemory
    from anton.core.session import ChatSession
    from anton.workspace import Workspace


class _SessionMeta(BaseModel):
    title: str = Field(
        description=(
            "A 5-7 word title in lowercase-with-hyphens, suitable as a filename slug. "
            "Example: 'pipeline-latency-root-cause-analysis'"
        )
    )
    summary: str = Field(
        description=(
            "A distilled narrative of the analytical session: the goal, key discoveries, "
            "any corrections or dead ends, and where the analysis currently stands. "
            "Each distinct finding appears exactly once. 2-4 sentences."
        )
    )


def _format_history_for_llm(history: list[dict], max_messages: int = 20) -> str:
    recent = history[-max_messages:]
    lines = []
    for msg in recent:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = " ".join(text_parts)
        lines.append(f"{role}: {str(content)[:400]}")
    return "\n".join(lines)


def _slugify(title: str) -> str:
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:60] or "session"


async def _generate_meta(
    llm_client: LLMClient,
    history: list[dict],
    session_id: str,
) -> tuple[str, str]:
    try:
        conversation_text = _format_history_for_llm(history)
        result = await llm_client.generate_object_code(
            _SessionMeta,
            system=(
                "You are producing a portable context distillation of an analytical session. "
                "For the summary: cover the goal, key discoveries, any corrections or dead ends, "
                "and where the analysis currently stands. Every distinct finding should appear once. "
                "No filler, no repetition, no omissions of meaningful conclusions."
            ),
            messages=[{
                "role": "user",
                "content": f"Distill this analytical session:\n\n{conversation_text}",
            }],
            max_tokens=300,
        )
        return _slugify(result.title), result.summary
    except Exception:
        return f"session-{session_id}", ""


async def handle_share_export(
    console: Console,
    session: "ChatSession",
    workspace: "Workspace",
    llm_client: "LLMClient",
    episodic: "EpisodicMemory | None",
    *,
    summary_only: bool = False,
) -> None:
    session_id = session._session_id
    if not session_id:
        console.print("[anton.warning]No active session to export.[/]")
        console.print()
        return

    if not episodic:
        console.print("[anton.muted]Episodic memory not enabled — memory snapshot will be empty.[/]")
        return

    history = [] if summary_only else list(session._history)

    msg_count = len(session._history)
    if not summary_only and msg_count > 100:
        console.print(
            f"[anton.warning]This session has {msg_count} messages. "
            "Consider [bold]/share export --summary[/] for a lighter file.[/]"
        )
        console.print()

    # memory snapshot
    episodes = episodic.get_memory_usage(
        session_id
    )
    exportable = [e for e in episodes if e.meta.get("kind") != "profile"]
    session_born = [
        {
            "content": e.content,
            "kind": e.meta.get("kind", ""),
            "topic": e.meta.get("topic", ""),
        }
        for e in exportable if e.role == "memory_write"
    ]
    project_accessed = [
        {
            "content": e.content,
            "kind": e.meta.get("kind", ""),
            "topic": e.meta.get("topic", ""),
        }
        for e in exportable if e.role == "memory_read"
    ]

    # scratchpad cells
    cells: list[dict] = []
    for pad_name, runtime in session._scratchpads.pads.items():
        for cell in runtime.cells:
            cells.append({
                "pad": pad_name,
                "code": cell.code,
                "stdout": cell.stdout,
                "stderr": cell.stderr,
                "error": cell.error,
                "description": cell.description,
            })

    console.print("[anton.muted]Generating session summary…[/]")
    title_slug, summary = await _generate_meta(llm_client, session._history, session_id)

    exported_at = datetime.now(timezone.utc).isoformat()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    filename = f"{title_slug}_{timestamp}.anton"

    payload = {
        "version": "0.1",
        "exported_by": getpass.getuser(),
        "exported_at": exported_at,
        "session": {
            "id": session_id,
            "title": title_slug.replace("-", " ").title(),
            "summary": summary,
            "conversation_history": history,
        },
        "memory": {
            "session_born": session_born,
            "project_accessed": project_accessed,
        },
        "scratchpad": {
            "cells": cells,
        },
    }

    output_dir = workspace.base / ".anton" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / filename

    tmp_fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, dest)
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise

    console.print()
    console.print("[bold][anton.cyan]Session exported[/][/]")
    console.print(f"  [bold]File:[/]    {dest}")
    console.print(f"  [bold]Title:[/]   {payload['session']['title']}")
    if summary:
        console.print(f"  [bold]Summary:[/] {summary}")
    console.print(
        f"  [bold]Memory:[/]  {len(session_born)} session-born, "
        f"{len(project_accessed)} project memories"
    )
    console.print(f"  [bold]Code:[/]    {len(cells)} scratchpad cells")
    if episodic and not session_born and not project_accessed:
        console.print(
            "[anton.muted]  No project memories were delivered in this session.[/]"
        )
    console.print()
