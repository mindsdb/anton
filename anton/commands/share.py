"""Slash-command handlers for /share."""
from __future__ import annotations

import ast
import getpass
import json
import os
import re
import tempfile
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from rich.console import Console

if TYPE_CHECKING:
    from anton.config.settings import AntonSettings
    from anton.core.llm.client import LLMClient
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

        return result.title, result.summary
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

    history = [] if summary_only else [asdict(ep) for ep in episodic.get_conversation()]

    msg_count = len(session._history)
    if not summary_only and msg_count > 100:
        console.print(
            f"[anton.warning]This session has {msg_count} messages. "
            "Consider [bold]/share export --summary[/] for a lighter file.[/]"
        )
        console.print()

    # memory snapshot
    episodes = episodic.get_memory_usage()
    session_born, project_accessed = [], []
    for e in episodes:
        item = {
            "content": e.content,
            "kind": e.meta.get("kind", ""),
            "topic": e.meta.get("topic", ""),
        }
        if e.role == "memory_write":
            session_born.append(item)
        if e.role == "memory_read":
            project_accessed.append(item)

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
    title, summary = await _generate_meta(llm_client, session._history, session_id)

    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    slug = slug[:60] or "session"

    exported_at = datetime.now(timezone.utc).isoformat()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{slug}_{timestamp}.anton"

    payload = {
        "version": "0.1",
        "exported_by": getpass.getuser(),
        "exported_at": exported_at,
        "session": {
            "id": session_id,
            "title": title,
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


# ── status ───────────────────────────────────────────────────────────────────


def _find_import_record(output_dir: Path, session_id: str) -> dict | None:
    """Return the .anton payload that was imported into session_id, or None."""
    if not output_dir.exists():
        return None
    for p in output_dir.glob("*.anton"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("imported", {}).get("session_id") == session_id:
                return data
        except Exception:
            continue
    return None

def handle_share_status(
    console: Console,
    session: "ChatSession",
    workspace: "Workspace",
) -> None:
    session_id = session._session_id

    console.print()
    console.print("[bold]Shared session status[/]")
    console.print()

    if not session_id:
        console.print("[anton.muted]  No active session.[/]")
        console.print()
        return

    output_dir = workspace.base / ".anton" / "output"
    record = _find_import_record(output_dir, session_id)

    if not record:
        console.print(f"  [bold]Status:[/]  Session is not imported")
        return

    sess = record.get("session", {})
    imp = record.get("imported", {})
    console.print(f"  [bold]Title:[/]       {sess.get('title', '—')}")
    console.print(
        f"  [bold]Exported by:[/] {record.get('exported_by', '?')} · "
        f"{record.get('exported_at', '')[:10]}"
    )
    if sess.get("summary"):
        console.print(f"  [bold]Summary:[/]     {sess['summary']}")
    console.print()
    console.print(
        f"  [bold]Imported by:[/] {imp.get('user', '?')} · "
        f"{imp.get('date', '')[:10]}"
    )

    console.print()

    from anton.core.datasources.data_vault import LocalDataVault
    connections = LocalDataVault().list_connections()
    connected_ds = {f"{c['engine']}_{c['name']}".lower() for c in connections}

    used_ds = set()
    for entry in record.get("session", {}).get("conversation_history", []):
        if not isinstance(entry, dict):
            continue
        for ds_name in (entry.get("meta") or {}).get("datasources") or []:
            used_ds.add(ds_name)

    if used_ds:
        console.print("  [bold]Data sources[/]")
        for ds_name in used_ds:
            if ds_name in connected_ds:
                mark = "[green]✓[/]"
                note = "connected"
            else:
                mark = "[yellow]![/]"
                note = "[anton.warning]not connected[/]"
            console.print(f"    {mark}  {ds_name} · {note}")
    else:
        console.print("[anton.muted]  No data sources referenced in this session.[/]")

    console.print()


# ── history ──────────────────────────────────────────────────────────────────


def handle_share_history(
    console: Console,
    workspace: "Workspace",
) -> None:
    output_dir = workspace.base / ".anton" / "output"

    console.print()

    files = []
    if output_dir.exists():
        files = sorted(output_dir.glob("*.anton"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not files:
        console.print("[anton.muted]No exported sessions found.[/]")
        console.print()
        return

    console.print(f"[bold]Exported sessions[/]  ({len(files)} files)")
    console.print()

    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            console.print(f"  [anton.warning]{p.name}[/] — corrupted or unreadable")
            console.print()
            continue

        sess = data.get("session", {})
        imp = data.get("imported", {})

        title = sess.get("title") or p.stem
        summary = sess.get("summary", "")

        if imp:
            date = imp.get("date", "")[:16].replace("T", " ")
            who = imp.get("user", "?")
            label = f"imported by {who} · {date}"
        else:
            date = data.get("exported_at", "")[:16].replace("T", " ")
            who = data.get("exported_by", "?")
            label = f"exported by {who} · {date}"

        console.print(f"  [bold]{title}[/]  [anton.muted]{label}[/]")
        if summary:
            short = summary[:120] + "…" if len(summary) > 120 else summary
            console.print(f"  {short}")
        console.print(f"  [anton.muted]→ {p}[/]")
        console.print()


# ── import ────────────────────────────────────────────────────────────────────


def _episodic_to_api_history(episodes: list[dict]) -> list[dict]:
    """Convert episodic episode list to Anthropic API message format for HistoryStore.

    Processes episodes sequentially:
      user -> {"role":"user","content":text}
      tool_call -> {"role":"assistant","content":[tool_use block]}  (generates id)
      scratchpad -> skipped (content captured in tool_result)
      tool_result -> {"role":"user","content":[tool_result block]}  (uses id from preceding tool_call)
      assistant -> {"role":"assistant","content":text}
    """
    history: list[dict] = []
    i = 0
    while i < len(episodes):
        ep = episodes[i]
        role = ep.get("role", "")

        if role == "user":
            history.append({"role": "user", "content": ep["content"]})
            i += 1

        elif role == "tool_call":
            tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
            tool_name = ep.get("meta", {}).get("tool", "unknown")
            content_str = ep.get("content", "{}")
            try:
                tool_input = json.loads(content_str)
            except Exception:
                try:
                    tool_input = ast.literal_eval(content_str)
                except Exception:
                    tool_input = {"raw": content_str}

            history.append({
                "role": "assistant",
                "content": [{"type": "tool_use", "id": tool_id, "name": tool_name, "input": tool_input}],
            })
            i += 1

            # Skip optional scratchpad episode
            if i < len(episodes) and episodes[i].get("role") == "scratchpad":
                i += 1

            # Consume matching tool_result
            if i < len(episodes) and episodes[i].get("role") == "tool_result":
                history.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": episodes[i]["content"]}],
                })
                i += 1

        elif role == "assistant":
            history.append({"role": "assistant", "content": ep["content"]})
            i += 1

        else:
            i += 1

    return history


async def import_v0_1(
        console: Console,
        session: "ChatSession",
        workspace: "Workspace",
        settings: "AntonSettings",
        state: dict,
        self_awareness,
        cortex: "Cortex | None",
        episodic: "EpisodicMemory | None",
        history_store: "HistoryStore | None",
        payload: dict,
        *,
        source_path: Path,
) -> "ChatSession":
    from anton.commands.session import restore_session
    from anton.utils.prompt import prompt_or_cancel

    # warn if active session
    if session._history:
        console.print(
            "[anton.warning]You have an active session in progress. "
            "Importing will create a new session — your current work is preserved in history.[/]"
        )
        console.print()
        choice = await prompt_or_cancel(
            "(anton) Continue?",
            choices=["y", "n"],
            choices_display="y/n",
            default="n",
        )
        if choice is None or choice != "y":
            console.print()
            return session

    raw_history = payload.get("session", {}).get("conversation_history", [])

    # 1. fill episodic from export
    if episodic and episodic.enabled:
        episodic.start_session()
    new_session_id = episodic._session_id if (episodic and episodic.enabled) else None

    # if episodic and episodic.enabled and new_session_id:
    #     _restore_episodes_to_episodic(episodic, raw_history, new_session_id)

    # 2. reconstruct API history and save to history_store
    api_history = _episodic_to_api_history(raw_history)
    if history_store and new_session_id:
        history_store.save(new_session_id, api_history)

    # stamp imported metadata and persist file to output/
    payload["imported"] = {
        "user": getpass.getuser(),
        "date": datetime.now(timezone.utc).isoformat(),
        "session_id": new_session_id,
    }
    output_dir = workspace.base / ".anton" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / source_path.name
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3. resume session (closes old scratchpads, rebuild_session, loads _history)
    new_session, _ = await restore_session(
        new_session_id, console, settings, state, self_awareness, cortex, workspace,
        session, episodic, history_store,
    )

    session_born = payload.get("memory", {}).get("session_born", [])
    project_accessed = payload.get("memory", {}).get("project_accessed", [])

    # log memories to episodic
    if episodic and episodic.enabled:
        for m in session_born:
            episodic.log_turn(0, "memory_write", m["content"],
                              kind=m.get("kind", ""), topic=m.get("topic", ""))
        for m in project_accessed:
            episodic.log_turn(0, "memory_read", m["content"],
                              kind=m.get("kind", ""), topic=m.get("topic", ""))

    # restore memories to cortex
    if cortex:
        for m in session_born + project_accessed:
            kind = m.get("kind", "")
            content = m.get("content", "")
            topic = m.get("topic", "")
            if kind in ("always", "never", "when"):
                cortex.project_hc.encode_rule(content, kind=kind, source="import")
            elif kind == "lesson":
                cortex.project_hc.encode_lesson(content, topic=topic, source="import")

    # restore scratchpad cells
    from anton.core.backends.base import Cell as _Cell  # noqa: PLC0415
    cells_data = payload.get("scratchpad", {}).get("cells", [])
    for cell_data in cells_data:
        pad_name = cell_data.get("pad", "main")
        pad = await new_session._scratchpads.get_or_create(pad_name)
        pad.cells.append(_Cell(
            code=cell_data.get("code", ""),
            stdout=cell_data.get("stdout", ""),
            stderr=cell_data.get("stderr", ""),
            error=cell_data.get("error"),
            description=cell_data.get("description", ""),
        ))

    # print briefing
    sess = payload.get("session", {})
    cells = payload.get("scratchpad", {}).get("cells", [])

    console.print()
    console.print(f"[bold][anton.cyan]Imported: {sess.get('title', 'Session')}[/][/]")
    console.print(
        f"  [bold]From:[/]    {payload.get('exported_by', '?')} · "
        f"{payload.get('exported_at', '')[:10]}"
    )
    if sess.get("summary"):
        console.print(f"  [bold]Summary:[/] {sess['summary']}")
    if new_session._turn_count:
        console.print(f"  [bold]Turns:[/]   {new_session._turn_count}")
    if session_born or project_accessed:
        console.print(
            f"  [bold]Memory:[/]  {len(session_born)} session-born, "
            f"{len(project_accessed)} project memories"
        )
    if cells:
        console.print(f"  [bold]Code:[/]    {len(cells)} scratchpad cells")
    console.print()
    console.print("[anton.muted]Session restored. Continue where it left off.[/]")
    console.print()

    return new_session


async def handle_share_import(
    console: Console,
    session: "ChatSession",
    workspace: "Workspace",
    settings: "AntonSettings",
    state: dict,
    self_awareness,
    cortex: "Cortex | None",
    episodic: "EpisodicMemory | None",
    history_store: "HistoryStore | None",
    filepath: str,
) -> "ChatSession":

    # parse & validate
    path = Path(filepath).expanduser()
    if not path.is_file():
        console.print(f"[anton.warning]File not found: {filepath}[/]")
        console.print()
        return session

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        console.print("[anton.warning]Could not read file — may be corrupted.[/]")
        console.print()
        return session

    version = payload.get("version")

    importers = {
        "0.1": import_v0_1
    }

    if version not in importers:
        console.print(
            f"[anton.warning]Unsupported version: {version}. Supported versions: {list(importers.keys())}.[/]"
        )
        console.print()
        return session

    return await importers[version](
        console, session, workspace, settings, state, self_awareness, cortex,
        episodic, history_store, payload,
        source_path=path,
    )
