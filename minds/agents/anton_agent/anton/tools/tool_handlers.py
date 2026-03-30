"""Tool definitions and handlers for the Anton agent.

Ported from anton/tools.py.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..chat_session import ChatSession


async def handle_recall(session: ChatSession, tc_input: dict) -> str:
    """Process a recall tool call — search episodic memory."""
    if session._episodic is None or not session._episodic.enabled:
        return "Episodic memory is not available."

    query = tc_input.get("query", "")
    if not query:
        return "No query provided."

    kwargs: dict = {}
    if "max_results" in tc_input:
        kwargs["max_results"] = int(tc_input["max_results"])
    if "days_back" in tc_input:
        kwargs["days_back"] = int(tc_input["days_back"])

    return session._episodic.recall_formatted(query, **kwargs)


async def handle_memorize(session: ChatSession, tc_input: dict) -> str:
    """Process a memorize tool call and return a result string."""
    import asyncio

    if session._cortex is None:
        return "Memory system not available."

    if session._cortex.mode == "off":
        return "Memory encoding is disabled."

    from ..memory.hippocampus import Engram

    raw_entries = tc_input.get("entries", [])
    if not raw_entries:
        return "No entries provided."

    engrams: list[Engram] = []
    for entry in raw_entries:
        if not isinstance(entry, dict) or "text" not in entry:
            continue

        kind = entry.get("kind", "lesson")
        if kind not in ("always", "never", "when", "lesson", "profile"):
            kind = "lesson"

        scope = entry.get("scope", "project")
        if scope not in ("global", "project"):
            scope = "project"

        engrams.append(
            Engram(
                text=entry["text"],
                kind=kind,
                scope=scope,
                confidence="high",
                topic=entry.get("topic", ""),
                source="user",
            )
        )

    if not engrams:
        return "No valid entries provided."

    async def _encode_bg(cortex, entries):
        with contextlib.suppress(Exception):
            await cortex.encode(entries)

    asyncio.create_task(_encode_bg(session._cortex, engrams))

    descriptions = [f"Encoded {e.kind}: {e.text}" for e in engrams]
    return "Memory updated: " + "; ".join(descriptions)


async def prepare_scratchpad_exec(session: ChatSession, tc_input: dict):
    """Validate and prepare a scratchpad exec call."""
    code = tc_input.get("code", "")
    if not code or not code.strip():
        return "No code provided."

    pad = await session._scratchpads.get_or_create()

    packages = tc_input.get("packages", [])
    if packages:
        install_result = await pad.install_packages(packages)
        if "Install failed" in install_result or "timed out" in install_result:
            return install_result

    description = tc_input.get("one_line_description", "")
    estimated_seconds = tc_input.get("estimated_execution_time_seconds", 0)
    if isinstance(estimated_seconds, str):
        try:
            estimated_seconds = int(estimated_seconds)
        except ValueError:
            estimated_seconds = 0

    estimated_time = f"{estimated_seconds}s" if estimated_seconds > 0 else ""
    return pad, code, description, estimated_time, estimated_seconds


def format_cell_result(cell) -> str:
    """Format a Cell into a tool result string."""
    parts: list[str] = []
    if cell.stdout:
        stdout = cell.stdout
        if len(stdout) > 10_000:
            stdout = stdout[:10_000] + f"\n\n... (truncated, {len(stdout)} chars total)"
        parts.append(f"[output]\n{stdout}")
    if cell.logs if hasattr(cell, "logs") else False:
        logs = cell.logs.strip()
        if len(logs) > 3_000:
            logs = logs[:3_000] + "\n... (logs truncated)"
        parts.append(f"[logs]\n{logs}")
    if cell.stderr:
        parts.append(f"[stderr]\n{cell.stderr}")
    if cell.error:
        parts.append(f"[error]\n{cell.error}")
    if not parts:
        return "Code executed successfully (no output)."
    return "\n".join(parts)


async def handle_scratchpad(session: ChatSession, tc_input: dict) -> str:
    """Dispatch a scratchpad tool call by action."""
    action = tc_input.get("action", "")

    if action == "exec":
        result = await prepare_scratchpad_exec(session, tc_input)
        if isinstance(result, str):
            return result
        pad, code, description, estimated_time, estimated_seconds = result

        cell = await pad.execute(
            code,
            description=description,
            estimated_time=estimated_time,
            estimated_seconds=estimated_seconds,
        )
        return format_cell_result(cell)

    elif action == "view":
        # The view op does not require a running backend.
        pad = await session._scratchpads.get_or_create(start=False)
        return pad.view()

    elif action == "reset":
        pad = await session._scratchpads.get_or_create(start=False)
        await pad.reset()
        return "Scratchpad reset. All state cleared."

    elif action == "remove":
        # The manager state will also need to be cleared, so it is handled there.
        return await session._scratchpads.remove()

    elif action == "dump":
        # The dump op does not require a running backend.
        pad = await session._scratchpads.get_or_create(start=False)
        return pad.render_notebook()

    elif action == "install":
        packages = tc_input.get("packages", [])
        if not packages:
            return "No packages specified."
        pad = await session._scratchpads.get_or_create()
        return await pad.install_packages(packages)

    else:
        return f"Unknown scratchpad action: {action}"


async def dispatch_tool(session: ChatSession, tool_name: str, tc_input: dict) -> str:
    """Dispatch a tool call by name. Returns result text."""
    if tool_name == "memorize":
        return await handle_memorize(session, tc_input)
    elif tool_name == "scratchpad":
        return await handle_scratchpad(session, tc_input)
    elif tool_name == "recall":
        return await handle_recall(session, tc_input)
    else:
        return f"Unknown tool: {tool_name}"
