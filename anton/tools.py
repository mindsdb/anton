"""Dynamic tool registry — decorator-based registration for chat tools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anton.chat import ChatSession


@dataclass
class ToolDef:
    name: str
    description: str
    input_schema: dict
    handler: Callable  # async (session, tc_input) -> str
    stream_handler: Callable | None = None  # async generator version


_registry: dict[str, ToolDef] = {}


def tool(name: str, *, description: str, input_schema: dict):
    """Decorator to register a tool with its handler."""
    def decorator(fn):
        _registry[name] = ToolDef(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=fn,
        )
        return fn
    return decorator


def tool_stream(name: str):
    """Decorator to register a streaming handler for an existing tool."""
    def decorator(fn):
        if name in _registry:
            _registry[name].stream_handler = fn
        return fn
    return decorator


def get_tool(name: str) -> ToolDef | None:
    return _registry.get(name)


def all_tools() -> list[ToolDef]:
    return list(_registry.values())


def build_tool_schemas(available: list[str]) -> list[dict]:
    """Build API-ready tool schema dicts for the given tool names."""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in _registry.values()
        if t.name in available
    ]


MEMORIZE_TOOL = {
    "name": "memorize",
    "description": (
        "Encode a rule or lesson into long-term memory for future sessions. "
        "Use this when you learn something important, discover a useful pattern, "
        "or the user asks you to remember something.\n\n"
        "Entry kinds:\n"
        "- always: Something to always do ('Use httpx instead of requests')\n"
        "- never: Something to never do ('Never use time.sleep() in scratchpad')\n"
        "- when: Conditional rule ('If paginated API → use async + progress()')\n"
        "- lesson: Factual knowledge ('CoinGecko rate-limits at 50/min')\n"
        "- profile: Fact about the user ('Name: Jorge', 'Prefers dark mode')"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "entries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The memory to encode",
                        },
                        "kind": {
                            "type": "string",
                            "enum": ["always", "never", "when", "lesson", "profile"],
                        },
                        "scope": {
                            "type": "string",
                            "enum": ["global", "project"],
                        },
                        "topic": {
                            "type": "string",
                            "description": "Topic slug for lessons (e.g. 'api-coingecko')",
                        },
                    },
                    "required": ["text", "kind", "scope"],
                },
            },
        },
        "required": ["entries"],
    },
}

RECALL_TOOL = {
    "name": "recall",
    "description": (
        "Search your episodic memory — an archive of past conversations. "
        "ONLY use this when the user explicitly asks about a previous conversation "
        "or session (e.g. 'what did we talk about last time?', 'remember when we...', "
        "'have we discussed X before?'). Do NOT use this for questions about code, "
        "files, or data in the workspace — use the scratchpad to explore those directly.\n\n"
        "Returns timestamped episodes matching the query (newest first). "
        "A single call is enough — do not call multiple times with different queries."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search term to find in past conversations.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum episodes to return (default 20).",
            },
            "days_back": {
                "type": "integer",
                "description": "Only search episodes from the last N days.",
            },
        },
        "required": ["query"],
    },
}

CONNECT_DATASOURCE_TOOL = {
    "name": "connect_new_datasource",
    "description": (
        "Connect a new data source to Anton's Local Vault. Call this when the user "
        "asks a question that requires data from a source that isn't connected yet "
        "(e.g. email, database, CRM, API). This starts an interactive connection flow "
        "where the user enters their credentials.\n\n"
        "Pass the datasource type/name (e.g. 'gmail', 'postgres', 'salesforce', 'hubspot'). "
        "Anton will match it to the right connector and guide the user through setup.\n\n"
        "Do NOT print any message before calling this tool — it handles the user-facing output."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "engine": {
                "type": "string",
                "description": "The datasource type or name (e.g. 'gmail', 'postgres', 'snowflake', 'hubspot')",
            },
            "reason": {
                "type": "string",
                "description": "Brief explanation of why this datasource is needed",
            },
        },
        "required": ["engine"],
    },
}

SCRATCHPAD_TOOL = {
    "name": "scratchpad",
    "description": (
        "Run Python code in a persistent scratchpad. Use this whenever you need to "
        "count characters, do math, parse data, transform text, or any task that "
        "benefits from precise computation rather than guessing. Variables, imports, "
        "and data persist across cells — like a notebook you drive programmatically.\n\n"
        "Actions:\n"
        "- exec: Run code in the scratchpad (creates it if needed)\n"
        "- view: See all cells and their outputs\n"
        "- reset: Restart the process, clearing all state (installed packages survive)\n"
        "- remove: Kill the scratchpad and delete its environment\n"
        "- dump: Show a clean notebook-style summary of cells (code + truncated output)\n"
        "- install: Install Python packages into the scratchpad's environment. "
        "Packages persist across resets.\n\n"
        "Use print() to produce output. Host Python packages are available by default. "
        "Include a 'packages' array on exec calls for any libraries your code needs — "
        "they'll be auto-installed before the cell runs (already-installed ones are skipped).\n"
        "get_llm() returns a pre-configured LLM client (sync) — call "
        "llm.complete(system=..., messages=[...]) for AI-powered computation.\n"
        "llm.generate_object(MyModel, system=..., messages=[...]) extracts structured "
        "data into Pydantic models. Supports single models and list[Model].\n"
        "agentic_loop(system=..., user_message=..., tools=[...], handle_tool=fn) "
        "runs a tool-call loop where the LLM reasons and calls your tools iteratively. "
        "handle_tool(name, inputs) -> str is a plain sync function.\n"
        "sample(var) inspects any variable with type-aware formatting — DataFrames get "
        "shape/dtypes/head, dicts get keys/values, lists get length/items. "
        "Defaults to 'preview' mode (compact); use sample(var, mode='full') for complete dump.\n"
        "All .anton/.env secrets are available as environment variables (os.environ).\n\n"
        "IMPORTANT: Cells have an inactivity timeout of 30 seconds — if a cell produces "
        "no output and no progress() calls for 30s, it is killed and all state is lost. "
        "For long-running code (API calls, data extraction, heavy computation), call "
        "progress(message) periodically to signal work is ongoing and reset the timer. "
        "The total timeout scales from your estimated_execution_time_seconds "
        "(roughly 2x the estimate). You MUST provide estimated_execution_time_seconds "
        "for every exec call. For very long operations, provide a realistic estimate "
        "and use progress() to keep the cell alive."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["exec", "view", "reset", "remove", "dump", "install"]},
            "name": {"type": "string", "description": "Scratchpad name"},
            "code": {
                "type": "string",
                "description": "Python code (exec only). Use print() for output.",
            },
            "packages": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Package names needed by this cell (exec or install). "
                "Listed after code so you know exactly what to include. "
                "Already-installed packages are skipped automatically.",
            },
            "one_line_description": {
                "type": "string",
                "description": "Brief description of what this cell does (e.g. 'Scrape listing prices'). Required for exec.",
            },
            "estimated_execution_time_seconds": {
                "type": "integer",
                "description": "Estimated execution time in seconds. Drives the total timeout (roughly 2x estimate). Use progress() for long cells.",
            },
        },
        "required": ["action", "name"],
    },
}

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
    """Process a memorize tool call and return a result string.

    Encoding is fire-and-forget so it never blocks scratchpad execution.
    """
    import asyncio

    if session._cortex is None:
        return "Memory system not available."

    if session._cortex.mode == "off":
        return "Memory encoding is disabled. Change memory mode via /setup to enable."

    from anton.memory.hippocampus import Engram

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

        # User-sourced memories (via explicit tool call) get high confidence
        engrams.append(Engram(
            text=entry["text"],
            kind=kind,
            scope=scope,
            confidence="high",
            topic=entry.get("topic", ""),
            source="user",
        ))

    if not engrams:
        return "No valid entries provided."

    # Always encode immediately via fire-and-forget — the LLM explicitly
    # chose to memorize these, so we never interrupt the user mid-turn
    # with confirmation prompts.  Confirmations are reserved for the
    # post-turn consolidator (lessons extracted from scratchpad sessions).
    async def _encode_bg(cortex, entries):
        try:
            await cortex.encode(entries)
        except Exception:
            pass  # Best-effort; don't disrupt the conversation

    asyncio.create_task(_encode_bg(session._cortex, engrams))

    descriptions = [f"Encoded {e.kind}: {e.text}" for e in engrams]
    return "Memory updated: " + "; ".join(descriptions)


async def prepare_scratchpad_exec(session: ChatSession, tc_input: dict):
    """Validate and prepare a scratchpad exec call.

    Returns (pad, code, description, estimated_time, estimated_seconds) or
    a str error message if validation fails.
    """
    name = tc_input.get("name", "")
    code = tc_input.get("code", "")
    if not code or not code.strip():
        return "No code provided."

    pad = await session._scratchpads.get_or_create(name)

    # Auto-install packages before running the cell
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
    """Format a Cell into a tool result string.

    Every section is labeled so the LLM can tell what came from where:
    [output] — print() / stdout from the cell code
    [logs]   — library logging (httpx, urllib3, etc.) captured at INFO+
    [stderr] — warnings and stderr writes
    [error]  — Python traceback if the cell raised an exception
    """
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
    name = tc_input.get("name", "")

    if not name:
        return "Scratchpad name is required."

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
        pad = session._scratchpads._pads.get(name)
        if pad is None:
            return f"No scratchpad named '{name}'."
        return pad.view()

    elif action == "reset":
        pad = session._scratchpads._pads.get(name)
        if pad is None:
            return f"No scratchpad named '{name}'."
        await pad.reset()
        return f"Scratchpad '{name}' reset. All state cleared."

    elif action == "remove":
        return await session._scratchpads.remove(name)

    elif action == "dump":
        pad = session._scratchpads._pads.get(name)
        if pad is None:
            return f"No scratchpad named '{name}'."
        return pad.render_notebook()

    elif action == "install":
        packages = tc_input.get("packages", [])
        if not packages:
            return "No packages specified."
        pad = await session._scratchpads.get_or_create(name)
        return await pad.install_packages(packages)

    else:
        return f"Unknown scratchpad action: {action}"


async def handle_connect_datasource(session: ChatSession, tc_input: dict) -> str:
    """Handle connect_new_datasource tool call — interactive connection flow."""
    engine = tc_input.get("engine", "")
    reason = tc_input.get("reason", "")
    if not engine:
        return "Engine name is required."

    console = session._console
    if console is None:
        return "Cannot connect datasource — no console available."

    console.print()
    console.print(
        f"[anton.prompt]anton>[/] I can help with that \u2014 let's connect [bold]{engine}[/] to Anton."
    )

    from anton.chat import _handle_connect_datasource
    from anton.prompt_utils import prompt_or_cancel
    from anton.data_vault import DataVault

    # Check which connections exist before
    vault = DataVault()
    before = {f"{c['engine']}-{c['name']}" for c in vault.list_connections()}

    await _handle_connect_datasource(
        console,
        session._scratchpads,
        session,
        prefill=engine,
    )

    # Check if a new connection was actually added
    after = {f"{c['engine']}-{c['name']}" for c in vault.list_connections()}
    new_connections = after - before

    if new_connections:
        slug = next(iter(new_connections))
        return (
            f"Successfully connected '{slug}'. The datasource is now available. "
            f"Continue helping the user with their original request using this data source."
        )
    else:
        # User cancelled or connection failed — show briefly with spinner
        # so user knows the agent is picking back up
        from rich.live import Live
        from rich.spinner import Spinner
        from rich.text import Text
        import asyncio

        console.print()
        console.print("[anton.muted]  No worries, let's continue where we left off.[/]")
        with Live(
            Spinner("dots", text=Text("", style="anton.muted"), style="anton.cyan"),
            console=console,
            refresh_per_second=10,
            transient=True,
        ):
            await asyncio.sleep(1.5)
        console.print()
        return (
            f"CANCELLED: The user pressed Escape and cancelled the '{engine}' connection. "
            f"STOP — do NOT call connect_new_datasource again. Do NOT retry. "
            f"Acknowledge the cancellation briefly and ask the user what they'd like to do instead. "
            f"Respond with TEXT ONLY — no tool calls."
        )


async def dispatch_tool(session: ChatSession, tool_name: str, tc_input: dict) -> str:
    """Dispatch a tool call by name. Returns result text."""
    if tool_name == "memorize":
        return await handle_memorize(session, tc_input)
    elif tool_name == "scratchpad":
        return await handle_scratchpad(session, tc_input)
    elif tool_name == "recall":
        return await handle_recall(session, tc_input)
    elif tool_name == "connect_new_datasource":
        return await handle_connect_datasource(session, tc_input)
    else:
        return f"Unknown tool: {tool_name}"
