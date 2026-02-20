from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import anthropic

from anton.channel.base import Channel
from anton.events.types import AntonEvent, StatusUpdate, TaskComplete, TaskFailed
from anton.llm.prompts import CHAT_SYSTEM_PROMPT
from anton.llm.provider import (
    StreamComplete,
    StreamEvent,
    StreamTaskProgress,
    StreamTextDelta,
    StreamToolResult,
    StreamToolUseStart,
)
from anton.minds import MindsClient
from anton.scratchpad import ScratchpadManager

if TYPE_CHECKING:
    from rich.console import Console

    from anton.config.settings import AntonSettings
    from anton.context.self_awareness import SelfAwarenessContext
    from anton.llm.client import LLMClient
    from anton.workspace import Workspace

EXECUTE_TASK_TOOL = {
    "name": "execute_task",
    "description": (
        "Execute a coding task autonomously through Anton's agent pipeline. "
        "Call this when you have enough context to act on the user's request."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "A clear, specific description of the task to execute.",
            },
        },
        "required": ["task"],
    },
}

UPDATE_CONTEXT_TOOL = {
    "name": "update_context",
    "description": (
        "Update self-awareness context files when you learn something important "
        "about the project or workspace. Use this to persist knowledge for future sessions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {
                            "type": "string",
                            "description": "Filename like 'project-overview.md'",
                        },
                        "content": {
                            "type": ["string", "null"],
                            "description": "New content, or null to delete the file",
                        },
                    },
                    "required": ["file", "content"],
                },
            },
        },
        "required": ["updates"],
    },
}

REQUEST_SECRET_TOOL = {
    "name": "request_secret",
    "description": (
        "Request a secret value (API key, token, password) from the user. "
        "The value is stored directly in .anton/.env and NEVER passed through the LLM. "
        "After calling this, you will be told the variable has been set — use it by name."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "variable_name": {
                "type": "string",
                "description": "Environment variable name to store (e.g. 'GITHUB_TOKEN', 'DATABASE_PASSWORD')",
            },
            "prompt_text": {
                "type": "string",
                "description": "What to ask the user (e.g. 'Please enter your GitHub personal access token')",
            },
        },
        "required": ["variable_name", "prompt_text"],
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
        "Packages persist across resets. Use this when you need a library that isn't "
        "already available.\n\n"
        "Use print() to produce output. Host Python packages are available by default. "
        "Use the install action to add more.\n"
        "run_skill(name, **kwargs) is available in code to call Anton skills.\n"
        "get_llm() returns a pre-configured LLM client (sync) — call "
        "llm.complete(system=..., messages=[...]) for AI-powered computation.\n"
        "llm.generate_object(MyModel, system=..., messages=[...]) extracts structured "
        "data into Pydantic models. Supports single models and list[Model].\n"
        "agentic_loop(system=..., user_message=..., tools=[...], handle_tool=fn) "
        "runs a tool-call loop where the LLM reasons and calls your tools iteratively. "
        "handle_tool(name, inputs) -> str is a plain sync function.\n"
        "All .anton/.env secrets are available as environment variables (os.environ).\n"
        "get_minds() returns a pre-configured Minds client (sync) for natural language "
        "database queries. Call minds.ask(question, mind) to query, then minds.export() "
        "to get CSV data — perfect for pd.read_csv(io.StringIO(csv)). Also: minds.data() "
        "for markdown tables, minds.catalog(datasource) to discover tables."
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
                "description": "Package names to install (install only).",
            },
        },
        "required": ["action", "name"],
    },
}


MINDS_TOOL = {
    "name": "minds",
    "description": (
        "Query databases using natural language via MindsDB. "
        "Minds translates your questions into SQL — you never write SQL directly. "
        "Data stays in MindsDB, only results come back.\n\n"
        "Actions:\n"
        "- ask: Ask a natural language question. Returns a text answer.\n"
        "- data: Fetch raw tabular results from the last ask (as a markdown table). "
        "Call this after ask when you need the actual data for analysis.\n"
        "- export: Export the full result set from the last ask as CSV. "
        "Perfect for loading into pandas with pd.read_csv(io.StringIO(csv)).\n"
        "- catalog: Discover available tables and columns for a datasource."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["ask", "data", "export", "catalog"],
            },
            "mind": {
                "type": "string",
                "description": "The mind (model) name to query. Required for 'ask'.",
            },
            "question": {
                "type": "string",
                "description": "Natural language question. Required for 'ask'.",
            },
            "datasource": {
                "type": "string",
                "description": "Datasource name. Required for 'catalog'.",
            },
            "limit": {
                "type": "integer",
                "description": "Max rows to return (default 100). For 'data' only.",
            },
            "offset": {
                "type": "integer",
                "description": "Row offset for pagination (default 0). For 'data' only.",
            },
        },
        "required": ["action"],
    },
}


MINDS_KNOWLEDGE_PROMPT = """\
You are summarizing what a database mind can do. You'll be given the mind name, \
its datasources, and a schema catalog of tables and columns.

Write a concise 2-4 paragraph summary covering:
1. What the data contains — describe the domain and key entities.
2. What questions this mind can answer — give 3-5 concrete example questions.
3. When to use this mind — what kinds of user requests should be routed here.

Be specific about table and column names so the reader knows exactly what's available, \
but do NOT include the raw schema. Write in plain English.

Mind: {mind_name}
Datasources: {datasources}

Schema catalog:
{catalog}
"""


class _ProgressChannel(Channel):
    """Channel that captures agent events into an asyncio.Queue instead of rendering."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[AntonEvent | None] = asyncio.Queue()

    async def emit(self, event: AntonEvent) -> None:
        await self.queue.put(event)

    async def prompt(self, question: str) -> str:
        return ""

    async def close(self) -> None:
        await self.queue.put(None)


class ChatSession:
    """Manages a multi-turn conversation with tool-call delegation."""

    def __init__(
        self,
        llm_client: LLMClient,
        run_task,
        *,
        run_task_stream=None,
        self_awareness: SelfAwarenessContext | None = None,
        runtime_context: str = "",
        workspace: Workspace | None = None,
        console: Console | None = None,
        skill_dirs: list[Path] | None = None,
        coding_provider: str = "anthropic",
        coding_api_key: str = "",
        minds_api_key: str = "",
        minds_base_url: str = "https://mdb.ai",
    ) -> None:
        self._llm = llm_client
        self._run_task = run_task
        self._run_task_stream = run_task_stream
        self._self_awareness = self_awareness
        self._runtime_context = runtime_context
        self._workspace = workspace
        self._console = console
        self._history: list[dict] = []
        self._scratchpads = ScratchpadManager(
            skill_dirs=skill_dirs,
            coding_provider=coding_provider,
            coding_model=getattr(llm_client, "coding_model", ""),
            coding_api_key=coding_api_key,
        )
        self._minds: MindsClient | None = (
            MindsClient(api_key=minds_api_key, base_url=minds_base_url)
            if minds_api_key
            else None
        )

    @property
    def history(self) -> list[dict]:
        return self._history

    @property
    def _minds_dir(self) -> Path | None:
        """Directory for connected-mind knowledge files."""
        if self._workspace is None:
            return None
        return self._workspace.base / ".anton" / "minds"

    def _build_minds_knowledge_section(self) -> str:
        """Read .anton/minds/*.md and return a formatted section for the system prompt."""
        minds_dir = self._minds_dir
        if minds_dir is None or not minds_dir.is_dir():
            return ""
        md_files = sorted(minds_dir.glob("*.md"))
        if not md_files:
            return ""
        parts: list[str] = ["\n\n## Connected Minds\n"]
        for f in md_files:
            mind_name = f.stem
            content = f.read_text(encoding="utf-8").strip()
            if content:
                parts.append(f"### {mind_name}\n{content}\n")
        return "\n".join(parts) if len(parts) > 1 else ""

    def _build_system_prompt(self) -> str:
        prompt = CHAT_SYSTEM_PROMPT.format(runtime_context=self._runtime_context)
        if self._self_awareness is not None:
            sa_section = self._self_awareness.build_prompt_section()
            if sa_section:
                prompt += sa_section
        # Inject anton.md project context
        if self._workspace is not None:
            md_context = self._workspace.build_anton_md_context()
            if md_context:
                prompt += md_context
        # Inject connected-minds knowledge
        minds_section = self._build_minds_knowledge_section()
        if minds_section:
            prompt += minds_section
        return prompt

    # Packages the LLM is most likely to care about when writing scratchpad code.
    _NOTABLE_PACKAGES: set[str] = {
        "numpy", "pandas", "matplotlib", "seaborn", "scipy", "scikit-learn",
        "requests", "httpx", "aiohttp", "beautifulsoup4", "lxml",
        "pillow", "sympy", "networkx", "sqlalchemy", "pydantic",
        "rich", "tqdm", "click", "fastapi", "flask", "django",
        "openai", "anthropic", "tiktoken", "transformers", "torch",
        "polars", "pyarrow", "openpyxl", "xlsxwriter",
        "plotly", "bokeh", "altair",
        "pytest", "hypothesis",
        "yaml", "pyyaml", "toml", "tomli", "tomllib",
        "jinja2", "markdown", "pygments",
        "cryptography", "paramiko", "boto3",
    }

    def _build_tools(self) -> list[dict]:
        scratchpad_tool = dict(SCRATCHPAD_TOOL)
        pkg_list = self._scratchpads._available_packages
        if pkg_list:
            notable = sorted(
                p for p in pkg_list
                if p.lower() in self._NOTABLE_PACKAGES
            )
            if notable:
                pkg_line = ", ".join(notable)
                extra = f"\n\nInstalled packages ({len(pkg_list)} total, notable: {pkg_line})."
            else:
                extra = f"\n\nInstalled packages: {len(pkg_list)} total (standard library plus dependencies)."
            scratchpad_tool["description"] = SCRATCHPAD_TOOL["description"] + extra

        tools = [EXECUTE_TASK_TOOL, scratchpad_tool]
        if self._self_awareness is not None:
            tools.append(UPDATE_CONTEXT_TOOL)
        if self._workspace is not None:
            tools.append(REQUEST_SECRET_TOOL)
        if self._minds is not None:
            minds_tool = dict(MINDS_TOOL)
            default_mind = os.environ.get("MINDS_DEFAULT_MIND", "")
            if default_mind:
                minds_tool["description"] = (
                    MINDS_TOOL["description"]
                    + f"\n\nDefault mind: {default_mind} (used when no mind is specified)."
                )
            tools.append(minds_tool)
        return tools

    def _handle_update_context(self, tc_input: dict) -> str:
        """Process an update_context tool call and return a result string."""
        if self._self_awareness is None:
            return "Context updates not available."

        from anton.context.self_awareness import ContextUpdate

        raw_updates = tc_input.get("updates", [])
        updates = [
            ContextUpdate(file=u["file"], content=u.get("content"))
            for u in raw_updates
            if isinstance(u, dict) and "file" in u
        ]

        if not updates:
            return "No valid updates provided."

        actions = self._self_awareness.apply_updates(updates)
        return "Context updated: " + "; ".join(actions)

    def _handle_request_secret(self, tc_input: dict) -> str:
        """Handle a request_secret tool call.

        Asks the user directly for the secret value, stores it in .env,
        and returns a confirmation — NEVER returns the actual secret value.
        """
        if self._workspace is None or self._console is None:
            return "Secret storage not available."

        var_name = tc_input.get("variable_name", "")
        prompt_text = tc_input.get("prompt_text", f"Enter value for {var_name}")

        if not var_name:
            return "No variable_name provided."

        # Check if already set
        if self._workspace.has_secret(var_name):
            return f"Variable {var_name} is already set in .anton/.env."

        # Ask user directly — this bypasses the LLM entirely
        self._console.print()
        value = self._console.input(f"[bold]{prompt_text}:[/] ")
        value = value.strip()

        if not value:
            return f"No value provided for {var_name}. Variable not set."

        # Store securely — value never touches the LLM
        self._workspace.set_secret(var_name, value)
        return f"Variable {var_name} has been set in .anton/.env. You can now use it."

    async def _handle_scratchpad(self, tc_input: dict) -> str:
        """Dispatch a scratchpad tool call by action."""
        action = tc_input.get("action", "")
        name = tc_input.get("name", "")

        if not name:
            return "Scratchpad name is required."

        if action == "exec":
            code = tc_input.get("code", "")
            if not code or not code.strip():
                return "No code provided."
            pad = await self._scratchpads.get_or_create(name)
            cell = await pad.execute(code)

            parts: list[str] = []
            if cell.stdout:
                stdout = cell.stdout
                if len(stdout) > 10_000:
                    stdout = stdout[:10_000] + f"\n\n... (truncated, {len(stdout)} chars total)"
                parts.append(stdout)
            if cell.stderr:
                parts.append(f"[stderr]\n{cell.stderr}")
            if cell.error:
                parts.append(f"[error]\n{cell.error}")
            if not parts:
                return "Code executed successfully (no output)."
            return "\n".join(parts)

        elif action == "view":
            pad = self._scratchpads._pads.get(name)
            if pad is None:
                return f"No scratchpad named '{name}'."
            return pad.view()

        elif action == "reset":
            pad = self._scratchpads._pads.get(name)
            if pad is None:
                return f"No scratchpad named '{name}'."
            await pad.reset()
            return f"Scratchpad '{name}' reset. All state cleared."

        elif action == "remove":
            return await self._scratchpads.remove(name)

        elif action == "dump":
            pad = self._scratchpads._pads.get(name)
            if pad is None:
                return f"No scratchpad named '{name}'."
            return pad.render_notebook()

        elif action == "install":
            packages = tc_input.get("packages", [])
            if not packages:
                return "No packages specified."
            pad = await self._scratchpads.get_or_create(name)
            return await pad.install_packages(packages)

        else:
            return f"Unknown scratchpad action: {action}"

    async def _handle_minds(self, tc_input: dict) -> str:
        """Dispatch a minds tool call by action."""
        if self._minds is None:
            return "Minds is not configured. Use /minds to set up."

        action = tc_input.get("action", "")

        try:
            if action == "ask":
                question = tc_input.get("question", "")
                if not question:
                    return "A 'question' is required for the ask action."
                mind = tc_input.get("mind", "") or os.environ.get("MINDS_DEFAULT_MIND", "")
                if not mind:
                    return (
                        "A 'mind' name is required. Specify it in the tool call "
                        "or set a default via /minds."
                    )
                return await self._minds.ask(question, mind)

            elif action == "data":
                limit = tc_input.get("limit", 100)
                offset = tc_input.get("offset", 0)
                return await self._minds.data(limit=limit, offset=offset)

            elif action == "export":
                return await self._minds.export()

            elif action == "catalog":
                datasource = tc_input.get("datasource", "")
                if not datasource:
                    return "A 'datasource' name is required for the catalog action."
                return await self._minds.catalog(datasource)

            else:
                return f"Unknown minds action: {action}"

        except ValueError as exc:
            return str(exc)
        except Exception as exc:
            return f"Minds error: {exc}"

    async def _handle_minds_connect(self, mind_name: str, console: Console) -> None:
        """Connect a mind: fetch schema, summarize with LLM, store knowledge file."""
        # Validate name
        if not mind_name or "/" in mind_name or "\\" in mind_name or mind_name.startswith("."):
            console.print("[anton.error]Invalid mind name.[/]")
            return

        if self._minds is None:
            console.print("[anton.error]Minds not configured. Run /minds setup first.[/]")
            return

        minds_dir = self._minds_dir
        if minds_dir is None:
            console.print("[anton.error]No workspace available.[/]")
            return

        console.print(f"[anton.cyan]Connecting mind '{mind_name}'...[/]")

        # 1. Fetch mind metadata
        try:
            mind_info = await self._minds.get_mind(mind_name)
        except Exception as exc:
            console.print(f"[anton.error]Failed to fetch mind '{mind_name}': {exc}[/]")
            return

        datasources = mind_info.get("datasources", [])
        if not datasources:
            console.print("[anton.warning]Mind has no datasources.[/]")

        # 2. Build catalog — try API first, fall back to get_mind metadata, then ask
        catalog_parts: list[str] = []
        catalog_ok = False
        for ds in datasources:
            ds_name = ds if isinstance(ds, str) else ds.get("name", str(ds))
            # 2a. Try the catalog API endpoint
            try:
                cat = await self._minds.catalog(ds_name, mind=mind_name)
                catalog_parts.append(f"# Datasource: {ds_name}\n{cat}")
                catalog_ok = True
                continue
            except Exception:
                pass

            # 2b. Fallback: extract table list from get_mind response
            tables = ds.get("tables", []) if isinstance(ds, dict) else []
            if tables:
                table_listing = "\n".join(f"  - {t}" for t in tables)
                catalog_parts.append(
                    f"# Datasource: {ds_name}\nAvailable tables:\n{table_listing}"
                )
                catalog_ok = True
            else:
                catalog_parts.append(f"# Datasource: {ds_name}\n(catalog unavailable)")

        # 2c. Last resort: if nothing worked, ask the mind itself
        if not catalog_ok:
            console.print("[anton.muted]Catalog unavailable — asking the mind directly...[/]")
            try:
                introspect = await self._minds.ask(
                    "List every table you have access to and their columns. "
                    "Be specific — include table names, column names, and data types.",
                    mind_name,
                )
                catalog_parts = [f"# Self-reported schema (from asking the mind)\n{introspect}"]
            except Exception:
                pass  # keep whatever we had

        combined_catalog = "\n\n".join(catalog_parts) if catalog_parts else "(no catalog available)"
        ds_names = ", ".join(
            ds if isinstance(ds, str) else ds.get("name", str(ds))
            for ds in datasources
        ) if datasources else "(none)"

        # 3. Ask LLM to summarize
        summary: str | None = None
        try:
            prompt = MINDS_KNOWLEDGE_PROMPT.format(
                mind_name=mind_name,
                datasources=ds_names,
                catalog=combined_catalog,
            )
            resp = await self._llm.plan(
                system="You are a technical writer producing concise database documentation.",
                messages=[{"role": "user", "content": prompt}],
            )
            summary = resp.content
        except Exception:
            pass

        # 4. Write knowledge file
        if not summary:
            summary = f"Mind: {mind_name}\nDatasources: {ds_names}\n\n{combined_catalog}"

        minds_dir.mkdir(parents=True, exist_ok=True)
        (minds_dir / f"{mind_name}.md").write_text(summary, encoding="utf-8")
        console.print(f"[anton.success]Mind '{mind_name}' connected. Knowledge stored in .anton/minds/{mind_name}.md[/]")

    def _handle_minds_disconnect(self, mind_name: str, console: Console) -> None:
        """Disconnect a mind by removing its knowledge file."""
        if not mind_name:
            console.print("[anton.error]Provide a mind name to disconnect.[/]")
            return

        minds_dir = self._minds_dir
        if minds_dir is None:
            console.print("[anton.error]No workspace available.[/]")
            return

        md_file = minds_dir / f"{mind_name}.md"
        if md_file.exists():
            md_file.unlink()
            console.print(f"[anton.success]Mind '{mind_name}' disconnected.[/]")
        else:
            console.print(f"[anton.warning]Mind '{mind_name}' is not connected.[/]")

    def _handle_minds_status(self, console: Console) -> None:
        """Show current Minds configuration and connected minds."""
        console.print()
        console.print("[anton.cyan]Minds status:[/]")

        # API key
        current_key = os.environ.get("MINDS_API_KEY", "")
        if current_key:
            masked = current_key[:4] + "..." + current_key[-4:] if len(current_key) > 8 else "***"
            console.print(f"  API key:  [bold]{masked}[/]")
        else:
            console.print("  API key:  [dim]not set[/]")

        # Base URL
        console.print(f"  Base URL: [bold]{os.environ.get('MINDS_BASE_URL', 'https://mdb.ai')}[/]")

        # Connected minds
        minds_dir = self._minds_dir
        if minds_dir is not None and minds_dir.is_dir():
            connected = sorted(f.stem for f in minds_dir.glob("*.md"))
            if connected:
                console.print(f"  Connected minds: [bold]{', '.join(connected)}[/]")
            else:
                console.print("  Connected minds: [dim]none[/]")
        else:
            console.print("  Connected minds: [dim]none[/]")
        console.print()

    async def close(self) -> None:
        """Clean up scratchpads and other resources."""
        await self._scratchpads.close_all()

    async def turn(self, user_input: str) -> str:
        self._history.append({"role": "user", "content": user_input})

        system = self._build_system_prompt()
        tools = self._build_tools()

        response = await self._llm.plan(
            system=system,
            messages=self._history,
            tools=tools,
        )

        # Handle tool calls (execute_task, update_context, request_secret)
        while response.tool_calls:
            # Build assistant message with content blocks
            assistant_content: list[dict] = []
            if response.content:
                assistant_content.append({"type": "text", "text": response.content})
            for tc in response.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input,
                })
            self._history.append({"role": "assistant", "content": assistant_content})

            # Process each tool call
            tool_results: list[dict] = []
            for tc in response.tool_calls:
                if tc.name == "execute_task":
                    task_desc = tc.input.get("task", "")
                    try:
                        await self._run_task(task_desc)
                        result_text = f"Task completed: {task_desc}"
                    except Exception as exc:
                        result_text = f"Task failed: {exc}"
                elif tc.name == "update_context":
                    result_text = self._handle_update_context(tc.input)
                elif tc.name == "request_secret":
                    result_text = self._handle_request_secret(tc.input)
                elif tc.name == "scratchpad":
                    result_text = await self._handle_scratchpad(tc.input)
                elif tc.name == "minds":
                    result_text = await self._handle_minds(tc.input)
                else:
                    result_text = f"Unknown tool: {tc.name}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result_text,
                })

            self._history.append({"role": "user", "content": tool_results})

            # Get follow-up from LLM
            response = await self._llm.plan(
                system=system,
                messages=self._history,
                tools=tools,
            )

        # Text-only response
        reply = response.content or ""
        self._history.append({"role": "assistant", "content": reply})
        return reply

    async def turn_stream(self, user_input: str) -> AsyncIterator[StreamEvent]:
        """Streaming version of turn(). Yields events as they arrive."""
        self._history.append({"role": "user", "content": user_input})

        async for event in self._stream_and_handle_tools():
            yield event

    async def _stream_and_handle_tools(self) -> AsyncIterator[StreamEvent]:
        """Stream one LLM call, handle tool loops, yield all events."""
        system = self._build_system_prompt()
        tools = self._build_tools()

        response: StreamComplete | None = None

        async for event in self._llm.plan_stream(
            system=system,
            messages=self._history,
            tools=tools,
        ):
            yield event
            if isinstance(event, StreamComplete):
                response = event

        if response is None:
            return

        llm_response = response.response

        # Tool-call loop
        while llm_response.tool_calls:
            # Build assistant message with content blocks
            assistant_content: list[dict] = []
            if llm_response.content:
                assistant_content.append({"type": "text", "text": llm_response.content})
            for tc in llm_response.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input,
                })
            self._history.append({"role": "assistant", "content": assistant_content})

            # Process each tool call
            tool_results: list[dict] = []
            for tc in llm_response.tool_calls:
                if tc.name == "execute_task":
                    task_desc = tc.input.get("task", "")
                    try:
                        if self._run_task_stream is not None:
                            async for progress in self._run_task_stream(task_desc):
                                yield progress
                        else:
                            await self._run_task(task_desc)
                        result_text = f"Task completed: {task_desc}"
                    except Exception as exc:
                        result_text = f"Task failed: {exc}"
                elif tc.name == "update_context":
                    result_text = self._handle_update_context(tc.input)
                elif tc.name == "request_secret":
                    result_text = self._handle_request_secret(tc.input)
                elif tc.name == "scratchpad":
                    result_text = await self._handle_scratchpad(tc.input)
                    if tc.input.get("action") == "dump":
                        yield StreamToolResult(content=result_text)
                elif tc.name == "minds":
                    result_text = await self._handle_minds(tc.input)
                else:
                    result_text = f"Unknown tool: {tc.name}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result_text,
                })

            self._history.append({"role": "user", "content": tool_results})

            # Stream follow-up
            response = None
            async for event in self._llm.plan_stream(
                system=system,
                messages=self._history,
                tools=tools,
            ):
                yield event
                if isinstance(event, StreamComplete):
                    response = event

            if response is None:
                return
            llm_response = response.response

        # Text-only final response — append to history
        reply = llm_response.content or ""
        self._history.append({"role": "assistant", "content": reply})


def _rebuild_session(
    *,
    settings: AntonSettings,
    state: dict,
    self_awareness,
    workspace,
    console: Console,
    skill_dirs: list[Path],
    do_run_task,
    do_run_task_stream,
) -> ChatSession:
    """Rebuild LLMClient + ChatSession after settings change."""
    from anton.llm.client import LLMClient

    state["llm_client"] = LLMClient.from_settings(settings)
    runtime_context = (
        f"- Provider: {settings.planning_provider}\n"
        f"- Planning model: {settings.planning_model}\n"
        f"- Coding model: {settings.coding_model}\n"
        f"- Workspace: {settings.workspace_path}\n"
    )
    api_key = (
        settings.anthropic_api_key if settings.coding_provider == "anthropic"
        else settings.openai_api_key
    ) or ""
    return ChatSession(
        state["llm_client"],
        do_run_task,
        run_task_stream=do_run_task_stream,
        self_awareness=self_awareness,
        runtime_context=runtime_context,
        workspace=workspace,
        console=console,
        skill_dirs=skill_dirs,
        coding_provider=settings.coding_provider,
        coding_api_key=api_key,
        minds_api_key=os.environ.get("MINDS_API_KEY", ""),
        minds_base_url=os.environ.get("MINDS_BASE_URL", "https://mdb.ai"),
    )


async def _handle_setup(
    console: Console,
    settings: AntonSettings,
    workspace: Workspace,
    state: dict,
    self_awareness,
    skill_dirs: list[Path],
    do_run_task,
    do_run_task_stream,
) -> ChatSession:
    """Interactive setup wizard — reconfigure provider, model, and API key."""
    from rich.prompt import Prompt

    console.print()
    console.print("[anton.cyan]Current configuration:[/]")
    console.print(f"  Provider (planning): [bold]{settings.planning_provider}[/]")
    console.print(f"  Provider (coding):   [bold]{settings.coding_provider}[/]")
    console.print(f"  Planning model:      [bold]{settings.planning_model}[/]")
    console.print(f"  Coding model:        [bold]{settings.coding_model}[/]")
    console.print()

    # --- Provider ---
    providers = {"1": "anthropic", "2": "openai"}
    current_num = "1" if settings.planning_provider == "anthropic" else "2"
    console.print("[anton.cyan]Available providers:[/]")
    console.print("  [bold]1[/]  Anthropic (Claude)")
    console.print("  [bold]2[/]  OpenAI (GPT / o-series)")
    console.print()

    choice = Prompt.ask(
        "Select provider",
        choices=["1", "2"],
        default=current_num,
        console=console,
    )
    provider = providers[choice]

    # --- API key ---
    key_attr = "anthropic_api_key" if provider == "anthropic" else "openai_api_key"
    current_key = getattr(settings, key_attr) or ""
    masked = current_key[:4] + "..." + current_key[-4:] if len(current_key) > 8 else "***"
    console.print()
    api_key = Prompt.ask(
        f"API key for {provider.title()} [dim](Enter to keep {masked})[/]",
        default="",
        console=console,
    )
    api_key = api_key.strip()

    # --- Models ---
    defaults = {
        "anthropic": ("claude-sonnet-4-6", "claude-opus-4-6"),
        "openai": ("gpt-4.1", "gpt-4.1"),
    }
    default_planning, default_coding = defaults.get(provider, ("", ""))

    console.print()
    planning_model = Prompt.ask(
        "Planning model",
        default=settings.planning_model if provider == settings.planning_provider else default_planning,
        console=console,
    )
    coding_model = Prompt.ask(
        "Coding model",
        default=settings.coding_model if provider == settings.coding_provider else default_coding,
        console=console,
    )

    # --- Persist ---
    settings.planning_provider = provider
    settings.coding_provider = provider
    settings.planning_model = planning_model
    settings.coding_model = coding_model

    workspace.set_secret("ANTON_PLANNING_PROVIDER", provider)
    workspace.set_secret("ANTON_CODING_PROVIDER", provider)
    workspace.set_secret("ANTON_PLANNING_MODEL", planning_model)
    workspace.set_secret("ANTON_CODING_MODEL", coding_model)

    if api_key:
        setattr(settings, key_attr, api_key)
        key_name = f"ANTON_{provider.upper()}_API_KEY"
        workspace.set_secret(key_name, api_key)

    console.print()
    console.print("[anton.success]Configuration updated.[/]")
    console.print()

    return _rebuild_session(
        settings=settings,
        state=state,
        self_awareness=self_awareness,
        workspace=workspace,
        console=console,
        skill_dirs=skill_dirs,
        do_run_task=do_run_task,
        do_run_task_stream=do_run_task_stream,
    )


async def _handle_minds_setup(
    console: Console,
    settings: AntonSettings,
    workspace: Workspace,
    state: dict,
    self_awareness,
    skill_dirs: list[Path],
    do_run_task,
    do_run_task_stream,
) -> ChatSession:
    """Interactive wizard to configure Minds (MindsDB) integration."""
    from rich.prompt import Prompt

    console.print()
    console.print("[anton.cyan]Minds (MindsDB) configuration:[/]")

    # Show current config
    current_key = os.environ.get("MINDS_API_KEY", "")
    current_url = os.environ.get("MINDS_BASE_URL", "https://mdb.ai")
    current_mind = os.environ.get("MINDS_DEFAULT_MIND", "")

    if current_key:
        masked = current_key[:4] + "..." + current_key[-4:] if len(current_key) > 8 else "***"
        console.print(f"  API key:      [bold]{masked}[/]")
    else:
        console.print("  API key:      [dim]not set[/]")
    console.print(f"  Base URL:     [bold]{current_url}[/]")
    if current_mind:
        console.print(f"  Default mind: [bold]{current_mind}[/]")
    else:
        console.print("  Default mind: [dim]not set[/]")
    console.print()

    # API key
    api_key = Prompt.ask(
        "Minds API key" + (f" [dim](Enter to keep {masked})[/]" if current_key else ""),
        default="",
        console=console,
    ).strip()

    # Base URL
    base_url = Prompt.ask(
        "Base URL",
        default=current_url,
        console=console,
    ).strip()

    # Default mind name
    default_mind = Prompt.ask(
        "Default mind name [dim](optional, Enter to skip)[/]",
        default=current_mind,
        console=console,
    ).strip()

    # Persist
    if api_key:
        workspace.set_secret("MINDS_API_KEY", api_key)
    if base_url != "https://mdb.ai":
        workspace.set_secret("MINDS_BASE_URL", base_url)
    if default_mind:
        workspace.set_secret("MINDS_DEFAULT_MIND", default_mind)

    console.print()
    console.print("[anton.success]Minds configuration updated.[/]")
    console.print()

    return _rebuild_session(
        settings=settings,
        state=state,
        self_awareness=self_awareness,
        workspace=workspace,
        console=console,
        skill_dirs=skill_dirs,
        do_run_task=do_run_task,
        do_run_task_stream=do_run_task_stream,
    )


def _print_slash_help(console: Console) -> None:
    """Print available slash commands."""
    console.print()
    console.print("[anton.cyan]Available commands:[/]")
    console.print("  [bold]/setup[/]              — Configure provider, model, and API key")
    console.print("  [bold]/minds[/]              — Show Minds status (API key, connected minds)")
    console.print("  [bold]/minds setup[/]        — Configure Minds API key and base URL")
    console.print("  [bold]/minds connect X[/]    — Connect a mind (fetches schema, stores knowledge)")
    console.print("  [bold]/minds disconnect X[/] — Disconnect a mind (removes stored knowledge)")
    console.print("  [bold]/help[/]               — Show this help message")
    console.print("  [bold]exit[/]                — Quit the chat")
    console.print()


def run_chat(console: Console, settings: AntonSettings) -> None:
    """Launch the interactive chat REPL."""
    asyncio.run(_chat_loop(console, settings))


async def _chat_loop(console: Console, settings: AntonSettings) -> None:
    from anton.channel.terminal import CLIChannel
    from anton.context.self_awareness import SelfAwarenessContext
    from anton.core.agent import Agent
    from anton.llm.client import LLMClient
    from anton.skill.registry import SkillRegistry
    from anton.workspace import Workspace

    # Use a mutable container so closures always see the current client
    state: dict = {"llm_client": LLMClient.from_settings(settings)}
    registry = SkillRegistry()

    builtin = Path(__file__).resolve().parent.parent / settings.skills_dir
    registry.discover(builtin)

    user_dir = Path(settings.user_skills_dir)
    registry.discover(user_dir)

    memory = None
    learnings_store = None
    if settings.memory_enabled:
        from anton.memory.learnings import LearningStore
        from anton.memory.store import SessionStore

        memory_dir = Path(settings.memory_dir)
        memory = SessionStore(memory_dir)
        learnings_store = LearningStore(memory_dir)

    channel = CLIChannel()

    # Self-awareness context
    self_awareness = SelfAwarenessContext(Path(settings.context_dir))

    # Workspace for anton.md and secret vault
    workspace = Workspace(settings.workspace_path)
    workspace.apply_env_to_process()

    async def _do_run_task(task: str) -> None:
        agent = Agent(
            channel=channel,
            llm_client=state["llm_client"],
            registry=registry,
            user_skills_dir=user_dir,
            memory=memory,
            learnings=learnings_store,
            self_awareness=self_awareness,
            skill_dirs=[builtin, user_dir],
        )
        await agent.run(task)

    async def _do_run_task_stream(task: str) -> AsyncIterator[StreamTaskProgress]:
        """Run agent task, yielding progress events as StreamTaskProgress."""
        progress_ch = _ProgressChannel()
        agent = Agent(
            channel=progress_ch,
            llm_client=state["llm_client"],
            registry=registry,
            user_skills_dir=user_dir,
            memory=memory,
            learnings=learnings_store,
            self_awareness=self_awareness,
            skill_dirs=[builtin, user_dir],
        )
        agent_task = asyncio.create_task(agent.run(task))

        try:
            while True:
                try:
                    event = await asyncio.wait_for(progress_ch.queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    if agent_task.done():
                        break
                    continue

                if event is None:
                    break

                if isinstance(event, StatusUpdate):
                    yield StreamTaskProgress(
                        phase=event.phase.value,
                        message=event.message,
                        eta_seconds=event.eta_seconds,
                    )
                elif isinstance(event, (TaskComplete, TaskFailed)):
                    break

            # Drain remaining events
            while not progress_ch.queue.empty():
                event = progress_ch.queue.get_nowait()
                if isinstance(event, StatusUpdate):
                    yield StreamTaskProgress(
                        phase=event.phase.value,
                        message=event.message,
                        eta_seconds=event.eta_seconds,
                    )

            # Re-raise any exception from the agent
            await agent_task
        except BaseException:
            if not agent_task.done():
                agent_task.cancel()
                try:
                    await agent_task
                except (asyncio.CancelledError, Exception):
                    pass
            raise

    # Build runtime context so the LLM knows what it's running on
    skill_names = [s.name for s in registry.list_all()]
    runtime_context = (
        f"- Provider: {settings.planning_provider}\n"
        f"- Planning model: {settings.planning_model}\n"
        f"- Coding model: {settings.coding_model}\n"
        f"- Workspace: {settings.workspace_path}\n"
        f"- Available skills: {', '.join(skill_names) if skill_names else 'none discovered'}\n"
        f"- Memory: {'enabled' if settings.memory_enabled else 'disabled'}"
    )

    skill_dirs = [builtin, user_dir]

    coding_api_key = (
        settings.anthropic_api_key if settings.coding_provider == "anthropic"
        else settings.openai_api_key
    ) or ""
    session = ChatSession(
        state["llm_client"],
        _do_run_task,
        run_task_stream=_do_run_task_stream,
        self_awareness=self_awareness,
        runtime_context=runtime_context,
        workspace=workspace,
        console=console,
        skill_dirs=skill_dirs,
        coding_provider=settings.coding_provider,
        coding_api_key=coding_api_key,
        minds_api_key=os.environ.get("MINDS_API_KEY", ""),
        minds_base_url=os.environ.get("MINDS_BASE_URL", "https://mdb.ai"),
    )

    console.print("[anton.muted]Chat with Anton. Type '/help' for commands or 'exit' to quit.[/]")
    console.print()

    from anton.chat_ui import StreamDisplay

    display = StreamDisplay(console)

    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import ANSI

    prompt_session: PromptSession[str] = PromptSession(mouse_support=False)

    try:
        while True:
            try:
                user_input = await prompt_session.prompt_async(ANSI("\033[1myou>\033[0m "))
            except EOFError:
                break

            stripped = user_input.strip()
            if not stripped:
                continue
            if stripped.lower() in ("exit", "quit", "bye"):
                break

            # Slash command dispatch
            if stripped.startswith("/"):
                parts = stripped.split()
                cmd = parts[0].lower()
                if cmd == "/setup":
                    session = await _handle_setup(
                        console, settings, workspace, state,
                        self_awareness, skill_dirs,
                        _do_run_task, _do_run_task_stream,
                    )
                elif cmd == "/minds":
                    subcmd = parts[1].lower() if len(parts) > 1 else ""
                    if subcmd == "setup":
                        session = await _handle_minds_setup(
                            console, settings, workspace, state,
                            self_awareness, skill_dirs,
                            _do_run_task, _do_run_task_stream,
                        )
                    elif subcmd == "connect":
                        if len(parts) < 3:
                            console.print("[anton.warning]Usage: /minds connect <name>[/]")
                        else:
                            await session._handle_minds_connect(parts[2], console)
                    elif subcmd == "disconnect":
                        if len(parts) < 3:
                            console.print("[anton.warning]Usage: /minds disconnect <name>[/]")
                        else:
                            session._handle_minds_disconnect(parts[2], console)
                    else:
                        session._handle_minds_status(console)
                elif cmd == "/help":
                    _print_slash_help(console)
                else:
                    console.print(f"[anton.warning]Unknown command: {cmd}[/]")
                continue

            display.start()
            t0 = time.monotonic()
            ttft: float | None = None
            total_input = 0
            total_output = 0

            try:
                async for event in session.turn_stream(stripped):
                    if isinstance(event, StreamTextDelta):
                        if ttft is None:
                            ttft = time.monotonic() - t0
                        display.append_text(event.text)
                    elif isinstance(event, StreamToolResult):
                        display.show_tool_result(event.content)
                    elif isinstance(event, StreamToolUseStart):
                        display.show_tool_execution(event.name)
                    elif isinstance(event, StreamTaskProgress):
                        display.update_progress(
                            event.phase, event.message, event.eta_seconds
                        )
                    elif isinstance(event, StreamComplete):
                        total_input += event.response.usage.input_tokens
                        total_output += event.response.usage.output_tokens

                elapsed = time.monotonic() - t0
                display.finish(total_input, total_output, elapsed, ttft)
            except anthropic.AuthenticationError:
                display.abort()
                console.print()
                console.print(
                    "[anton.error]Invalid API key. Let's set up a new one.[/]"
                )
                settings.anthropic_api_key = None
                from anton.cli import _ensure_api_key
                _ensure_api_key(settings)
                session = _rebuild_session(
                    settings=settings,
                    state=state,
                    self_awareness=self_awareness,
                    workspace=workspace,
                    console=console,
                    skill_dirs=skill_dirs,
                    do_run_task=_do_run_task,
                    do_run_task_stream=_do_run_task_stream,
                )
            except KeyboardInterrupt:
                display.abort()
                console.print()
                break
            except Exception as exc:
                display.abort()
                console.print(f"[anton.error]Error: {exc}[/]")
                console.print()
    except KeyboardInterrupt:
        pass

    console.print()
    console.print("[anton.muted]See you.[/]")
    await session.close()
    await channel.close()
