"""Extra tools for the open source terminal agent."""

from __future__ import annotations
from typing import TYPE_CHECKING

from anton.core.tools.tool_defs import ToolDef

if TYPE_CHECKING:
    from anton.core.session import ChatSession


async def handle_connect_datasource(session: ChatSession, tc_input: dict) -> str:
    """Handle connect_new_datasource tool call — interactive connection flow."""
    engine = tc_input.get("engine", "")
    if not engine:
        return "Engine name is required."

    raw_known = tc_input.get("known_variables") or {}
    known_variables: dict[str, str] = (
        {str(k): str(v) for k, v in raw_known.items() if v is not None and v != ""}
        if isinstance(raw_known, dict) else {}
    )

    console = session._console
    if console is None:
        return "Cannot connect datasource — no console available."

    # ── Telemetry: connection attempt ────────────────────────────────
    _settings = getattr(session, "_settings", None)
    if _settings is None:
        try:
            from anton.config.settings import AntonSettings
            _settings = AntonSettings()
        except Exception:
            _settings = None

    if _settings:
        from anton.analytics import send_event
        send_event(_settings, "ds_connect_attempt", engine=engine)

    console.print()
    console.print(
        f"[anton.prompt]anton>[/] I can help with that \u2014 let's connect [bold]{engine}[/] to Anton."
    )

    from anton.commands.datasource import handle_connect_datasource

    from anton.core.datasources.data_vault import LocalDataVault
    vault = session._data_vault or LocalDataVault()
    before = {f"{c['engine']}-{c['name']}" for c in vault.list_connections()}

    # Clear any stale status from a previous run
    setattr(session, "_pending_connect_redirect", None)
    setattr(session, "_pending_connect_status", None)

    await handle_connect_datasource(
        console,
        session._scratchpads,
        session,
        prefill=engine,
        known_variables=known_variables or None,
        from_tool_call=True,
        vault=vault,
    )

    # Check if a new connection was actually added
    after = {f"{c['engine']}-{c['name']}" for c in vault.list_connections()}
    new_connections = after - before

    if new_connections:
        slug = next(iter(new_connections))
        # ── Telemetry: connection succeeded ──────────────────────────
        if _settings:
            send_event(_settings, "ds_connect_success", engine=engine)
        return (
            f"Successfully connected '{slug}'. The datasource is now available. "
            f"Continue helping the user with their original request using this data source."
        )

    # Did the flow record a mid-flow redirect? Read it from the session
    # attribute stashed by _build_redirect_message. We CANNOT append to
    # session._history from within the handler — we're between the
    # tool_use and tool_result blocks and doing so breaks the Anthropic
    # API invariant that every tool_use must be immediately followed by
    # its tool_result.
    redirect_text = getattr(session, "_pending_connect_redirect", None)
    if redirect_text:
        setattr(session, "_pending_connect_redirect", None)
        return redirect_text

    # No new connection was saved. Distinguish *why* — the LLM should
    # not be told "user pressed Escape" when really the test failed.
    status = getattr(session, "_pending_connect_status", None)
    setattr(session, "_pending_connect_status", None)

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

    if status == "test_failed":
        # ── Telemetry: connection failed ─────────────────────────────
        if _settings:
            from anton.analytics import send_event
            send_event(_settings, "ds_connect_failed", engine=engine)
        return (
            f"CONNECTION TEST FAILED: The connection test for '{engine}' did not "
            f"succeed and the user declined to re-enter credentials. Nothing was "
            f"saved.\n\n"
            f"You have exactly TWO mutually exclusive options — pick ONE, do NOT "
            f"mix them:\n\n"
            f"OPTION A — Retry silently (only if you suspect a transient issue "
            f"like a network glitch or first-connection cold start):\n"
            f"  Emit ZERO text in your response. Output ONLY a tool_use block "
            f"calling connect_new_datasource again with the same known_variables. "
            f"The user will only see the final result — clean and uncluttered.\n\n"
            f"OPTION B — Give up and troubleshoot (if you believe the failure is "
            f"real — bad credentials, wrong host, firewall, etc.):\n"
            f"  Respond with TEXT ONLY, NO tool calls. Briefly explain what "
            f"likely went wrong and ask the user what to do.\n\n"
            f"CRITICAL: Mixing text + a retry tool call in the same response "
            f"produces a confusing two-message stack for the user (failure text "
            f"followed by success text). Pick A or B, never both."
        )

    # Default: user cancelled (pressed Escape) at some point
    return (
        f"CANCELLED: The user cancelled the '{engine}' connection setup before "
        f"it completed. Ask the user what they'd like to do instead. "
        f"Do NOT immediately call connect_new_datasource again unless they "
        f"explicitly ask for it. Respond with TEXT ONLY — no tool calls."
    )


CONNECT_DATASOURCE_TOOL = ToolDef(
    name = "connect_new_datasource",
    description = (
        "Connect a new data source to Anton's Local Vault. Call this when the user "
        "asks a question that requires data from a source that isn't connected yet "
        "(e.g. email, database, CRM, API). This starts an interactive connection flow "
        "where the user enters their credentials.\n\n"
        "Pass the datasource type/name (e.g. 'gmail', 'postgres', 'salesforce', 'hubspot'). "
        "Anton will match it to the right connector and guide the user through setup.\n\n"
        "If the user has ALREADY mentioned credential values in the conversation "
        "(e.g. 'connect to dynamodb, my access key is AKIA... and region is us-east-1'), "
        "pass them as `known_variables` so the user is not asked again.\n\n"
        "Do NOT print any message before calling this tool — it handles the user-facing output."
    ),
    input_schema = {
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
            "known_variables": {
                "type": "object",
                "description": (
                    "Pre-extracted credential field values from the conversation. "
                    "Use snake_case field names (e.g. {\"host\": \"db.example.com\", "
                    "\"port\": \"5432\", \"user\": \"admin\"}). Only pass fields the "
                    "user actually mentioned — never invent values."
                ),
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["engine"],
    },
    handler = handle_connect_datasource,
)


async def handle_publish_or_preview(session: ChatSession, tc_input: dict) -> str:
    """Interactive preview/publish flow after dashboard creation."""
    import os
    import webbrowser
    from pathlib import Path

    console = session._console

    raw_path = tc_input.get("file_path", "")
    title = tc_input.get("title", "Dashboard")
    action = tc_input.get("action", "ask")
    file_path = Path(raw_path)
    if not file_path.is_absolute() and session._workspace:
        file_path = Path(session._workspace.base) / raw_path

    if not file_path.exists():
        return f"File not found: {file_path}"

    # Direct preview — just open and return, no prompts
    if action in ("preview", "ask"):
        abs_path = os.path.abspath(str(file_path))
        webbrowser.open(f"file://{abs_path}")
        return f"Opened {title} in browser. The user can ask for changes or say /publish to publish it to the web."

    # Publish flow
    from anton.config.settings import AntonSettings
    from anton.publisher import publish

    settings = AntonSettings()

    if not settings.minds_api_key:
        console.print()
        console.print("  [anton.muted]To publish you need a free Minds account.[/]")
        console.print("  [anton.muted]Run [bold]/publish[/bold] to set up your API key and publish.[/]")
        console.print()
        return (
            "STOP: No Minds API key configured. Do NOT call this tool again. "
            "Tell the user to run the /publish command to set up their mdb.ai API key "
            "and publish their dashboard. The /publish command handles the interactive "
            "API key setup flow."
        )

    import json as _json

    from rich.live import Live
    from rich.spinner import Spinner

    # Check if this file was previously published — reuse report_id to
    # update instead of creating a new report every time.
    output_dir = file_path.parent
    published_json = output_dir / ".published.json"
    published_map: dict = {}
    try:
        if published_json.is_file():
            published_map = _json.loads(published_json.read_text())
    except Exception:
        pass

    file_key = file_path.name
    prev = published_map.get(file_key)
    report_id = prev.get("report_id") if isinstance(prev, dict) else None

    action_text = "  Updating..." if report_id else "  Publishing..."
    with Live(Spinner("dots", text=action_text, style="anton.cyan"), console=console, transient=True):
        try:
            result = publish(
                file_path,
                api_key=settings.minds_api_key,
                report_id=report_id,
                publish_url=settings.publish_url,
                ssl_verify=settings.minds_ssl_verify,
            )
        except Exception as e:
            if report_id:
                # The report may have been deleted server-side — retry
                # without report_id to create a fresh one.
                try:
                    result = publish(
                        file_path,
                        api_key=settings.minds_api_key,
                        publish_url=settings.publish_url,
                        ssl_verify=settings.minds_ssl_verify,
                    )
                except Exception as e2:
                    console.print(f"  [anton.error]Publish failed: {e2}[/]")
                    console.print()
                    return f"PUBLISH FAILED: {e2}"
            else:
                console.print(f"  [anton.error]Publish failed: {e}[/]")
                console.print()
                return f"PUBLISH FAILED: {e}"

    view_url = result.get("view_url", "")
    returned_report_id = result.get("report_id", "")
    version = result.get("version", 1)
    unchanged = result.get("unchanged", False)

    if unchanged:
        console.print(f"  [anton.muted]Already up to date (v{version})[/]")
    elif report_id:
        console.print(f"  [anton.success]Updated! (v{version})[/]")
    else:
        console.print(f"  [anton.success]Published![/]")
    console.print(f"  [link={view_url}]{view_url}[/link]")
    console.print()

    # Persist the mapping so future publishes of the same file update
    # instead of creating a new report.
    if returned_report_id:
        published_map[file_key] = {
            "report_id": returned_report_id,
            "url": view_url,
            "last_md5": result.get("md5", ""),
        }
        try:
            published_json.write_text(_json.dumps(published_map, indent=2))
        except Exception:
            pass

    if view_url:
        webbrowser.open(view_url)

    status = "Updated" if report_id else "Published"
    return f"{status} successfully!\nView URL: {view_url}"


PUBLISH_TOOL = ToolDef(
    name = "publish_or_preview",
    description = (
        "Call this after generating an HTML dashboard or report in .anton/output/. "
        "Actions: 'ask' (default) prompts the user to preview/publish/skip interactively. "
        "'preview' opens the file in the browser immediately. "
        "'publish' publishes to the web immediately. "
        "Use 'preview' or 'publish' when the user has already stated their intent. "
        "Use 'ask' after generating a new dashboard to let the user choose."
    ),
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the HTML file (e.g. .anton/output/dashboard.html)",
            },
            "title": {
                "type": "string",
                "description": "Short title describing the dashboard (e.g. 'BTC & Macro Dashboard')",
            },
            "action": {
                "type": "string",
                "enum": ["ask", "preview", "publish"],
                "description": "What to do: 'ask' prompts user, 'preview' opens locally, 'publish' publishes to web",
            },
        },
        "required": ["file_path"],
    },
    handler = handle_publish_or_preview,
    prompt = (
        "CONTENT SHARING POLICY:\n"
        "- Publishing dashboards or reports to the web is done ONLY via the `publish_or_preview` tool. \n"
        "- Do NOT upload, post, or share generated files (HTML, data, images) to external hosting \n"
        "- services (paste sites, gists, CDNs, file hosts) via scratchpad code — unless the user \n"
        "- explicitly names the service and confirms. Reading from public APIs and writing to the \n"
        "- user's connected datasources (databases, CRMs, etc.) is fine — this rule only applies to \n"
        "- sharing generated output with the public internet."
    ),
)
