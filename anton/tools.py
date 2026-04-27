"""Extra tools for the open source terminal agent."""

from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING

from anton.core.tools.tool_defs import ToolDef


if TYPE_CHECKING:
    from anton.core.datasources.datasource_registry import DatasourceEngine, DatasourceField
    from anton.core.session import ChatSession


SECRET_NAME_TOKENS = (
    "password", "secret", "token", "api_key", "key",
    "auth", "credential", "private",
)
SCRUBBED_VALUE_RE = re.compile(r"^\[DS_\w+\]$")


def looks_secret(field_name: str) -> bool:
    """Heuristic: treat fields whose name suggests a secret as `secret=True`."""
    lower = field_name.lower()
    return any(tok in lower for tok in SECRET_NAME_TOKENS)


def _resolve_active_fields(
    engine_def: "DatasourceEngine", known_variables: dict[str, str]
) -> list["DatasourceField"]:
    """Pick the active field set for an engine based on provided variables.

    For engines with ``auth_method == "choice"``, match by largest overlap
    with ``known_variables`` so a YOLO save with (e.g.) ``private_key``
    targets the key-pair auth method rather than password auth.
    """
    if engine_def.auth_method == "choice" and engine_def.auth_methods:
        best = engine_def.auth_methods[0]
        best_score = -1
        for am in engine_def.auth_methods:
            am_names = {f.name for f in am.fields}
            score = sum(1 for k in known_variables if k in am_names)
            if score > best_score:
                best_score = score
                best = am
        return list(best.fields)
    return list(engine_def.fields)


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

    dropped_scrubbed = [
        k for k, v in known_variables.items() if SCRUBBED_VALUE_RE.match(v)
    ]
    if dropped_scrubbed:
        known_variables = {
            k: v
            for k, v in known_variables.items()
            if not SCRUBBED_VALUE_RE.match(v)
        }
        console.print()
        console.print(
            f"[anton.warning](anton)[/] Ignoring scrubbed-placeholder values "
            f"for {', '.join(dropped_scrubbed)} — those [DS_...] strings are "
            f"scrub-markers, not real credentials. Pass the actual secret "
            f"values instead."
        )

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

    from anton.core.datasources.data_vault import LocalDataVault
    vault = session._data_vault or LocalDataVault()

    if known_variables:
        from anton.core.datasources.datasource_registry import (
            DatasourceEngine,
            DatasourceField,
            DatasourceRegistry,
        )
        from anton.utils.datasources import (
            find_matching_connection,
            persist_custom_engine,
            save_connection,
        )
        from anton.commands.datasource.verify import run_connection_test
        registry = DatasourceRegistry()
        engine_def = registry.find_by_name(engine)
        if engine_def is None:
            adhoc_fields = [
                DatasourceField(
                    name=k,
                    required=False,
                    secret=looks_secret(k),
                    description="",
                )
                for k in known_variables
            ]
            engine_def = persist_custom_engine(
                registry, engine, adhoc_fields
            )
            if engine_def is None:
                engine_def = DatasourceEngine(
                    engine=engine,
                    display_name=engine,
                    fields=adhoc_fields,
                    custom=True,
                )
        if engine_def is not None:
            active_fields = _resolve_active_fields(engine_def, known_variables)
            active_names = {f.name for f in active_fields}
            fields_to_save = {
                k: v for k, v in known_variables.items() if k in active_names
            }
            filtered_out = sorted(
                k for k in known_variables if k not in active_names
            )
            for f in active_fields:
                if f.required and f.default and not fields_to_save.get(f.name):
                    fields_to_save[f.name] = f.default
            missing_required = [
                f.name
                for f in active_fields
                if f.required and not fields_to_save.get(f.name)
            ]

            if filtered_out:
                console.print()
                console.print(
                    f"[anton.warning](anton)[/] Ignoring keys that don't "
                    f"belong to [bold]{engine_def.display_name}[/]: "
                    f"{', '.join(filtered_out)}."
                )

            if fields_to_save and not missing_required:
                test_credentials = dict(fields_to_save)
                if engine_def.test_snippet:
                    ok = await run_connection_test(
                        console,
                        session._scratchpads,
                        vault,
                        engine_def,
                        test_credentials,
                        active_fields,
                    )
                    if not ok:
                        if _settings:
                            from anton.analytics import send_event
                            send_event(_settings, "ds_connect_failed", engine=engine)
                        return (
                            f"Connection test failed for '{engine}'. Nothing "
                            f"was saved. Either retry with corrected "
                            f"known_variables or explain the issue to the user."
                        )

                conn_name = find_matching_connection(
                    vault, engine_def, test_credentials
                )
                if conn_name is None:
                    conn_name = uuid.uuid4().hex[:8]
                    while vault.load(engine_def.engine, conn_name) is not None:
                        conn_name = uuid.uuid4().hex[:8]
                existing = vault.load(engine_def.engine, conn_name) or {}
                merged = {**existing, **test_credentials}
                slug = save_connection(vault, engine_def, conn_name, merged)
                session._active_datasource = slug
                if _settings:
                    from anton.analytics import send_event
                    send_event(_settings, "ds_connect_success", engine=engine)

                if existing:
                    changed = [
                        k for k, v in test_credentials.items()
                        if existing.get(k) != v
                    ]
                    preserved = [k for k in existing if k not in test_credentials]
                    if changed:
                        _msg = (
                            f"Updated connection `{slug}` in vault. "
                            f"Fields changed: {', '.join(sorted(changed))}."
                        )
                    else:
                        _msg = (
                            f"Connection `{slug}` already matched the provided "
                            f"values — nothing changed."
                        )
                    if preserved:
                        _msg += (
                            f" Preserved existing fields: "
                            f"{', '.join(sorted(preserved))}."
                        )
                    _msg += (
                        " Future turns can reference this connection by its "
                        "slug. Access credentials via DS_<FIELD> environment "
                        "variables in scratchpad code — never embed raw values."
                    )
                    return _msg
                return (
                    f"Saved connection `{slug}` to vault with fields: "
                    f"{', '.join(sorted(test_credentials.keys()))}. "
                    f"Future turns can reference this connection by its slug. "
                    f"Access credentials via DS_<FIELD> environment variables "
                    f"in scratchpad code — never embed raw values."
                )

    console.print()
    console.print(
        f"[anton.prompt]anton>[/] I can help with that \u2014 let's connect [bold]{engine}[/] to Anton."
    )

    from anton.commands.datasource import handle_connect_datasource

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
            f"Connection test failed for '{engine}'. Nothing was saved. "
            f"Either retry with corrected known_variables or explain the "
            f"issue to the user."
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
        "Connect a data source to Anton's Local Vault. Two modes:\n\n"
        "(a) Non-interactive: call this tool IMMEDIATELY when the user shares "
        "credentials in chat (host, password, API token, service account JSON, "
        "etc.). Pass all extracted values as known_variables. The tool saves "
        "to the vault without any prompts and returns a confirmation. This "
        "ensures credentials are persisted before being used anywhere — never "
        "reference chat-supplied credentials directly in scratchpad code; always "
        "go through the vault.\n\n"
        "(b) Interactive: call with just engine and no known_variables when the "
        "user has no credentials in context yet. Anton runs the same flow as "
        "/connect, prompting for fields one at a time.\n\n"
        "Supported engines: see the built-in registry (PostgreSQL, MySQL, Snowflake, "
        "BigQuery, Redshift, Databricks, MariaDB, MSSQL, Oracle, HubSpot, Salesforce, "
        "Shopify, Gmail, and more). Unknown engines (not in the built-in registry) "
        "are also saved silently as ad-hoc connections when known_variables are "
        "provided — no prompts, no auth-method interrogation. A minimal engine "
        "definition is appended to ~/.anton/datasources.md so future sessions "
        "recognize it. Reference credentials via DS_<ENGINE>_<NAME>__<FIELD> env "
        "vars like any other connection.\n\n"
        "Partial credentials are fine — save what the user provided. Ask for missing "
        "pieces in a later turn only if needed. Never invent values.\n\n"
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
