"""Extra tools for the open source terminal agent."""

from __future__ import annotations

import dataclasses
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
# All marker forms `scrub_credentials` emits: [DS_*] for vault secrets,
# [<ENV_VAR>] labels for provider keys, [REDACTED_API_KEY] for shape matches.
# User messages are scrubbed too, so the model may see these and echo them
# back as known_variables — they must never be saved as real credentials.
SCRUBBED_VALUE_RE = re.compile(r"^\[(?:DS_\w+|[A-Z][A-Z0-9_]*)\]$")


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

    # Mode (a) below needs no terminal — it writes straight to the vault. A
    # quiet console swallows its status prints instead of crashing on
    # `None.print()`. The real console (if any) is required only for mode
    # (b)'s prompts, gated further down.
    from rich.console import Console
    console = session._console or Console(quiet=True)

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
            f"for {', '.join(dropped_scrubbed)} — those bracketed strings are "
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
                        # Mode (a) is driven by the model, not a human at a
                        # keyboard — there's no one to answer a "retry?"
                        # prompt, and prompt_or_cancel would otherwise drive
                        # a real terminal regardless of `console`.
                        interactive=False,
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
                    from anton.utils.datasources import default_user_label, ensure_unique_user_label
                    requested_label = tc_input.get("user_label") or default_user_label(vault, engine_def.engine)
                    test_credentials["_user_label"] = ensure_unique_user_label(vault, requested_label)
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
                label = merged.get("_user_label", "")
                return (
                    f"Saved connection `{slug}` (label \"{label}\") to vault with fields: "
                    f"{', '.join(sorted(test_credentials.keys()))}. "
                    f"Future turns can reference this connection by its slug. "
                    f"Access credentials via DS_<FIELD> environment variables "
                    f"in scratchpad code — never embed raw values."
                )

    # Below this point is mode (b), the interactive field-by-field prompt
    # flow. It needs a real terminal (Rich prompts, Live spinners); a quiet
    # console can't service it, so bail out with an actionable message
    # instead of hanging or crashing on a prompt read.
    if session._console is None:
        # dropped_scrubbed's console.print warning above never reaches the
        # model (nor, in this no-console host, anyone else) — repeat it here
        # so the model knows *why* known_variables wasn't enough, rather
        # than re-sending the same scrub-marker placeholder again.
        note = ""
        if dropped_scrubbed:
            note = (
                f" Scrubbed-placeholder values for "
                f"{', '.join(sorted(dropped_scrubbed))} were ignored — those "
                f"are not real credentials; pass the actual secret values "
                f"instead."
            )
        return (
            "Interactive connection setup isn't available in this environment "
            "(no terminal to prompt in)." + note + " Ask the user for the "
            "credential values directly in chat, then call "
            "connect_new_datasource again with known_variables set."
        )

    console.print()
    console.print(
        f"[anton.prompt]anton>[/] I can help with that \u2014 let's connect [bold]{engine}[/] to Anton."
    )

    from anton.commands.datasource import handle_connect_datasource

    # Track (engine, name) pairs rather than joined slugs: engine names may
    # contain a dash (custom engines are free-form), so a joined string
    # cannot be split back into its parts reliably.
    before = {(c["engine"], c["name"]) for c in vault.list_connections()}

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
        prefill_label=tc_input.get("user_label"),
    )

    # Check if a new connection was actually added
    after = {(c["engine"], c["name"]) for c in vault.list_connections()}
    new_connections = after - before

    if new_connections:
        engine_saved, name_saved = sorted(new_connections)[0]
        slug = f"{engine_saved}-{name_saved}"
        saved_fields = vault.load(engine_saved, name_saved) or {}
        saved_label = saved_fields.get("_user_label", "")
        # ── Telemetry: connection succeeded ──────────────────────────
        if _settings:
            send_event(_settings, "ds_connect_success", engine=engine)
        return (
            f"Successfully connected '{slug}' (label \"{saved_label}\"). The datasource is "
            f"now available. Continue helping the user with their original request using "
            f"this data source."
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


# Shared by both description variants below — what mode (a) actually does.
# One copy so the two descriptions can't drift out of sync with each other.
_CONNECT_DATASOURCE_MODE_A_BODY = (
    "this tool IMMEDIATELY when the user shares credentials in chat (host, "
    "password, API token, service account JSON, etc.). Pass all extracted "
    "values as known_variables. The tool saves to the vault without any "
    "prompts and returns a confirmation. This ensures credentials are "
    "persisted before being used anywhere — never reference chat-supplied "
    "credentials directly in scratchpad code; always go through the vault."
)

# Shared by both description variants below — everything that doesn't
# depend on whether an interactive terminal is available.
_CONNECT_DATASOURCE_DESCRIPTION_TAIL = (
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
)

CONNECT_DATASOURCE_TOOL = ToolDef(
    name = "connect_new_datasource",
    description = (
        "Connect a data source to Anton's Local Vault. Two modes:\n\n"
        "(a) Non-interactive: call " + _CONNECT_DATASOURCE_MODE_A_BODY + "\n\n"
        "(b) Interactive: call with just engine and no known_variables when the "
        "user has no credentials in context yet. Anton runs the same flow as "
        "/connect, prompting for fields one at a time.\n\n"
        + _CONNECT_DATASOURCE_DESCRIPTION_TAIL
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
            "user_label": {
                "type": "string",
                "description": (
                    "Optional human-readable label for this connection (e.g. 'prod-db'). "
                    "If the user mentioned a name for it in chat, pass it here. If omitted, "
                    "a default based on the engine is used (e.g. 'postgres', 'postgres 2' "
                    "for a second connection to the same engine)."
                ),
            },
        },
        "required": ["engine"],
    },
    handler = handle_connect_datasource,
)

# Console-less hosts (Cowork desktop: session._console is None) can't service
# mode (b) — there's no terminal to prompt in, and handle_connect_datasource
# refuses it with an actionable error rather than crashing (ENG-1849). Don't
# advertise a mode the model would then have to be told, mid-turn, doesn't
# work here — swapped in by ChatSession.__init__ (anton/core/session.py)
# once session._console is known.
CONNECT_DATASOURCE_TOOL_NO_CONSOLE = dataclasses.replace(
    CONNECT_DATASOURCE_TOOL,
    description = (
        "Connect a data source to Anton's Local Vault. Call "
        + _CONNECT_DATASOURCE_MODE_A_BODY + "\n\n"
        "There is no interactive prompt flow in this environment — do NOT "
        "call this tool with just an engine and no known_variables; there is "
        "nothing to prompt with and the call will fail. If the user hasn't "
        "shared credentials yet, ask for them in chat first, then call this "
        "tool once you have real values to pass.\n\n"
        + _CONNECT_DATASOURCE_DESCRIPTION_TAIL
    ),
    # Own copy, not shared with CONNECT_DATASOURCE_TOOL: this variant has no
    # prompt flow to fall back on, so known_variables is mandatory here,
    # matching the description above.
    input_schema = {
        **CONNECT_DATASOURCE_TOOL.input_schema,
        "required": ["engine", "known_variables"],
    },
)


def _previewable_html(path: "Path"):
    """Best-effort HTML file to open for local preview.

    For a file, returns it as-is; for a folder (e.g. a fullstack artifact),
    prefers index.html / static/index.html, else the first ``*.html`` found."""
    from pathlib import Path

    path = Path(path)
    if path.is_file():
        return path
    for cand in (path / "index.html", path / "static" / "index.html"):
        if cand.is_file():
            return cand
    try:
        htmls = sorted(path.rglob("*.html"))
    except OSError:
        htmls = []
    return htmls[0] if htmls else None


async def handle_publish_or_preview(session: ChatSession, tc_input: dict) -> str:
    """Interactive preview/publish flow after dashboard creation.

    Accepts the **artifact folder** (preferred) or any file inside it. The
    artifact type is read from its ``metadata.json`` via
    ``resolve_publish_target`` — fullstack apps publish the whole folder, static
    reports publish the primary file — so callers never need to know the type."""
    import os
    import webbrowser
    from pathlib import Path

    from anton.config.settings import AntonSettings
    from anton.publish_access import (
        access_from_owner_side,
        resolve_access,
        resolve_publish_target,
    )

    console = session._console

    raw_path = tc_input.get("file_path", "")
    title = tc_input.get("title", "Dashboard")
    action = tc_input.get("action", "ask")

    settings = AntonSettings()
    artifacts_root = Path(settings.artifacts_dir)

    # Accept the artifact folder (preferred) or any file inside it. Resolve a
    # relative path against the artifacts dir first (so "my-app" works), then
    # the workspace base.
    file_path = Path(raw_path)
    if not file_path.is_absolute():
        base = Path(session._workspace.base) if session._workspace else artifacts_root
        file_path = next(
            (c for c in (artifacts_root / raw_path, base / raw_path) if c.exists()),
            base / raw_path,
        )

    if not file_path.exists():
        return f"Path not found: {file_path}"

    # Decide WHAT to publish from the artifact's metadata.json: fullstack → the
    # whole folder, static → the primary file. Also yields the canonical
    # .published.json location + key (shared with /publish and cowork-server).
    publish_target, published_dir, published_key, _is_fullstack = resolve_publish_target(
        file_path, [artifacts_root]
    )

    # Direct preview — open a previewable HTML and return, no prompts.
    if action in ("preview", "ask"):
        preview_file = _previewable_html(publish_target)
        if preview_file is not None:
            webbrowser.open(f"file://{os.path.abspath(str(preview_file))}")
            return f"Opened {title} in browser. The user can ask for changes or say /publish to publish it to the web."
        return f"{title} is at {file_path} but no previewable HTML was found."

    # Publish flow
    from anton.publisher import publish

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

    from anton.utils.prompt import prompt_or_cancel

    # Check if this artifact was previously published — reuse report_id to
    # update instead of creating a new report every time. published_dir /
    # published_key were resolved above via the unified convention (shared with
    # /publish and cowork-server) so the entry points never disagree.
    published_json = published_dir / ".published.json"
    published_map: dict = {}
    try:
        if published_json.is_file():
            published_map = _json.loads(published_json.read_text())
    except Exception:
        pass

    prev = published_map.get(published_key)
    report_id = prev.get("report_id") if isinstance(prev, dict) else None

    # Resolve the access spec: explicit tool fields > preserve previous >
    # public. NB: password input (when the user asked for a password but gave
    # no value) is collected BEFORE the Live spinner below — prompt_toolkit
    # and rich.Live must not run at the same time.
    access_mode = tc_input.get("access_mode")
    if access_mode:
        req_access = {"mode": access_mode}
        if access_mode == "password":
            pw = (tc_input.get("password") or "").strip()
            if not pw:
                import sys
                if sys.stdin.isatty():
                    entered = await prompt_or_cancel("  Set a password for this report", password=True)
                    if entered is None or not entered.strip():
                        return "CANCELLED: publish aborted by user (no password entered)."
                    pw = entered.strip()
                else:
                    return (
                        "CANCELLED: password required but no TTY. Ask the user to run "
                        "/publish to set a password interactively."
                    )
            req_access["password"] = pw
        elif access_mode == "restricted":
            from anton.publish_access import parse_emails
            valid, _invalid = parse_emails(tc_input.get("emails") or [])
            req_access["emails"] = valid
            req_access["org_allowed"] = bool(tc_input.get("org_allowed"))
    elif isinstance(prev, dict) and prev.get("report_id"):
        req_access = access_from_owner_side(prev)  # preserve previous, do NOT reset to public
    else:
        req_access = {"mode": "public"}

    eff_access, pwd_version, access_version, owner_side = resolve_access(None, req_access, prev)

    action_text = "  Updating..." if report_id else "  Publishing..."
    with Live(Spinner("dots", text=action_text, style="anton.cyan"), console=console, transient=True):
        try:
            result = publish(
                publish_target,
                api_key=settings.minds_api_key,
                report_id=report_id,
                publish_url=settings.publish_url,
                ssl_verify=settings.minds_ssl_verify,
                access=eff_access,
                pwd_version=pwd_version,
                access_version=access_version,
            )
        except Exception as e:
            if report_id:
                # The report may have been deleted server-side — retry
                # without report_id to create a fresh one.
                try:
                    result = publish(
                        publish_target,
                        api_key=settings.minds_api_key,
                        publish_url=settings.publish_url,
                        ssl_verify=settings.minds_ssl_verify,
                        access=eff_access,
                        pwd_version=pwd_version,
                        access_version=access_version,
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
    # instead of creating a new report (owner-side; unified location + key).
    if returned_report_id:
        entry = dict(owner_side)
        entry.update({
            "report_id": returned_report_id,
            "url": view_url,
            "last_md5": result.get("md5", ""),
        })
        published_map[published_key] = entry
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
        "Call this after generating a dashboard/report/app to preview or publish it. "
        "Pass the artifact FOLDER as file_path — the tool reads metadata.json and picks "
        "the right thing to publish (whole folder for fullstack apps, primary file for "
        "static reports). "
        "Actions: 'ask' (default) prompts the user to preview/publish/skip interactively. "
        "'preview' opens it in the browser immediately. "
        "'publish' publishes to the web immediately. "
        "Use 'preview' or 'publish' when the user has already stated their intent. "
        "Use 'ask' after generating a new artifact to let the user choose."
    ),
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": (
                    "Path to the artifact FOLDER (e.g. artifacts/<slug>). The tool reads "
                    "the folder's metadata.json to decide how to publish: fullstack apps "
                    "publish the whole folder, static reports publish their primary file. "
                    "A path to a file inside the artifact also works."
                ),
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
            "access_mode": {
                "type": "string",
                "enum": ["public", "password", "restricted"],
                "description": "Access level the user chose. If the user hasn't said which access they want, ASK them first (see the tool guidance) rather than guessing; only set this once you know.",
            },
            "password": {
                "type": "string",
                "description": "Password for access_mode='password'. ONLY if the user stated it verbatim; otherwise leave empty and it will be asked interactively. NEVER invent one.",
            },
            "emails": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Allowed viewer emails for access_mode='restricted'.",
            },
            "org_allowed": {
                "type": "boolean",
                "description": "Whether the whole organization may view (access_mode='restricted').",
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
        "- sharing generated output with the public internet.\n"
        "ACCESS MODE:\n"
        "- Before publishing you MUST know the access mode. If the user stated it "
        "(e.g. 'publish with password', 'share only with a@x.com', 'make it public'), use it: "
        "set access_mode (+ password/emails/org_allowed) accordingly.\n"
        "- If the user did NOT specify an access mode, ASK them in chat which one they want — "
        "public (anyone with the link), password-protected, or restricted to specific emails / "
        "the whole organization — and wait for their answer before calling this tool with "
        "action='publish'. Do NOT silently default to public.\n"
        "- NEVER invent a password. If the user chooses password but gives no value, set "
        "access_mode='password' and leave password empty — the app will prompt them.\n"
        "- Exception: when re-publishing an artifact that already has access on record and the "
        "user says nothing new about access, omit these fields to keep its previous access "
        "(no need to ask again)."
    ),
)


# Extra (non-core) tools every interactive anton session registers. Kept as a
# single source so the fresh-session builder (chat.py) and the resume/settings
# rebuild builder (chat_session.py::rebuild_session) can't drift — ENG-1166,
# where rebuild_session omitted these and every resumed session silently lost
# publish_or_preview + connect_new_datasource. Callers pass list(...) so each
# session gets its own copy rather than sharing this module-level list.
DEFAULT_SESSION_TOOLS = [CONNECT_DATASOURCE_TOOL, PUBLISH_TOOL]
