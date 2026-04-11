"""Slash-command handlers for datasource commands."""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding

from anton.connect_collector import ConnectionCollector, extract_variables
from anton.data_vault import DataVault
from anton.datasource_registry import (
    AuthMethod,
    DatasourceEngine,
    DatasourceField,
    DatasourceRegistry,
)
from anton.utils.datasources import (
    register_secret_vars,
    remove_engine_block,
    restore_namespaced_env,
    parse_connection_slug,
)
from anton.utils.prompt import prompt_or_cancel
from anton.core.backends.manager import ScratchpadManager

if TYPE_CHECKING:
    from anton.chat import ChatSession


# ─────────────────────────────────────────────────────────────────────────────
# LLM-facing schema (Pydantic) for handle_add_custom_datasource
# ─────────────────────────────────────────────────────────────────────────────


class _CustomDatasourceField(BaseModel):
    """One credential field in a custom-datasource spec."""

    name: str = Field(
        ...,
        description=(
            "snake_case field name (e.g. 'host', 'api_key'). Must be a "
            "valid Python identifier; this becomes both the on-disk key "
            "and the env var suffix (DS_<NAME>)."
        ),
    )
    value: str = Field(
        default="",
        description=(
            "Inline value if the user already provided one in their "
            "description, otherwise empty string."
        ),
    )
    secret: bool = Field(
        default=False,
        description=(
            "True if the field is sensitive (passwords, API keys, "
            "tokens) — affects how it's stored and prompted for."
        ),
    )
    required: bool = Field(
        default=True,
        description="True if the connection cannot be tested without this field.",
    )
    description: str = Field(
        default="",
        description=(
            "One-line description shown to the user when prompting "
            "for this field."
        ),
    )


class _CustomDatasourceSpec(BaseModel):
    """Structured output of the LLM call in handle_add_custom_datasource."""

    display_name: str = Field(
        ...,
        description="Human-readable name for the service (e.g. 'GitHub API').",
    )
    pip: str = Field(
        default="",
        description=(
            "pip-installable package name (or space-separated names) "
            "needed to interact with this service. Empty string if no "
            "extra package is required (e.g. plain HTTPS via stdlib)."
        ),
    )
    test_snippet: str = Field(
        default="",
        description=(
            "Python code that tests the connection using os.environ "
            "vars DS_FIELDNAME (uppercase field name with DS_ prefix) "
            "and prints 'ok' on success. Empty string if untestable."
        ),
    )
    fields: list[_CustomDatasourceField] = Field(
        default_factory=list,
        description=(
            "Credential fields the user will need to provide. List in "
            "the order they should be prompted."
        ),
    )

_PROMPT_RECONNECT_CANCEL = "(reconnect/cancel)"


def handle_list_data_sources(console: Console) -> None:
    """Print all saved Local Vault connections in a table with status."""
    from rich.table import Table

    vault = DataVault()
    registry = DatasourceRegistry()
    conns = vault.list_connections()
    console.print()
    if not conns:
        console.print("[anton.muted]No data sources connected yet.[/]")
        console.print("[anton.muted]Use /connect to add one.[/]")
        console.print()
        return

    table = Table(title="Local Vault — Saved Connections", show_lines=False)
    table.add_column("Name", style="bold")
    table.add_column("Source")
    table.add_column("Status")

    for c in conns:
        slug = f"{c['engine']}-{c['name']}"
        engine_def = registry.get(c["engine"])
        source = engine_def.display_name if engine_def else c["engine"]
        fields = vault.load(c["engine"], c["name"]) or {}

        if not fields:
            status = "[yellow]incomplete[/]"
        elif engine_def and engine_def.auth_method != "choice":
            required = [f.name for f in engine_def.fields if f.required]
            missing = [name for name in required if name not in fields]
            status = "[yellow]incomplete[/]" if missing else "[green]saved[/]"
        else:
            status = "[green]saved[/]"

        table.add_row(slug, source, status)

    console.print(table)
    console.print()


async def handle_remove_data_source(console: Console, slug: str) -> None:
    """Delete a connection from the Local Vault by slug (engine-name)."""
    vault = DataVault()
    registry = DatasourceRegistry()

    if not slug:
        connections = vault.list_connections()
        if not connections:
            console.print("[anton.muted]No saved connections to remove.[/]")
            console.print()
            return
        console.print()
        console.print("[anton.cyan](anton)[/] Which connection do you want to remove?\n")
        for i, c in enumerate(connections, 1):
            conn_slug = f"{c['engine']}-{c['name']}"
            engine_def = registry.get(c["engine"])
            label = engine_def.display_name if engine_def else c["engine"]
            console.print(f"          [bold]{i:>2}.[/bold] {conn_slug} [dim]({label})[/]")
        console.print()
        choices = [str(i) for i in range(1, len(connections) + 1)]
        pick = await prompt_or_cancel("(anton) Enter a number", choices=choices)
        if pick is None:
            console.print("[anton.muted]Cancelled.[/]")
            console.print()
            return
        picked = connections[int(pick) - 1]
        slug = f"{picked['engine']}-{picked['name']}"

    parsed = parse_connection_slug(slug, [e.engine for e in registry.all_engines()], vault=vault)
    if parsed is None:
        console.print(
            f"[anton.warning]Invalid name '{slug}'. Use engine-name format.[/]"
        )
        console.print()
        return
    engine, name = parsed
    if vault.load(engine, name) is None:
        console.print(f"[anton.warning]No connection '{slug}' found.[/]")
        console.print()
        return

    confirm = await prompt_or_cancel(
        f"(anton) Remove '{slug}' from Local Vault?",
        choices=["y", "n"], default="n",
    )
    if confirm is not None and confirm.strip().lower() == "y":
        vault.delete(engine, name)
        restore_namespaced_env(vault)
        engine_def = registry.get(engine)
        if engine_def is not None and engine_def.custom:
            remaining = [
                c for c in vault.list_connections() if c["engine"] == engine
            ]
            if not remaining:
                user_path = DatasourceRegistry._USER_PATH
                if user_path.is_file():
                    updated = remove_engine_block(
                        user_path.read_text(encoding="utf-8"), engine
                    )
                    user_path.write_text(updated, encoding="utf-8")
                    registry.reload()
        console.print(f"[anton.success]Removed {slug}.[/]")
    else:
        console.print("[anton.muted]Cancelled.[/]")
    console.print()


async def show_credential_help(
    console: Console,
    session: "ChatSession",
    service_name: str,
    current_field,
    all_fields: list,
) -> None:
    """Use the LLM to explain how to obtain credentials."""
    field_descriptions = ", ".join(
        f"{f.name} ({f.description})" for f in all_fields
    )
    storage_note = (
        "The credentials will be stored securely in Anton's Local Vault — "
        "do NOT suggest storage tips, password managers, or safe-keeping advice."
    )
    if current_field is not None:
        prompt = (
            f"I'm connecting to {service_name} and need to provide: {field_descriptions}\n\n"
            f"I need help with the '{current_field.name}' field"
            f" ({current_field.description}).\n\n"
            "Give me a brief step-by-step guide on where and how to get this credential. "
            f"Be concise — numbered steps, no fluff. {storage_note}"
        )
        heading = f"[anton.cyan](anton)[/] How to get [bold]{current_field.name}[/]:"
    else:
        prompt = (
            f"I'm connecting to {service_name} and need these credentials: {field_descriptions}\n\n"
            "Give me a brief step-by-step guide on where and how to obtain each of these. "
            f"Be concise — numbered steps, no fluff. {storage_note}"
        )
        heading = f"[anton.cyan](anton)[/] How to get credentials for [bold]{service_name}[/]:"

    console.print()
    console.print("[anton.muted]        Looking up instructions…[/]")

    try:
        resp = await session._llm.plan(
            system="You are a helpful assistant that guides users through obtaining credentials for services.",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=512,
        )
        help_text = (resp.content or "").strip()
    except Exception:
        help_text = "Sorry, couldn't fetch help right now. Try checking the service's documentation."

    console.print()
    console.print(heading)
    console.print()
    console.print(Padding(Markdown(help_text), (0, 0, 0, 8)))
    console.print()


async def run_connection_test(
    console: "Console",
    scratchpads: "ScratchpadManager",
    vault: "DataVault",
    engine_def: "DatasourceEngine",
    credentials: dict[str, str],
    retry_fields: "list[DatasourceField]",
) -> bool:
    """Inject flat DS_* vars, run engine_def.test_snippet, restore env.

    Returns True on success, False if the user declines retry after failure.
    Mutates credentials in-place when the user re-enters secrets on retry.
    """
    while True:
        console.print()
        console.print("[anton.cyan](anton)[/] Got it. Testing connection…")

        vault.clear_ds_env()
        for key, value in credentials.items():
            os.environ[f"DS_{key.upper()}"] = value
        register_secret_vars(engine_def)  # flat mode, for scrubbing during test

        try:
            pad = await scratchpads.get_or_create("__datasource_test__")
            await pad.reset()
            if engine_def.pip:
                if isinstance(engine_def.pip, list):
                    pip_pkgs = engine_def.pip
                else:
                    # Split space-separated package strings into individual packages
                    pip_pkgs = engine_def.pip.split()
                install_result = await pad.install_packages(pip_pkgs)
                if "failed" in (install_result or "").lower():
                    console.print()
                    console.print(f"[anton.warning](anton)[/] Package install issue: {install_result[:200]}")

            # Run the test, retry up to 2 times on ModuleNotFoundError
            cell = None
            for attempt in range(3):
                cell = await pad.execute(engine_def.test_snippet)
                if cell.error and "ModuleNotFoundError" in cell.error:
                    # Extract the missing module and try to install it
                    match = re.search(r"No module named '([^']+)'", cell.error)
                    if match:
                        missing = match.group(1).split(".")[0]
                        await pad.install_packages([missing])
                        continue
                break
        finally:
            restore_namespaced_env(vault)

        if cell.error or (cell.stdout.strip() != "ok" and cell.stderr.strip()):
            error_text = cell.error or cell.stderr.strip() or cell.stdout.strip()
            last_line = next(
                (ln for ln in reversed(error_text.splitlines()) if ln.strip()), error_text
            )
            console.print()
            console.print("[anton.warning](anton)[/] ✗ Connection failed.")
            console.print()
            console.print(f"        Error: {last_line}")
            console.print()
            retry = await prompt_or_cancel(
                "(anton) Would you like to re-enter your credentials?",
                choices=["y", "n"], default="n",
            )
            if retry is None or retry.strip().lower() != "y":
                return False
            console.print()
            for f in retry_fields:
                if not f.secret:
                    continue
                value = await prompt_or_cancel(f"(anton) {f.name}", password=True)
                if value is None:
                    return False
                if value:
                    credentials[f.name] = value
            continue

        console.print("[anton.success]        ✓ Connected successfully![/]")
        return True


async def handle_add_custom_datasource(
    console: Console,
    name: str,
    registry,
    session: "ChatSession",
    *,
    known_service: bool = False,
):
    """Ask for the tool name, use the LLM to identify required fields, then collect credentials."""

    console.print()
    if name:
        tool_name = name
    else:
        tool_name = await prompt_or_cancel(
            "(anton) What is the name of the tool or service?",
        )
        if not tool_name or not tool_name.strip():
            return None
        tool_name = tool_name.strip()

    if known_service:
        # LLM already recognised this service — skip the auth question
        user_answer = ""
        console.print("[anton.muted]        Working out the connection details…[/]")
    else:
        user_answer = await prompt_or_cancel(
            f"(anton) How do you authenticate with {tool_name}? "
            "Describe what credentials you have (don't paste actual values)",
        )
        if not user_answer or not user_answer.strip():
            return None
        console.print()
        console.print("[anton.muted]    Got it — working out the connection details…[/]")

    llm_prompt = f"The user wants to connect to {repr(tool_name)}."
    if user_answer:
        llm_prompt += f" They said: {user_answer}"
    else:
        llm_prompt += " Determine the standard authentication fields for this service."
    llm_prompt += (
        "\n\nReturn the connection spec following the schema you've been given. "
        "For test_snippet, write Python that uses os.environ['DS_<FIELDNAME>'] "
        "vars (uppercase, DS_ prefix) and prints 'ok' on success."
    )

    try:
        spec: _CustomDatasourceSpec = await session._llm.generate_object(
            _CustomDatasourceSpec,
            system="You are a data source connection expert.",
            messages=[{"role": "user", "content": llm_prompt}],
            max_tokens=1024,
        )
    except Exception:
        console.print(
            "[anton.warning]        Couldn't identify connection details. Try again.[/]"
        )
        console.print()
        return None

    test_snippet = spec.test_snippet.strip()
    fields: list[DatasourceField] = []
    for f in spec.fields:
        if not f.name:
            continue
        fields.append(
            DatasourceField(
                name=f.name,
                required=f.required,
                secret=f.secret,
                description=f.description,
            )
        )

    if not fields:
        console.print("[anton.warning]    Couldn't identify any connection fields.[/]")
        console.print()
        return None

    display_name = spec.display_name or name
    pip_pkg = spec.pip

    # Show summary
    console.print()
    console.print("      [bold]── What I'll save ──────────────────────────[/]")
    credentials: dict[str, str] = {}
    for f, raw in zip(fields, spec.fields):
        inline_value = (raw.value or "").strip()
        if f.secret and inline_value:
            console.print(
                f"        • [bold]{f.name:<14}[/] (secret — provided, stored securely)"
            )
            credentials[f.name] = inline_value
        elif f.secret:
            console.print(
                f"        • [bold]{f.name:<14}[/] (secret — I'll ask for this)"
            )
        else:
            val_display = inline_value or "[anton.muted]<to be collected>[/]"
            console.print(f"        • [bold]{f.name:<14}[/] {val_display}")
            if inline_value:
                credentials[f.name] = inline_value
    console.print()

    # Offer help before collecting credentials
    help_answer = await prompt_or_cancel(
        "(anton) Do you need instructions on how to obtain these credentials?",
        choices=["y", "n"], default="n",
    )
    if help_answer is None:
        return None
    if help_answer.strip().lower() == "y":
        await show_credential_help(
            console, session, display_name, None, fields,
        )

    # Prompt for any secret fields not provided inline
    for f, raw in zip(fields, spec.fields):
        if not f.secret:
            continue
        if (raw.value or "").strip():
            continue
        value = await prompt_or_cancel(f"(anton) {f.name}", password=True)
        if value is None:
            return None
        if value:
            credentials[f.name] = value

    # Prompt for any required non-secret fields not provided inline
    for f, raw in zip(fields, spec.fields):
        if f.secret:
            continue
        if not f.required:
            continue
        if f.name in credentials:
            continue
        value = await prompt_or_cancel(f"(anton) {f.name}")
        if value is None:
            return None
        if value:
            credentials[f.name] = value

    # Offer to collect optional non-secret fields
    for f, raw in zip(fields, spec.fields):
        if f.secret or f.required or f.name in credentials:
            continue
        value = await prompt_or_cancel(f"(anton) {f.name} (optional — press Enter to skip)")
        if value is None:
            return None
        if value:
            credentials[f.name] = value

    if not credentials:
        console.print("[anton.warning]        No credentials collected. Aborting.[/]")
        console.print()
        return None

    # Build engine slug and write definition to ~/.anton/datasources.md
    slug = re.sub(r"[^\w]", "_", display_name.lower()).strip("_")
    field_lines = "\n".join(
        f"  - {{ name: {f.name}, required: {str(f.required).lower()}, "
        f'secret: {str(f.secret).lower()}, description: "{f.description}" }}'
        for f in fields
    )
    test_snippet_yaml = ""
    if test_snippet:
        indented = "\n".join(f"  {line}" for line in test_snippet.splitlines())
        test_snippet_yaml = f"test_snippet: |\n{indented}\n"

    yaml_block = (
        f"\n---\n\n## {display_name}\n"
        "```yaml\n"
        f"engine: {slug}\n"
        f"display_name: {display_name}\n"
        + (f"pip: {pip_pkg}\n" if pip_pkg else "")
        + f"fields:\n{field_lines}\n"
        + test_snippet_yaml
        + "```\n"
    )
    user_ds_path = Path("~/.anton/datasources.md").expanduser()
    tmp_path = user_ds_path.with_suffix(".tmp")

    # Write to temp, validate it parses, then rename atomically
    existing = (
        user_ds_path.read_text(encoding="utf-8") if user_ds_path.is_file() else ""
    )

    existing = remove_engine_block(existing, slug)

    tmp_path.write_text(existing + yaml_block, encoding="utf-8")

    parsed = registry.validate_file(tmp_path)
    if slug in parsed:
        import shutil

        shutil.move(str(tmp_path), str(user_ds_path))
    else:
        tmp_path.unlink(missing_ok=True)
        console.print(
            "[anton.warning]Could not validate engine definition — "
            "credentials saved but engine not written to datasources.md.[/]"
        )

    registry.reload()
    engine_def = registry.get(slug)
    if engine_def is None:
        # Fallback: construct inline so the flow can continue even if parse failed
        engine_def = DatasourceEngine(
            engine=slug,
            display_name=display_name,
            pip=pip_pkg,
            fields=fields,
            test_snippet=test_snippet,
        )

    # All required fields must be present before the caller saves credentials
    missing_required = [f.name for f in fields if f.required and f.name not in credentials]
    if missing_required:
        console.print(
            "[anton.warning]    Cannot save — missing required fields: "
            f"{', '.join(missing_required)}. Aborting.[/]"
        )
        console.print()
        return None

    return engine_def, credentials


async def _reconnect_to_saved(
    console: Console,
    session: "ChatSession",
    vault: "DataVault",
    registry: "DatasourceRegistry",
    slug: str,
    conn: dict,
    *,
    from_tool_call: bool = False,
) -> "ChatSession":
    """Inject env for a saved connection and mark it as the active datasource."""
    restore_namespaced_env(vault)
    session._active_datasource = slug
    recon_engine_def = registry.get(conn["engine"])
    if recon_engine_def:
        register_secret_vars(recon_engine_def, engine=conn["engine"], name=conn["name"])
        engine_label = recon_engine_def.display_name
    else:
        engine_label = conn["engine"]
    console.print()
    console.print(
        f'[anton.success]        ✓ Reconnected to [bold]"{slug}"[/bold].[/]'
    )
    console.print()
    if not from_tool_call:
        # When invoked via the LLM tool call, we must not append to
        # session._history here — it would land between a tool_use and
        # its tool_result. The tool wrapper returns a fresh message.
        session._history.append(
            {
                "role": "assistant",
                "content": (
                    f'I\'ve reconnected to the {engine_label} connection "{slug}" '
                    f"in the Local Vault. I can now query this data source when needed."
                ),
            }
        )
    return session


def _build_redirect_message(
    collector: ConnectionCollector,
    user_message: str,
    target_engine: str | None = None,
) -> str:
    """Build a structured REDIRECT message for the main agent.

    Returns a string describing what was collected so far, what's still
    missing, and what the user said. The caller decides where to put it
    (session history for slash-command path, or tool-result return for
    the LLM tool-call path — never both, to keep tool_use/tool_result
    ordering intact).
    """
    collector.redirect_message = user_message.strip()
    payload = collector.to_redirect_result()
    parts = [
        f"REDIRECT during {payload['engine_display']} connection setup.",
        f"Collected so far: {json.dumps(payload['collected_variables'])}.",
    ]
    if payload["missing_required"]:
        parts.append(
            f"Still missing: {', '.join(payload['missing_required'])}."
        )
    if target_engine:
        parts.append(f"User wants to switch to: {target_engine}.")
    parts.append(f'User said: "{collector.redirect_message}".')
    parts.append(
        "Decide what to do next — you may call connect_new_datasource "
        "again with the correct engine and pass known_variables to "
        "pre-fill what's already collected."
    )
    return " ".join(parts)


async def handle_connect_datasource(
    console: Console,
    scratchpads: ScratchpadManager,
    session: "ChatSession",
    datasource_name: str | None = None,
    prefill: str | None = None,
    known_variables: dict[str, str] | None = None,
    from_tool_call: bool = False,
) -> "ChatSession":
    """
    Connect a data source by entering credentials, either for a new name or re-entering for an existing one.

    `known_variables` may pre-fill credential fields (e.g. when called as a
    tool by the LLM, which may have already extracted host/port/etc. from
    the conversation).

    `from_tool_call=True` when invoked via the LLM `connect_new_datasource`
    tool. In that case we must NOT append assistant messages to
    `session._history` — we are sitting between a `tool_use` block and its
    `tool_result` block, and appending messages there violates the
    Anthropic API invariant. The tool wrapper builds its own return
    message from the vault diff instead.
    """

    vault = DataVault()
    registry = DatasourceRegistry()

    if datasource_name is not None:
        parsed = parse_connection_slug(
            datasource_name, [e.engine for e in registry.all_engines()], vault=vault
        )
        if parsed is None:
            console.print(
                f"[anton.warning]Invalid slug '{datasource_name}'. "
                "Expected format: engine-name.[/]"
            )
            console.print()
            return session
        edit_engine, edit_name = parsed
        existing = vault.load(edit_engine, edit_name)
        if existing is None:
            console.print(
                f"[anton.warning]No connection '{datasource_name}' found in Local Vault.[/]"
            )
            console.print()
            return session
        engine_def = registry.get(edit_engine)
        if engine_def is None:
            console.print(
                f"[anton.warning]Unknown engine '{edit_engine}'. "
                "Cannot update credentials.[/]"
            )
            console.print()
            return session

        console.print()
        console.print(
            f"[anton.cyan](anton)[/] Editing [bold]\"{datasource_name}\"[/bold]"
            f" ({engine_def.display_name})."
        )
        console.print("[anton.muted]        Press Enter to keep the current value.[/]")
        console.print()

        # Detect which fields to present (handle auth_method=choice)
        active_fields = engine_def.fields
        if engine_def.auth_method == "choice" and engine_def.auth_methods:
            for am in engine_def.auth_methods:
                am_field_names = {af.name for af in am.fields}
                if any(k in am_field_names for k in existing):
                    active_fields = am.fields
                    break
            if not active_fields:
                active_fields = engine_def.auth_methods[0].fields

        # Start from existing values; let user update field-by-field
        credentials: dict[str, str] = dict(existing)
        for f in active_fields:
            current = existing.get(f.name, "")
            field_label = f"(anton) {f.name}"
            if not f.required:
                field_label += " (optional)"

            if f.secret:
                masked = "••••••••" if current else ""
                label = f"{field_label} [{masked}]" if masked else field_label
                value = await prompt_or_cancel(label, password=True)
                if value is None:
                    return session
                if value:
                    credentials[f.name] = value
                # else: keep existing (already in credentials)
            elif current:
                value = await prompt_or_cancel(
                    f"{field_label}",
                    default=current,
                )
                if value is None:
                    return session
                credentials[f.name] = value if value else current
            elif f.default:
                value = await prompt_or_cancel(
                    f"{field_label}",
                    default=f.default,
                )
                if value is None:
                    return session
                if value:
                    credentials[f.name] = value
            else:
                value = await prompt_or_cancel(field_label)
                if value is None:
                    return session
                if value:
                    credentials[f.name] = value

        if engine_def.test_snippet:
            if not await run_connection_test(
                console, scratchpads, vault, engine_def, credentials, active_fields
            ):
                return session

        vault.save(edit_engine, edit_name, credentials)
        restore_namespaced_env(vault)
        register_secret_vars(engine_def, engine=edit_engine, name=edit_name)
        console.print()
        console.print(
            f'        Credentials updated for [bold]"{datasource_name}"[/bold].'
        )
        console.print()
        console.print(
            "[anton.muted]        You can now ask me questions about your data.[/]"
        )
        console.print()
        if not from_tool_call:
            session._history.append(
                {
                    "role": "assistant",
                    "content": (
                        f"I've updated the credentials for the {engine_def.display_name} connection "
                        f'"{datasource_name}" in the Local Vault.'
                    ),
                }
            )
        return session

    console.print()
    all_engines = registry.all_engines()
    popular_engines = [e for e in all_engines if e.popular and not e.custom]
    other_engines = [e for e in all_engines if not e.popular and not e.custom]
    custom_engines = [e for e in all_engines if e.custom]
    display_engines = popular_engines + other_engines + custom_engines

    saved_connections = vault.list_connections()
    # Build deduplicated list of engine types from saved connections (one per engine)
    seen_engines: set[str] = set()
    recent_engine_entries: list[tuple[str, str]] = []  # (engine_slug, display_name)
    for c in saved_connections:
        if c["engine"] not in seen_engines:
            seen_engines.add(c["engine"])
            engine_obj = registry.get(c["engine"])
            label = engine_obj.display_name if engine_obj else c["engine"]
            recent_engine_entries.append((c["engine"], label))

    def print_sections() -> None:
        console.print(
            "[anton.cyan](anton)[/] Select a data source to create a new connection:\n"
        )
        console.print("       [bold]  Primary")
        console.print(
            "         [bold]  0.[/bold] Custom datasource"
            " (connect anything via API, SQL, or MCP)\n"
        )
        if popular_engines:
            console.print("       [bold]  Most popular")
            for i, e in enumerate(popular_engines, 1):
                console.print(f"          [bold]{i:>2}.[/bold] {e.display_name}")
            console.print()
        if recent_engine_entries:
            start = len(popular_engines) + 1
            console.print("       [bold]  Recently used data sources")
            for i, (_, label) in enumerate(recent_engine_entries, start):
                console.print(f"          [bold]{i:>2}.[/bold] {label}")
            console.print()

    def print_all() -> None:
        console.print(
            "[anton.cyan](anton)[/] All data sources (★ = popular):\n"
        )
        console.print("       [bold]  Primary")
        console.print(
            "         [bold]  0.[/bold] Custom datasource"
            " (connect anything via API, SQL, or MCP)\n"
        )
        for i, e in enumerate(display_engines, 1):
            star = " ★" if e.popular else ""
            console.print(f"          [bold]{i:>2}.[/bold] {e.display_name}{star}")
        console.print()

    async def get_create_new_answer() -> str | None:
        print_sections()
        console.print(
            "       [anton.muted]Don't see yours? Type a datasource name (e.g., GitHub, Gmail, Jira, ...)\n"
            "       It can be virtually any datasource — we'll figure out the details together.[/]"
        )
        console.print()
        ans = await prompt_or_cancel(
            "(anton) Enter a number or type a datasource name",
        )
        if ans is None:
            return None
        if ans.strip().lower() == "all":
            console.print()
            print_all()
            ans = await prompt_or_cancel(
                "(anton) Enter a number or type a name",
            )
        return ans

    if prefill:
        answer = prefill
    elif saved_connections:
        console.print()
        console.print("[anton.cyan](anton)[/] What would you like to do?\n")
        console.print("          [bold]  1.[/bold] Use an existing connection")
        console.print("          [bold]  2.[/bold] Create a new connection")
        console.print()
        top_choice = await prompt_or_cancel(
            "(anton) Enter a number", choices=["1", "2"]
        )
        if top_choice is None:
            return session

        if top_choice == "1":
            console.print()
            console.print("[anton.cyan](anton)[/] Your saved connections:\n")
            for i, c in enumerate(saved_connections, 1):
                conn_slug = f"{c['engine']}-{c['name']}"
                engine_obj = registry.get(c["engine"])
                engine_label = engine_obj.display_name if engine_obj else c["engine"]
                console.print(
                    f"          [bold]{i:>2}.[/bold] {conn_slug}"
                    f" [dim]— {engine_label}[/]"
                )
            console.print()
            pick = await prompt_or_cancel(
                "(anton) Enter a number",
                choices=[str(i) for i in range(1, len(saved_connections) + 1)],
            )
            if pick is None:
                return session
            picked_conn = saved_connections[int(pick) - 1]
            picked_slug = f"{picked_conn['engine']}-{picked_conn['name']}"
            return await _reconnect_to_saved(
                console, session, vault, registry, picked_slug, picked_conn,
                from_tool_call=from_tool_call,
            )

        # top_choice == "2": create new connection
        answer = await get_create_new_answer()
        if answer is None:
            return session
    else:
        answer = await get_create_new_answer()
        if answer is None:
            return session

    stripped_answer = answer.strip()
    known_slugs = {
        f"{c['engine']}-{c['name']}": c for c in vault.list_connections()
    }
    if stripped_answer in known_slugs:
        conn = known_slugs[stripped_answer]
        return await _reconnect_to_saved(
            console, session, vault, registry, stripped_answer, conn,
            from_tool_call=from_tool_call,
        )

    engine_def: DatasourceEngine | None = None
    custom_source = False
    llm_recognised = False
    # Recently used data sources are numbered after popular engines
    saved_start = len(popular_engines) + 1
    max_num = len(popular_engines) + len(recent_engine_entries)

    if stripped_answer.isdigit() or (stripped_answer.lstrip("-").isdigit()):
        pick_num = int(stripped_answer)
        if pick_num == 0:
            custom_source = True
        elif 1 <= pick_num <= len(popular_engines):
            engine_def = popular_engines[pick_num - 1]
        elif recent_engine_entries and saved_start <= pick_num <= max_num:
            # User picked a recently used data source — start a new connection of that engine
            picked_engine_slug, _ = recent_engine_entries[pick_num - saved_start]
            engine_def = registry.get(picked_engine_slug)
            if engine_def is None:
                custom_source = True
        else:
            console.print(
                f"[anton.warning](anton)[/] '{stripped_answer}' is out of range. "
                f"Please enter 0\u2013{max_num}.[/]"
            )
            console.print()
            return session

    if engine_def is None and not custom_source:
        engine_def = registry.find_by_name(stripped_answer)
        # if exact match not found, try substring match against display and engine names
        if engine_def is None:
            needle = stripped_answer.lower()
            candidates = [
                e
                for e in all_engines
                if needle in e.display_name.lower() or needle in e.engine.lower()
            ]
            if len(candidates) == 1:
                engine_def = candidates[0]
            elif len(candidates) > 1:
                console.print()
                console.print(
                    f"[anton.warning](anton)[/] '{stripped_answer}' matches multiple engines — "
                    "which one did you mean?"
                )
                console.print()
                for i, e in enumerate(candidates, 1):
                    console.print(f"        {i}. {e.display_name}")
                console.print()
                pick = await prompt_or_cancel("(anton) Enter a number")
                if pick is None:
                    return session
                pick = (pick or "").strip()
                try:
                    engine_def = candidates[int(pick) - 1]
                except (ValueError, IndexError):
                    console.print("[anton.warning]Invalid choice. Aborting.[/]")
                    console.print()
                    return session
        # Ask the LLM to identify the datasource
        if engine_def is None:
            engine_names = [e.display_name for e in all_engines]
            try:
                console.print()
                console.print("[anton.muted]        Looking up datasource…[/]")
                llm_resp = await session._llm.plan(
                    system="You are a datasource identification assistant.",
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"The user typed: {stripped_answer!r}\n"
                                f"Known datasources: {engine_names!r}\n\n"
                                "If the user input clearly matches one of the known datasources, "
                                "reply with EXACTLY: MATCH:<display_name>\n"
                                "If it does NOT match any known datasource but you recognise it "
                                "as a real service/tool, reply with EXACTLY: CUSTOM\n"
                                "If you don't recognise it at all, reply with EXACTLY: UNKNOWN\n"
                                "Reply with only one of those three forms, nothing else."
                            ),
                        }
                    ],
                    max_tokens=64,
                )
                llm_text = (llm_resp.content or "").strip()
            except Exception:
                llm_text = "UNKNOWN"

            llm_recognised = llm_text == "CUSTOM" or llm_text.startswith("MATCH:")

            if llm_text.startswith("MATCH:"):
                matched_name = llm_text[len("MATCH:"):].strip()
                matched_engine = next(
                    (e for e in all_engines if e.display_name == matched_name), None
                )
                if matched_engine is not None:
                    if matched_name.lower() != stripped_answer.lower():
                        confirm = await prompt_or_cancel(
                            f'(anton) Did you mean "{matched_name}"?',
                            choices=["y", "n"], default="y",
                        )
                        if confirm is not None and confirm.strip().lower() == "y":
                            engine_def = matched_engine
                    else:
                        engine_def = matched_engine

            if engine_def is None:
                custom_source = True

    if custom_source:
        result = await handle_add_custom_datasource(
            console, stripped_answer if not stripped_answer.isdigit() else "", registry, session,
            known_service=llm_recognised,
        )
        if result is None:
            return session
        engine_def, credentials = result
        if engine_def.test_snippet:
            if not await run_connection_test(
                console, scratchpads, vault, engine_def, credentials, engine_def.fields
            ):
                return session
        conn_name = uuid.uuid4().hex[:8]
        vault.save(engine_def.engine, conn_name, credentials)
        slug = f"{engine_def.engine}-{conn_name}"
        restore_namespaced_env(vault)
        session._active_datasource = slug
        register_secret_vars(engine_def, engine=engine_def.engine, name=conn_name)
        console.print(
            f'        Credentials saved to Local Vault as [bold]"{slug}"[/bold].'
        )
        console.print()
        console.print(
            "[anton.muted]        You can now ask me questions about your data.[/]"
        )
        console.print()
        if not from_tool_call:
            session._history.append(
                {
                    "role": "assistant",
                    "content": (
                        f'I\'ve saved a {engine_def.display_name} connection named "{slug}" '
                        f"to the Local Vault. I can now query this data source when needed."
                    ),
                }
            )
        return session

    assert engine_def is not None  # custom_source path always returns before this line
    active_fields = engine_def.fields
    chosen_method: "AuthMethod | None" = None
    if engine_def.auth_method == "choice" and engine_def.auth_methods:
        console.print()
        console.print(
            f"[anton.cyan](anton)[/] How would you like to authenticate with "
            f"[bold]{engine_def.display_name}[/]?"
        )
        console.print()
        for i, am in enumerate(engine_def.auth_methods, 1):
            console.print(f"        {i}. {am.display}")
        console.print()
        choice_str = await prompt_or_cancel("(anton) Enter a number")
        if choice_str is None:
            return session
        choice_str = (choice_str or "").strip()
        try:
            choice_idx = int(choice_str) - 1
            chosen_method = engine_def.auth_methods[choice_idx]
        except (ValueError, IndexError):
            console.print("[anton.warning]Invalid choice. Aborting.[/]")
            console.print()
            return session
        active_fields = chosen_method.fields

    # ── Smart credential collection ────────────────────────────────────
    # Track filled vs. missing fields as a puzzle. Each user response is
    # parsed via the LLM to extract any variables mentioned, so users can
    # fill multiple fields at once, paste a connection string, or change
    # direction mid-flow.
    collector = ConnectionCollector(
        engine_def=engine_def,
        auth_method=chosen_method,
    )
    if known_variables:
        collector.fill_many(known_variables)

    known_engine_slugs = [e.engine for e in registry.all_engines()]
    partial = False
    required_fields = [f for f in active_fields if f.required]
    optional_fields = [f for f in active_fields if not f.required]

    if collector.is_complete:
        # Pre-fill already covered everything — skip the field list and
        # the help prompt and go straight to testing + saving. Show a
        # brief confirmation of what was received.
        filled_names = [
            f.name for f in active_fields if collector.collected.get(f.name)
        ]
        console.print()
        console.print(
            f"[anton.cyan](anton)[/] Got everything for [bold]"
            f"{engine_def.display_name}[/] from context: "
            f"{', '.join(filled_names)}."
        )
        console.print()
    else:
        # Show the field list so the user sees what's expected.
        console.print()
        console.print(
            f"[anton.cyan](anton)[/] To connect [bold]"
            f"{engine_def.display_name}[/], I'll need the following:"
        )
        console.print()

        if required_fields:
            console.print("        [bold]Required[/]      " + "─" * 39)
            for f in required_fields:
                marker = (
                    "[green]✓[/] " if collector.collected.get(f.name) else "• "
                )
                console.print(
                    f"        {marker}[bold]{f.name:<12}[/] "
                    f"[anton.muted]— {f.description}[/]"
                )

        if optional_fields:
            console.print()
            console.print("        [bold]Optional[/]      " + "─" * 39)
            for f in optional_fields:
                marker = (
                    "[green]✓[/] " if collector.collected.get(f.name) else "• "
                )
                console.print(
                    f"        {marker}[bold]{f.name:<12}[/] "
                    f"[anton.muted]— {f.description}[/]"
                )

        console.print()

        # Offer instructions — but only if nothing has been pre-filled.
        # If the user already provided some credentials (via the tool's
        # `known_variables` or a paste), they clearly know what they're
        # doing and don't need guidance — just prompt for what's missing.
        if not collector.collected:
            help_answer = await prompt_or_cancel(
                "(anton) Do you need instructions on how to obtain these credentials?",
                choices_display="y/n", default="n",
            )
            if help_answer is None:
                return session
            normalized = help_answer.strip().lower()
            if normalized == "y":
                await show_credential_help(
                    console, session, engine_def.display_name, None, active_fields,
                )
            elif normalized and normalized != "n":
                # Non-y/n answer — maybe the user pasted credentials here.
                extracted = await extract_variables(
                    help_answer,
                    expected_fields=collector.active_fields,
                    current_engine=engine_def.engine,
                    current_engine_display=engine_def.display_name,
                    known_engine_slugs=known_engine_slugs,
                    session=session,
                )
                if extracted.is_redirect:
                    redirect_text = _build_redirect_message(
                        collector, help_answer, extracted.redirect_engine
                    )
                    session._pending_connect_redirect = redirect_text
                    if not from_tool_call:
                        session._history.append(
                            {"role": "assistant", "content": redirect_text}
                        )
                    return session
                if extracted.variables:
                    filled = collector.fill_many(extracted.variables)
                    if filled:
                        console.print(
                            f"[anton.muted]        Got: {', '.join(filled)}[/]"
                        )
                        console.print()

    while not collector.is_complete:
        collector.format_status(console)
        console.print()

        next_field = collector.next_field
        # When only one required field remains, ask for it directly with
        # the matching prompt style (password masking, default value,
        # etc.). No LLM extraction needed — the answer IS the value.
        only_one_required = (
            next_field is not None
            and next_field.required
            and len(collector.missing_required) == 1
        )

        if only_one_required and next_field is not None:
            label = f"(anton) {next_field.name}"
            if next_field.secret:
                value = await prompt_or_cancel(label, password=True)
            elif next_field.default:
                value = await prompt_or_cancel(label, default=next_field.default)
            else:
                value = await prompt_or_cancel(label)
            if value is None:
                return session
            if not value:
                # Empty answer for the only missing required field —
                # treat as a partial save signal.
                partial = True
                break
            collector.fill(next_field.name, value)
            continue

        # Multiple fields remain — open prompt that accepts bulk input
        missing_names = ", ".join(f.name for f in collector.missing_required)
        prompt_label = (
            f"(anton) Provide values for {missing_names} "
            f"(one at a time, or 'key=value key2=value2', or 'skip')"
        )
        value = await prompt_or_cancel(prompt_label)
        if value is None:
            return session
        if value.strip().lower() == "skip":
            partial = True
            break
        if not value.strip():
            continue

        extracted = await extract_variables(
            value,
            expected_fields=collector.active_fields,
            current_engine=engine_def.engine,
            current_engine_display=engine_def.display_name,
            known_engine_slugs=known_engine_slugs,
            session=session,
        )

        if extracted.is_redirect:
            redirect_text = _build_redirect_message(
                collector, value, extracted.redirect_engine
            )
            # Stash for the tool wrapper; also mirror to history only if
            # we're NOT inside a tool_use/tool_result pair.
            session._pending_connect_redirect = redirect_text
            if not from_tool_call:
                session._history.append(
                    {"role": "assistant", "content": redirect_text}
                )
            return session

        if extracted.variables:
            filled = collector.fill_many(extracted.variables)
            if filled:
                console.print(
                    f"[anton.muted]        Got: {', '.join(filled)}[/]"
                )
                console.print()
                continue

        # LLM returned nothing structured — fall back to treating the
        # input as the value for the next missing required field.
        if next_field is not None:
            collector.fill(next_field.name, value.strip())
        else:
            console.print(
                "[anton.warning]        Couldn't parse that. "
                "Try 'key=value' or one value at a time.[/]"
            )
            console.print()

    credentials: dict[str, str] = dict(collector.collected)

    if partial:
        auto_name = uuid.uuid4().hex[:8]
        vault.save(engine_def.engine, auto_name, credentials)
        slug = f"{engine_def.engine}-{auto_name}"
        console.print()
        console.print(
            f"[anton.muted]Partial connection saved to Local Vault as "
            f'[bold]"{slug}"[/bold]. '
            f"Run [bold]/edit {slug}[/bold] to complete it when you're ready.[/]"
        )
        console.print()
        return session

    if engine_def.test_snippet:
        if not await run_connection_test(
            console, scratchpads, vault, engine_def, credentials, active_fields
        ):
            # Either the test failed and the user declined to re-enter
            # credentials, or the user pressed Escape during the retry
            # prompt. Mark this so the tool wrapper can return an
            # accurate (non-misleading) message to the LLM.
            session._pending_connect_status = "test_failed"
            return session

    conn_name = registry.derive_name(engine_def, credentials)
    if not conn_name:
        conn_name = uuid.uuid4().hex[:8]

    slug = f"{engine_def.engine}-{conn_name}"

    if vault.load(engine_def.engine, conn_name) is not None:
        console.print()
        console.print(
            f'[anton.warning](anton)[/] A connection [bold]"{slug}"[/bold] already exists.'
        )
        console.print()
        choice = await prompt_or_cancel(
            f"(anton) {_PROMPT_RECONNECT_CANCEL}",
        )
        if choice is None or choice.strip().lower() != "reconnect":
            console.print("[anton.muted]Cancelled.[/]")
            console.print()
            return session
        restore_namespaced_env(vault)
        register_secret_vars(engine_def, engine=engine_def.engine, name=conn_name)
        console.print()
        console.print(
            f'[anton.success]        ✓ Reconnected to [bold]"{slug}"[/bold].[/]'
        )
        console.print()
        if not from_tool_call:
            session._history.append(
                {
                    "role": "assistant",
                    "content": (
                        f'I\'ve reconnected to the {engine_def.display_name} connection "{slug}" '
                        f"in the Local Vault. I can now query this data source when needed."
                    ),
                }
            )
        return session

    vault.save(engine_def.engine, conn_name, credentials)
    restore_namespaced_env(vault)
    session._active_datasource = slug
    register_secret_vars(engine_def, engine=engine_def.engine, name=conn_name)
    console.print(f'        Credentials saved to Local Vault as [bold]"{slug}"[/bold].')

    console.print()
    console.print(
        "[anton.muted]        You can now ask me questions about your data.[/]"
    )
    console.print()

    # Inject a brief assistant message so the LLM is aware of the new
    # connection — but only when NOT in a tool call (in that case the
    # tool wrapper constructs its own return message; appending here
    # would break tool_use/tool_result pairing).
    if not from_tool_call:
        session._history.append(
            {
                "role": "assistant",
                "content": (
                    f'I\'ve saved a {engine_def.display_name} connection named "{slug}" '
                    f"to the Local Vault. I can now query this data source when needed."
                ),
            }
        )
    return session


async def handle_test_datasource(
    console: Console,
    scratchpads: ScratchpadManager,
    slug: str,
) -> None:
    """Test an existing Local Vault connection by running its test_snippet."""
    if not slug:
        console.print(
            "[anton.warning]Usage: /test <engine-name>[/]"
        )
        console.print()
        return

    vault = DataVault()
    registry = DatasourceRegistry()
    parsed = parse_connection_slug(slug, [e.engine for e in registry.all_engines()], vault=vault)
    if parsed is None:
        console.print(
            f"[anton.warning]Invalid name '{slug}'. Use engine-name format.[/]"
        )
        console.print()
        return
    engine, name = parsed
    fields = vault.load(engine, name)
    if fields is None:
        console.print(
            f"[anton.warning]No connection '{slug}' found in Local Vault.[/]"
        )
        console.print()
        return

    engine_def = registry.get(engine)
    if engine_def is None:
        console.print(
            f"[anton.warning]Unknown engine '{engine}'. Cannot test.[/]"
        )
        console.print()
        return

    if not engine_def.test_snippet:
        console.print(
            f"[anton.warning]No test snippet defined for '{engine}'. Cannot test.[/]"
        )
        console.print()
        return

    console.print()
    console.print(
        f"[anton.cyan](anton)[/] Testing connection [bold]{slug}[/bold]…"
    )

    vault.clear_ds_env()
    vault.inject_env(engine, name, flat=True)
    register_secret_vars(engine_def)  # flat names for scrubbing during test

    cell = None
    try:
        pad = await scratchpads.get_or_create("__datasource_test__")
        await pad.reset()
        if engine_def.pip:
            await pad.install_packages([engine_def.pip])
        cell = await pad.execute(engine_def.test_snippet)
    finally:
        restore_namespaced_env(vault)

    if cell is None or cell.error or (
        cell.stdout.strip() != "ok" and cell.stderr.strip()
    ):
        error_text = ""
        if cell is not None:
            error_text = cell.error or cell.stderr.strip() or cell.stdout.strip()
        first_line = (
            next((ln for ln in error_text.splitlines() if ln.strip()), error_text)
            if error_text
            else "unknown error"
        )
        console.print()
        console.print(
            f"[anton.warning](anton)[/] ✗ Connection test failed for"
            f" [bold]{slug}[/bold]."
        )
        console.print()
        console.print(f"        Error: {first_line}")
    else:
        console.print(
            f"[anton.success]        ✓ Connection test passed for"
            f" [bold]{slug}[/bold]![/]"
        )
    console.print()
