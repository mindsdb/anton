"""Main datasource connection flow."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING

from anton.connect_collector import ConnectionCollector, extract_variables
from anton.core.datasources.data_vault import DataVault, LocalDataVault
from anton.core.datasources.datasource_registry import DatasourceRegistry
from anton.utils.datasources import (
    find_matching_connection,
    parse_connection_slug,
    register_secret_vars,
    restore_namespaced_env,
    save_connection,
)
from anton.utils.prompt import prompt_or_cancel
from anton.commands.datasource.helpers import prompt_field_value, show_credential_help
from anton.commands.datasource.custom import collect_custom_credentials, handle_add_custom_datasource
from anton.commands.datasource.verify import run_connection_test

_PROMPT_RECONNECT_CANCEL = "(reconnect/cancel)"


def _build_redirect_message(
    collector: ConnectionCollector,
    user_message: str,
    target_engine: str | None = None,
) -> str:
    """Build a structured REDIRECT message for the main agent."""
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

if TYPE_CHECKING:
    from rich.console import Console
    from anton.chat import ChatSession
    from anton.core.backends.manager import ScratchpadManager


async def _reconnect_to_saved(
    console: "Console",
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


async def handle_connect_datasource(
    console: "Console",
    scratchpads: "ScratchpadManager",
    session: "ChatSession",
    datasource_name: str | None = None,
    prefill: str | None = None,
    known_variables: dict[str, str] | None = None,
    from_tool_call: bool = False,
    vault: "DataVault | None" = None,
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

    vault = vault or LocalDataVault()
    registry = DatasourceRegistry()

    try:
        from anton.config.settings import AntonSettings
        _settings = AntonSettings()
    except Exception:
        _settings = None

    def _telemetry(event_name: str, engine: str = "") -> None:
        if _settings and not from_tool_call:
            from anton.analytics import send_event
            send_event(_settings, event_name, engine=engine)

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

        _telemetry("ds_connect_attempt", engine=edit_engine)

        console.print()
        console.print(
            f"[anton.cyan](anton)[/] Editing [bold]\"{datasource_name}\"[/bold]"
            f" ({engine_def.display_name})."
        )
        console.print("[anton.muted]        Enter to keep current value.[/]")
        console.print()

        active_fields = engine_def.fields
        if engine_def.auth_method == "choice" and engine_def.auth_methods:
            for am in engine_def.auth_methods:
                am_field_names = {af.name for af in am.fields}
                if any(k in am_field_names for k in existing):
                    active_fields = am.fields
                    break
            if not active_fields:
                active_fields = engine_def.auth_methods[0].fields

        credentials: dict[str, str] = dict(existing)
        for f in active_fields:
            if not await prompt_field_value(f, credentials):
                return session

        if engine_def.test_snippet:
            if not await run_connection_test(
                console, scratchpads, vault, engine_def, credentials, active_fields
            ):
                _telemetry("ds_connect_failed", engine=edit_engine)
                return session

        vault.save(edit_engine, edit_name, credentials)
        _telemetry("ds_connect_success", engine=edit_engine)
        restore_namespaced_env(vault)
        register_secret_vars(engine_def, engine=edit_engine, name=edit_name)
        console.print()
        console.print(
            f'        Credentials updated for [bold]"{datasource_name}"[/bold].'
        )
        console.print()
        console.print("[anton.muted]        Ready to query your data.[/]")
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

    saved_connections = vault.list_connections()

    if prefill:
        stripped_answer = prefill.strip()
        if not stripped_answer:
            return session
    else:
        console.print()
        console.print(
            "[anton.cyan](anton)[/] What would you like to connect?"
        )
        console.print(
            "  [anton.muted]Examples: PostgreSQL, MySQL, Snowflake, BigQuery, "
            "Gmail, GitHub, HubSpot, Salesforce, Jira, REST API.[/]"
        )
        if saved_connections:
            slugs = ", ".join(f"{c['engine']}-{c['name']}" for c in saved_connections[:5])
            more = "" if len(saved_connections) <= 5 else f" (+{len(saved_connections) - 5} more, see /list)"
            console.print(
                f"  [anton.muted]Or reconnect: {slugs}{more}[/]"
            )
        console.print()

        answer = await prompt_or_cancel("(anton) Connect to")
        if answer is None:
            return session

        stripped_answer = answer.strip()
        if not stripped_answer:
            return session

    known_slugs = {f"{c['engine']}-{c['name']}": c for c in saved_connections}
    if stripped_answer in known_slugs:
        conn = known_slugs[stripped_answer]
        return await _reconnect_to_saved(
            console, session, vault, registry, stripped_answer, conn,
            from_tool_call=from_tool_call,
        )

    engine_def = registry.find_by_name(stripped_answer)

    if engine_def is None:
        _telemetry("ds_connect_attempt", engine=stripped_answer)
        result = await handle_add_custom_datasource(
            console, stripped_answer, registry, session, known_service=False,
        )
        if result is None:
            return session
        engine_def, credentials = result
        if engine_def.test_snippet:
            async def _retry_custom_credentials() -> bool:
                console.print("[anton.muted]        Let's update those details.[/]")
                return await collect_custom_credentials(
                    console,
                    session,
                    registry,
                    engine_def.display_name,
                    engine_def.fields,
                    credentials,
                    edit_existing=True,
                )

            if not await run_connection_test(
                console,
                scratchpads,
                vault,
                engine_def,
                credentials,
                engine_def.fields,
                retry_edit_callback=_retry_custom_credentials,
            ):
                _telemetry("ds_connect_failed", engine=engine_def.engine)
                return session
        conn_name = uuid.uuid4().hex[:8]
        slug = save_connection(vault, engine_def, conn_name, credentials)
        _telemetry("ds_connect_success", engine=engine_def.engine)
        session._active_datasource = slug
        console.print(
            f'        Saved to Local Vault as [bold]"{slug}"[/bold].'
        )
        console.print()
        console.print("[anton.muted]        Ready to query your data.[/]")
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

    assert engine_def is not None
    # _telemetry("ds_connect_attempt", engine=engine_def.engine)
    active_fields = engine_def.fields
    chosen_method = None
    if engine_def.auth_method == "choice" and engine_def.auth_methods:
        console.print()
        console.print(
            f"[anton.cyan](anton)[/] Auth method for [bold]{engine_def.display_name}[/]:"
        )
        for i, am in enumerate(engine_def.auth_methods, 1):
            console.print(f"        {i}. {am.display}")
        console.print()
        choice_str = await prompt_or_cancel("(anton) Choose")
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
        filled_names = [
            f.name for f in active_fields if collector.collected.get(f.name)
        ]
        console.print()
        console.print(
            f"[anton.cyan](anton)[/] Got [bold]{engine_def.display_name}[/] "
            f"details from context: {', '.join(filled_names)}."
        )
        console.print()
    else:
        console.print()
        console.print(
            f"[anton.cyan](anton)[/] [bold]{engine_def.display_name}[/] needs:"
        )
        for f in required_fields:
            marker = "[green]✓[/] " if collector.collected.get(f.name) else "• "
            console.print(
                f"        {marker}[bold]{f.name}[/] "
                f"[anton.muted]— {f.description}[/]"
            )
        for f in optional_fields:
            marker = "[green]✓[/] " if collector.collected.get(f.name) else "• "
            console.print(
                f"        {marker}[bold]{f.name}[/] "
                f"[anton.muted]— {f.description} (optional)[/]"
            )
        console.print()

    while not collector.is_complete:
        next_field = collector.next_field
        remaining_count = len(collector.missing_required)
        use_sequential = next_field is not None and next_field.required

        if use_sequential and next_field is not None:
            pretty = next_field.name.replace("_", " ").title()
            label = f"(anton) {pretty}"
            if next_field.secret:
                value = await prompt_or_cancel(label, password=True)
            elif next_field.default:
                value = await prompt_or_cancel(label, default=next_field.default)
            else:
                value = await prompt_or_cancel(label)
            if value is None:
                return session
            if not value:
                partial = True
                break
            stripped = value.strip()
            lowered = stripped.lower()
            if lowered == "skip":
                partial = True
                break
            if lowered == "help":
                await show_credential_help(
                    console, session, engine_def.display_name, next_field, active_fields,
                )
                continue
            if remaining_count > 1 and (
                "=" in stripped or "://" in stripped or "\n" in stripped
            ):
                extracted = await extract_variables(
                    stripped,
                    expected_fields=collector.active_fields,
                    current_engine=engine_def.engine,
                    current_engine_display=engine_def.display_name,
                    known_engine_slugs=known_engine_slugs,
                    session=session,
                )
                if extracted.is_redirect:
                    redirect_text = _build_redirect_message(
                        collector, stripped, extracted.redirect_engine
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
                        continue
            collector.fill(next_field.name, value)
            continue

        collector.format_status(console)
        console.print()

        missing_names = ", ".join(f.name for f in collector.missing_required)
        prompt_label = f"(anton) {missing_names} (value, 'key=value …', 'help', or 'skip')"
        value = await prompt_or_cancel(prompt_label)
        if value is None:
            return session
        if value.strip().lower() == "skip":
            partial = True
            break
        if value.strip().lower() == "help":
            await show_credential_help(
                console, session, engine_def.display_name, None, active_fields,
            )
            continue
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

        if next_field is not None:
            collector.fill(next_field.name, value.strip())
        else:
            console.print(
                "[anton.warning]        Couldn't parse. "
                "Try 'key=value' or one value.[/]"
            )
            console.print()

    credentials: dict[str, str] = dict(collector.collected)

    if partial:
        auto_name = uuid.uuid4().hex[:8]
        vault.save(engine_def.engine, auto_name, credentials)
        slug = f"{engine_def.engine}-{auto_name}"
        console.print()
        console.print(
            f'[anton.muted]Partial save: [bold]"{slug}"[/bold]. '
            f"Run [bold]/edit {slug}[/bold] to finish.[/]"
        )
        console.print()
        return session

    if engine_def.test_snippet:
        if not await run_connection_test(
            console, scratchpads, vault, engine_def, credentials, active_fields
        ):
            _telemetry("ds_connect_failed", engine=engine_def.engine)
            session._pending_connect_status = "test_failed"
            return session

    existing_name = find_matching_connection(vault, engine_def, credentials)
    if existing_name is not None:
        conn_name = existing_name
        is_update = True
    else:
        conn_name = uuid.uuid4().hex[:8]
        while vault.load(engine_def.engine, conn_name) is not None:
            conn_name = uuid.uuid4().hex[:8]
        is_update = False

    slug = f"{engine_def.engine}-{conn_name}"

    save_connection(vault, engine_def, conn_name, credentials)
    _telemetry("ds_connect_success", engine=engine_def.engine)
    session._active_datasource = slug
    if is_update:
        console.print(
            f'        Updated existing connection [bold]"{slug}"[/bold].'
        )
    else:
        console.print(f'        Saved to Local Vault as [bold]"{slug}"[/bold].')

    console.print()
    console.print("[anton.muted]        Ready to query your data.[/]")
    console.print()

    if not from_tool_call:
        verb = "updated" if is_update else "saved"
        session._history.append(
            {
                "role": "assistant",
                "content": (
                    f'I\'ve {verb} the {engine_def.display_name} connection "{slug}" '
                    f"in the Local Vault. I can now query this data source when needed."
                ),
            }
        )
    return session
