"""Main datasource connection flow."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING

from anton.connect_collector import ConnectionCollector, extract_variables
from anton.core.datasources.data_vault import DataVault, LocalDataVault
from anton.core.datasources.datasource_registry import DatasourceRegistry
from anton.utils.datasources import parse_connection_slug, register_secret_vars, restore_namespaced_env
from anton.utils.prompt import prompt_or_cancel
from anton.commands.datasource.helpers import show_credential_help
from anton.commands.datasource.custom import handle_add_custom_datasource
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
        console.print("[anton.muted]        Press Enter to keep the current value.[/]")
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
    seen_engines: set[str] = set()
    recent_engine_entries: list[tuple[str, str]] = []
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

    engine_def = None
    custom_source = False
    llm_recognised = False
    saved_start = len(popular_engines) + 1
    max_num = len(popular_engines) + len(recent_engine_entries)

    if stripped_answer.isdigit() or (stripped_answer.lstrip("-").isdigit()):
        pick_num = int(stripped_answer)
        if pick_num == 0:
            custom_source = True
        elif 1 <= pick_num <= len(popular_engines):
            engine_def = popular_engines[pick_num - 1]
        elif recent_engine_entries and saved_start <= pick_num <= max_num:
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
        _telemetry("ds_connect_attempt", engine=stripped_answer if not stripped_answer.isdigit() else "custom")
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
                _telemetry("ds_connect_failed", engine=engine_def.engine)
                return session
        conn_name = uuid.uuid4().hex[:8]
        vault.save(engine_def.engine, conn_name, credentials)
        _telemetry("ds_connect_success", engine=engine_def.engine)
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

    assert engine_def is not None
    _telemetry("ds_connect_attempt", engine=engine_def.engine)
    active_fields = engine_def.fields
    chosen_method = None
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
            f"[anton.cyan](anton)[/] Got everything for [bold]"
            f"{engine_def.display_name}[/] from context: "
            f"{', '.join(filled_names)}."
        )
        console.print()
    else:
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
                partial = True
                break
            collector.fill(next_field.name, value)
            continue

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
            _telemetry("ds_connect_failed", engine=engine_def.engine)
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
    _telemetry("ds_connect_success", engine=engine_def.engine)
    restore_namespaced_env(vault)
    session._active_datasource = slug
    register_secret_vars(engine_def, engine=engine_def.engine, name=conn_name)
    console.print(f'        Credentials saved to Local Vault as [bold]"{slug}"[/bold].')

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
