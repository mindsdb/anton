"""Slash-command handlers for datasource commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console

from anton.data_vault import DataVault
from anton.datasource_registry import DatasourceRegistry
from anton.prompt_utils import prompt_or_cancel

if TYPE_CHECKING:
    pass


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
    from anton.chat import _restore_namespaced_env, _remove_engine_block, parse_connection_slug

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

    _parsed = parse_connection_slug(slug, [e.engine for e in registry.all_engines()], vault=vault)
    if _parsed is None:
        console.print(
            f"[anton.warning]Invalid name '{slug}'. Use engine-name format.[/]"
        )
        console.print()
        return
    engine, name = _parsed
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
        _restore_namespaced_env(vault)
        engine_def = registry.get(engine)
        if engine_def is not None and engine_def.custom:
            remaining = [
                c for c in vault.list_connections() if c["engine"] == engine
            ]
            if not remaining:
                user_path = DatasourceRegistry._USER_PATH
                if user_path.is_file():
                    updated = _remove_engine_block(
                        user_path.read_text(encoding="utf-8"), engine
                    )
                    user_path.write_text(updated, encoding="utf-8")
                    registry.reload()
        console.print(f"[anton.success]Removed {slug}.[/]")
    else:
        console.print("[anton.muted]Cancelled.[/]")
    console.print()
