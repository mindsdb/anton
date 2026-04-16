"""Slash-command handlers for /theme, /explain, and /help."""

from __future__ import annotations

from rich.console import Console

from anton.explainability import ExplainabilityStore


def handle_theme(console: Console, arg: str) -> None:
    """Switch the color theme (light/dark)."""
    import os
    from anton.channel.theme import detect_color_mode, build_rich_theme

    current = detect_color_mode()

    if not arg:
        new_mode = "light" if current == "dark" else "dark"
    elif arg in ("light", "dark"):
        new_mode = arg
    else:
        console.print(
            f"[anton.warning]Unknown theme '{arg}'. Use: /theme light | /theme dark[/]"
        )
        console.print()
        return

    os.environ["ANTON_THEME"] = new_mode
    console._theme_stack.push_theme(build_rich_theme(new_mode))
    console.print(f"[anton.success]Theme set to {new_mode}.[/]")
    console.print()


def print_slash_help(console: Console) -> None:
    """Print available slash commands."""
    console.print()

    console.print("[anton.cyan]Available commands:[/]")

    console.print("\n[bold]LLM Provider[/]")
    console.print("  [bold]/llm[/]      — Change LLM provider or API key")

    console.print("\n[bold]Data Connections[/]")
    console.print(
        "  [bold]/connect[/]   — Connect a database or API to your Local Vault"
    )
    console.print("  [bold]/list[/]      — List all saved connections")
    console.print("  [bold]/edit[/]      — Edit credentials for an existing connection")
    console.print("  [bold]/remove[/]    — Remove a saved connection")
    console.print("  [bold]/test[/]      — Test a saved connection")

    console.print("\n[bold]Workspace[/]")
    console.print("  [bold]/setup[/]     — Configure models and memory settings")
    console.print("  [bold]/memory[/]    — View memory status and usage")
    console.print("  [bold]/theme[/]     — Switch theme (light/dark)")

    console.print("\n[bold]Chat Tools[/]")
    console.print("  [bold]/paste[/]     — Attach an image from your clipboard")
    console.print("  [bold]/resume[/]    — Continue a previous session")
    console.print("  [bold]/publish[/]   — Publish an HTML report to the web")
    console.print("  [bold]/unpublish[/] — Remove a published report")
    console.print(
        "  [bold]/explain[/]   — Show explainability details for the latest answer"
    )

    console.print("\n[bold]General[/]")
    console.print("  [bold]/help[/]      — Show this help menu")
    console.print("  [bold]exit[/]       — Exit the chat")

    console.print()


def handle_explain(console: Console, workspace_path) -> None:
    """Print explainability details for the latest answer in the workspace."""
    store = ExplainabilityStore(workspace_path)
    record = store.load_latest()
    if record is None:
        console.print(
            "[anton.warning]No explainability record found yet for this workspace.[/]"
        )
        console.print()
        return

    console.print()
    console.print("[anton.cyan]Explain This Answer[/]")
    console.print(f"[anton.muted]Turn {record.turn} • {record.created_at}[/]")
    console.print()

    console.print("[bold]Summary[/]")
    console.print(record.summary or "No summary available.")
    console.print()

    console.print("[bold]Data Sources Used[/]")
    if record.data_sources:
        for source in record.data_sources:
            engine = source.get("engine")
            if engine:
                console.print(f"  - {source.get('name', 'Unknown')} ({engine})")
            else:
                console.print(f"  - {source.get('name', 'Unknown')}")
    else:
        console.print("  - None captured")
    console.print()

    console.print("[bold]Generated SQL[/]")
    if record.sql_queries:
        for i, query in enumerate(record.sql_queries, 1):
            header = f"  Query {i}: {query.get('datasource', 'Unknown datasource')}"
            if query.get("engine"):
                header += f" ({query['engine']})"
            console.print(header)
            console.print("```sql")
            console.print(query.get("sql", ""))
            console.print("```")
            if query.get("status") == "error" and query.get("error_message"):
                console.print(f"[anton.warning]{query['error_message']}[/]")
            console.print()
    else:
        console.print("  - No SQL generated")
        console.print()
