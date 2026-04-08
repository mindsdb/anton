"""Slash-command handlers for /theme and /help."""

from __future__ import annotations

from rich.console import Console


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
        console.print(f"[anton.warning]Unknown theme '{arg}'. Use: /theme light | /theme dark[/]")
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
    console.print("  [bold]/connect[/]   — Connect a database or API to your Local Vault")
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

    console.print("\n[bold]General[/]")
    console.print("  [bold]/help[/]      — Show this help menu")
    console.print("  [bold]exit[/]       — Exit the chat")

    console.print()
