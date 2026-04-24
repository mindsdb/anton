"""Slash-command handlers for /theme, /explain, and /help."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rich.console import Console
from prompt_toolkit.completion import Completer, Completion

from anton.explainability import ExplainabilityStore


@dataclass
class Command:
    command: str
    description: str = None


COMMANDS = [
    "LLM Provider",
    Command("/llm",    "Change LLM provider or API key"),
    Command("/minds",  "Connect to Minds server"),
    None,
    "Data Connections",
    Command("/connect",   "Connect a database or API to your Local Vault"),
    Command("/list",      "List all saved connections"),
    Command("/edit",      "Edit credentials for an existing connection"),
    Command("/remove",    "Remove a saved connection"),
    Command("/test",      "Test a saved connection"),
    None,
    "Workspace",
    Command("/setup",   "Configure models and memory settings"),
    Command("/memory",  "View memory status and usage"),
    Command("/theme",   "Switch theme (light/dark)"),
    None,
    "Skills",
    Command("/skill", "Manage skills"),
    None,
    "Chat Tools",
    Command("/paste",     "Attach an image from your clipboard"),
    Command("/resume",    "Continue a previous session"),
    Command("/publish",   "Publish an HTML report to the web"),
    Command("/unpublish", "Remove a published report"),
    Command("/explain",   "Show explainability details for the latest answer"),
    None,
    "General",
    Command("/help", "Show this help menu"),
    Command("exit",  "Exit the chat"),
]

THEME_COMMANDS = [
    Command("/theme light"),
    Command("/theme dark")
]

SKILLS_COMMANDS = [
    Command("/skill save <name>"),
    Command("/skill list"),
    Command("/skill show <label>"),
    Command("/skill remove <label>")
]


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
        cmds = [cmd.command for cmd in THEME_COMMANDS]
        console.print(
            f"[anton.warning]Unknown theme '{arg}'. Use: " + " | ".join(cmds) + "[/]"
        )
        console.print()
        return

    os.environ["ANTON_THEME"] = new_mode
    console._theme_stack.push_theme(build_rich_theme(new_mode))
    console.print(f"[anton.success]Theme set to {new_mode}.[/]")
    console.print()


def print_slash_help(console: Console) -> None:
    """Print available slash commands."""
    width = max(len(cmd.command) for cmd in COMMANDS if isinstance(cmd, Command))
    console.print()
    console.print("[anton.cyan]Available commands:[/]")
    console.print()
    for item in COMMANDS:
        if item is None:
            console.print()
        elif isinstance(item, str):
            console.print(f"  [bold dim]{item}[/]")
        else:
            console.print(f"  [bold]{item.command}[/]{' ' * (width - len(item.command) + 2)} — {item.description}")
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


def make_completer(command_sources: list[list]) -> Completer:
    """Return a Completer that suggests slash commands extracted from *sources*."""
    commands = set()
    for source in command_sources:
        for item in source:
            if not isinstance(item, Command):
                continue
            match = re.search(r'[\[<]([^>\]]+)[>\]]', item.command)
            if match and "|" in match.group(1):
                # split options [a|b|c] to separated commands
                prefix = item.command[:match.start()]
                for variant in match.group(1).split("|"):
                    commands.add(prefix + variant)
            else:
                commands.add(item.command)

    class _Completer(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/"):
                return

            word_start = text.rfind(" ") + 1
            current_word = text[word_start:]
            seen = set()
            for cmd in commands:
                if not cmd.startswith(text):
                    continue
                next_word = cmd[word_start:].split(" ")[0]
                if next_word == current_word:
                    continue
                if next_word and next_word[0] not in ("<", "[") and next_word not in seen:
                    seen.add(next_word)
                    yield Completion(next_word, start_position=-len(current_word))

    return _Completer()

