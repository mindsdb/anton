from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from anton import __version__

app = typer.Typer(
    name="anton",
    help="Anton — autonomous coding copilot",
)


def _make_console() -> Console:
    from anton.channel.theme import build_rich_theme, detect_color_mode

    mode = detect_color_mode()
    return Console(theme=build_rich_theme(mode))


console = _make_console()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Anton — autonomous coding copilot."""
    if ctx.invoked_subcommand is None:
        from anton.channel.branding import render_dashboard

        render_dashboard(console)


def _has_api_key(settings) -> bool:
    """Check if any API key is available."""
    if settings.anthropic_api_key:
        return True
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    return False


def _ensure_api_key(settings) -> None:
    """Prompt the user to configure a provider and API key if none is set."""
    if _has_api_key(settings):
        return

    console.print()
    console.print("[anton.warning]No API key configured.[/]")
    console.print()

    providers = {"1": "anthropic"}
    console.print("[anton.cyan]Available providers:[/]")
    console.print("  [bold]1[/]  Anthropic (Claude)")
    console.print()

    choice = Prompt.ask(
        "Select provider",
        choices=list(providers.keys()),
        default="1",
        console=console,
    )
    provider = providers[choice]

    console.print()
    api_key = Prompt.ask(
        f"Enter your {provider.title()} API key",
        console=console,
    )

    if not api_key.strip():
        console.print("[anton.error]No API key provided. Exiting.[/]")
        raise typer.Exit(1)

    api_key = api_key.strip()

    # Save to ~/.anton/.env for persistence
    env_dir = Path("~/.anton").expanduser()
    env_dir.mkdir(parents=True, exist_ok=True)
    env_file = env_dir / ".env"

    lines: list[str] = []
    key_name = f"ANTON_{provider.upper()}_API_KEY"
    replaced = False
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith(f"{key_name}="):
                lines.append(f"{key_name}={api_key}")
                replaced = True
            else:
                lines.append(line)
    if not replaced:
        lines.append(f"{key_name}={api_key}")

    env_file.write_text("\n".join(lines) + "\n")

    # Apply to current process so this run works
    os.environ[key_name] = api_key
    if provider == "anthropic":
        settings.anthropic_api_key = api_key

    console.print()
    console.print(f"[anton.success]Saved to {env_file}[/]")
    console.print()


@app.command()
def run(
    task: str = typer.Argument(..., help="The task for Anton to complete"),
) -> None:
    """Give Anton a task and let it work autonomously."""
    from anton.channel.branding import render_banner

    render_banner(console)
    asyncio.run(_run_task(task))


async def _run_task(task: str) -> None:
    from anton.channel.terminal import CLIChannel
    from anton.config.settings import AntonSettings
    from anton.core.agent import Agent
    from anton.llm.client import LLMClient
    from anton.skill.registry import SkillRegistry

    settings = AntonSettings()
    _ensure_api_key(settings)
    channel = CLIChannel()

    try:
        llm_client = LLMClient.from_settings(settings)
        registry = SkillRegistry()

        # Discover built-in skills
        builtin = Path(__file__).resolve().parent.parent / settings.skills_dir
        registry.discover(builtin)

        # Discover user skills
        user_dir = Path(settings.user_skills_dir).expanduser()
        registry.discover(user_dir)

        # Set up memory if enabled
        memory = None
        learnings_store = None
        if settings.memory_enabled:
            from anton.memory.learnings import LearningStore
            from anton.memory.store import SessionStore

            memory_dir = Path(settings.memory_dir).expanduser()
            memory = SessionStore(memory_dir)
            learnings_store = LearningStore(memory_dir)

        agent = Agent(
            channel=channel,
            llm_client=llm_client,
            registry=registry,
            user_skills_dir=user_dir,
            memory=memory,
            learnings=learnings_store,
        )
        await agent.run(task)
    finally:
        await channel.close()


@app.command("skills")
def list_skills() -> None:
    """List all discovered skills."""
    from anton.config.settings import AntonSettings
    from anton.skill.registry import SkillRegistry

    settings = AntonSettings()
    registry = SkillRegistry()

    builtin = Path(__file__).resolve().parent.parent / settings.skills_dir
    registry.discover(builtin)

    user_dir = Path(settings.user_skills_dir).expanduser()
    registry.discover(user_dir)

    skills = registry.list_all()
    if not skills:
        console.print("[dim]No skills discovered.[/]")
        return

    table = Table(title="Discovered Skills")
    table.add_column("Name", style="anton.cyan")
    table.add_column("Description")
    table.add_column("Parameters")

    for s in skills:
        params = ", ".join(
            f"{k}: {v.get('type', '?')}"
            for k, v in s.parameters.get("properties", {}).items()
        )
        table.add_row(s.name, s.description, params or "—")

    console.print(table)


@app.command("sessions")
def list_sessions() -> None:
    """List recent sessions."""
    from anton.config.settings import AntonSettings
    from anton.memory.store import SessionStore

    settings = AntonSettings()
    memory_dir = Path(settings.memory_dir).expanduser()
    store = SessionStore(memory_dir)

    sessions = store.list_sessions()
    if not sessions:
        console.print("[dim]No sessions found.[/]")
        return

    table = Table(title="Recent Sessions")
    table.add_column("ID", style="anton.cyan")
    table.add_column("Task")
    table.add_column("Status")
    table.add_column("Summary")

    for s in sessions:
        preview = s.get("summary_preview") or ""
        if len(preview) > 60:
            preview = preview[:60] + "..."
        table.add_row(s["id"], s.get("task", "")[:50], s.get("status", ""), preview)

    console.print(table)


@app.command("session")
def show_session(
    session_id: str = typer.Argument(..., help="Session ID to display"),
) -> None:
    """Show session details and summary."""
    from anton.config.settings import AntonSettings
    from anton.memory.store import SessionStore

    settings = AntonSettings()
    memory_dir = Path(settings.memory_dir).expanduser()
    store = SessionStore(memory_dir)

    session = store.get_session(session_id)
    if session is None:
        console.print(f"[red]Session {session_id} not found.[/]")
        raise typer.Exit(1)

    console.print(f"[bold]Session:[/] {session['id']}")
    console.print(f"[bold]Task:[/] {session.get('task', 'N/A')}")
    console.print(f"[bold]Status:[/] {session.get('status', 'N/A')}")

    summary = session.get("summary")
    if summary:
        console.print(f"\n[bold]Summary:[/]\n{summary}")


@app.command("learnings")
def list_learnings() -> None:
    """List all learnings with summaries."""
    from anton.config.settings import AntonSettings
    from anton.memory.learnings import LearningStore

    settings = AntonSettings()
    memory_dir = Path(settings.memory_dir).expanduser()
    store = LearningStore(memory_dir)

    items = store.list_all()
    if not items:
        console.print("[dim]No learnings recorded yet.[/]")
        return

    table = Table(title="Learnings")
    table.add_column("Topic", style="anton.cyan")
    table.add_column("Summary")

    for item in items:
        table.add_row(item["topic"], item["summary"])

    console.print(table)


@app.command("channels")
def list_channels() -> None:
    """List available communication channels."""
    from anton.channel.registry import ChannelRegistry
    from anton.channel.terminal import CLIChannel, TerminalChannel
    from anton.channel.types import ChannelCapability, ChannelInfo, ChannelMeta

    registry = ChannelRegistry()
    registry.register(
        ChannelInfo(
            meta=ChannelMeta(
                id="cli",
                label="Terminal CLI",
                description="Rich interactive terminal",
                icon="\U0001f4bb",
                capabilities=[
                    ChannelCapability.TEXT_OUTPUT,
                    ChannelCapability.TEXT_INPUT,
                    ChannelCapability.INTERACTIVE,
                    ChannelCapability.RICH_FORMATTING,
                ],
                aliases=["terminal", "term"],
            ),
            factory=CLIChannel,
        )
    )

    channels = registry.list_all()
    if not channels:
        console.print("[dim]No channels registered.[/]")
        return

    table = Table(title="Channels")
    table.add_column("ID", style="anton.cyan")
    table.add_column("Label")
    table.add_column("Description")
    table.add_column("Capabilities")

    for ch in channels:
        caps = ", ".join(c.value for c in ch.meta.capabilities)
        table.add_row(
            f"{ch.meta.icon}  {ch.meta.id}" if ch.meta.icon else ch.meta.id,
            ch.meta.label,
            ch.meta.description,
            caps,
        )

    console.print(table)


@app.command("version")
def version() -> None:
    """Show Anton version."""
    console.print(f"Anton v{__version__}")
