from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from anton import __version__


def _reexec() -> None:
    """Re-execute the current process from scratch using the original binary."""
    import shutil

    # Prefer the installed `anton` binary so the uv tool wrapper re-runs correctly.
    binary = shutil.which("anton") or sys.argv[0]
    os.execv(binary, [binary] + sys.argv[1:])


# ---------------------------------------------------------------------------
# Dependency checking — runs before anything that needs the heavy imports
# ---------------------------------------------------------------------------

# Core dependencies from pyproject.toml that anton needs at runtime
_REQUIRED_PACKAGES: dict[str, str] = {
    "anthropic": "anthropic>=0.42.0",
    "ollama": "ollama>=0.6.1",
    "openai": "openai>=1.0",
    "pydantic": "pydantic>=2.0",
    "pydantic_settings": "pydantic-settings>=2.0",
    "prompt_toolkit": "prompt-toolkit>=3.0",
}
# typer and rich are already imported above — if they were missing we'd
# never reach this point, so no need to check them.


def _check_dependencies() -> list[str]:
    """Return list of missing package install specs."""
    import importlib

    missing: list[str] = []
    for module_name, install_spec in _REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(install_spec)
    return missing


def _find_uv() -> str | None:
    """Find the uv binary."""
    import shutil

    uv = shutil.which("uv")
    if uv:
        return uv

    if sys.platform == "win32":
        candidates = (
            os.path.expanduser("~/.local/bin/uv.exe"),
            os.path.expanduser("~/.cargo/bin/uv.exe"),
        )
    else:
        candidates = (
            os.path.expanduser("~/.local/bin/uv"),
            os.path.expanduser("~/.cargo/bin/uv"),
        )

    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _ensure_dependencies(console: Console) -> None:
    """Check for missing dependencies and offer to install them."""
    missing = _check_dependencies()
    if not missing:
        return

    console.print()
    console.print("[anton.warning]Missing dependencies detected:[/]")
    for pkg in missing:
        console.print(f"  [bold]- {pkg}[/]")
    console.print()

    # Check if install script is available locally (dev checkout)
    repo_root = Path(__file__).resolve().parent.parent
    if sys.platform == "win32":
        install_script = repo_root / "install.ps1"
    else:
        install_script = repo_root / "install.sh"
    uv = _find_uv()

    if uv:
        if Confirm.ask(
            f"Install missing packages with uv?",
            default=True,
            console=console,
        ):
            import subprocess

            console.print(f"[anton.muted]  Running: uv pip install {' '.join(missing)}[/]")
            result = subprocess.run(
                [uv, "pip", "install", "--python", sys.executable, *missing],
                capture_output=True,
            )
            if result.returncode == 0:
                console.print("[anton.success]  Dependencies installed.[/]")
                _reexec()
            else:
                console.print(f"[anton.error]  Install failed:[/]")
                console.print(result.stderr.decode() if result.stderr else result.stdout.decode())
                if install_script.is_file():
                    if sys.platform == "win32":
                        console.print(f"\n[anton.muted]  Or run the install script: powershell -File {install_script}[/]")
                    else:
                        console.print(f"\n[anton.muted]  Or run the install script: sh {install_script}[/]")
            raise typer.Exit(0)
    elif install_script.is_file():
        console.print(f"To install all dependencies, run:")
        if sys.platform == "win32":
            console.print(f"  [bold]powershell -File {install_script}[/]")
        else:
            console.print(f"  [bold]sh {install_script}[/]")
        console.print()
        raise typer.Exit(1)
    else:
        console.print("To install missing dependencies, run:")
        console.print(f"  [bold]pip install {' '.join(missing)}[/]")
        console.print()
        if sys.platform == "win32":
            console.print("[anton.muted]Or reinstall anton: irm https://raw.githubusercontent.com/mindsdb/anton/main/install.ps1 | iex[/]")
        else:
            console.print('[anton.muted]Or reinstall anton: curl -sSf https://raw.githubusercontent.com/mindsdb/anton/main/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"[/]')
        console.print()
        raise typer.Exit(1)

app = typer.Typer(
    name="anton",
    help="Anton — a self-evolving autonomous system",
)


def _make_console() -> Console:
    from anton.channel.theme import build_rich_theme, detect_color_mode

    mode = detect_color_mode()
    return Console(theme=build_rich_theme(mode))


console = _make_console()


def _get_settings(ctx: typer.Context):
    """Retrieve the resolved AntonSettings from context."""
    return ctx.obj["settings"]


def _ensure_workspace(settings) -> None:
    """Check workspace state and initialize if needed.

    Boot logic:
    1. If $PWD/.anton exists → use it (local project), boot straight away
    2. If $HOME/.anton exists → use it (global project), boot straight away
    3. Neither exists → create local $PWD/.anton and boot
    """
    from anton.workspace import Workspace

    local_path = settings.workspace_path
    global_path = Path.home()

    local_ws = Workspace(local_path)
    global_ws = Workspace(global_path)

    # Always ensure local .anton exists so project memory has a home
    if not local_ws.is_initialized():
        local_ws.initialize()
        console.print(f"[anton.muted]  workspace is {local_path}/.anton[/]")

    # Local env wins, then global fills in anything missing (API keys, etc.)
    local_ws.apply_env_to_process()
    if local_path != global_path and global_ws.is_initialized():
        global_ws.apply_env_to_process()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    folder: str | None = typer.Option(
        None, "--folder", "-f", help="Workspace folder (defaults to cwd)"
    ),
    resume: bool = typer.Option(
        False, "--resume", "-r", help="Resume a previous chat session"
    ),
) -> None:
    """Anton — a self-evolving autonomous system."""
    _ensure_dependencies(console)

    from anton.config.settings import AntonSettings

    settings = AntonSettings()
    settings.resolve_workspace(folder)

    from anton.updater import check_and_update
    if check_and_update(console, settings):
        # Re-exec with the freshly installed code so no old modules remain in memory.
        _reexec()

    ctx.ensure_object(dict)
    ctx.obj["settings"] = settings

    if ctx.invoked_subcommand is None:
        from anton.channel.branding import render_banner
        from anton.chat import run_chat

        render_banner(console)
        _ensure_workspace(settings)
        _ensure_api_key(settings)
        run_chat(console, settings, resume=resume)


def _has_api_key(settings) -> bool:
    """Check if the configured providers are ready to use."""
    providers = {settings.planning_provider, settings.coding_provider}
    for p in providers:
        if p == "ollama":
            continue
        if p == "anthropic" and not (settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")):
            return False
        if p in ("openai", "openai-compatible") and not (settings.openai_api_key or os.environ.get("OPENAI_API_KEY")):
            return False
    return True


def _ensure_api_key(settings) -> None:
    """Prompt the user to configure an LLM provider if needed."""
    if _has_api_key(settings):
        return

    from anton.llm.setup import configure_llm_settings
    from anton.workspace import Workspace

    ws = Workspace(Path.home())

    if settings.minds_enabled:
        _ensure_minds_api_key(settings, ws)
    else:
        applied = configure_llm_settings(
            console,
            settings,
            ws,
            show_current_config=False,
        )
        if not applied:
            raise typer.Exit(1)

    # Reload env vars into the process so the scratchpad subprocess inherits them
    ws.apply_env_to_process()

    console.print()
    console.print(f"[anton.success]Saved to {ws.env_path}[/]")
    console.print()


def _ensure_anthropic_api_key(settings, ws) -> None:
    """Prompt for Anthropic API key (default flow)."""
    from rich.prompt import Prompt

    console.print()
    console.print("[anton.cyan]Anthropic configuration[/]")
    console.print()

    api_key = Prompt.ask("Anthropic API key", console=console)
    if not api_key.strip():
        console.print("[anton.error]No API key provided. Exiting.[/]")
        raise typer.Exit(1)
    api_key = api_key.strip()

    settings.anthropic_api_key = api_key
    settings.planning_provider = "anthropic"
    settings.coding_provider = "anthropic"
    settings.planning_model = "claude-sonnet-4-6"
    settings.coding_model = "claude-haiku-4-5-20251001"
    ws.set_secret("ANTON_ANTHROPIC_API_KEY", api_key)
    ws.set_secret("ANTON_PLANNING_PROVIDER", "anthropic")
    ws.set_secret("ANTON_CODING_PROVIDER", "anthropic")
    ws.set_secret("ANTON_PLANNING_MODEL", "claude-sonnet-4-6")
    ws.set_secret("ANTON_CODING_MODEL", "claude-haiku-4-5-20251001")


def _ensure_minds_api_key(settings, ws) -> None:
    """Prompt for Minds API key and configure LLM endpoints (opt-in flow)."""
    from rich.prompt import Prompt

    console.print()
    console.print("[anton.cyan]Minds configuration[/]")
    console.print()

    api_key = Prompt.ask("Minds API key", console=console)
    if not api_key.strip():
        console.print("[anton.error]No API key provided. Exiting.[/]")
        raise typer.Exit(1)
    api_key = api_key.strip()

    minds_url = Prompt.ask(
        "Minds URL",
        default="https://mdb.ai",
        console=console,
    ).strip()
    if not minds_url.startswith("http://") and not minds_url.startswith("https://"):
        minds_url = "https://" + minds_url
    minds_url = minds_url.rstrip("/")

    # Store Minds credentials
    settings.minds_api_key = api_key
    settings.minds_url = minds_url
    ws.set_secret("ANTON_MINDS_API_KEY", api_key)
    ws.set_secret("ANTON_MINDS_URL", minds_url)

    # Test if the Minds server supports LLM endpoints (_code_/_reason_)
    # (silenced: was printing "Testing LLM endpoints..." and "not available" messages)
    from anton.chat import _minds_test_llm
    llm_ok = _minds_test_llm(minds_url, api_key, verify=True)
    if not llm_ok:
        llm_ok = _minds_test_llm(minds_url, api_key, verify=False)

    if llm_ok:
        console.print("[anton.success]LLM endpoints available — using Minds server as LLM provider.[/]")
        base_url = f"{minds_url}/api/v1"
        settings.openai_api_key = api_key
        settings.openai_base_url = base_url
        settings.planning_provider = "openai-compatible"
        settings.coding_provider = "openai-compatible"
        settings.planning_model = "_reason_"
        settings.coding_model = "_code_"
        ws.set_secret("ANTON_OPENAI_API_KEY", api_key)
        ws.set_secret("ANTON_OPENAI_BASE_URL", base_url)
        ws.set_secret("ANTON_PLANNING_PROVIDER", "openai-compatible")
        ws.set_secret("ANTON_CODING_PROVIDER", "openai-compatible")
        ws.set_secret("ANTON_PLANNING_MODEL", "_reason_")
        ws.set_secret("ANTON_CODING_MODEL", "_code_")
    else:
        # LLM endpoints not available — fall back to Anthropic
        _ensure_anthropic_api_key(settings, ws)


@app.command("setup")
def setup(ctx: typer.Context) -> None:
    """Configure provider, model, and API key."""
    settings = _get_settings(ctx)
    _ensure_workspace(settings)
    _ensure_api_key(settings)
    console.print("[anton.success]Setup complete.[/]")


@app.command("dashboard")
def dashboard() -> None:
    """Show the Anton status dashboard."""
    from anton.channel.branding import render_dashboard

    render_dashboard(console)


@app.command("sessions")
def list_sessions(ctx: typer.Context) -> None:
    """List recent sessions."""
    from anton.memory.store import SessionStore

    settings = _get_settings(ctx)
    memory_dir = Path(settings.memory_dir)
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
    ctx: typer.Context,
    session_id: str = typer.Argument(..., help="Session ID to display"),
) -> None:
    """Show session details and summary."""
    from anton.memory.store import SessionStore

    settings = _get_settings(ctx)
    memory_dir = Path(settings.memory_dir)
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
def list_learnings(ctx: typer.Context) -> None:
    """List all learnings with summaries."""
    from anton.memory.learnings import LearningStore

    settings = _get_settings(ctx)
    memory_dir = Path(settings.memory_dir)
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


@app.command("version")
def version() -> None:
    """Show Anton version."""
    console.print(f"Anton v{__version__}")
