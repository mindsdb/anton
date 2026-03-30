from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from anton import __version__
from anton.chat import _prompt_or_cancel


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

            console.print(
                f"[anton.muted]  Running: uv pip install {' '.join(missing)}[/]"
            )
            result = subprocess.run(
                [uv, "pip", "install", "--python", sys.executable, *missing],
                capture_output=True,
            )
            if result.returncode == 0:
                console.print("[anton.success]  Dependencies installed.[/]")
                _reexec()
            else:
                console.print(f"[anton.error]  Install failed:[/]")
                console.print(
                    result.stderr.decode() if result.stderr else result.stdout.decode()
                )
                if install_script.is_file():
                    if sys.platform == "win32":
                        console.print(
                            f"\n[anton.muted]  Or run the install script: powershell -File {install_script}[/]"
                        )
                    else:
                        console.print(
                            f"\n[anton.muted]  Or run the install script: sh {install_script}[/]"
                        )
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
            console.print(
                "[anton.muted]Or reinstall anton: irm https://raw.githubusercontent.com/mindsdb/anton/main/install.ps1 | iex[/]"
            )
        else:
            console.print(
                '[anton.muted]Or reinstall anton: curl -sSf https://raw.githubusercontent.com/mindsdb/anton/main/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"[/]'
            )
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
        from anton.chat import run_chat

        _ensure_workspace(settings)
        if not _has_api_key(settings):
            _onboard(settings)
        else:
            from anton.channel.branding import render_banner
            render_banner(console)
        run_chat(console, settings, resume=resume)


def _has_api_key(settings) -> bool:
    """Check if any LLM provider is fully configured."""
    providers = {settings.planning_provider, settings.coding_provider}
    for p in providers:
        if p == "anthropic" and not (
            settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        ):
            return False
        if p in ("openai", "openai-compatible") and not (
            settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
        ):
            return False
    return True


def _onboard(settings) -> None:
    """First-time onboarding: animated robot talking the intro + LLM provider selection."""
    import sys
    import time

    from rich.prompt import Prompt

    from anton import __version__
    from anton.workspace import Workspace

    ws = Workspace(Path.home())
    g = "anton.glow"

    _INTRO_LINES = [
        "Hi Boss! I'm Anton, your AI coworker.",
        "",
        "For the best experience, I recommend Minds-Enterprise-Cloud as your LLM Provider:",
        "",
        "  \u2713 Smart model routing",
        "  \u2713 Faster responses",
        "  \u2713 Cost optimized",
        "  \u2713 Secure data connectors",
    ]

    if sys.stdout.isatty():
        import asyncio
        asyncio.run(_animate_onboard(console, __version__, _INTRO_LINES, settings=settings, ws=ws))
    else:
        # Static fallback for non-interactive terminals
        from anton.channel.branding import render_banner

        render_banner(console, animate=False)
        console.print()
        for line in _INTRO_LINES:
            console.print(line)


def _ensure_api_key(settings) -> None:
    if not _has_api_key(settings):
        _onboard(settings)

    
async def _animate_onboard(console, version: str, intro_lines: list[str], *, settings, ws) -> None:
    """Animate the robot talking while typing out the intro text below."""
    import time

    from rich.live import Live
    from rich.text import Text

    from anton.channel.branding import (
        _MOUTH_SMILE,
        _MOUTH_TALK,
        _build_robot_text,
        pick_tagline,
    )

    tagline = pick_tagline()
    char_delay = 0.02
    line_pause = 0.15
    char_count = 0  # drives mouth animation

    def _build_frame(mouth: str, typed_lines: list[str]) -> Text:
        """Build robot + separator + typed text as a single renderable."""
        frame = _build_robot_text(mouth, "\u2661\u2661\u2661\u2661")
        frame.append(f" {'━' * 40}\n", style="bold cyan")
        frame.append(f" v{version} \u2014 \"{tagline}\"\n", style="dim")
        frame.append("\n")
        frame.append("anton> ", style="bold cyan")
        for line in typed_lines:
            frame.append(line)
        return frame

    with Live(
        _build_frame(_MOUTH_SMILE, []),
        console=console,
        refresh_per_second=30,
        transient=True,
    ) as live:
        time.sleep(0.4)

        typed_so_far: list[str] = []

        for line_idx, line in enumerate(intro_lines):
            if line == "":
                typed_so_far.append("\n")
                live.update(_build_frame(_MOUTH_SMILE, typed_so_far))
                time.sleep(line_pause)
                continue

            # Type out each character
            current = ""
            for ch in line:
                current += ch
                char_count += 1
                mouth = _MOUTH_TALK[char_count % 2]
                live.update(_build_frame(mouth, typed_so_far + [current]))
                time.sleep(char_delay)

            typed_so_far.append(current + "\n")
            live.update(_build_frame(_MOUTH_SMILE, typed_so_far))
            time.sleep(line_pause)

        # Hold final frame briefly
        time.sleep(0.3)

    # Print the static final state
    from anton.channel.branding import _render_robot_static

    _render_robot_static(console, "\u2661\u2661\u2661\u2661")
    console.print(f"[anton.glow] {'━' * 40}[/]")
    console.print(f" v{version} \u2014 [anton.muted]\"{tagline}\"[/]")
    console.print()
    console.print("[anton.cyan]anton>[/] ", end="")
    first_text = True
    for line in intro_lines:
        if line == "":
            if not first_text:
                console.print()
        elif line.startswith("  \u2713"):
            first_text = False
            console.print(f"  [anton.success]\u2713[/] {line[4:]}")
        else:
            first_text = False
            console.print(line)

    console.print()
    console.print(f"[anton.glow] {'━' * 40}[/]")
    console.print()
    console.print("  [bold]1[/]  [link=https://mdb.ai][anton.cyan]Minds-Enterprise-Cloud[/][/link] [anton.success](recommended)[/]")
    console.print("  [bold]2[/]  [anton.cyan]Bring your own key[/] [anton.muted]Anthropic / OpenAI[/]")
    console.print()

    while True:
        # choice = Prompt.ask(
        #     "Choose LLM Provider",
        #     choices=["1", "2", "3"],
        #     default="1",
        #     console=console,
        # )
        choice = await _prompt_or_cancel("(anton) Choose LLM Provider",
                                   choices=["1", "2", "3", "q"],
                                   default="1")

        try:
            if choice == "1":
                _setup_minds(settings, ws)
            elif choice == "2":
                _setup_other_provider(settings, ws)
            elif choice == "3":
                _setup_minds(settings, ws, default_url=None)
            break  # success
        except _SetupRetry:
            console.print()
            console.print("  [bold]1[/]  [link=https://mdb.ai][anton.cyan]Minds-Enterprise-Cloud[/][/link] [anton.success](recommended)[/]")
            console.print("  [bold]2[/]  [anton.cyan]Bring your own key[/] [anton.muted]Anthropic / OpenAI[/]")
            console.print()
            continue

    # Reload env vars so the scratchpad subprocess inherits them
    ws.apply_env_to_process()

    # Summary
    console.print()
    console.print(f"[anton.glow] {'━' * 40}[/]")
    console.print()
    provider_label = settings.planning_provider
    model_label = settings.planning_model
    if provider_label == "openai-compatible":
        if settings.minds_url and "mdb.ai" in settings.minds_url:
            provider_label = "Minds-Enterprise-Cloud"
        else:
            provider_label = "Minds-Enterprise Server"
        model_label = "smart_router"
    console.print(f"  [anton.muted]Provider:[/] [anton.cyan]{provider_label}[/]")
    console.print(f"  [anton.muted]Model:[/]    [anton.cyan]{model_label}[/]")
    console.print()


class _SetupRetry(Exception):
    """Raised by setup functions to go back to provider selection."""
    pass


def _setup_prompt(label: str, default: str | None = None, is_password: bool = False) -> str:
    """Prompt for input with ESC-to-go-back and a bottom toolbar hint.

    Returns the user's input string.
    Raises _SetupRetry if the user presses ESC.
    Works both from sync context (onboarding) and async context (/setup).
    """
    import asyncio

    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style as PTStyle

    _esc_pressed = False

    bindings = KeyBindings()

    @bindings.add("escape")
    def _on_esc(event):
        nonlocal _esc_pressed
        _esc_pressed = True
        event.app.exit(result="")

    pt_style = PTStyle.from_dict({
        "bottom-toolbar": "noreverse nounderline bg:default",
    })

    def _toolbar():
        return HTML("<style fg='#ff69b4'>\u23f5\u23f5 Esc to go back</style>")

    suffix = f" ({default}): " if default else ": "
    session: PromptSession[str] = PromptSession(
        mouse_support=False,
        bottom_toolbar=_toolbar,
        style=pt_style,
        key_bindings=bindings,
        is_password=is_password,
    )

    # Use async prompt if inside a running event loop, sync otherwise
    try:
        asyncio.get_running_loop()
        in_async = True
    except RuntimeError:
        in_async = False

    if in_async:
        # We're inside an async context (e.g. /setup from chat loop)
        # Run prompt_toolkit in a thread to avoid nested event loop conflict
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(session.prompt, f"  {label}{suffix}")
            result = future.result()
    else:
        result = session.prompt(f"  {label}{suffix}")

    if _esc_pressed:
        console.print("  [anton.muted]Going back...[/]")
        raise _SetupRetry()

    if not result and default:
        return default
    return result


def _setup_minds(settings, ws, *, default_url: str | None = "https://mdb.ai") -> None:
    """Set up Minds as the LLM provider (cloud or enterprise)."""
    from rich.prompt import Confirm, Prompt

    import webbrowser

    console.print()

    is_cloud = default_url == "https://mdb.ai"

    if is_cloud:
        minds_url = "https://mdb.ai"
    else:
        minds_url = _setup_prompt("Server URL", default=default_url).strip()
        if not minds_url.startswith("http://") and not minds_url.startswith("https://"):
            minds_url = "https://" + minds_url
        minds_url = minds_url.rstrip("/")

    if is_cloud:
        console.print("  [anton.muted]If you don't have an API key yet, we'll help you create one — it takes a few seconds.[/]")
        console.print()
        has_key = Confirm.ask(
            "  Do you have an mdb.ai API key?",
            default=True,
            console=console,
        )
        if not has_key:
            webbrowser.open(f"{minds_url}/apiKeys")
            console.print()

    while True:
        api_key = _setup_prompt("API key", is_password=True)
        if api_key.strip():
            break
        console.print("  [anton.warning]Please enter your API key.[/]")
    api_key = api_key.strip()

    # Store Minds credentials
    settings.minds_api_key = api_key
    settings.minds_url = minds_url
    ws.set_secret("ANTON_MINDS_API_KEY", api_key)
    ws.set_secret("ANTON_MINDS_URL", minds_url)

    # Test connection with a spinner
    from anton.chat import _minds_test_llm

    from rich.live import Live
    from rich.spinner import Spinner

    ssl_verify = True
    llm_ok = False
    rate_limited = False

    with Live(Spinner("dots", text="  Connecting...", style="anton.cyan"), console=console, transient=True):
        result = _minds_test_llm(minds_url, api_key, verify=True)
        if result == "rate_limited":
            rate_limited = True
        elif not result:
            result_no_ssl = _minds_test_llm(minds_url, api_key, verify=False)
            if result_no_ssl == "rate_limited":
                rate_limited = True
            elif result_no_ssl:
                ssl_verify = False
                llm_ok = True
        else:
            llm_ok = True

    if llm_ok and not ssl_verify:
        console.print("  [anton.warning]SSL certificate verification failed.[/]")
        skip_ssl = Confirm.ask(
            "  Continue without SSL verification?",
            default=False,
            console=console,
        )
        if not skip_ssl:
            llm_ok = False

    if llm_ok:
        console.print("  [anton.success]Connected[/]")
        base_url = f"{minds_url}/api/v1"
        settings.openai_api_key = api_key
        settings.openai_base_url = base_url
        settings.planning_provider = "openai-compatible"
        settings.coding_provider = "openai-compatible"
        settings.planning_model = "_reason_"
        settings.coding_model = "_code_"
        settings.minds_ssl_verify = ssl_verify
        ws.set_secret("ANTON_OPENAI_API_KEY", api_key)
        ws.set_secret("ANTON_OPENAI_BASE_URL", base_url)
        ws.set_secret("ANTON_PLANNING_PROVIDER", "openai-compatible")
        ws.set_secret("ANTON_CODING_PROVIDER", "openai-compatible")
        ws.set_secret("ANTON_PLANNING_MODEL", "_reason_")
        ws.set_secret("ANTON_CODING_MODEL", "_code_")
        if not ssl_verify:
            ws.set_secret("ANTON_MINDS_SSL_VERIFY", "false")
    elif rate_limited:
        console.print("[anton.error]Token limit exceeded. Visit https://mdb.ai to upgrade or to top up your tokens.[/]")
        raise _SetupRetry()
    else:
        console.print("  [anton.error]Could not connect. Check your API key and URL.[/]")
        retry = Confirm.ask("  Try again?", default=True, console=console)
        if retry:
            _setup_minds(settings, ws, default_url=default_url)
        else:
            raise _SetupRetry()


def _setup_other_provider(settings, ws) -> None:
    """Set up Anthropic or OpenAI as the LLM provider."""
    from rich.text import Text

    console.print()
    for label, idx in [("Anthropic", "1"), ("OpenAI", "2")]:
        line = Text()
        line.append(f"  {idx} ", style="bold")
        line.append(label, style="anton.cyan")
        console.print(line)
    console.print()

    choice = _setup_prompt("Provider", default="1").strip().lower()

    if choice in ("1", "anthropic"):
        _setup_anthropic(settings, ws)
    elif choice in ("2", "openai"):
        _setup_openai(settings, ws)
    else:
        console.print(f"  [anton.warning]Unknown provider '{choice}', using Anthropic.[/]")
        _setup_anthropic(settings, ws)


def _validate_with_spinner(console, label: str, fn) -> None:
    """Run a validation function with a spinner, print result."""
    from rich.live import Live
    from rich.spinner import Spinner

    with Live(Spinner("dots", text=f"  Validating {label}...", style="anton.cyan"), console=console, transient=True):
        fn()
    console.print(f"  [anton.success]Validated[/] [anton.muted]{label}[/]")


def _setup_anthropic(settings, ws) -> None:
    """Set up Anthropic with a single model for both reasoning and coding."""
    from rich.prompt import Confirm

    console.print()
    while True:
        api_key = _setup_prompt("API key", is_password=True)
        if api_key.strip():
            break
        console.print("  [anton.warning]Please enter your API key.[/]")
    api_key = api_key.strip()

    model = _setup_prompt("Model", default="claude-sonnet-4-6").strip()

    try:
        def _test():
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            client.messages.create(model=model, max_tokens=1, messages=[{"role": "user", "content": "ping"}])

        _validate_with_spinner(console, model, _test)
    except Exception as exc:
        console.print(f"  [anton.error]Failed:[/] {exc}")
        retry = Confirm.ask("  Try again?", default=True, console=console)
        if retry:
            _setup_anthropic(settings, ws)
            return
        else:
            raise _SetupRetry()

    settings.anthropic_api_key = api_key
    settings.planning_provider = "anthropic"
    settings.coding_provider = "anthropic"
    settings.planning_model = model
    settings.coding_model = model
    ws.set_secret("ANTON_ANTHROPIC_API_KEY", api_key)
    ws.set_secret("ANTON_PLANNING_PROVIDER", "anthropic")
    ws.set_secret("ANTON_CODING_PROVIDER", "anthropic")
    ws.set_secret("ANTON_PLANNING_MODEL", model)
    ws.set_secret("ANTON_CODING_MODEL", model)


def _setup_openai(settings, ws) -> None:
    """Set up OpenAI with a single model for both reasoning and coding."""
    from rich.prompt import Confirm

    console.print()
    while True:
        api_key = _setup_prompt("API key", is_password=True)
        if api_key.strip():
            break
        console.print("  [anton.warning]Please enter your API key.[/]")
    api_key = api_key.strip()

    model = _setup_prompt("Model", default="gpt-4o").strip()

    try:
        def _test():
            import openai
            client = openai.OpenAI(api_key=api_key)
            client.chat.completions.create(model=model, max_tokens=1, messages=[{"role": "user", "content": "ping"}])

        _validate_with_spinner(console, model, _test)
    except Exception as exc:
        console.print(f"  [anton.error]Failed:[/] {exc}")
        retry = Confirm.ask("  Try again?", default=True, console=console)
        if retry:
            _setup_openai(settings, ws)
            return
        else:
            raise _SetupRetry()

    settings.openai_api_key = api_key
    settings.planning_provider = "openai"
    settings.coding_provider = "openai"
    settings.planning_model = model
    settings.coding_model = model
    ws.set_secret("ANTON_OPENAI_API_KEY", api_key)
    ws.set_secret("ANTON_PLANNING_PROVIDER", "openai")
    ws.set_secret("ANTON_CODING_PROVIDER", "openai")
    ws.set_secret("ANTON_PLANNING_MODEL", model)
    ws.set_secret("ANTON_CODING_MODEL", model)


@app.command("setup")
def setup(ctx: typer.Context) -> None:
    """Configure provider, model, and API key."""
    settings = _get_settings(ctx)
    _ensure_workspace(settings)
    _onboard(settings)
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


@app.command("connect")
def connect_data_source(
    ctx: typer.Context,
    slug: str = typer.Argument(
        default="", help="Existing connection slug to reconnect (e.g. postgres-mydb)."
    ),
) -> None:
    """Connect a database or API to the Local Vault.

    Pass an existing connection slug (e.g. postgres-mydb) to reconnect using
    stored credentials without re-entering them.  Use /edit to
    update credentials for an existing connection.
    """
    import asyncio

    from anton.chat import ChatSession, _handle_connect_datasource
    from anton.llm.client import LLMClient
    from anton.scratchpad import ScratchpadManager

    settings = _get_settings(ctx)
    _ensure_workspace(settings)
    _ensure_api_key(settings)

    llm_client = LLMClient.from_settings(settings)
    scratchpads = ScratchpadManager(
        coding_provider=settings.coding_provider,
        coding_model=settings.coding_model,
        coding_api_key=(
            settings.anthropic_api_key
            if settings.coding_provider == "anthropic"
            else settings.openai_api_key
        )
        or "",
    )
    session = ChatSession(llm_client)

    async def _run() -> None:
        await _handle_connect_datasource(
            console,
            scratchpads,
            session,
            datasource_name=slug or None,
        )
        await scratchpads.close_all()

    asyncio.run(_run())


@app.command("list")
def list_data_sources(ctx: typer.Context) -> None:
    """List all saved data source connections in the Local Vault."""
    from anton.chat import _handle_list_data_sources

    _handle_list_data_sources(console)


@app.command("edit")
def edit_data_source(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Connection slug to edit (e.g. postgres-mydb)."),
) -> None:
    """Edit credentials for an existing Local Vault connection."""
    import asyncio

    from anton.chat import ChatSession, _handle_connect_datasource
    from anton.llm.client import LLMClient
    from anton.scratchpad import ScratchpadManager

    settings = _get_settings(ctx)
    _ensure_workspace(settings)
    _ensure_api_key(settings)

    llm_client = LLMClient.from_settings(settings)
    scratchpads = ScratchpadManager(
        coding_provider=settings.coding_provider,
        coding_model=settings.coding_model,
        coding_api_key=(
            settings.anthropic_api_key
            if settings.coding_provider == "anthropic"
            else settings.openai_api_key
        )
        or "",
    )
    session = ChatSession(llm_client)

    async def _run() -> None:
        await _handle_connect_datasource(
            console,
            scratchpads,
            session,
            datasource_name=name,
        )
        await scratchpads.close_all()

    asyncio.run(_run())


@app.command("remove")
def remove_data_source(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Connection slug to remove (e.g. postgres-mydb)."),
) -> None:
    """Remove a saved connection from the Local Vault."""
    from anton.chat import _handle_remove_data_source

    _handle_remove_data_source(console, name)


@app.command("test")
def test_data_source(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Connection slug to test (e.g. postgres-mydb)."),
) -> None:
    """Test a saved Local Vault connection using its test snippet."""
    import asyncio

    from anton.chat import _handle_test_datasource
    from anton.scratchpad import ScratchpadManager

    settings = _get_settings(ctx)
    _ensure_workspace(settings)
    _ensure_api_key(settings)

    scratchpads = ScratchpadManager(
        coding_provider=settings.coding_provider,
        coding_model=settings.coding_model,
        coding_api_key=(
            settings.anthropic_api_key
            if settings.coding_provider == "anthropic"
            else settings.openai_api_key
        )
        or "",
    )

    async def _run() -> None:
        await _handle_test_datasource(console, scratchpads, name)
        await scratchpads.close_all()

    asyncio.run(_run())
