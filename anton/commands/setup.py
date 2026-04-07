"""Slash-command handlers for /setup and /memory."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse

from rich.console import Console

from anton.config.settings import AntonSettings
from anton.prompt_utils import prompt_or_cancel

if TYPE_CHECKING:
    from anton.chat import ChatSession
    from anton.memory.episodic import EpisodicMemory
    from anton.memory.history_store import HistoryStore
    from anton.memory.cortex import Cortex
    from anton.workspace import Workspace


def handle_memory(
    console: Console,
    settings: AntonSettings,
    cortex: "Cortex | None",
    episodic: "EpisodicMemory | None" = None,
) -> None:
    """Show memory status — read-only dashboard."""
    console.print()
    console.print("[anton.cyan]Memory Status[/]")
    console.print()

    mode_labels = {
        "autopilot": "Autopilot — Anton decides what to remember",
        "copilot": "Co-pilot — save obvious, confirm ambiguous",
        "off": "Off — never save (still reads existing)",
    }
    mode_label = mode_labels.get(settings.memory_mode, settings.memory_mode)
    console.print(f"  Mode:  [bold]{mode_label}[/]")
    console.print()

    if cortex is None:
        console.print("  [anton.warning]Memory system not initialized.[/]")
        console.print()
        return

    def _show_scope(label: str, hc) -> int:
        identity = hc.recall_identity()
        rules = hc.recall_rules()
        lessons_raw = hc._read_full_lessons()
        rule_count = (
            sum(1 for ln in rules.splitlines() if ln.strip().startswith("- "))
            if rules
            else 0
        )
        lesson_count = (
            sum(1 for ln in lessons_raw.splitlines() if ln.strip().startswith("- "))
            if lessons_raw
            else 0
        )
        topics: list[str] = []
        if hc._topics_dir.is_dir():
            topics = [
                p.stem for p in sorted(hc._topics_dir.iterdir()) if p.suffix == ".md"
            ]

        console.print(f"  [anton.cyan]{label}[/] [dim]({hc._dir})[/]")
        if identity:
            entries = [
                ln.strip()[2:]
                for ln in identity.splitlines()
                if ln.strip().startswith("- ")
            ]
            if entries:
                console.print(
                    f"    Identity:  {', '.join(entries[:3])}"
                    + (" ..." if len(entries) > 3 else "")
                )
            else:
                console.print("    Identity:  [dim](set)[/]")
        else:
            console.print("    Identity:  [dim](empty)[/]")
        console.print(f"    Rules:     {rule_count}")
        console.print(f"    Lessons:   {lesson_count}")
        if topics:
            console.print(f"    Topics:    {', '.join(topics)}")
        else:
            console.print("    Topics:    [dim](none)[/]")
        console.print()
        return rule_count + lesson_count

    global_total = _show_scope("Global Memory", cortex.global_hc)
    project_total = _show_scope("Project Memory", cortex.project_hc)

    total = global_total + project_total
    console.print(f"  Total entries: [bold]{total}[/]")
    if cortex.needs_compaction():
        console.print("  [anton.warning]Compaction needed (>50 entries in a scope)[/]")
    console.print()

    if episodic is not None:
        status = "[bold]ON[/]" if episodic.enabled else "[dim]OFF[/]"
        sessions = episodic.session_count()
        console.print(f"  [anton.cyan]Episodic Memory[/]")
        console.print(f"    Status:    {status}")
        console.print(f"    Sessions:  {sessions}")
        console.print()

    console.print("[dim]  Use /setup > Memory to change configuration.[/]")
    console.print()


async def handle_setup_memory(
    console: Console,
    settings: AntonSettings,
    workspace: "Workspace",
    cortex: "Cortex | None",
    episodic: "EpisodicMemory | None" = None,
) -> None:
    """Setup sub-menu: memory mode and episodic memory toggle."""
    console.print()
    console.print("[anton.cyan]Memory configuration[/]")
    console.print()

    console.print("  Memory mode:")
    console.print(
        r"    [bold]1[/]  Autopilot — Anton decides what to remember       [dim]\[recommended][/]"
    )
    console.print(
        r"    [bold]2[/]  Co-pilot — save obvious, confirm ambiguous        [dim]\[selective][/]"
    )
    console.print(
        r"    [bold]3[/]  Off — never save memory (still reads existing)    [dim]\[suppressed][/]"
    )
    console.print()

    mode_map = {"1": "autopilot", "2": "copilot", "3": "off"}
    current_mode_num = {"autopilot": "1", "copilot": "2", "off": "3"}.get(
        settings.memory_mode, "1"
    )
    mode_choice = await prompt_or_cancel(
        "(anton) Memory mode", choices=["1", "2", "3"], default=current_mode_num
    )
    if mode_choice is None:
        console.print()
        return
    memory_mode = mode_map[mode_choice]
    settings.memory_mode = memory_mode
    workspace.set_secret("ANTON_MEMORY_MODE", memory_mode)
    if cortex is not None:
        cortex.mode = memory_mode

    if episodic is not None:
        console.print()
        ep_status = "ON" if episodic.enabled else "OFF"
        console.print(
            f"  Episodic memory (conversation archive): Currently [bold]{ep_status}[/]"
        )
        toggle = await prompt_or_cancel(
            "(anton) Toggle episodic memory?", choices=["y", "n"], default="n"
        )
        if toggle is None:
            toggle = "n"
        if toggle == "y":
            new_state = not episodic.enabled
            episodic.enabled = new_state
            settings.episodic_memory = new_state
            workspace.set_secret(
                "ANTON_EPISODIC_MEMORY", "true" if new_state else "false"
            )
            console.print(f"  Episodic memory: [bold]{'ON' if new_state else 'OFF'}[/]")

    console.print()
    console.print("[anton.success]Configuration updated.[/]")
    console.print()


async def handle_setup_models(
    console: Console,
    settings: AntonSettings,
    workspace: "Workspace",
    state: dict,
    self_awareness,
    cortex: "Cortex | None",
    session: "ChatSession",
    episodic: "EpisodicMemory | None" = None,
    history_store: "HistoryStore | None" = None,
    session_id: str | None = None,
) -> "ChatSession":
    """Setup sub-menu: provider, API key, and models."""
    from pathlib import Path
    from anton.workspace import Workspace as _Workspace
    from anton.cli import _SetupRetry, _setup_minds, _setup_other_provider
    from anton.chat_session import rebuild_session

    # Always persist API keys and model settings to global ~/.anton/.env
    global_ws = _Workspace(Path.home())

    def _provider_label(provider: str) -> str:
        if provider == "openai-compatible":
            base = settings.openai_base_url or ""
            if settings.minds_url and "mdb.ai" in settings.minds_url:
                return "Minds-Enterprise-Cloud"
            else:
                hostname = None
                if base:
                    parsed = urlparse(base)
                    hostname = parsed.hostname
                if hostname and (
                    hostname == "generativelanguage.googleapis.com"
                    or hostname.endswith(".generativelanguage.googleapis.com")
                ):
                    return "Google Gemini"
                elif base:
                    return f"OpenAI-compatible ({base})"
            return "OpenAI-compatible"
        return provider.capitalize()

    def _model_label(model: str, role: str) -> str:
        if model in ("_reason_", "_code_"):
            return f"smart_router({role})"
        return model

    provider_display = _provider_label(settings.planning_provider)
    planning_display = _model_label(settings.planning_model, "planning")
    coding_display = _model_label(settings.coding_model, "coding")

    console.print()
    console.print("[anton.cyan]Current configuration:[/]")
    console.print(f"  Provider: [bold]{provider_display}[/]")
    if planning_display == coding_display:
        console.print(f"  Model:    [bold]{planning_display}[/]")
    else:
        console.print(f"  Planning: [bold]{planning_display}[/]")
        console.print(f"  Coding:   [bold]{coding_display}[/]")
    console.print()

    def _print_choices():
        console.print("  [bold]1[/]  [link=https://mdb.ai][anton.cyan]Minds-Enterprise-Cloud[/][/link] [anton.success](recommended)[/]")
        console.print("  [bold]2[/]  [anton.cyan]Minds-Enterprise-Server[/] [anton.muted]self-hosted[/]")
        console.print("  [bold]3[/]  [anton.cyan]Bring your own key[/] [anton.muted]Anthropic / OpenAI / Gemini[/]")
        console.print("  [bold]q[/]  [anton.muted]Back[/]")
        console.print()

    _print_choices()

    while True:
        choice = await prompt_or_cancel(
            "(anton) Choose LLM Provider",
            choices=["1", "2", "3", "q"],
            default="1",
        )
        if choice is None or choice == "q":
            return session

        try:
            if choice == "1":
                _setup_minds(settings, global_ws)
            elif choice == "2":
                _setup_minds(settings, global_ws, default_url=None)
            elif choice == "3":
                _setup_other_provider(settings, global_ws)
            break
        except _SetupRetry:
            console.print()
            _print_choices()
            continue

    global_ws.apply_env_to_process()

    console.print()
    console.print("[anton.success]Configuration updated.[/]")
    console.print()

    return rebuild_session(
        settings=settings,
        state=state,
        self_awareness=self_awareness,
        cortex=cortex,
        workspace=workspace,
        console=console,
        episodic=episodic,
        history_store=history_store,
        session_id=session_id,
    )


async def handle_setup(
    console: Console,
    settings: AntonSettings,
    workspace: "Workspace",
    state: dict,
    self_awareness,
    cortex: "Cortex | None",
    session: "ChatSession",
    episodic: "EpisodicMemory | None" = None,
    history_store: "HistoryStore | None" = None,
    session_id: str | None = None,
) -> "ChatSession":
    """Interactive setup wizard with sub-menu: Models or Memory."""
    console.print()
    console.print("[anton.cyan]/setup[/]")
    console.print()
    console.print("  What do you want to configure?")
    console.print("    [bold]1[/]  LLM — provider, API key, and models")
    console.print("    [bold]2[/]  Memory — memory mode and episodic memory")
    console.print("    [bold]q[/]  Back")
    console.print()

    top_choice = await prompt_or_cancel(
        "(anton) Select", choices=["1", "2", "q"], default="q"
    )
    if top_choice is None or top_choice == "q":
        console.print()
        return session

    if top_choice == "1":
        return await handle_setup_models(
            console,
            settings,
            workspace,
            state,
            self_awareness,
            cortex,
            session,
            episodic=episodic,
            history_store=history_store,
            session_id=session_id,
        )
    else:
        await handle_setup_memory(
            console, settings, workspace, cortex, episodic=episodic
        )
        return session
