from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse

from rich.console import Console

from anton.config.settings import AntonSettings
from anton.utils.prompt import prompt_or_cancel
from anton.memory.manage import handle_setup_memory

if TYPE_CHECKING:
    from anton.chat import ChatSession
    from anton.core.memory.episodes import EpisodicMemory
    from anton.memory.history_store import HistoryStore
    from anton.core.memory.cortex import Cortex
    from anton.workspace import Workspace



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
