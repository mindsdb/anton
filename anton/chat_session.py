"""Session factory — builds and rebuilds ChatSession after settings changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console

from anton.config.settings import AntonSettings
from anton.minds_client import refresh_knowledge

if TYPE_CHECKING:
    from anton.chat import ChatSession
    from anton.core.memory.cortex import Cortex
    from anton.core.memory.episodes import EpisodicMemory
    from anton.memory.history_store import HistoryStore
    from anton.workspace import Workspace


def build_runtime_context(settings: AntonSettings) -> str:
    """Build runtime context string including Minds datasource info if configured."""
    ctx = (
        f"- Provider: {settings.planning_provider}\n"
        f"- Planning model: {settings.planning_model}\n"
        f"- Coding model: {settings.coding_model}\n"
        f"- Workspace: {settings.workspace_path}\n"
        f"- Memory mode: {settings.memory_mode}"
    )
    if settings.minds_api_key and (
        settings.minds_mind_name or settings.minds_datasource
    ):
        engine = settings.minds_datasource_engine or "unknown"
        ctx += f"\n\n**CONNECTED MIND (Minds):**\n"
        if settings.minds_mind_name:
            ctx += f"- Mind: {settings.minds_mind_name}\n"
        if settings.minds_datasource:
            ctx += (
                f"- Datasource: {settings.minds_datasource}\n"
                f"- Engine: {engine}\n"
            )
        ctx += (
            f"- Minds URL: {settings.minds_url}\n"
            f"- To query data, use the scratchpad with the built-in `query_minds_data()` function.\n"
            f"  It is pre-loaded in the scratchpad namespace — DO NOT import it. Just call it directly.\n"
            f'  Example: result = query_minds_data("SELECT * FROM users LIMIT 5")\n'
            f"  Returns dict with 'type', 'data' (list of rows), 'column_names', 'error_message'.\n"
            f'  Optional: query_minds_data("SELECT ...", datasource="other_ds")\n'
        )
        if settings.minds_datasource:
            ctx += f"- Write SQL appropriate for the {engine} engine.\n"
    return ctx


def rebuild_session(
    *,
    settings: AntonSettings,
    state: dict,
    self_awareness,
    cortex: "Cortex | None",
    workspace: "Workspace | None",
    console: Console,
    episodic: "EpisodicMemory | None" = None,
    history_store: "HistoryStore | None" = None,
    session_id: str | None = None,
) -> "ChatSession":
    """Rebuild LLMClient + ChatSession after settings change."""
    from anton.core.llm.client import LLMClient
    from anton.chat import ChatSession
    from anton.core.session import ChatSessionConfig

    state["llm_client"] = LLMClient.from_settings(settings)

    # Update cortex with new LLM client and memory mode
    if cortex is not None:
        cortex._llm = state["llm_client"]
        cortex.mode = settings.memory_mode

    # Refresh mind knowledge from remote server
    refresh_knowledge(settings, cortex)

    runtime_context = build_runtime_context(settings)
    api_key = (
        settings.anthropic_api_key
        if settings.coding_provider == "anthropic"
        else settings.openai_api_key
    ) or ""
    return ChatSession(ChatSessionConfig(
        llm_client=state["llm_client"],
        self_awareness=self_awareness,
        cortex=cortex,
        episodic=episodic,
        runtime_context=runtime_context,
        workspace=workspace,
        console=console,
        history_store=history_store,
        session_id=session_id,
        proactive_dashboards=settings.proactive_dashboards,
        output_dir=settings.output_dir,
    ))
