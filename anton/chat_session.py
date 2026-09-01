"""Session factory — builds and rebuilds ChatSession after settings changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console

from anton.config.settings import AntonSettings
from anton.core.llm.identity import build_runtime_context  # noqa: F401 — re-export
from anton.core.llm.prompt_builder import SystemPromptContext
from anton.minds_client import refresh_knowledge

if TYPE_CHECKING:
    from anton.chat import ChatSession
    from anton.core.memory.cortex import Cortex
    from anton.core.memory.episodes import EpisodicMemory
    from anton.memory.history_store import HistoryStore
    from anton.workspace import Workspace


# build_runtime_context lives in anton.core.llm.identity (import-light, so the
# cloud pod can use it); re-exported here because cowork-server imports it from
# this module: `from anton.chat_session import build_runtime_context`.
__all__ = ["build_runtime_context", "get_runtime_factory", "rebuild_session"]


def get_runtime_factory(settings: AntonSettings):
    """Return the appropriate scratchpad runtime factory based on settings.

    If backend is set to "remote" (and minds_api_key available),
    returns a remote factory. Otherwise returns the local factory.
    """
    if settings.backend == "remote":
        from functools import partial
        from anton.core.backends.remote import remote_scratchpad_runtime_factory

        return partial(
            remote_scratchpad_runtime_factory,
            endpoint_url=settings.minds_url,
            api_key=settings.minds_api_key,
        )

    from anton.core.backends.local import local_scratchpad_runtime_factory
    return local_scratchpad_runtime_factory


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
    from anton.core.llm.tracing import HARNESS_ANTON, SURFACE_CLI
    from anton.core.session import ChatSessionConfig
    from anton.tools import DEFAULT_SESSION_TOOLS

    state["llm_client"] = LLMClient.from_settings(settings)

    # Update cortex with new LLM client and memory mode
    if cortex is not None:
        cortex._llm = state["llm_client"]
        cortex.mode = settings.memory_mode

    # Refresh mind knowledge from remote server
    refresh_knowledge(settings, cortex)

    runtime_context = build_runtime_context(settings)
    return ChatSession(ChatSessionConfig(
        llm_client=state["llm_client"],
        runtime_factory=get_runtime_factory(settings),
        settings=settings,
        self_awareness=self_awareness,
        cortex=cortex,
        episodic=episodic,
        system_prompt_context=SystemPromptContext(
            runtime_context=runtime_context,
        ),
        workspace=workspace,
        console=console,
        history_store=history_store,
        session_id=session_id,
        # ENG-1495: identify the host explicitly. Left unset, `harness` reached
        # telemetry as "" — which meant BOTH "this is the CLI" and "the host
        # didn't identify itself", so the two could never be told apart and the
        # ambiguity got worse with every new host.
        # WHICH AGENT: the CLI runs anton, so that is what it reports
        # (ENG-1694). It said "cli" until then, which described the host and
        # left "did this run anton or hermes?" unanswerable.
        harness=HARNESS_ANTON,
        # WHERE it ran — the other axis, and now the only place "cli" appears.
        surface=SURFACE_CLI,
        proactive_dashboards=settings.proactive_dashboards,
        act_first=settings.act_first,
        output_dir=settings.artifacts_dir,
        # ENG-1166: without this, resumed / post-settings-change sessions
        # register only core tools and silently lose publish_or_preview +
        # connect_new_datasource. Mirror the fresh-session builder (chat.py).
        tools=list(DEFAULT_SESSION_TOOLS),
        web_search_enabled=settings.web_search_enabled,
        web_fetch_enabled=settings.web_fetch_enabled,
    ))
