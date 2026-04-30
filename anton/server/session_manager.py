"""Chat session manager for the local HTTP server.

Manages ChatSession lifecycle keyed by conversation_id, with capped concurrent
live sessions and on-disk history persistence (resumable across restarts).

Adapted from anton_servicesrepo/scratchpad_service/chat_session_manager.py —
the local variant uses AntonSettings and Anton's existing HistoryStore instead
of the container-specific runtime module.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import AsyncIterator

from anton.config.settings import AntonSettings


MAX_SESSIONS = int(os.environ.get("ANTON_SERVER_MAX_SESSIONS", "3"))


_settings: AntonSettings | None = None
_sessions: dict[str, object] = {}  # conversation_id -> ChatSession
_history_store = None  # type: ignore[assignment]


def configure(settings: AntonSettings) -> None:
    """Bind the manager to a resolved AntonSettings instance.

    Must be called once at server startup before any chat_stream() calls.
    """
    global _settings, _history_store
    _settings = settings

    from anton.memory.history_store import HistoryStore
    episodes_dir = settings.workspace_path / ".anton" / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    _history_store = HistoryStore(episodes_dir)


def _require_settings() -> AntonSettings:
    if _settings is None:
        raise RuntimeError("session_manager.configure() must be called before use")
    return _settings


def _build_session(
    conversation_id: str,
    initial_history: list[dict] | None = None,
):
    """Create a new ChatSession bound to the configured workspace + LLM."""
    from anton.core.llm.client import LLMClient
    from anton.core.session import ChatSession, ChatSessionConfig
    from anton.core.llm.prompt_builder import SystemPromptContext
    from anton.core.backends.local import local_scratchpad_runtime_factory
    from anton.workspace import Workspace
    from anton.chat_session import build_runtime_context

    settings = _require_settings()
    llm_client = LLMClient.from_settings(settings)

    workspace = Workspace(settings.workspace_path)
    if not workspace.is_initialized():
        workspace.initialize()

    runtime_context = build_runtime_context(settings)
    output_path = f"{settings.output_dir.rstrip('/')}/"

    config = ChatSessionConfig(
        llm_client=llm_client,
        runtime_factory=local_scratchpad_runtime_factory,
        system_prompt_context=SystemPromptContext(
            runtime_context=runtime_context,
            output_context=f"Save output to `{output_path}` (create it if needed).",
        ),
        workspace=workspace,
        history_store=_history_store,
        session_id=conversation_id,
        initial_history=initial_history,
    )

    return ChatSession(config)


def _evict_oldest() -> None:
    if _sessions:
        oldest_id = next(iter(_sessions))
        _sessions.pop(oldest_id, None)


def _new_conversation_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def get_or_create_session(conversation_id: str | None = None):
    """Return (session, conversation_id), creating or resuming as needed."""
    if not conversation_id:
        conversation_id = _new_conversation_id()

    if conversation_id in _sessions:
        return _sessions[conversation_id], conversation_id

    if len(_sessions) >= MAX_SESSIONS:
        _evict_oldest()

    history = _history_store.load(conversation_id) if _history_store else None
    session = _build_session(conversation_id, initial_history=history)
    _sessions[conversation_id] = session
    return session, conversation_id


async def chat_stream(
    user_input: str | list[dict],
    conversation_id: str | None = None,
) -> tuple[AsyncIterator, str]:
    """Send a message and yield stream events. Returns (event_stream, conversation_id)."""
    session, cid = get_or_create_session(conversation_id)

    async def _stream():
        async for event in session.turn_stream(user_input):
            yield event
        # HistoryStore auto-persists via ChatSession when history_store is set,
        # but save here too for safety.
        if _history_store is not None:
            _history_store.save(cid, session.history)

    return _stream(), cid


async def close_all() -> None:
    """Close all live sessions (called on server shutdown)."""
    for cid in list(_sessions):
        session = _sessions.pop(cid, None)
        if session is None:
            continue
        try:
            await session.close()
        except Exception:
            pass


def list_sessions() -> list[str]:
    return list(_sessions.keys())
