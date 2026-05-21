"""Shared ChatSession builder for any host (Cowork desktop, dispatch, CLI integrations).

Wraps the boilerplate that used to live inline in `anton-cowork/server/routes/anton_bridge.py`
so the same workspace + memory + vault wiring is reused by every host. Hosts customize
behavior via parameters (extra tools, system-prompt suffix, output-context template) rather
than forking the builder.

Public API:
    build_chat_session(...)        — async builder returning a ready ChatSession
    resolve_workspace_base(...)    — workspace path normalizer
    safe_redact_error(exc)         — error-message redactor that strips API keys
    AntonConfigurationError        — raise when setup is missing/invalid
    AntonRuntimeError              — raise when an Anton call fails after configuration is OK
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

REDACTED_ENV_KEYS: tuple[str, ...] = (
    "ANTON_ANTHROPIC_API_KEY",
    "ANTON_OPENAI_API_KEY",
    "ANTON_MINDS_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)


class AntonConfigurationError(RuntimeError):
    """Setup is missing or invalid — Anton cannot run yet."""


class AntonRuntimeError(RuntimeError):
    """A real Anton call failed after configuration passed."""


def safe_redact_error(exc: Exception) -> str:
    """Stringify an exception, replacing any live API-key values with [redacted]."""
    message = str(exc).strip() or exc.__class__.__name__
    for key in REDACTED_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            message = message.replace(value, "[redacted]")
    return message


def resolve_workspace_base(workspace_path: Optional[str]) -> Path:
    """Normalize a user-provided workspace path; default to the current working directory."""
    if workspace_path:
        return Path(workspace_path).expanduser().resolve()
    return Path.cwd().resolve()


async def build_chat_session(
    *,
    session_id: str,
    workspace_path: Optional[str] = None,
    model: Optional[str] = None,
    extra_tools: Optional[Sequence[Any]] = None,
    system_prompt_suffix: Optional[str] = None,
    output_context: Optional[str] = None,
):
    """Build a ChatSession scoped to one workspace.

    Replicates the same wiring the Anton CLI uses: settings → workspace → LLM client →
    memory (cortex + episodes + history) → data vault env injection → ChatSession.

    Parameters
    ----------
    session_id
        Stable id used for episodic-memory resume + history-store load.
    workspace_path
        Project workspace root. None → current working directory.
    model
        Override planning model on the resolved settings.
    extra_tools
        Tools added on top of the default Anton tool set (e.g. publishing, datasource connect).
        Hosts pass their own tool set; pass None for the bare default.
    system_prompt_suffix
        Free-form text appended to the system prompt. Hosts use this to nudge tone or
        describe their UI affordances. None → no suffix.
    output_context
        Override for the per-session output-folder hint. None → use the default template
        pointing at `settings.artifacts_dir`.

    Returns
    -------
    anton.core.session.ChatSession
        Ready to call `turn_stream(text)`.
    """
    # Imports kept inside the function so this module is importable without anton's
    # heavy deps (typer, rich) at module load time.
    from anton.chat_session import build_runtime_context
    from anton.config.settings import AntonSettings
    from anton.context.self_awareness import SelfAwarenessContext
    from anton.core.llm.client import LLMClient
    from anton.core.memory.cortex import Cortex
    from anton.core.memory.episodes import EpisodicMemory
    from anton.core.memory.hippocampus import Hippocampus
    from anton.core.session import ChatSession, ChatSessionConfig, SystemPromptContext
    from anton.memory.history_store import HistoryStore
    from anton.workspace import Workspace

    try:
        from anton.core.datasources.data_vault import LocalDataVault
    except Exception:  # pragma: no cover — old Anton builds may not expose it
        LocalDataVault = None  # type: ignore[assignment]

    base = resolve_workspace_base(workspace_path)
    settings = AntonSettings()
    settings.resolve_workspace(str(base))
    if model:
        settings.planning_model = model

    workspace = Workspace(base)
    workspace.initialize()
    workspace.apply_env_to_process()

    anton_dir = base / ".anton"
    output_dir = Path(settings.artifacts_dir)
    context_dir = Path(settings.context_dir)
    episodes_dir = anton_dir / "episodes"
    project_memory_dir = anton_dir / "memory"
    for directory in (output_dir, context_dir, episodes_dir, project_memory_dir):
        directory.mkdir(parents=True, exist_ok=True)

    llm_client = LLMClient.from_settings(settings)
    self_awareness = SelfAwarenessContext(context_dir)

    global_memory_dir = Path.home() / ".anton" / "memory"
    global_memory_dir.mkdir(parents=True, exist_ok=True)
    cortex = Cortex(
        global_hc=Hippocampus(global_memory_dir),
        project_hc=Hippocampus(project_memory_dir),
        mode=settings.memory_mode if settings.memory_enabled else "off",
        llm_client=llm_client,
    )
    episodic = EpisodicMemory(episodes_dir, enabled=settings.episodic_memory)
    episodic.resume_session(session_id)
    history_store = HistoryStore(episodes_dir)
    initial_history = history_store.load(session_id)

    resolved_output_context = output_context or (
        f"Save generated files and dashboards to `{output_dir}`. "
        "When you create a user-facing HTML dashboard or report, save it there."
    )

    data_vault = LocalDataVault() if LocalDataVault is not None else None
    google_drive_oauth_connected = False
    if data_vault is not None:
        try:
            for conn in data_vault.list_connections():
                engine = conn.get("engine")
                name = conn.get("name")
                if not (engine and name):
                    continue
                data_vault.inject_env(engine, name)
                if engine == "google_drive":
                    fields = data_vault.load(engine, name) or {}
                    if fields.get("auth_type") == "oauth":
                        google_drive_oauth_connected = True
        except Exception:
            logger.debug("Could not inject Anton data vault env", exc_info=True)

    integration_guidance = ""
    if google_drive_oauth_connected:
        integration_guidance = (
            " Connected Google Drive accounts are available through Google OAuth credentials "
            "in the injected `DS_GOOGLE_DRIVE_<CONNECTION>__...` environment variables. "
            "Only claim Google Drive access if you can actually use those credentials successfully."
        )

    suffix_parts = [s for s in (system_prompt_suffix, integration_guidance) if s]
    final_suffix = "".join(suffix_parts) if suffix_parts else None

    config = ChatSessionConfig(
        llm_client=llm_client,
        settings=settings,
        self_awareness=self_awareness,
        cortex=cortex,
        episodic=episodic,
        system_prompt_context=SystemPromptContext(
            runtime_context=build_runtime_context(settings),
            suffix=final_suffix,
            output_context=resolved_output_context,
        ),
        workspace=workspace,
        data_vault=data_vault,
        initial_history=initial_history,
        history_store=history_store,
        session_id=session_id,
        proactive_dashboards=settings.proactive_dashboards,
        tools=list(extra_tools) if extra_tools else [],
    )
    return ChatSession(config)


__all__ = [
    "AntonConfigurationError",
    "AntonRuntimeError",
    "REDACTED_ENV_KEYS",
    "build_chat_session",
    "resolve_workspace_base",
    "safe_redact_error",
]
