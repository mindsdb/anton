"""Cloud-safe ChatSession builder.

Assembles a :class:`~anton.core.session.ChatSession` scoped to the tenant's
mounted workspace with the desktop-only, tenant-leaky behaviours OFF. It does
NOT call the desktop ``build_chat_session`` (which loads workspace ``.env``,
uses ``~/.anton`` personal memory, and injects vault creds into ``os.environ``).

Safety posture (all internal — nothing here is on the wire):

* Trusted pod-side workspace mount, never taken from the request.
* No dotenv loading (``AntonSettings(_env_file=None)``, shared into Workspace).
* Personal memory / connectors / data-vault / disk history OFF.
* Scratchpad subprocess env built from a non-secret allowlist, so generated
  code can't read the provider key (interim until the Plan-5 gateway removes
  the key from the pod entirely).
* Only reviewed, headless-safe tools are exposed (scratchpad + artifacts).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from anton.cloud_turn.contract import TurnRequestV1

if TYPE_CHECKING:
    from anton.core.session import ChatSession

logger = logging.getLogger(__name__)

#: Trusted mount path — pod-side config, never from the wire request.
DEFAULT_CLOUD_WORKSPACE_PATH = "/workspace"
#: Operator/CI override for the mount path (pod-side env var, not request data).
_WORKSPACE_PATH_ENV = "ANTON_CLOUD_WORKSPACE_PATH"

#: The only tools exposed in a cloud turn: scratchpad + the workspace-scoped
#: artifact tools. Everything else core registers is dropped.
CLOUD_TOOL_ALLOWLIST = frozenset(
    {
        "scratchpad",
        "create_artifact",
        "list_artifacts",
        "open_artifact",
        "update_artifact",
    }
)


def resolve_trusted_workspace_path() -> Path:
    """Resolve the trusted workspace mount path (never from the wire request).

    Reads :data:`_WORKSPACE_PATH_ENV` or falls back to
    :data:`DEFAULT_CLOUD_WORKSPACE_PATH`; rejects relative paths and ``..``,
    then canonicalises so downstream containment checks compare a real path.
    """
    raw = os.environ.get(_WORKSPACE_PATH_ENV) or DEFAULT_CLOUD_WORKSPACE_PATH
    if not os.path.isabs(raw):
        raise ValueError(
            f"trusted workspace path must be absolute, got {raw!r} "
            f"(set {_WORKSPACE_PATH_ENV} to an absolute path)"
        )
    if ".." in Path(raw).parts:
        raise ValueError(f"trusted workspace path must not contain '..': {raw!r}")
    resolved = Path(raw).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    if not resolved.is_dir():
        raise ValueError(f"trusted workspace path is not a directory: {resolved}")
    return resolved


def build_cloud_chat_session(request: TurnRequestV1) -> "ChatSession":
    """Assemble a cloud-safe ChatSession for one turn.

    History is DB-authoritative (from the request). The workspace path is the
    trusted pod mount, NOT ``request.workspace_path``.
    """
    from anton.config.settings import AntonSettings
    from anton.core.backends.local import sanitized_scratchpad_runtime_factory
    from anton.core.llm.client import LLMClient
    from anton.core.session import ChatSession, ChatSessionConfig
    from anton.workspace import Workspace

    base = resolve_trusted_workspace_path()

    # `_env_file=None`: never load the AntonSettings .env chain (~/.anton/.env,
    # ~/.cowork/.env, /workspace/.env). Same object passed to Workspace so it
    # doesn't build a second, dotenv-loading one.
    settings = AntonSettings(_env_file=None)
    settings.resolve_workspace(str(base))
    if request.model:
        settings.planning_model = request.model
    # Skills stay in the workspace, never the pod-shared ~/.anton.
    settings.skills_root = base / ".anton" / "skills"

    workspace = Workspace(base, settings=settings)
    workspace.initialize()
    # No apply_env_to_process(): loading workspace .env into the process env
    # would expose tenant secrets to cell code.

    llm_client = LLMClient.from_settings(settings)

    config = ChatSessionConfig(
        llm_client=llm_client,
        settings=settings,
        workspace=workspace,
        session_id=request.conversation_id,
        harness="cloud",
        # DB-authoritative history; the pod never loads its own.
        initial_history=list(request.history) if request.history else None,
        console=None,                       # headless
        cortex=None,                        # personal memory OFF
        episodic=None,
        self_awareness=None,
        data_vault=None,                    # connectors OFF
        history_store=None,                 # disk history OFF (DB authoritative)
        tools=[],                           # no host connector/publish tools
        tool_allowlist=CLOUD_TOOL_ALLOWLIST,          # only reviewed tools survive the build
        runtime_factory=sanitized_scratchpad_runtime_factory,  # secret-free scratchpad env
        web_search_enabled=False,
        web_fetch_enabled=False,
    )

    session = ChatSession(config)
    logger.info(
        "cloud session built conversation=%s workspace=%s tools=%s",
        request.conversation_id, base, sorted(CLOUD_TOOL_ALLOWLIST),
    )
    return session
