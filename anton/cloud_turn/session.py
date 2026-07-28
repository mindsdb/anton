"""Cloud-safe ChatSession builder.

Assembles ``ChatSessionConfig`` directly instead of the desktop
``build_chat_session`` (which loads workspace ``.env``, uses ``~/.anton``
personal memory, and injects vault creds into ``os.environ`` — all cross-tenant
hazards in a shared pod). Milestone-1 posture:

* Trusted pod-side workspace path, never a wire field.
* No dotenv loading.
* Scratchpad env from a non-secret allowlist.
* Explicit tool allowlist; new core tools are not auto-enabled.
* Personal memory / connectors / data-vault / local-file history OFF.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from anton.cloud_turn.errors import (
    UnsupportedCapabilityError,
    UnsupportedModelError,
)
from anton.cloud_turn.protocol import CapabilitiesV1, TurnRequestV1

if TYPE_CHECKING:
    from anton.config.settings import AntonSettings
    from anton.core.session import ChatSession

logger = logging.getLogger(__name__)

#: Trusted mount path — pod-side config, never from the wire request.
DEFAULT_CLOUD_WORKSPACE_PATH = "/workspace"

#: Operator/CI override for the mount path (pod-side env var, not request data).
_WORKSPACE_PATH_ENV = "ANTON_CLOUD_WORKSPACE_PATH"

#: Pod-side trusted model allowlist (comma-separated). Default empty = the
#: request may NOT override the model; the trusted settings default is used.
_MODEL_ALLOWLIST_ENV = "ANTON_CLOUD_MODEL_ALLOWLIST"

#: The only tools exposed in a milestone-1 cloud turn: scratchpad + the
#: workspace-scoped artifact tools. Everything else core registers is dropped.
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

    # Canonicalise (follows symlinks) so later checks compare the real location.
    resolved = Path(raw).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    if not resolved.is_dir():
        raise ValueError(f"trusted workspace path is not a directory: {resolved}")
    return resolved


def build_cloud_chat_session(request: TurnRequestV1) -> "ChatSession":
    """Assemble a cloud-safe ChatSession for one turn.

    The LLM credential is read by ``LLMClient.from_settings`` from the
    ``ANTON_*`` environment (a synthetic-data dev key today; a gateway-issued
    run-scoped credential before beta). History comes from the request (the DB
    is authoritative) — never from the workspace.
    """
    from anton.cloud_turn.messages import to_anton_history
    from anton.config.settings import AntonSettings
    from anton.core.backends.local import sanitized_scratchpad_runtime_factory
    from anton.core.llm.client import LLMClient
    from anton.core.session import ChatSession, ChatSessionConfig
    from anton.workspace import Workspace

    caps = request.capabilities

    # Milestone 1 is the locked-down posture only; any capability on = fail loud.
    _reject_unsupported_capabilities(caps)

    # Trusted pod-side mount path — NOT request-controlled.
    base = resolve_trusted_workspace_path()

    # `_env_file=None`: never load the AntonSettings .env chain (~/.anton/.env,
    # ~/.cowork/.env, /workspace/.env). Same object is passed to Workspace so it
    # doesn't build a second, dotenv-loading one.
    settings = AntonSettings(_env_file=None)
    settings.resolve_workspace(str(base))
    _apply_model_policy(settings, request.model)
    # Skills stay in the workspace, never the pod-shared ~/.anton.
    settings.skills_root = base / ".anton" / "skills"

    workspace = Workspace(base, settings=settings)
    workspace.initialize()
    # No apply_env_to_process(): loading workspace .env into the process env
    # would expose tenant secrets to cell code. No capability enables it.

    llm_client = LLMClient.from_settings(settings)

    config = ChatSessionConfig(
        llm_client=llm_client,
        settings=settings,
        workspace=workspace,
        session_id=request.conversation_id,
        harness="cloud",
        # DB-authoritative history; the pod never loads its own. Typed wire
        # messages are converted to Anton's internal dict shape.
        initial_history=to_anton_history(request.history) or None,
        console=None,                       # headless — no Rich console
        cortex=None,                        # personal memory OFF
        episodic=None,
        self_awareness=None,
        data_vault=None,                    # connectors OFF
        history_store=None,                 # local-file history OFF (DB authoritative)
        tools=[],                           # no host connector/publish tools
        tool_allowlist=CLOUD_TOOL_ALLOWLIST,          # only reviewed tools survive the build
        runtime_factory=sanitized_scratchpad_runtime_factory,  # secret-free scratchpad env
        # Milestone 1: text + scratchpad only; web tools land with a later capability.
        web_search_enabled=False,
        web_fetch_enabled=False,
    )

    session = ChatSession(config)
    logger.info(
        "cloud session built run_id=%s workspace=%s tool_allowlist=%s",
        request.run_id, base, sorted(CLOUD_TOOL_ALLOWLIST),
    )
    return session


def _reject_unsupported_capabilities(caps: CapabilitiesV1) -> None:
    """Fail loud on any capability that has no cloud-safe implementation yet."""
    enabled = [name for name in CapabilitiesV1.model_fields if getattr(caps, name)]
    if enabled:
        raise UnsupportedCapabilityError(
            "cloud session does not implement capabilities: "
            + ", ".join(sorted(enabled))
        )


def _trusted_model_allowlist() -> frozenset[str]:
    raw = os.environ.get(_MODEL_ALLOWLIST_ENV, "")
    return frozenset(m.strip() for m in raw.split(",") if m.strip())


def _apply_model_policy(settings: "AntonSettings", requested: str | None) -> None:
    """Model selection is NOT request-controlled by default.

    ``None`` → keep the trusted settings default. A requested model is honoured
    only if it is in the pod-side trusted allowlist (default empty), else
    rejected with a structured error."""
    if requested is None:
        return
    allowed = _trusted_model_allowlist()
    if requested not in allowed:
        raise UnsupportedModelError(
            f"model {requested!r} is not permitted for cloud turns"
        )
    settings.planning_model = requested
