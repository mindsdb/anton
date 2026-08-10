"""Cloud-safe ChatSession builder.

Assembles a :class:`~anton.core.session.ChatSession` scoped to the tenant's
mounted workspace with the desktop-only, tenant-leaky behaviours OFF. It does
NOT call the desktop ``build_chat_session`` (which loads workspace ``.env``,
uses ``~/.anton`` personal memory, and injects vault creds into ``os.environ``).

Safety posture (all internal — nothing here is on the wire):

* Trusted pod-side workspace mount, never taken from the request.
* No dotenv loading (``AntonSettings(_env_file=None)``, shared into Workspace).
* Connectors / data-vault / disk history OFF.
* Memory never persists pod-side: cowork sends the tenant's slots per turn, and
  writes are reported back for cowork to apply (see :func:`_build_cortex`).
* Only reviewed, headless-safe tools are exposed (scratchpad + artifacts).
"""

from __future__ import annotations

import logging
import os
import tempfile
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
        # Safe to list only because a cortex is always built — core registers
        # `memorize` with one, and an allowlist name matching no tool is fatal.
        "memorize",
    }
)

#: Per-turn staging: Hippocampus reads slots from disk, so the payload has to land
#: as files. Not the workspace — its PVC outlives the turn and cells can read it.
_MEMORY_DIR = Path(tempfile.gettempdir()) / "anton-cloud-memory"

#: Wire slot name -> the filename Hippocampus reads. Unknown slots are ignored.
_MEMORY_SLOT_FILES = {"profile": "profile.md", "rules": "rules.md", "lessons": "lessons.md"}


def _write_memory_slots(dest: Path, slots: dict | None) -> None:
    """Write `slots` into `dest`, clearing any this turn didn't send — the dir is
    reused across turns, so a server-side deletion must not linger."""
    dest.mkdir(parents=True, exist_ok=True)
    values = slots if isinstance(slots, dict) else {}
    for slot, filename in _MEMORY_SLOT_FILES.items():
        path = dest / filename
        text = values.get(slot)
        if isinstance(text, str) and text.strip():
            path.write_text(text, encoding="utf-8")
        elif path.exists():
            path.unlink()


def _cloud_cortex_class():
    """Cortex that records writes for the entrypoint to emit instead of storing
    them. cowork applies them under its own scope; nothing persists pod-side."""
    from anton.core.memory.cortex import Cortex

    class _CloudCortex(Cortex):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.pending_memory: list[dict] = []

        async def encode(self, engrams: list) -> list[str]:
            # Normalize only — cowork validates, being the trust boundary.
            accepted = [
                {"text": e.text, "kind": e.kind, "scope": e.scope or "global",
                 "topic": e.topic or "", "confidence": e.confidence or "medium",
                 "source": e.source or "llm"}
                for e in engrams
                if isinstance(e.text, str) and e.text.strip()
            ]
            self.pending_memory.extend(accepted)
            return [f"Encoded {e['kind']}: {e['text']}" for e in accepted]

    return _CloudCortex


def _build_cortex(memory: dict | None, llm_client):
    """Cortex over cowork's memory. Built even for an empty store, or an org could
    never record its first memory — core only registers `memorize` with a cortex."""
    from anton.core.memory.hippocampus import Hippocampus

    memory = memory if isinstance(memory, dict) else {}

    global_dir = _MEMORY_DIR / "global"
    project_dir = _MEMORY_DIR / "project"
    _write_memory_slots(global_dir, memory.get("global"))
    _write_memory_slots(project_dir, memory.get("project"))
    # Not "off", or `memorize` refuses to encode; the automatic passes that mode
    # also unlocks are disabled via `background_memory`.
    return _cloud_cortex_class()(
        global_hc=Hippocampus(global_dir),
        project_hc=Hippocampus(project_dir),
        mode="autopilot",
        llm_client=llm_client,
    )


def drain_pending_memory(session) -> list[dict]:
    """Engrams this turn asked to remember. Clears as it reads, so a second call
    can't duplicate them."""
    cortex = getattr(session, "_cortex", None)
    pending = getattr(cortex, "pending_memory", None)
    if not pending:
        return []
    drained = list(pending)
    pending.clear()
    return drained


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
    from anton.core.backends.local import local_scratchpad_runtime_factory
    from anton.core.llm.client import LLMClient
    from anton.core.session import ChatSession, ChatSessionConfig
    from anton.workspace import Workspace

    base = resolve_trusted_workspace_path()

    # `_env_file=None`: never load the AntonSettings .env chain (~/.anton/.env,
    # ~/.cowork/.env, /workspace/.env). Same object passed to Workspace so it
    # doesn't build a second, dotenv-loading one.
    #
    # When the request carries a per-turn `llm` block (cowork's short-TTL
    # MindsHub turn key), fold it in here so it wins over any env-based
    # config: the existing minds-cloud -> openai-compatible derivation in
    # AntonSettings.model_post_init then points the client at MindsHub with
    # the turn key, never a long-lived pod credential.
    llm = request.llm or {}
    settings_kwargs: dict = {"_env_file": None}
    if llm:
        settings_kwargs.update(
            planning_provider=llm["provider"],
            coding_provider=llm["provider"],
            minds_api_key=llm["api_key"],
            minds_url=llm["base_url"],
        )
        # Coding model follows the turn credential; otherwise the pod keeps
        # the built-in default, which the turn key may not be allowed to pay for.
        if llm.get("coding_model"):
            settings_kwargs["coding_model"] = llm["coding_model"]
    settings = AntonSettings(**settings_kwargs)
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

    cortex = _build_cortex(request.memory, llm_client)

    config = ChatSessionConfig(
        llm_client=llm_client,
        settings=settings,
        workspace=workspace,
        session_id=request.conversation_id,
        harness="cloud",
        # DB-authoritative history; the pod never loads its own.
        initial_history=list(request.history) if request.history else None,
        console=None,                       # headless
        cortex=cortex,                      # org memory; writes are reported, not stored
        episodic=None,
        self_awareness=None,
        data_vault=None,                    # connectors OFF
        history_store=None,                 # disk history OFF (DB authoritative)
        tools=[],                           # no host connector/publish tools
        tool_allowlist=CLOUD_TOOL_ALLOWLIST,  # only reviewed tools survive the build
        background_memory=False,            # one turn per pod: no end-of-turn LLM passes
        runtime_factory=local_scratchpad_runtime_factory,
        web_search_enabled=False,
        web_fetch_enabled=False,
    )

    session = ChatSession(config)
    logger.info(
        "cloud session built conversation=%s workspace=%s tools=%s",
        request.conversation_id, base, sorted(CLOUD_TOOL_ALLOWLIST),
    )
    return session
