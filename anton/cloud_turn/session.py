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

import contextlib
import logging
import os
import tempfile
from pathlib import Path, PurePosixPath
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
        # Read-only over staged skills + builtins. The prompt mandates recalling
        # some builtins, so stripping the tool would point the model at a
        # missing tool. Only writes a stats counter in the discarded staging dir.
        "recall_skill",
    }
)

#: Per-turn staging: Hippocampus reads slots from disk, so the payload has to land
#: as files. Used only as a fallback when no shared mount is configured (desktop,
#: CI); see _MEMORY_GLOBAL_ROOT_ENV below for the mounted, cross-turn-persistent path.
_MEMORY_DIR = Path(tempfile.gettempdir()) / "anton-cloud-memory"

#: Wire slot name -> the filename Hippocampus reads. Unknown slots are ignored.
_MEMORY_SLOT_FILES = {"profile": "profile.md", "rules": "rules.md", "lessons": "lessons.md"}

#: Pod-side mount of this (org, user)'s global memory. When set, global memory
#: slots are read straight off the mount and the wire payload's `global` block is
#: ignored: cowork-server owns that tree and the pod sees the live copy, not a
#: per-turn snapshot. Writes still go out as `turn_memory` events regardless;
#: cowork-server remains the trust boundary for what gets persisted.
_MEMORY_GLOBAL_ROOT_ENV = "ANTON_CLOUD_MEMORY_GLOBAL_ROOT"

#: Per-turn skills staging: a FRESH mkdtemp each turn (unlike _MEMORY_DIR).
#: Skills carry no cross-turn state, and the unpredictable path stops prior-turn
#: cell code planting a symlink or two turns wiping each other's tree. Used only
#: as a fallback when no shared mount is configured; see _SKILLS_ROOT_ENV below.
_SKILLS_DIR_PREFIX = "anton-cloud-skills-"

#: Pod-side mount of the org's skills tree. When set, skills are read from it
#: directly and the wire payload's `skills` block is ignored: cowork-server owns
#: that tree and the pod sees the live copy, not a per-turn snapshot.
_SKILLS_ROOT_ENV = "ANTON_CLOUD_SKILLS_ROOT"


def _safe_skill_file(skill_dir: Path, rel: str) -> Path | None:
    """Resolve a wire-supplied relative path inside `skill_dir`, or None.

    resolve() also rejects NUL/surrogate filename bytes — keep it."""
    pure = PurePosixPath(rel)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        return None
    candidate = skill_dir.joinpath(*pure.parts)
    try:
        candidate.resolve().relative_to(skill_dir.resolve())
    except ValueError:
        return None
    return candidate


def _stage_skills(skills: dict | None) -> Path:
    """The session's skills_root.

    With the org tree mounted (_SKILLS_ROOT_ENV set), this is the mount itself
    and nothing is staged: cowork-server owns that tree and the pod reads it
    live, so the wire payload's `skills` block is ignored entirely.

    Without a mount (desktop, CI), the request's skills are materialized into a
    fresh per-turn dir for SkillStore to read, as before. Wire data is
    untrusted (like memory): bad slugs and escaping paths are dropped, no
    single bad entry fails the turn. Builtins resolve via SkillStore
    regardless; a wire skill shadowing one is logged (the prompt mandates some).
    """
    mounted = os.environ.get(_SKILLS_ROOT_ENV, "").strip()
    if mounted:
        root = Path(mounted)
        root.mkdir(parents=True, exist_ok=True)
        return root

    from anton.core.memory import skills as skills_memory
    from anton.core.tools.skill_format import validate_name

    dest = Path(tempfile.mkdtemp(prefix=_SKILLS_DIR_PREFIX))

    entries = skills if isinstance(skills, dict) else {}
    for slug, entry in entries.items():
        try:
            validate_name(str(slug))
        except ValueError:
            logger.warning("skills: dropping invalid slug %r", slug)
            continue
        if (skills_memory._BUILTIN_SKILLS_ROOT / str(slug)).is_dir():
            logger.warning("skills: %r overrides the packaged builtin skill", slug)
        files = entry.get("files") if isinstance(entry, dict) else None
        if not isinstance(files, dict):
            continue
        skill_dir = dest / str(slug)
        for rel, text in files.items():
            if not isinstance(rel, str) or not isinstance(text, str):
                continue
            path = _safe_skill_file(skill_dir, rel)
            if path is None:
                logger.warning("skills: dropping unsafe path %r in %r", rel, slug)
                continue
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            except (OSError, ValueError):
                # OSError: "a" staged as a file, then "a/b". ValueError: lone
                # surrogates (valid JSON, unencodable UTF-8). Drop the file, not
                # the turn; clean up the empty file write_text left behind.
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
                logger.warning("skills: could not stage %r in %r", rel, slug, exc_info=True)
    return dest


def _write_memory_slots(dest: Path, slots: dict | None) -> None:
    """Write `slots` into `dest`, clearing any this turn didn't send — the dir is
    reused across turns, so a server-side deletion must not linger."""
    dest.mkdir(parents=True, exist_ok=True)
    values = slots if isinstance(slots, dict) else {}
    for slot, filename in _MEMORY_SLOT_FILES.items():
        path = dest / filename
        text = values.get(slot)
        if isinstance(text, str) and text.strip():
            try:
                path.write_text(text, encoding="utf-8")
            except ValueError:
                # Lone surrogates (valid JSON, unencodable UTF-8). Skip the
                # slot, not the turn; clean up the empty file write_text left.
                path.unlink(missing_ok=True)
                logger.warning("memory: slot %r is not UTF-8-encodable; skipped", slot)
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


def _global_memory_dir(memory: dict | None) -> Path:
    """Directory Hippocampus reads global slots from.

    With the org+user tree mounted (_MEMORY_GLOBAL_ROOT_ENV set), this is that
    mount, read live; the wire payload's `global` block is ignored. Without a
    mount (desktop, CI), the per-turn tmp staging dir is written from the wire,
    as before.
    """
    mounted = os.environ.get(_MEMORY_GLOBAL_ROOT_ENV, "").strip()
    if mounted:
        root = Path(mounted)
        root.mkdir(parents=True, exist_ok=True)
        return root

    global_dir = _MEMORY_DIR / "global"
    _write_memory_slots(global_dir, (memory or {}).get("global"))
    return global_dir


def _build_cortex(memory: dict | None, llm_client):
    """Cortex over cowork's memory. Built even for an empty store, or an org could
    never record its first memory: core only registers `memorize` with a cortex."""
    from anton.core.memory.hippocampus import Hippocampus

    memory = memory if isinstance(memory, dict) else {}

    global_dir = _global_memory_dir(memory)
    project_dir = _MEMORY_DIR / "project"
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
    # Mounted (_SKILLS_ROOT_ENV set): the organization's own live skills tree,
    # owned by cowork-server, read directly. Unmounted (desktop, CI): the
    # request's skills are staged outside the workspace instead, since that PVC
    # outlives the turn and cells could read anything left under it. Builtins
    # resolve via SkillStore's package root regardless.
    settings.skills_root = _stage_skills(request.skills)

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
