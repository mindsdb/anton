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
import hashlib
import logging
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from anton.cloud_turn.contract import TurnRequestV1
from anton.core.tools.skill_format import SKILL_FILE

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
        # Safe to list only because `skill_drafts_root` is always set below —
        # core registers the tool with one, and an allowlist name matching no
        # tool is fatal. Writes only into this conversation's drafts dir; the
        # tenant's skill store is never touched from the pod.
        "create_skill_draft",
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


#: Per-file cap on a reported draft. Skills are small text, so the cap only
#: catches a mispackaged blob before it reaches the reply stream. An oversized
#: SKILL.md drops the whole draft: a truncated procedure is worse than none.
_DRAFT_FILE_MAX = 200_000
#: Drafts reported per turn. A prompt-confused agent looping over
#: `create_skill_draft` must not flood the wire; the excess is logged, not silent.
_MAX_DRAFTS_PER_TURN = 20
#: Whole-draft budget. The per-file cap alone bounds nothing: a skill may carry
#: any number of siblings, so 20 drafts x N files x 200 KB is unbounded on the
#: reply stream. SKILL.md is read first and always fits, so exhausting the budget
#: costs siblings, never the procedure itself.
_DRAFT_TOTAL_MAX = 1_000_000


def _draft_files(folder: Path) -> dict[str, str] | None:
    """One draft folder as ``{filename: text}``, or None to skip it.

    Top-level text files only, mirroring the desktop draft card (skills are flat
    text). The drafts path is predictable and cell code can write anywhere in
    the workspace, so symlinks and anything resolving outside `folder` go.
    """
    if folder.is_symlink() or not (folder / SKILL_FILE).is_file():
        return None
    resolved = folder.resolve()
    files: dict[str, str] = {}
    budget = _DRAFT_TOTAL_MAX
    # SKILL.md first so the procedure is never the thing the budget squeezes out,
    # then by name — sorting on the flag alone is stable, so siblings would keep
    # readdir order and the budget would drop different files on different hosts.
    for child in sorted(folder.iterdir(), key=lambda p: (p.name != SKILL_FILE, p.name)):
        if child.is_symlink() or not child.is_file():
            continue
        if child.resolve().parent != resolved:
            logger.warning("skill drafts: %r skipping out-of-tree file %r", folder.name, child.name)
            continue
        try:
            text = child.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            logger.warning("skill drafts: %r skipping unreadable file %r", folder.name, child.name)
            continue
        size = len(text.encode("utf-8", "replace"))
        if size > _DRAFT_FILE_MAX:
            if child.name == SKILL_FILE:
                logger.warning("skill drafts: dropping %r (SKILL.md over %d bytes)",
                               folder.name, _DRAFT_FILE_MAX)
                return None
            logger.warning("skill drafts: %r skipping oversized file %r", folder.name, child.name)
            continue
        if size > budget:
            logger.warning("skill drafts: %r over the %d-byte draft budget, dropping %r "
                           "and every later sibling", folder.name, _DRAFT_TOTAL_MAX, child.name)
            break
        budget -= size
        files[child.name] = text
    # A draft is its SKILL.md — siblings alone are a folder, not a procedure.
    # Checked on the collected files, not on disk: SKILL.md can exist and still
    # be dropped above for being oversized.
    return files if SKILL_FILE in files else None


def _snapshot_skill_drafts(root: Path) -> dict[str, str]:
    """``slug -> content hash`` for every staged draft, for the end-of-turn diff.

    Keyed on content, not just the folder name: drafts live on the workspace and
    outlive the turn, so a draft refined in place has to re-report and not only
    one appearing for the first time. Hashes every top-level file, so a
    sibling-only edit counts as a change too.
    """
    if not root.is_dir():
        return {}
    snapshot: dict[str, str] = {}
    for child in sorted(root.iterdir()):
        if child.is_symlink() or not child.is_dir():
            continue
        digest = hashlib.sha256()
        for f in sorted(child.iterdir()):
            if f.is_symlink() or not f.is_file():
                continue
            # Length-prefixed: concatenating name and body unseparated makes
            # {"ab": "c"} and {"a": "bc"} hash the same, so a rename plus a
            # compensating edit would read as "unchanged".
            name = f.name.encode("utf-8", "replace")
            digest.update(f"{len(name)}:".encode())
            digest.update(name)
            with contextlib.suppress(OSError):
                body = f.read_bytes()
                digest.update(f"{len(body)}:".encode())
                digest.update(body)
        snapshot[child.name] = digest.hexdigest()
    return snapshot


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


def drain_pending_skills(session) -> list[dict]:
    """Skill drafts this turn created or changed, as ``[{"slug", "files"}]``.

    Raw file text rather than a parsed skill: cowork is the trust boundary and
    re-validates slugs and paths. Sizes are bounded here, at the producer, since
    the reply stream is what an unbounded payload would flood.

    The baseline advances only for drafts actually returned, so a second call
    reports nothing (the folder survives the turn, and re-reporting an untouched
    draft would re-raise a card the user already dismissed) while anything the
    caps held back stays pending and goes out next turn instead of being
    silently dropped forever. Best-effort: a lost draft must not cost the turn
    its reply.
    """
    root = getattr(session, "_skill_drafts_root", None)
    if not root:
        return []
    before = getattr(session, "_skill_drafts_before", None) or {}
    try:
        after = _snapshot_skill_drafts(Path(root))
    except OSError:
        logger.warning("skill drafts: could not diff the staging dir", exc_info=True)
        return []
    changed = sorted(slug for slug, digest in after.items() if before.get(slug) != digest)
    this_turn = changed[:_MAX_DRAFTS_PER_TURN]
    if len(changed) > len(this_turn):
        logger.warning("skill drafts: %d changed, reporting %d this turn, the rest next",
                       len(changed), len(this_turn))

    entries: list[dict] = []
    delivered: dict[str, str] = {}
    for slug in this_turn:
        try:
            files = _draft_files(Path(root) / slug)
        except OSError:
            logger.warning("skill drafts: could not read %r", slug, exc_info=True)
            continue
        if files is not None:
            entries.append({"slug": slug, "files": files})
            delivered[slug] = after[slug]

    # Unchanged drafts keep their hash; changed ones advance ONLY if they went
    # out. A draft held back by the per-turn cap, or unreadable this time, keeps
    # its old hash and so is still "changed" next turn.
    session._skill_drafts_before = {
        slug: digest for slug, digest in after.items() if slug not in changed
    } | delivered
    return entries


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


#: Attachments cowork-server stages into the workspace for this conversation.
_ATTACHMENTS_DIRNAME = "attachments"


def _sniff_image_format(head: bytes) -> str | None:
    """The real image format from magic bytes, or None if not a known image.
    Content, never the extension — a BMP named ``.png`` must not ship as PNG,
    and a text file named ``.png`` must not ship at all."""
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if head.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "GIF"
    if head.startswith(b"BM"):
        return "BMP"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "WEBP"
    return None


def _image_block_from_file(path: Path) -> dict | None:
    """A base64 image content block for *path*, or None if it can't be shown.

    Format is detected from magic bytes (not the extension: a mislabeled or
    non-image file must not be shipped and rejected by the model, which would
    fail every turn). Size is capped BEFORE the file is read, so a huge upload
    can't OOM the pod. Pillow is optional in the deployed image, so it's used
    only to convert BMP (via the shared ``clipboard.image_content_block``); a
    BMP where Pillow is absent is skipped rather than crashing."""
    from anton.utils.clipboard import MAX_IMAGE_BYTES, image_content_block

    try:
        if path.stat().st_size * 4 // 3 > MAX_IMAGE_BYTES:  # cap before reading into memory
            return None
        raw = path.read_bytes()
    except OSError:
        return None
    fmt = _sniff_image_format(raw[:16])
    if fmt is None:
        return None
    try:
        return image_content_block(raw, fmt, oversize="skip")
    except Exception:  # e.g. BMP with Pillow absent, or a decode failure
        return None


def build_turn_content(base: Path, user_text: str) -> "str | list[dict]":
    """Turn input augmented with the conversation's attachments.

    cowork-server stages uploads into ``<workspace>/attachments/`` on the shared
    mount (they never cross the stdin wire). Returns the plain string when there
    are none — unchanged behaviour. Otherwise a multimodal user message: an
    image block per image (so the model can actually see it) plus a text block
    that lists every attachment's path and carries the user's text. Never
    raises: a bad attachment is skipped, the turn still runs.
    """
    attach_dir = base / _ATTACHMENTS_DIRNAME
    try:
        files = sorted(p for p in attach_dir.rglob("*") if p.is_file()) if attach_dir.is_dir() else []
    except OSError:
        files = []
    if not files:
        return user_text

    from anton.clipboard import is_image_path

    image_blocks: list[dict] = []
    lines: list[str] = []
    for f in files:
        # One bad attachment must never sink the whole set — degrade it to a
        # path listing line so the docstring's "never raises" actually holds.
        try:
            if is_image_path(f.name):
                block = _image_block_from_file(f)
                if block is not None:
                    image_blocks.append(block)
                    lines.append(f"  - {f.name} (image, shown above)")
                else:
                    lines.append(f"  - {f} (image could not be shown inline; read from this path if needed)")
            else:
                lines.append(f"  - {f}")
        except Exception:
            logger.warning("cloud turn: skipping unreadable attachment %s", f, exc_info=True)

    listing = (
        "The user attached the following files to this conversation. Any images "
        "are included in this message; read other files from their absolute paths:\n"
        + "\n".join(lines)
    )
    text = f"{listing}\n\n{user_text}" if user_text.strip() else listing
    return [*image_blocks, {"type": "text", "text": text}]


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
    # Skills the agent builds stage here instead of the store, and the entrypoint
    # reports them for cowork to surface. ON the workspace, unlike everything
    # staged above — editing a skill spans turns, and the workspace PVC is the
    # only per-conversation storage that outlives the pod. Same layout as the
    # desktop harness (`<project>/.anton/skill_drafts`).
    settings.skill_drafts_root = base / settings.memory_dir / "skill_drafts"
    settings.skill_drafts_root.mkdir(parents=True, exist_ok=True)

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
        # WHERE the user was, which this pod cannot know on its own — only the
        # deployment does, so cowork sends it (ENG-1459). Absent when the pod is
        # driven directly rather than by cowork; an unset surface reads as
        # "nobody declared one", which is the honest answer for a standalone run.
        surface=(request.trace or {}).get("surface"),
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
    # Baseline for `drain_pending_skills`, taken before the turn runs. Drafts
    # from earlier turns are already on the workspace, so without this every
    # turn would re-report all of them.
    session._skill_drafts_before = _snapshot_skill_drafts(settings.skill_drafts_root)
    logger.info(
        "cloud session built conversation=%s workspace=%s tools=%s",
        request.conversation_id, base, sorted(CLOUD_TOOL_ALLOWLIST),
    )
    return session
