"""Single source of truth for publish access resolution + target resolution.

Ported verbatim (behaviour-preserving) from cowork-server so that anton's
`/publish`, anton's `publish_or_preview` tool, and cowork-server all resolve
access, versioning, and `.published.json` location the same way.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_EMAIL_SPLIT_RE = re.compile(r"[\s,;]+")

# Keep in sync with anton.publisher._FULLSTACK_EXCLUDED (publisher.py:42):
# backend.log is the running backend's runtime log — excluded from the
# published bundle there, so it must not count as user content here either.
_HOUSEKEEPING_FILES = {"metadata.json", "README.md", "backend.log", ".published.json", ".revisions"}


def normalize_emails(values) -> list[str]:
    """Strip + lowercase + de-dupe, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in values or []:
        email = str(raw).strip().lower()
        if email and email not in seen:
            seen.add(email)
            out.append(email)
    return out


def resolve_access(
    password: str | None, access: dict | None, previous: Any
) -> tuple[dict, int, int, dict]:
    """Resolve effective publish access from request + prior state.

    Returns ``(effective_access, pwd_version, access_version, owner_side)``.
    A request with no usable selection (empty password, or restricted with no
    emails, no org and no owner_only) degrades to ``public`` — a server-side
    safety net for programmatic callers; interactive callers must gate empty
    input earlier.
    NOTE: with ``access=None`` and ``password=None`` the mode defaults to
    ``public`` — the prior mode is NOT inherited (``previous`` only feeds the
    version counters). Callers that want to preserve a prior non-public mode
    must reconstruct it via ``access_from_owner_side`` first.
    """
    prev = previous if isinstance(previous, dict) else {}
    password = (password or "").strip() or None

    mode = (access or {}).get("mode") if access else None
    if not mode:
        mode = "password" if password else "public"

    prev_pwd_version = prev.get("pwd_version", 0) or 0
    prev_access_version = prev.get("access_version", 0) or 0
    pwd_version = prev_pwd_version or 1
    access_version = prev_access_version or 1

    if mode == "password":
        pw = ((access or {}).get("password") or password or "").strip() or None
        if pw:
            prev_password = prev.get("access_password")
            pwd_version = (prev_pwd_version + 1) if pw != prev_password else (prev_pwd_version or 1)
            owner_side = {
                "mode": "password",
                "requires_password": True,
                "access_password": pw,
                "pwd_version": pwd_version,
            }
            return {"mode": "password", "password": pw}, pwd_version, access_version, owner_side
        mode = "public"  # empty password → public

    if mode == "restricted":
        emails = normalize_emails((access or {}).get("emails"))
        org_allowed = bool((access or {}).get("org_allowed"))
        # The owner always matches (the FK condition in auth), so an explicit
        # owner_only next to emails/org carries no information — canonicalise it
        # away so two equivalent selections don't differ in access_version.
        owner_only = bool((access or {}).get("owner_only")) and not emails and not org_allowed
        if emails or org_allowed or owner_only:
            prev_restricted = prev.get("mode") == "restricted"
            prev_emails = prev.get("emails") if prev_restricted else None
            prev_org = prev.get("org_allowed") if prev_restricted else None
            # bool(): pre-ENG-1769 entries have no owner_only key, and a bare
            # `False != None` would bump the version on an unchanged re-publish.
            prev_owner_only = bool(prev.get("owner_only")) if prev_restricted else None
            changed = (
                (emails != prev_emails)
                or (org_allowed != prev_org)
                or (owner_only != prev_owner_only)
            )
            access_version = (prev_access_version + 1) if changed else (prev_access_version or 1)
            owner_side = {
                "mode": "restricted",
                "requires_password": False,
                "emails": emails,
                "org_allowed": org_allowed,
                "owner_only": owner_only,
                "access_version": access_version,
            }
            return (
                {"mode": "restricted", "emails": emails, "org_allowed": org_allowed},
                pwd_version,
                access_version,
                owner_side,
            )
        mode = "public"  # nothing selected → public

    return {"mode": "public"}, pwd_version, access_version, {"mode": "public", "requires_password": False}


def parse_emails(text_or_list) -> tuple[list[str], list[str]]:
    """Split + validate emails (mirrors AccessChooser.jsx parseEmailList).

    Accepts a raw string ("a@x.com, b@x.com") or a list. Returns
    ``(valid, invalid)`` — valid are stripped/lowercased/de-duped, invalid
    keep their original text (for a "Skipped invalid: ..." warning).
    """
    if isinstance(text_or_list, str):
        tokens = _EMAIL_SPLIT_RE.split(text_or_list)
    else:
        tokens = [str(t) for t in (text_or_list or [])]
    valid: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        low = tok.lower()
        if _EMAIL_RE.match(low):
            if low not in seen:
                seen.add(low)
                valid.append(low)
        else:
            invalid.append(tok)
    return valid, invalid


def access_from_owner_side(entry: Any) -> dict:
    """Rebuild the input access shape from a stored `.published.json` entry.

    Canonicalises the inline reconstruction that used to live in
    cowork-server's ``update_artifact``. Used by: /publish "keep", the tool's
    "preserve previous" default, and cowork-server's update_artifact.
    """
    entry = entry if isinstance(entry, dict) else {}
    mode = entry.get("mode") or ("password" if entry.get("requires_password") else "public")
    if mode == "password":
        return {"mode": "password", "password": entry.get("access_password", "") or ""}
    if mode == "restricted":
        return {
            "mode": "restricted",
            "emails": entry.get("emails", []) or [],
            "org_allowed": bool(entry.get("org_allowed")),
            # LOAD-BEARING: without this key a "keep" re-publish of an
            # owner-only artifact reconstructs an empty selection, which
            # degrades to public — silently un-privating the artifact.
            "owner_only": bool(entry.get("owner_only")),
        }
    return {"mode": "public"}


def _fullstack_types() -> frozenset[str]:
    """Artifact types anton's publisher bundles as fullstack apps.

    Imported lazily to avoid a publish_access ↔ publisher import cycle."""
    from anton.publisher import FULLSTACK_ARTIFACT_TYPES
    return FULLSTACK_ARTIFACT_TYPES


def _load_metadata(folder: Path) -> dict | None:
    path = folder / "metadata.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Skipping artifact with unreadable metadata: %s", path, exc_info=True)
        return None


def _user_files(folder: Path) -> list[Path]:
    """All non-housekeeping files inside an artifact folder, mtime desc."""
    out: list[Path] = []
    try:
        for p in folder.rglob("*"):
            if not p.is_file() or p.is_symlink():
                continue
            rel = p.relative_to(folder)
            top = rel.parts[0] if rel.parts else ""
            if top in _HOUSEKEEPING_FILES:
                continue
            out.append(p)
    except OSError:
        return []
    try:
        out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        pass
    return out


def _pick_primary(folder: Path, files: list[Path], primary_hint: str | None = None) -> Path | None:
    """The "open this" file for an artifact."""
    if primary_hint:
        try:
            target = (folder / primary_hint).resolve()
            target.relative_to(folder.resolve())
            if target.is_file():
                return target
        except (ValueError, OSError):
            pass
    if not files:
        return None
    index = next((f for f in files if f.name == "index.html"), None)
    if index is not None:
        return index
    html = next((f for f in files if f.suffix.lower() == ".html"), None)
    if html is not None:
        return html
    return files[0]


def _artifact_root_for(path: Path, container_dirs: list[Path]) -> Path:
    """Climb from an artifact file to the folder holding its metadata.json,
    bounded by the given container dirs (anton: [artifacts_dir]; cowork:
    _scan_artifact_dirs()). Falls back to path.parent."""
    containers = {str(Path(d).resolve()) for d in container_dirs}
    current = path.parent.resolve()
    while True:
        if (current / "metadata.json").is_file():
            return current
        if str(current) in containers or current.parent == current:
            return path.parent.resolve()
        current = current.parent


def resolve_publish_target(
    artifact: Path, container_dirs: list[Path]
) -> tuple[Path, Path, str, bool]:
    """Decide what to publish + where `.published.json` lives + its key.

    Returns (publish_target, published_dir, published_key, is_fullstack).
    - fullstack → publish the artifact *directory*; `.published.json` at root;
    - static → publish the single primary *file*; `.published.json` in its parent.
    Always keyed by the primary file name.
    """
    if artifact.is_dir():
        artifact_root = artifact
        meta = _load_metadata(artifact_root) or {}
        primary = _pick_primary(artifact_root, _user_files(artifact_root), primary_hint=meta.get("primary"))
    else:
        artifact_root = _artifact_root_for(artifact, container_dirs)
        meta = _load_metadata(artifact_root) if (artifact_root / "metadata.json").is_file() else None
        primary = artifact

    if (meta or {}).get("type") in _fullstack_types():
        key = primary.name if primary else "index.html"
        return artifact_root, artifact_root, key, True
    if primary:
        return primary, primary.parent, primary.name, False
    return artifact_root, artifact_root, "index.html", False


async def prompt_access(prompt_fn, *, previous=None, allow_keep=False) -> dict | None:
    """Interactively collect an access spec via a prompt_or_cancel-like coro.

    Returns an input access dict ({"mode": ...}) or None if the user cancels
    (Esc at any step). An empty password re-prompts rather than silently
    degrading to public; an empty restricted selection is an explicit
    owner-only publish, while malformed addresses re-prompt — parity with
    cowork's isAccessDraftValid(). ``prompt_fn`` is injected so this is
    unit-testable without a TTY.
    """
    choices = ["public", "password", "restricted"]
    if allow_keep:
        choices = ["keep"] + choices
    mode = await prompt_fn(
        "  Access",
        choices=choices,
        choices_display="/".join(choices),
        default=("keep" if allow_keep else "public"),
    )
    if mode is None:
        return None
    mode = (mode or "").strip().lower() or ("keep" if allow_keep else "public")

    if mode == "keep":
        return access_from_owner_side(previous) if previous else {"mode": "public"}
    if mode == "public":
        return {"mode": "public"}

    if mode == "password":
        while True:
            pw = await prompt_fn("  Password", password=True)
            if pw is None:
                return None
            pw = pw.strip()
            if pw:
                return {"mode": "password", "password": pw}
            # empty → re-prompt (do NOT degrade to public silently)

    if mode == "restricted":
        hint = ""
        while True:
            raw = await prompt_fn(
                f"{hint}  Allowed emails (comma or space separated; empty = only you)"
            )
            if raw is None:
                return None
            valid, invalid = parse_emails(raw)
            if invalid:
                # Re-prompt instead of dropping them: a typo'd address would
                # otherwise turn into an owner-only publish the user never asked
                # for. prompt_fn is the only output channel available here.
                hint = f"  Invalid: {', '.join(invalid)}\n"
                continue
            org_ans = await prompt_fn(
                "  Allow everyone in your organization?",
                choices=["y", "n"], choices_display="y/n", default="n",
            )
            if org_ans is None:
                return None
            org_allowed = org_ans.strip().lower() in ("y", "yes")
            return {
                "mode": "restricted",
                "emails": valid,
                "org_allowed": org_allowed,
                "owner_only": not valid and not org_allowed,
            }

    return {"mode": "public"}
