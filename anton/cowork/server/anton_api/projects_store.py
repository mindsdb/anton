"""Projects store — folder-as-id workspaces under a single common directory.

Mirrors the original antontron model in src/main/index.ts:
  <projects_dir>/<name>/.anton/...   one folder per project
  <projects_dir>/../state.json       { "activeProject": "<name>" }

The projects dir is taken from ANTON_PROJECTS_DIR (set by Electron to
app.getPath('userData')/projects), with ~/.antontron/projects as the fallback
for standalone server runs.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import TypedDict


logger = logging.getLogger(__name__)


# `general` is the system baseline — the project that's auto-created
# on first launch AND the orphan-fallback the UI routes unassigned
# tasks to. One name, one role.
#
# Pre-existing installs may still have a `default/` directory from
# the era when antontron auto-provisioned it. Those keep working
# untouched: the directory is treated as a regular user project,
# the user can rename or delete it like anything else.
GENERAL_PROJECT = "general"

# Folder name policy. Whitelist letters/digits/dot/underscore/hyphen so a
# single sanitizer is safe on macOS, Linux, and Windows. Anything outside
# the whitelist is collapsed to a single hyphen; runs are deduped and
# leading/trailing punctuation is stripped so we never emit names like
# `..` or `-foo`.
_NAME_DISALLOWED = re.compile(r"[^A-Za-z0-9._-]+")
_NAME_HYPHEN_RUNS = re.compile(r"-{2,}")
# Windows reserved device names (case-insensitive). A folder named CON,
# PRN, NUL, COM1, etc. is rejected by NTFS even with no extension.
_WIN_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}
# Cap below the per-component limit so we always have room to append a
# `-NN` collision suffix without crossing the 64-char practical ceiling
# we want for clean URLs and ls output.
_NAME_MAX_LEN = 48
_NAME_FALLBACK = "untitled-project"


class Project(TypedDict):
    name: str
    path: str


def projects_dir() -> Path:
    env = os.environ.get("ANTON_PROJECTS_DIR")
    base = Path(env).expanduser() if env else Path.home() / ".antontron" / "projects"
    return base.resolve()


def _state_path() -> Path:
    return projects_dir().parent / "state.json"


def project_path(name: str) -> Path:
    return projects_dir() / name


def sanitize_name(name: str) -> str:
    """Map any user input to a folder-safe project name.

    Always returns a non-empty string. Strange characters (`Hello, World!`,
    `proj/2025`, emoji, etc.) collapse to single hyphens. Windows-reserved
    names get an `-x` tail so NTFS will accept them. Names that sanitize
    to nothing fall back to `untitled-project`.
    """
    raw = (name or "").strip()
    cleaned = _NAME_DISALLOWED.sub("-", raw)
    cleaned = _NAME_HYPHEN_RUNS.sub("-", cleaned)
    cleaned = cleaned.strip("-._")
    if len(cleaned) > _NAME_MAX_LEN:
        cleaned = cleaned[:_NAME_MAX_LEN].rstrip("-._")
    if not cleaned:
        cleaned = _NAME_FALLBACK
    if cleaned.lower() in _WIN_RESERVED:
        cleaned = f"{cleaned}-x"
    return cleaned


def unique_name(base: str, *, exclude: str | None = None) -> str:
    """Find a non-colliding folder name by appending `-2`, `-3`, …

    `exclude` lets a rename keep its current folder name without treating
    it as a collision (so renaming `foo` to `foo` is a no-op rather than
    `foo-2`). Existence is checked against the actual filesystem, so this
    honours macOS/Windows case-insensitivity automatically.
    """
    ensure_projects_dir()
    if base != exclude and not project_path(base).exists():
        return base
    if base == exclude:
        return base
    i = 2
    while True:
        candidate = f"{base}-{i}"
        if candidate == exclude or not project_path(candidate).exists():
            return candidate
        i += 1


def ensure_projects_dir() -> Path:
    p = projects_dir()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _scaffold(target: Path) -> None:
    (target / ".anton").mkdir(parents=True, exist_ok=True)


def ensure_general_project() -> None:
    """Provision the baseline `general/` project on first launch.

    Idempotent: re-running on an existing install is a no-op. New
    installs land here on first boot; old installs that still have
    `default/` keep it untouched (no migration), so they end up
    with both `default/` and `general/` side by side — same as
    they have today.
    """
    ensure_projects_dir()
    general_dir = project_path(GENERAL_PROJECT)
    if not general_dir.exists():
        general_dir.mkdir(parents=True, exist_ok=True)
    _scaffold(general_dir)


def list_projects() -> list[Project]:
    ensure_projects_dir()
    out: list[Project] = []
    for child in sorted(projects_dir().iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        out.append({"name": child.name, "path": str(child)})
    return out


def create_project(name: str) -> Project:
    sanitized = sanitize_name(name)
    final_name = unique_name(sanitized)
    target = project_path(final_name)
    target.mkdir(parents=True)
    _scaffold(target)
    return {
        "name": final_name,
        "path": str(target),
        "requested": name,
        "renamed": final_name != (name or "").strip(),
    }


def rename_project(old_name: str, new_name: str) -> Project:
    # The system-baseline `general` project is the only one that
    # can't be renamed — the orphan-fallback path needs a stable
    # name. Pre-existing `default/` directories from older installs
    # are treated as regular user projects and can be renamed
    # freely.
    if old_name == GENERAL_PROJECT:
        raise ValueError("Cannot rename the General project")
    old_dir = project_path(old_name)
    if not old_dir.exists():
        raise FileNotFoundError("Project not found")
    sanitized = sanitize_name(new_name)
    final_name = unique_name(sanitized, exclude=old_name)
    new_dir = project_path(final_name)
    if old_dir != new_dir:
        old_dir.rename(new_dir)
        state = _read_state()
        if state.get("activeProject") == old_name:
            _write_state({"activeProject": final_name})
    return {
        "name": final_name,
        "path": str(new_dir),
        "requested": new_name,
        "renamed": final_name != (new_name or "").strip(),
    }


def delete_project(name: str) -> bool:
    # `general` is the system-baseline + orphan-fallback project.
    # The UI always needs a safe place to route unassigned tasks,
    # so it can't be deleted. Every other project (including any
    # pre-existing `default/` from older installs) is fair game.
    if name == GENERAL_PROJECT:
        raise ValueError("Cannot delete the General project")
    target = project_path(name)
    if not target.exists():
        return False
    shutil.rmtree(target)
    state = _read_state()
    if state.get("activeProject") == name:
        # Active project just got deleted — fall back to `general`,
        # re-provisioning it if somehow absent.
        _write_state({"activeProject": GENERAL_PROJECT})
        ensure_general_project()
    return True


def _read_state() -> dict:
    path = _state_path()
    if not path.is_file():
        return {"activeProject": GENERAL_PROJECT}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"activeProject": GENERAL_PROJECT}
        if not data.get("activeProject"):
            data["activeProject"] = GENERAL_PROJECT
        return data
    except Exception:
        return {"activeProject": GENERAL_PROJECT}


def _write_state(state: dict) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def get_active() -> str:
    name = _read_state().get("activeProject") or GENERAL_PROJECT
    if not project_path(name).exists():
        # Recovery path — the active-project pointer is dangling
        # (its directory was deleted out from under us). Fall back
        # to the system baseline and re-provision it.
        ensure_general_project()
        _write_state({"activeProject": GENERAL_PROJECT})
        return GENERAL_PROJECT
    return name


def set_active(name: str) -> str:
    if not project_path(name).exists():
        raise FileNotFoundError("Project not found")
    _write_state({"activeProject": name})
    return name


def resolve_project(name: str | None) -> tuple[str, Path]:
    """Return (name, path) for an explicit name or the active project."""
    target_name = name or get_active()
    target_path = project_path(target_name)
    if not target_path.exists():
        raise FileNotFoundError(f"Project not found: {target_name}")
    return target_name, target_path
