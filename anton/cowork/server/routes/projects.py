"""Projects — folder-as-id workspaces under a single common directory.

Mirrors the original antontron IPC handlers (src/main/index.ts):
  GET    /v1/projects                       → list
  POST   /v1/projects                       → create
  GET    /v1/projects/active                → current active project name
  PUT    /v1/projects/active                → set active project name
  PATCH  /v1/projects/{name}                → rename
  DELETE /v1/projects/{name}                → delete

Project file CRUD (files live at the project root; `anton.md` alone is
stored under `.context/` and exposed in the API as `anton.md`):
  GET    /v1/projects/{name}/files          → list files
  GET    /v1/projects/{name}/files/{path}   → read file body (text)
  PUT    /v1/projects/{name}/files/{path}   → write/replace file body
  POST   /v1/projects/{name}/files/upload   → multipart upload
  DELETE /v1/projects/{name}/files/{path}   → remove file
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from ..anton_api import conversation_manager, projects_store


router = APIRouter()
logger = logging.getLogger(__name__)


class CreateProjectRequest(BaseModel):
    name: str


class RenameProjectRequest(BaseModel):
    name: str


class SetActiveRequest(BaseModel):
    name: str


@router.get("")
async def list_projects():
    return {"projects": projects_store.list_projects()}


@router.post("")
async def create_project(req: CreateProjectRequest):
    # Sanitization always yields a usable name (falls back to
    # "untitled-project") and a `-NN` suffix is appended on collision,
    # so this endpoint never errors on naming. The response carries
    # `requested` + `renamed` so the client can tell the user when the
    # stored name differs from what they typed.
    return projects_store.create_project(req.name)


@router.get("/active")
async def get_active_project():
    return {"name": projects_store.get_active()}


@router.put("/active")
async def set_active_project(req: SetActiveRequest):
    try:
        return {"name": projects_store.set_active(req.name)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/{name}")
async def rename_project(name: str, req: RenameProjectRequest):
    try:
        result = projects_store.rename_project(name, req.name)
    except ValueError as exc:
        # Reserved for domain rules (e.g. "Cannot rename default project").
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Walk the renamed directory and rewrite every conversation's
    # `_meta.json` so the `project` field matches the new name. Without
    # this, the listing route keeps reporting tasks under the old
    # project name (which no longer exists), so the UI loses every
    # task / conversation under the renamed project.
    try:
        conversation_manager.relabel_project(result["name"])
    except Exception:
        # Don't fail the rename on relabel issues — the directory move
        # itself succeeded. Worst case the next list call shows stale
        # project labels until something else triggers a rewrite.
        logging.getLogger(__name__).debug("relabel_project failed", exc_info=True)

    return result


@router.delete("/{name}")
async def delete_project(name: str):
    try:
        deleted = projects_store.delete_project(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "deleted", "name": name}


# ── Project files (context directory) ──────────────────────────────
#
# Each project owns a `.context/` subdir for human-curated reference
# files: `anton.md` (working instructions surfaced to the LLM) plus
# whatever the user drops in (lessons, schemas, sample data, etc).
# Kept separate from `.anton/` so the runtime state and the user's
# context never collide.

CONTEXT_DIRNAME = ".context"
ANTON_INSTRUCTIONS_FILENAME = "anton.md"

# Cap text-file reads/writes at this size — we don't want the API to
# dump multi-megabyte blobs across HTTP if a user uploads a giant log.
# Multipart upload (binary) has its own larger limit.
TEXT_MAX_BYTES = 2 * 1024 * 1024  # 2 MiB


class FileWriteRequest(BaseModel):
    content: str


def _project_dir(name: str) -> Path:
    """Resolve a project name to its on-disk directory or 404 if gone."""
    try:
        _, base = projects_store.resolve_project(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return base


def _context_dir(project_name: str) -> Path:
    base = _project_dir(project_name)
    ctx = base / CONTEXT_DIRNAME
    ctx.mkdir(parents=True, exist_ok=True)
    return ctx


def _anton_md_disk_path(project_name: str) -> Path:
    return _context_dir(project_name) / ANTON_INSTRUCTIONS_FILENAME


def _safe_relpath(rel: str, base: Path) -> Path:
    """Resolve `rel` against `base` and reject any path that escapes
    the base directory (`../`, absolute paths, symlink trickery).

    Returns the absolute Path. Raises HTTPException 400 on any
    attempt to break out — defensive: in a desktop loopback service
    the threat model is limited but still worth gating since file
    APIs are easy to misuse.
    """
    if not rel:
        raise HTTPException(status_code=400, detail="path required")
    cleaned = rel.replace("\\", "/").lstrip("/")
    candidate = (base / cleaned).resolve()
    base_resolved = base.resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid path") from exc
    return candidate


def _file_meta(p: Path, base: Path) -> dict[str, Any] | None:
    """Compact descriptor for the listing response.

    Skips entries whose resolved path is not under `base` (e.g. symlinks
    that point outside the project — `rglob` + `resolve()` would otherwise
    raise ValueError in `relative_to`).
    """
    try:
        st = p.stat()
    except FileNotFoundError:
        return None
    try:
        resolved = p.resolve()
        base_resolved = base.resolve()
        rel = resolved.relative_to(base_resolved)
    except ValueError:
        return None
    return {
        "path":     str(rel).replace("\\", "/"),
        "name":     p.name,
        "size":     st.st_size,
        "modified": st.st_mtime,
        "is_dir":   p.is_dir(),
    }


@router.get("/{name}/files")
async def list_project_files(name: str):
    """List every file under the project's context dir, recursively.

    Anton.md is always included in the response — when it doesn't
    exist on disk yet we emit a synthetic entry with size=0 so the UI
    can render it as "empty, click to author" instead of having to
    branch on its absence.
    """
    base = _project_dir(name)
    files: list[dict[str, Any]] = []
    for p in sorted(base.rglob("*")):
        if p.is_dir():
            continue
        meta = _file_meta(p, base)
        if meta:
            files.append(meta)

    if not any(
        f["path"] == _anton_md_disk_path(name).relative_to(base).as_posix() for f in files
    ):
        files.insert(0, {
            "path":     _anton_md_disk_path(name).relative_to(base).as_posix(),
            "name":     ANTON_INSTRUCTIONS_FILENAME,
            "size":     0,
            "modified": None,
            "is_dir":   False,
            "synthetic": True,  # not on disk yet
        })
    else:
        # Surface anton.md first regardless of name sort order.
        files.sort(
            key=lambda f: (f["path"] != _anton_md_disk_path(name).relative_to(base).as_posix(), f["path"])
        )

    return {"files": files}


@router.get("/{name}/files/{path:path}")
async def read_project_file(name: str, path: str):
    base = _project_dir(name)
    target = _safe_relpath(path, base)
    if not target.exists():
        # `anton.md` is allowed to be requested before it's been
        # written — return an empty body so the editor can open in
        # author mode. Any other missing path is a 404.
        if path == _anton_md_disk_path(name).relative_to(base).as_posix():
            return {"path": path, "content": "", "size": 0, "modified": None}
        raise HTTPException(status_code=404, detail="File not found")
    if target.is_dir():
        raise HTTPException(status_code=400, detail="Path is a directory")
    if target.stat().st_size > TEXT_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large to read inline")
    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=415, detail="File is not valid UTF-8 text") from exc
    st = target.stat()
    return {
        "path":     path,
        "content":  content,
        "size":     st.st_size,
        "modified": st.st_mtime,
    }


@router.put("/{name}/files/{path:path}")
async def write_project_file(name: str, path: str, req: FileWriteRequest):
    base = _project_dir(name)
    target = _safe_relpath(path, base)
    if target.exists() and target.is_dir():
        raise HTTPException(status_code=400, detail="Path is a directory")
    body = req.content or ""
    if len(body.encode("utf-8")) > TEXT_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Content exceeds 2 MiB cap")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    st = target.stat()
    return {
        "path":     path,
        "size":     st.st_size,
        "modified": st.st_mtime,
    }


@router.post("/{name}/files/upload")
async def upload_project_files(
    name: str,
    files: list[UploadFile] = File(...),
):
    """Multipart upload — saves each file under the context dir using
    its original filename (sanitised). Returns a per-file result so
    the caller can tell which uploads succeeded.
    """
    base = _project_dir(name)
    results: list[dict[str, Any]] = []
    for f in files:
        if not f.filename:
            results.append({"name": "", "ok": False, "error": "filename missing"})
            continue
        # Strip directory components from the upload filename so a
        # malicious or sloppy client can't write to ../whatever.
        safe_name = os.path.basename(f.filename).strip()
        if not safe_name or safe_name.startswith("."):
            # Don't allow dotfiles via upload (would shadow .context
            # housekeeping). The user can write `.foo` via the PUT
            # endpoint with an explicit path if they really want.
            results.append({"name": f.filename, "ok": False, "error": "invalid filename"})
            continue
        target = base / safe_name
        try:
            data = await f.read()
            target.write_bytes(data)
            results.append({
                "name": safe_name,
                "ok":   True,
                "size": len(data),
            })
        except Exception as exc:
            logger.error("Failed to write file %s: %s", safe_name, exc)
            results.append({"name": safe_name, "ok": False, "error": "File write failed"})
    return {"results": results}


@router.delete("/{name}/files/{path:path}")
async def delete_project_file(name: str, path: str):
    base = _project_dir(name)
    target = _safe_relpath(path, base)
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if target.is_dir():
        raise HTTPException(status_code=400, detail="Path is a directory")
    target.unlink()
    return {"status": "deleted", "path": path}
