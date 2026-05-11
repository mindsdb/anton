"""
Artifacts — outputs Anton produces, surfaced to the user.

Each artifact is a folder under `<project>/artifacts/<slug>/`. The
folder owns its `metadata.json` (Pydantic-validated source of truth)
and `README.md` (rendered from metadata). Multi-file outputs (HTML +
CSS + JS, app + dataset) cluster together; single-file outputs
(document, image) live in their own folder anyway so provenance can
attach.

This module:
  - Lists every artifact across registered projects, newest first.
  - Resolves a request path (relative slug-anchored or absolute) to
    a real file on disk for opening / previewing.
  - Mounts an HTML artifact's parent dir as a token-keyed asset URL
    so the in-app iframe preview can resolve relative `<script>` /
    `<link>` references the same way a browser would.
"""
from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..anton_api import projects_store

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory registry mapping a short, deterministic token to the parent
# directory of an HTML artifact's primary file. Used by the in-app
# iframe preview so relative `<script src=…>` / `<link href=…>`
# references can resolve against a real URL (srcdoc has no base URL
# → relative refs 404). Token is sha256(parent_dir)[:16] — stable
# across calls so reopens don't allocate fresh entries.
_PREVIEW_MOUNTS: dict[str, Path] = {}


# ─── Type / kind mapping ───────────────────────────────────────────────────
#
# Stored `type` values (from `Artifact.type`) map onto display kinds
# used by the renderer's card chrome. Kept in sync with the closed
# enum in anton-core's `core/artifacts/models.py`.

ARTIFACT_TYPES = {
    "html-app",
    "document",
    "dataset",
    "image",
    "mixed",
    "fullstack-stateless-app",
    "fullstack-stateful-app",
}

KIND_BY_TYPE = {
    "html-app": "Dashboard",
    "document": "Document",
    "dataset": "Data",
    "image": "Image",
    "mixed": "Bundle",
    "fullstack-stateless-app": "App",
    "fullstack-stateful-app": "App",
}

# Fallback kind by extension when an artifact has no metadata yet
# (shouldn't happen with the new flow, but tolerated for resilience).
KIND_BY_EXT = {
    ".html": "Dashboard",
    ".md": "Document",
    ".txt": "Document",
    ".pdf": "Document",
    ".csv": "Data",
    ".json": "Data",
    ".png": "Image",
    ".jpg": "Image",
    ".jpeg": "Image",
    ".svg": "Image",
}

BG_CYCLE = [
    "linear-gradient(135deg, var(--stone-100), var(--surface-03))",
    "linear-gradient(135deg, var(--ocean-50), #fff)",
    "linear-gradient(135deg, var(--sage-50), #fff)",
    "linear-gradient(135deg, #fff, var(--stone-150))",
]

# Files under each artifact folder that are housekeeping rather than
# user-content. They're listed in metadata for the renderer but not
# considered when picking the "primary" file to open.
_HOUSEKEEPING_FILES = {"metadata.json", "README.md", ".published.json"}

# Extensions we'll preview as text in the artifact viewer.
TEXT_EXTENSIONS = {
    ".html", ".md", ".txt", ".csv", ".json", ".py", ".js",
    ".ts", ".tsx", ".css", ".log",
}


# ─── Helpers ───────────────────────────────────────────────────────────────


def _human_mtime(path: Path) -> str:
    secs = time.time() - path.stat().st_mtime
    if secs < 60:    return "updated just now"
    if secs < 3600:  return f"updated {int(secs // 60)}m ago"
    if secs < 86400: return f"updated {int(secs // 3600)}h ago"
    return f"updated {int(secs // 86400)}d ago"


def _scan_artifact_dirs() -> list[Path]:
    """Every registered project's `<base>/artifacts/` dir that exists.

    The legacy `.anton/output/` flat dump is intentionally NOT
    scanned anymore — the renamed model demands per-folder metadata
    and old files have neither. Users migrate by moving their files
    into a proper `artifacts/<slug>/` subfolder; until they do, the
    files just stay where they are and stop showing up here.
    """
    dirs: dict[str, Path] = {}
    for project in projects_store.list_projects():
        candidate = Path(project["path"]) / "artifacts"
        if candidate.is_dir():
            dirs[str(candidate.resolve())] = candidate
    return list(dirs.values())


def _iter_artifact_folders() -> Iterator[Path]:
    """Yield every direct subfolder of every project's artifacts dir.

    Only folders containing a readable `metadata.json` are passed
    through to callers; bare folders are skipped (incomplete writes,
    or user-stashed dirs the agent hasn't claimed).
    """
    for root in _scan_artifact_dirs():
        try:
            for child in sorted(root.iterdir()):
                if not child.is_dir():
                    continue
                if not (child / "metadata.json").is_file():
                    continue
                yield child
        except OSError:
            continue


def _load_metadata(folder: Path) -> dict | None:
    path = folder / "metadata.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Skipping artifact with unreadable metadata: %s", path, exc_info=True)
        return None


def _user_files(folder: Path) -> list[Path]:
    """All non-housekeeping files inside an artifact folder.

    Walks recursively (so `data/prices.csv` shows up alongside
    `dashboard.html`). Sorted by mtime descending so the "primary"
    pick lands on whatever was written most recently.
    """
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
    """The "open this" file for an artifact card.

    Resolution order:
      0. `metadata.primary` (when set and the file exists) — the
         agent declared this explicitly at create_artifact time, so
         it wins over the heuristic.
      1. `index.html` if present — the universal app entry point.
      2. The newest `.html` file — dashboards, reports.
      3. The newest non-housekeeping file of any type — covers
         documents, datasets, images, bundles.
    Returns None when the folder has no user-content yet (artifact
    just claimed, no writes), or when the agent's declared primary
    points at a missing file AND no fallback exists.
    """
    # Honor the agent's declared primary when the file actually
    # exists. Path-traversal guard: must resolve inside `folder`.
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


def _published_url_for(folder: Path, primary: Path | None) -> str:
    """Look up the published URL recorded for this artifact, if any.

    The publisher writes a `.published.json` map keyed by file name
    inside the artifact folder. We surface only the URL for the
    primary file so the card shows a single Published pill instead
    of one per file.
    """
    if primary is None:
        return ""
    published_index = folder / ".published.json"
    if not published_index.is_file():
        return ""
    try:
        pmap = json.loads(published_index.read_text(encoding="utf-8"))
        entry = pmap.get(primary.name)
        if isinstance(entry, dict):
            return entry.get("url", "") or ""
    except Exception:
        pass
    return ""


# ─── Listing ───────────────────────────────────────────────────────────────


@router.get("")
async def list_artifacts():
    """Every artifact across all projects, newest first.

    Card payload mirrors the shape the legacy listing returned, plus
    a few new fields the renderer can consume (slug, type,
    description, fileCount, folder). Older renderers that only know
    about `path` / `kind` / `updated` keep working.
    """
    cards: list[dict] = []
    for folder in _iter_artifact_folders():
        meta = _load_metadata(folder)
        if meta is None:
            continue
        files = _user_files(folder)
        primary = _pick_primary(folder, files, primary_hint=meta.get("primary"))
        primary_path = str(primary) if primary is not None else str(folder)
        primary_ext = primary.suffix.lower() if primary is not None else ""
        artifact_type = meta.get("type") or "mixed"
        kind = KIND_BY_TYPE.get(artifact_type) or KIND_BY_EXT.get(primary_ext, "File")
        is_live = False
        if primary is not None:
            try:
                is_live = (time.time() - primary.stat().st_mtime) < 300
            except OSError:
                is_live = False
        idx = len(cards) % len(BG_CYCLE)

        # Sort key — prefer the folder's own updatedAt (deterministic,
        # what the metadata advertises), falling back to the primary
        # file's mtime so very-old artifacts still order sensibly.
        sort_ts: float
        try:
            sort_ts = (folder / "metadata.json").stat().st_mtime
        except OSError:
            sort_ts = 0.0

        cards.append({
            "id": meta.get("id") or folder.name,
            "slug": meta.get("slug") or folder.name,
            "title": meta.get("name") or folder.name,
            "description": meta.get("description") or "",
            "type": artifact_type,
            "kind": kind,
            "ext": primary_ext,
            "updated": _human_mtime(folder / "metadata.json"),
            "live": is_live,
            "bg": BG_CYCLE[idx],
            "fileCount": len(files),
            "folder": str(folder),
            "path": primary_path,
            # Surfaces whether the agent declared a primary or the
            # server fell back to the heuristic — the renderer can
            # show a small "auto" hint in either direction if useful.
            "primary": meta.get("primary") or None,
            "publishedUrl": _published_url_for(folder, primary),
            "_sortTs": sort_ts,
        })

    cards.sort(key=lambda c: c["_sortTs"], reverse=True)
    for c in cards:
        c.pop("_sortTs", None)
    # Cap at 80 — same order-of-magnitude as the previous 40 cap on
    # flat files but a touch higher since each artifact is denser.
    return cards[:80]


# ─── Path resolution ───────────────────────────────────────────────────────


def _candidate_relative_artifacts(raw_path: str) -> list[Path]:
    """Resolve a relative path against every project's artifacts/ dir.

    Accepts any of:
      - `<slug>/dashboard.html`        → matches `<base>/artifacts/<slug>/dashboard.html`
      - `artifacts/<slug>/dashboard.html` (legacy callers may include the prefix)
    """
    text = (raw_path or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    parts = [p for p in text.split("/") if p]
    if not text or any(p in (".", "..") for p in parts):
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    if text.startswith("artifacts/"):
        text = text[len("artifacts/"):]
    matches: dict[str, Path] = {}
    for art_root in _scan_artifact_dirs():
        try:
            target = (art_root / text).resolve()
            target.relative_to(art_root.resolve())
        except ValueError:
            continue
        if target.is_file():
            matches[str(target)] = target
    return list(matches.values())


def _resolve_artifact_path(raw_path: str) -> Path:
    """Turn an artifact request path into an absolute path on disk.

    Accepts:
      - Absolute paths under any registered project's `artifacts/` dir.
      - Relative paths anchored at any artifact root (slug-prefixed or
        with a leading `artifacts/`).
    Path-traversal guarded; non-existent files yield 404.
    """
    # Reject null bytes, which are used in path injection attacks.
    if "\x00" in raw_path:
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    try:
        target = Path(raw_path).expanduser()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid artifact path") from exc
    if not str(target).strip():
        raise HTTPException(status_code=400, detail="Invalid artifact path")

    if target.is_absolute():
        resolved = target.resolve()
        for art_root in _scan_artifact_dirs():
            try:
                resolved.relative_to(art_root.resolve())
            except ValueError:
                continue
            if resolved.is_file():
                return resolved
        raise HTTPException(status_code=404, detail="Artifact is not in a known artifacts directory")

    matches = _candidate_relative_artifacts(raw_path)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise HTTPException(
            status_code=400,
            detail="Artifact path matches multiple project artifact roots; pass an absolute path",
        )
    raise HTTPException(status_code=404, detail="Artifact is not in a known artifacts directory")


def _reveal_in_file_manager(artifact: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(artifact)], check=False)
    elif sys.platform == "win32":
        subprocess.run(["explorer", f"/select,{artifact}"], check=False)
    else:
        subprocess.run(["xdg-open", str(artifact.parent)], check=False)


# ─── Preview / open / reveal ───────────────────────────────────────────────


@router.get("/preview")
async def preview_artifact(path: str = Query(...)):
    artifact = _resolve_artifact_path(path)
    suffix = artifact.suffix.lower()
    if suffix not in TEXT_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Preview is available for text, Markdown, code, JSON, CSV, and HTML files",
        )
    try:
        text = artifact.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not read artifact") from exc

    return {
        "path": str(artifact),
        "title": artifact.name,
        "kind": KIND_BY_EXT.get(suffix, "File"),
        "mime": mimetypes.guess_type(str(artifact))[0] or "text/plain",
        "content": text[:200_000],
        "truncated": len(text) > 200_000,
    }


# ─── Iframe preview mount ────────────────────────────────────────────────
#
# Two-step flow used by ArtifactViewer to render HTML with relative
# asset references intact:
#
#   1. POST /v1/artifacts/preview-mount {path} → register the artifact's
#      parent dir under a deterministic token, return the entry
#      filename and a relative URL the iframe should load.
#   2. GET  /v1/artifacts/preview-asset/{token}/{rel_path} → serve files
#      from that mounted dir, restricted to descendants of the parent so
#      a malicious artifact can't traverse into the rest of the disk.

class PreviewMountRequest(BaseModel):
    path: str


@router.post("/preview-mount")
async def preview_mount(req: PreviewMountRequest):
    artifact = _resolve_artifact_path(req.path)
    if artifact.suffix.lower() != ".html":
        raise HTTPException(status_code=415, detail="Preview mount is only available for HTML artifacts")
    parent = artifact.parent.resolve()
    token = hashlib.sha256(str(parent).encode("utf-8")).hexdigest()[:16]
    _PREVIEW_MOUNTS[token] = parent

    published_url = ""
    published_path = parent / ".published.json"
    if published_path.is_file():
        try:
            pmap = json.loads(published_path.read_text(encoding="utf-8"))
            entry = pmap.get(artifact.name)
            if isinstance(entry, dict):
                published_url = entry.get("url", "") or ""
        except Exception:
            published_url = ""

    return {
        "token": token,
        "entry": artifact.name,
        "relUrl": f"/artifacts/preview-asset/{token}/{artifact.name}",
        "publishedUrl": published_url,
    }


@router.get("/preview-asset/{token}/{rel_path:path}")
async def preview_asset(token: str, rel_path: str):
    parent = _PREVIEW_MOUNTS.get(token)
    if parent is None:
        raise HTTPException(status_code=404, detail="Preview mount has expired or is unknown")
    try:
        target = (parent / rel_path).resolve()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid asset path") from exc
    try:
        target.relative_to(parent)
    except ValueError:
        raise HTTPException(status_code=403, detail="Asset is outside the artifact directory")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    return FileResponse(target, media_type=media_type, headers={
        "Cache-Control": "private, max-age=300",
    })


class ArtifactAction(BaseModel):
    path: str


@router.post("/open")
async def open_artifact(req: ArtifactAction):
    artifact = _resolve_artifact_path(req.path)
    try:
        subprocess.run(["open", str(artifact)], check=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not open artifact") from exc
    return {"status": "ok", "path": str(artifact)}


@router.post("/reveal")
async def reveal_artifact(req: ArtifactAction):
    artifact = _resolve_artifact_path(req.path)
    try:
        _reveal_in_file_manager(artifact)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not reveal artifact") from exc
    return {"status": "ok", "path": str(artifact)}
