"""Attachment and URL context APIs for Anton CoWork."""

from __future__ import annotations

import mimetypes
import re
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from .cowork_state import attachments_dir, load_state, save_state, utc_now_iso


router = APIRouter(prefix="/v1/attachments", tags=["attachments"])

TEXT_LIMIT = 120_000


class SnippetAttachmentRequest(BaseModel):
    title: str = Field(default="Snippet", max_length=160)
    content: str
    language: str | None = Field(default=None, max_length=40)
    session_id: str | None = None
    project_path: str | None = None


class ProjectFileAttachmentRequest(BaseModel):
    project_path: str
    path: str
    session_id: str | None = None


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip()
    return cleaned[:140] or "attachment"


def _new_id(prefix: str = "att") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _text_preview(text: str | None) -> str:
    if not text:
        return ""
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:320]


def _truncate_text(text: str, limit: int = TEXT_LIMIT) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _store_metadata(metadata: dict) -> dict:
    state = load_state()
    state["attachments"][metadata["id"]] = metadata
    save_state(state)
    return metadata


def _make_attachment(
    *,
    kind: str,
    name: str,
    source: str,
    path: str | None = None,
    source_url: str | None = None,
    session_id: str | None = None,
    project_path: str | None = None,
    mime: str | None = None,
    size: int | None = None,
    text: str = "",
    extraction_status: str = "ready",
    truncated: bool = False,
    note: str | None = None,
    language: str | None = None,
) -> dict:
    attachment_id = _new_id()
    now = utc_now_iso()
    metadata = {
        "id": attachment_id,
        "kind": kind,
        "name": name,
        "mime": mime or mimetypes.guess_type(name)[0] or "application/octet-stream",
        "size": size or 0,
        "path": path,
        "source": source,
        "sourceUrl": source_url,
        "sessionId": session_id,
        "projectPath": project_path,
        "language": language,
        "createdAt": now,
        "updatedAt": now,
        "status": "ready" if extraction_status != "error" else "error",
        "extractionStatus": extraction_status,
        "text": text,
        "textPreview": _text_preview(text),
        "truncated": truncated,
        "note": note,
    }
    return _store_metadata(metadata)


def get_attachments(ids: list[str] | None = None) -> list[dict]:
    state = load_state()
    attachments = state.get("attachments", {})
    if ids is None:
        return sorted(attachments.values(), key=lambda item: item.get("createdAt", ""), reverse=True)
    return [attachments[item_id] for item_id in ids if item_id in attachments]


def assign_attachments(ids: list[str] | None, session_id: str) -> list[dict]:
    if not ids:
        return []
    state = load_state()
    updated: list[dict] = []
    for item_id in ids:
        metadata = state.get("attachments", {}).get(item_id)
        if not metadata:
            continue
        metadata["sessionId"] = session_id
        metadata["updatedAt"] = utc_now_iso()
        updated.append(metadata)
    save_state(state)
    return updated


def attachment_context(ids: list[str] | None) -> str:
    selected = get_attachments(ids or [])
    if not selected:
        return ""

    sections = ["Attached context supplied by the user:"]
    for item in selected:
        header_bits = [item.get("kind") or "attachment", item.get("mime") or "unknown type"]
        if item.get("sourceUrl"):
            header_bits.append(item["sourceUrl"])
        elif item.get("source"):
            header_bits.append(item["source"])
        header = f"### {item.get('name') or item['id']} ({'; '.join(header_bits)})"
        sections.append(header)
        path = f"File path: {item['path']}"
        sections.append(path)

    return "\n\n".join(sections)


@router.get("")
def list_attachments(
    session_id: str | None = Query(default=None),
    ids: list[str] | None = Query(default=None),
):
    attachments = get_attachments(ids)
    if session_id:
        attachments = [item for item in attachments if item.get("sessionId") == session_id]
    return {"attachments": attachments}


@router.post("/upload")
async def upload_attachments(
    files: list[UploadFile] = File(...),
    session_id: str | None = Form(default=None),
    project_path: str | None = Form(default=None),
):
    created: list[dict] = []
    for file in files:
        attachment_id = _new_id()
        filename = _safe_name(file.filename or "attachment")
        target_dir = attachments_dir() / attachment_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / filename
        data = await file.read()
        target.write_bytes(data)
        mime = file.content_type or mimetypes.guess_type(filename)[0]

        metadata = {
            "id": attachment_id,
            "kind": "file",
            "name": filename,
            "mime": mime or "application/octet-stream",
            "size": len(data),
            "path": str(target),
            "sessionId": session_id,
            "projectPath": project_path,
            "createdAt": utc_now_iso(),
            "updatedAt": utc_now_iso(),
        }
        _store_metadata(metadata)
        created.append(metadata)
    return {"attachments": created}


@router.post("/snippet")
def create_snippet(request: SnippetAttachmentRequest):
    text, truncated = _truncate_text(request.content)
    metadata = _make_attachment(
        kind="snippet",
        name=request.title or "Snippet",
        source="snippet",
        session_id=request.session_id,
        project_path=request.project_path,
        mime="text/plain",
        size=len(request.content.encode("utf-8")),
        text=text,
        extraction_status="ready",
        truncated=truncated,
        language=request.language,
    )
    return {"attachment": metadata}


@router.delete("/{attachment_id}")
def delete_attachment(attachment_id: str):
    state = load_state()
    metadata = state.get("attachments", {}).pop(attachment_id, None)
    if not metadata:
        raise HTTPException(status_code=404, detail="Attachment not found.")
    save_state(state)
    path = metadata.get("path")
    if path:
        parent = Path(path).parent
        if parent.name == attachment_id:
            shutil.rmtree(parent, ignore_errors=True)
    return {"ok": True}
