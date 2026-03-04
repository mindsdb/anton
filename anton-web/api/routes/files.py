from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

router = APIRouter(prefix="/api")


def _get_workspace() -> Path:
    from api.main import WORKSPACE_PATH

    return WORKSPACE_PATH


@router.post("/files")
async def upload_file(file: UploadFile = File(...)):
    workspace = _get_workspace()
    uploads_dir = workspace / ".anton" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    file_id = uuid.uuid4().hex[:8]
    ext = Path(file.filename).suffix if file.filename else ""
    dest = uploads_dir / f"{file_id}{ext}"

    content = await file.read()
    dest.write_bytes(content)

    image_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
    file_type = "image" if ext.lower() in image_exts else "file"

    return {
        "id": file_id,
        "name": file.filename,
        "path": str(dest),
        "type": file_type,
    }
