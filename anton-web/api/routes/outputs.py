from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, Response

router = APIRouter(prefix="/api")

_MEDIA_TYPES = {
    ".html": "text/html",
    ".htm": "text/html",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".csv": "text/csv",
    ".json": "application/json",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def _get_workspace() -> Path:
    from api.main import WORKSPACE_PATH

    return WORKSPACE_PATH


@router.get("/outputs/{path:path}")
async def serve_output(path: str):
    workspace = _get_workspace()
    file_path = workspace / path
    if not file_path.is_file():
        return Response(status_code=404, content="Not found")
    media_type = _MEDIA_TYPES.get(file_path.suffix.lower(), "application/octet-stream")
    return FileResponse(file_path, media_type=media_type)
