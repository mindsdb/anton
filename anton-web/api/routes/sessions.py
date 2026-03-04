from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api")


def _get_manager():
    from api.main import session_manager

    return session_manager


@router.get("/sessions")
async def list_sessions():
    return _get_manager().list_sessions()
