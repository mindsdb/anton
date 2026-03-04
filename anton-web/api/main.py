from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure the anton package is importable (repo root is parent of anton-web/)
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.session_manager import SessionManager

WORKSPACE_PATH = Path.home() / ".anton-web"

session_manager = SessionManager(WORKSPACE_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await session_manager.initialize()
    yield
    await session_manager.shutdown()


app = FastAPI(title="Anton Web API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.routes import chat, files, outputs, sessions  # noqa: E402

app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(outputs.router)
app.include_router(files.router)
