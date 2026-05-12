"""
Anton CoWork — FastAPI backend server.

Runs on http://127.0.0.1:8765
Wraps Anton's Python API and exposes /v1/* REST + SSE endpoints.

Usage:
  python main.py
  # or: uvicorn main:app --host 127.0.0.1 --port 8765 --reload
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Load ~/.anton/.env if it exists (before anything else)
_env_path = Path.home() / ".anton" / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))


def _maybe_self_update_and_reexec() -> None:
    """Run anton's self-update flow before the server boots.

    Mirrors what `anton/cli.py` does at the top of its root command,
    so antontron-launched servers stay current the same way the CLI
    does. The Node side previously duplicated this logic in
    TypeScript; the Python check is the single source of truth now.

    Behaviour:
      * `_ANTON_UPDATED=1` env var → already updated this process
        tree, skip (matches `anton/updater.py` loop guard).
      * `disable_autoupdates` setting → user opted out, skip.
      * Otherwise call `anton.updater.check_and_update`. If it
        installed a newer version, set the loop-guard env var and
        `os.execv` the same python interpreter in module mode.
        The kernel replaces the current process in-place — same PID,
        same stdio pipes — so antontron's `server-process.ts`
        doesn't need to know an update happened. The 15s /health
        probe absorbs the extra cold-start time.

    Errors here are non-fatal: any exception drops us through to
    starting the server on the existing version. Better to be one
    release behind than fail to boot because GitHub was down.
    """
    if os.environ.get("_ANTON_UPDATED") == "1":
        return
    try:
        from anton.config.settings import AntonSettings
        from anton.updater import check_and_update
    except Exception:
        # Anton not importable yet (first launch race, broken venv) —
        # nothing for us to update against. Let the rest of main.py
        # raise the proper diagnostic.
        return
    try:
        settings = AntonSettings()
    except Exception:
        return
    if getattr(settings, "disable_autoupdates", False):
        return

    class _NullConsole:
        """Stand-in for rich.Console — `check_and_update` only calls
        `console.print(msg)` to show status, which we capture via
        stdout instead so antontron's log buffer picks it up."""
        def print(self, *args, **kwargs):
            try:
                print(*args)
            except Exception:
                pass

    try:
        if check_and_update(_NullConsole(), settings, extras=["cowork-server"]):
            os.environ["_ANTON_UPDATED"] = "1"
            # Re-exec through module mode explicitly. When launched as
            # `python -m anton.cowork.server`, sys.argv cannot be replayed
            # blindly to preserve `-m` semantics.
            os.execv(sys.executable, [sys.executable, "-m", "anton.cowork.server"])
    except Exception:
        # Updater raised — keep going on the existing version.
        return

from fastapi import FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .anton_api import conversation_manager, projects_store, scratchpad_runtime
from .routes.responses import router as responses_router
from .routes.conversations import router as conversations_router
from .routes.scratchpad import router as scratchpad_router
from .routes.projects import router as projects_router
from .routes.settings import router as settings_router
from .routes.settings import get_config_status
from .routes.artifacts import router as artifacts_router
from .routes.utilities import router as utilities_router
from .routes.attachments import router as attachments_router
from .routes.search import router as search_router
from .routes.pins import router as pins_router
from .routes.schedules import router as schedules_router, start_scheduler
from .routes.browse import router as browse_router
from .routes.integrations import router as integrations_router, refresh_google_oauth_tokens
from .routes.datavault import router as datavault_router
from .routes.dispatch import (
    router as dispatch_router,
    close_repo as close_dispatch_repo,
    start_dispatch,
    stop_dispatch,
)
from .routes.dispatch_slack import router as dispatch_slack_router
from .routes.connectors import router as connectors_router
from . import COWORK_SERVER_PROTOCOL_VERSION, COWORK_SERVER_VERSION

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger("anton-server")

# When ANTON_SERVE_SPA=1, serve the cowork web SPA at / from ANTON_SPA_DIR.
# Used by the Docker image (cowork:web) — the same FastAPI process answers
# both /v1/* (API) and / (SPA) on one port. Unset by default so the
# Electron path is byte-identical to before.
ANTON_SERVE_SPA = os.environ.get("ANTON_SERVE_SPA", "").lower() in ("1", "true", "yes")
_spa_dir_env = os.environ.get("ANTON_SPA_DIR", "/app/dist/renderer-web")
SPA_DIR: Path | None = Path(_spa_dir_env).resolve() if ANTON_SERVE_SPA else None
if ANTON_SERVE_SPA and (SPA_DIR is None or not SPA_DIR.exists()):
    logger.warning(
        "ANTON_SERVE_SPA=1 but SPA bundle not found at %s; SPA serving disabled.",
        _spa_dir_env,
    )
    ANTON_SERVE_SPA = False
    SPA_DIR = None

    
async def _google_token_refresh_loop() -> None:
    await asyncio.sleep(60)  # let the vault settle after startup
    while True:
        try:
            await asyncio.to_thread(refresh_google_oauth_tokens)
        except Exception:
            pass
        await asyncio.sleep(30 * 60)  # every 30 minutes


@asynccontextmanager
async def lifespan(app: FastAPI):
    projects_store.ensure_general_project()
    start_scheduler()
    await start_dispatch()
    refresh_task = asyncio.create_task(_google_token_refresh_loop())
    yield
    refresh_task.cancel()
    try:
        await refresh_task
    except asyncio.CancelledError:
        pass
    await stop_dispatch()
    await close_dispatch_repo()
    await conversation_manager.close_all()
    await scratchpad_runtime.close_all()


app = FastAPI(title="Anton CoWork API", version="1.0.0", lifespan=lifespan)

_renderer_url = os.environ.get("VITE_RENDERER_URL", "").rstrip("/")
_allow_origins = ["http://localhost:5173", "http://127.0.0.1:5173", "app://-", "null"]
if _renderer_url and _renderer_url not in _allow_origins:
    _allow_origins.append(_renderer_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chat layer (OpenAI-style)
app.include_router(responses_router)
app.include_router(conversations_router)
app.include_router(scratchpad_router)

# Cowork resources — all under /v1/*
app.include_router(projects_router, prefix="/v1/projects",  tags=["projects"])
app.include_router(settings_router, prefix="/v1/settings",  tags=["settings"])
app.include_router(artifacts_router, prefix="/v1/artifacts", tags=["artifacts"])
# utilities.py exposes /memory, /skills, /datasources, /publish at the prefix root
app.include_router(utilities_router, prefix="/v1", tags=["utilities"])
app.include_router(attachments_router)
app.include_router(search_router)
app.include_router(pins_router)
app.include_router(schedules_router)
app.include_router(browse_router)
app.include_router(integrations_router, prefix="/v1/integrations", tags=["integrations"])
app.include_router(datavault_router, prefix="/v1/datavault", tags=["datavault"])
app.include_router(dispatch_router, prefix="/v1/dispatch", tags=["dispatch"])
app.include_router(dispatch_slack_router, prefix="/v1/dispatch", tags=["dispatch-slack"])
# Predefined connector registry — server/connectors/*.json
app.include_router(connectors_router)


@app.get("/health")
async def health():
    config = get_config_status()
    anton_available = conversation_manager.is_anton_available()
    return {
        "status": "ok",
        "anton_version": COWORK_SERVER_VERSION,
        "cowork_server_version": COWORK_SERVER_VERSION,
        "cowork_server_protocol_version": COWORK_SERVER_PROTOCOL_VERSION,
        "anton_available": anton_available,
        "mode": "anton" if anton_available else "demo",
        "config_ready": config["config_ready"],
        "config_error": config["config_error"],
        "provider": config["provider"],
        "model": config["model"],
        "provider_label": config["provider_label"],
        "live_conversations": conversation_manager.list_live(),
        "live_pads": scratchpad_runtime.list_pads(),
    }


@app.get("/")
async def root():
    if ANTON_SERVE_SPA and SPA_DIR is not None:
        return FileResponse(str(SPA_DIR / "index-web.html"))
    return {
        "message": "Anton CoWork API",
        "anton_available": conversation_manager.is_anton_available(),
    }


# SPA static + client-side-routing fallback. Mounted AFTER all API routers
# so they take priority; the catch-all only fires for paths none of them
# claimed. /v1/* and /health are explicitly excluded so wrong API paths
# return a clean 404 instead of falling back to index-web.html.
if ANTON_SERVE_SPA and SPA_DIR is not None:
    # Subdirectories that the SPA build emits — served by StaticFiles so
    # they get correct MIME-types and range-request handling for free.
    # Any future top-level subdir of SPA_DIR that should be served must
    # be added to this tuple (or as its own dedicated mount).
    for _sub in ("assets", "fonts", "gravity-field"):
        _sub_path = SPA_DIR / _sub
        if _sub_path.exists():
            app.mount(f"/{_sub}", StaticFiles(directory=str(_sub_path)), name=f"spa-{_sub}")

    # Pre-resolve every top-level file in SPA_DIR into a string→Path
    # allowlist. The fallback handler below looks the request up in this
    # dict — the user-controlled `full_path` is used only as a dict key,
    # never as a path component. That removes the entire untrusted-input
    # → path-expression dataflow that CodeQL py/path-injection flagged
    # against earlier realpath/is_relative_to-based guards: there is no
    # path being built from `full_path` at all, so traversal sequences
    # (`..`, encoded variants, absolute paths) simply miss the dict and
    # fall through to the SPA shell. Computed at startup so the dict is
    # immutable for the request handler's lifetime.
    _spa_files: dict[str, Path] = {
        _entry.name: _entry for _entry in SPA_DIR.iterdir() if _entry.is_file()
    }
    _spa_shell: Path = SPA_DIR / "index-web.html"

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        # Wrong /v1/* paths must 404 cleanly — never serve the SPA shell
        # in place of a missing API endpoint or callers will silently get
        # HTML where they expected JSON.
        if full_path == "v1" or full_path.startswith("v1/") or full_path == "health":
            raise HTTPException(status_code=404)
        # Top-level file the build emitted? Serve it. Otherwise this is
        # either a client-side route (/projects/foo, /artifacts/...) or a
        # bogus path — both get the SPA shell so the renderer's router
        # can take over (or render its own 404 view) on the client.
        served = _spa_files.get(full_path)
        if served is not None:
            return FileResponse(str(served))
        return FileResponse(str(_spa_shell))


def main() -> None:
    _maybe_self_update_and_reexec()

    import uvicorn

    # Default port 26866 spells "ANTON" on a phone T9 keypad (A=2, N=6,
    # T=8, O=6, N=6). Lives in the dynamic/private port range and avoids
    # the usual web-dev / database collisions (3000, 5000, 5173, 5432,
    # 6379, 8000, 8080, 8765, etc.). Override with ANTON_SERVER_PORT.
    port = int(os.environ.get("ANTON_SERVER_PORT", 26866))
    host = os.environ.get("ANTON_SERVER_HOST", "127.0.0.1")

    logger.info("─" * 50)
    logger.info("Anton CoWork server starting on %s:%d", host, port)
    logger.info("Anton available: %s", conversation_manager.is_anton_available())
    if not conversation_manager.is_anton_available():
        logger.info("Anton is not installed; chat endpoints will return 503")
        logger.info("To install: pip install anton  (from github.com/mindsdb/anton)")
    logger.info("─" * 50)

    uvicorn.run(app, host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
