"""FastAPI app exposing Anton over HTTP.

Currently exposes:
  - OpenAI Responses API   (/v1/responses)
  - Scratchpad control API (/v1/scratchpad/*)

Both surfaces match the shapes served by anton_servicesrepo/scratchpad_service
so a single client can target either backend.

Build the app with `create_app(settings)` rather than importing a module-level
`app`, so the host process owns settings resolution.
"""

from __future__ import annotations

import json
import traceback
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from anton import __version__
from anton.config.settings import AntonSettings
from anton.server import scratchpad_runtime, session_manager
from anton.server.formatter import format_responses_stream
from anton.server.models import (
    ResponseObject,
    ResponseOutput,
    ResponseOutputContent,
    ResponseStatus,
    ResponsesRequest,
    ScratchpadExecRequest,
    ScratchpadInstallRequest,
    ScratchpadPadRequest,
    ScratchpadStartRequest,
)


def create_app(settings: AntonSettings) -> FastAPI:
    """Build a FastAPI instance bound to the given AntonSettings."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        session_manager.configure(settings)
        scratchpad_runtime.configure(settings)
        yield
        await session_manager.close_all()
        await scratchpad_runtime.close_all()

    app = FastAPI(
        title="Anton",
        version=__version__,
        lifespan=lifespan,
    )

    # Permissive CORS — local desktop app talks over loopback; tighten if exposed.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def _global_exception_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "traceback": traceback.format_exc()},
        )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "version": __version__,
            "workspace": str(settings.workspace_path),
            "sessions": session_manager.list_sessions(),
            "pads": scratchpad_runtime.list_pads(),
            "pad_count": len(scratchpad_runtime.list_pads()),
            "max_pads": scratchpad_runtime.MAX_PADS,
            "last_activity": scratchpad_runtime.last_activity,
        }

    @app.post("/v1/responses")
    async def responses(req: ResponsesRequest):
        """OpenAI Responses API — streaming SSE or single-shot JSON."""
        if isinstance(req.input, str):
            user_input = req.input
        elif isinstance(req.input, list):
            user_messages = [m for m in req.input if m.role == "user"]
            user_input = user_messages[-1].content if user_messages else ""
            if not isinstance(user_input, str):
                raise HTTPException(status_code=400, detail="Only string content is supported")
        else:
            raise HTTPException(status_code=400, detail="Invalid input")

        conversation_id = req.conversation or None

        if req.stream:
            async def stream():
                try:
                    event_stream, cid = await session_manager.chat_stream(
                        user_input, conversation_id=conversation_id,
                    )
                    async for chunk in format_responses_stream(
                        event_stream, model=req.model, conversation_id=cid,
                    ):
                        yield chunk
                except Exception as e:
                    yield (
                        "event: response.failed\n"
                        f"data: {json.dumps({'error': str(e), 'traceback': traceback.format_exc()})}\n\n"
                    )

            return StreamingResponse(
                stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                },
            )

        from anton.core.llm.provider import StreamTextDelta

        collected: list[str] = []
        event_stream, cid = await session_manager.chat_stream(
            user_input, conversation_id=conversation_id,
        )
        async for event in event_stream:
            if isinstance(event, StreamTextDelta):
                collected.append(event.text)

        return ResponseObject(
            model=req.model,
            status=ResponseStatus.completed,
            output=[ResponseOutput(
                status=ResponseStatus.completed,
                content=[ResponseOutputContent(text="".join(collected))],
            )],
        )

    # ------------------------------------------------------------------
    # Scratchpad API — direct control over named LocalScratchpadRuntimes.
    # Mirrors the /scratchpad/* surface of anton_servicesrepo.
    # ------------------------------------------------------------------

    @app.post("/v1/scratchpad/start")
    async def scratchpad_start(req: ScratchpadStartRequest):
        """Create and start a named scratchpad."""
        pad = scratchpad_runtime.get_or_create(
            req.name,
            coding_provider=req.coding_provider,
            coding_model=req.coding_model,
            coding_api_key=req.coding_api_key,
            coding_base_url=req.coding_base_url,
        )
        await pad.start()
        return {"status": "started", "name": req.name}

    @app.post("/v1/scratchpad/execute")
    async def scratchpad_execute(req: ScratchpadExecRequest):
        """Execute code and return the final Cell (non-streaming)."""
        pad = scratchpad_runtime.get(req.name)
        if not pad:
            raise HTTPException(
                status_code=404,
                detail=f"Scratchpad '{req.name}' not found. Call /v1/scratchpad/start first.",
            )

        cell = await pad.execute(
            req.code,
            description=req.description,
            estimated_time=req.estimated_time,
            estimated_seconds=req.estimated_seconds,
        )
        return {"cell": asdict(cell)}

    @app.post("/v1/scratchpad/execute-stream")
    async def scratchpad_execute_stream(req: ScratchpadExecRequest):
        """Execute code with SSE streaming — progress events then the final Cell."""
        pad = scratchpad_runtime.get(req.name)
        if not pad:
            raise HTTPException(
                status_code=404,
                detail=f"Scratchpad '{req.name}' not found. Call /v1/scratchpad/start first.",
            )

        from anton.core.backends.base import Cell

        async def event_stream():
            try:
                async for item in pad.execute_streaming(
                    req.code,
                    description=req.description,
                    estimated_time=req.estimated_time,
                    estimated_seconds=req.estimated_seconds,
                ):
                    if isinstance(item, str):
                        yield f"data: {json.dumps({'type': 'progress', 'message': item})}\n\n"
                    elif isinstance(item, Cell):
                        yield f"data: {json.dumps({'type': 'cell', 'cell': asdict(item)})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/v1/scratchpad/install")
    async def scratchpad_install(req: ScratchpadInstallRequest):
        """Install Python packages into a scratchpad's environment."""
        pad = scratchpad_runtime.get(req.name)
        if not pad:
            raise HTTPException(status_code=404, detail=f"Scratchpad '{req.name}' not found.")

        result = await pad.install_packages(req.packages)
        return {"result": result}

    @app.post("/v1/scratchpad/reset")
    async def scratchpad_reset(req: ScratchpadPadRequest):
        """Kill the runtime, clear cells, and restart."""
        pad = scratchpad_runtime.get(req.name)
        if not pad:
            raise HTTPException(status_code=404, detail=f"Scratchpad '{req.name}' not found.")

        await pad.reset()
        return {"status": "reset", "name": req.name}

    @app.post("/v1/scratchpad/cancel")
    async def scratchpad_cancel(req: ScratchpadPadRequest):
        """Cancel the currently running cell."""
        pad = scratchpad_runtime.get(req.name)
        if not pad:
            raise HTTPException(status_code=404, detail=f"Scratchpad '{req.name}' not found.")

        await pad.cancel()
        return {"status": "cancelled", "name": req.name}

    @app.get("/v1/scratchpad/view")
    async def scratchpad_view(name: str = "default"):
        """View all cells and their outputs."""
        pad = scratchpad_runtime.get(name)
        if not pad:
            raise HTTPException(status_code=404, detail=f"Scratchpad '{name}' not found.")

        return {"view": pad.view()}

    @app.get("/v1/scratchpad/notebook")
    async def scratchpad_notebook(name: str = "default"):
        """Get a clean markdown notebook-style summary."""
        pad = scratchpad_runtime.get(name)
        if not pad:
            raise HTTPException(status_code=404, detail=f"Scratchpad '{name}' not found.")

        return {"notebook": pad.render_notebook()}

    @app.get("/v1/scratchpad/cells")
    async def scratchpad_cells(name: str = "default"):
        """Get all cells as structured data."""
        pad = scratchpad_runtime.get(name)
        if not pad:
            raise HTTPException(status_code=404, detail=f"Scratchpad '{name}' not found.")

        return {"cells": [asdict(c) for c in pad.cells]}

    @app.post("/v1/scratchpad/close")
    async def scratchpad_close(req: ScratchpadPadRequest):
        """Shut down a scratchpad, preserving persistent resources."""
        pad = scratchpad_runtime.get(req.name)
        if not pad:
            raise HTTPException(status_code=404, detail=f"Scratchpad '{req.name}' not found.")

        await pad.close()
        scratchpad_runtime.remove(req.name)
        return {"status": "closed", "name": req.name}

    @app.post("/v1/scratchpad/cleanup")
    async def scratchpad_cleanup(req: ScratchpadPadRequest):
        """Fully destroy a scratchpad and its environment."""
        pad = scratchpad_runtime.get(req.name)
        if not pad:
            raise HTTPException(status_code=404, detail=f"Scratchpad '{req.name}' not found.")

        await pad.cleanup()
        scratchpad_runtime.remove(req.name)
        return {"status": "cleaned", "name": req.name}

    @app.get("/v1/scratchpad/list")
    async def scratchpad_list():
        """List all active scratchpad names."""
        return {"pads": scratchpad_runtime.list_pads()}

    return app
