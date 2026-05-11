"""/v1/scratchpad/* — direct control over named LocalScratchpadRuntime pads."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..anton_api import scratchpad_runtime
from ..anton_api.models import (
    ScratchpadExecRequest,
    ScratchpadInstallRequest,
    ScratchpadPadRequest,
    ScratchpadStartRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/scratchpad", tags=["scratchpad"])


@router.post("/start")
async def scratchpad_start(req: ScratchpadStartRequest):
    pad = scratchpad_runtime.get_or_create(
        req.name,
        coding_provider=req.coding_provider,
        coding_model=req.coding_model,
        coding_api_key=req.coding_api_key,
        coding_base_url=req.coding_base_url,
    )
    await pad.start()
    return {"status": "started", "name": req.name}


@router.post("/execute")
async def scratchpad_execute(req: ScratchpadExecRequest):
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


@router.post("/execute-stream")
async def scratchpad_execute_stream(req: ScratchpadExecRequest):
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
        except Exception:
            logger.exception("scratchpad execution failed")
            yield f"data: {json.dumps({'type': 'error', 'error': 'Execution failed'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/install")
async def scratchpad_install(req: ScratchpadInstallRequest):
    pad = scratchpad_runtime.get(req.name)
    if not pad:
        raise HTTPException(status_code=404, detail=f"Scratchpad '{req.name}' not found.")
    result = await pad.install_packages(req.packages)
    return {"result": result}


@router.post("/reset")
async def scratchpad_reset(req: ScratchpadPadRequest):
    pad = scratchpad_runtime.get(req.name)
    if not pad:
        raise HTTPException(status_code=404, detail=f"Scratchpad '{req.name}' not found.")
    await pad.reset()
    return {"status": "reset", "name": req.name}


@router.post("/cancel")
async def scratchpad_cancel(req: ScratchpadPadRequest):
    pad = scratchpad_runtime.get(req.name)
    if not pad:
        raise HTTPException(status_code=404, detail=f"Scratchpad '{req.name}' not found.")
    await pad.cancel()
    return {"status": "cancelled", "name": req.name}


@router.get("/view")
async def scratchpad_view(name: str = "default"):
    pad = scratchpad_runtime.get(name)
    if not pad:
        raise HTTPException(status_code=404, detail=f"Scratchpad '{name}' not found.")
    return {"view": pad.view()}


@router.get("/notebook")
async def scratchpad_notebook(name: str = "default"):
    pad = scratchpad_runtime.get(name)
    if not pad:
        raise HTTPException(status_code=404, detail=f"Scratchpad '{name}' not found.")
    return {"notebook": pad.render_notebook()}


@router.get("/cells")
async def scratchpad_cells(name: str = "default"):
    pad = scratchpad_runtime.get(name)
    if not pad:
        raise HTTPException(status_code=404, detail=f"Scratchpad '{name}' not found.")
    return {"cells": [asdict(c) for c in pad.cells]}


@router.post("/close")
async def scratchpad_close(req: ScratchpadPadRequest):
    pad = scratchpad_runtime.get(req.name)
    if not pad:
        raise HTTPException(status_code=404, detail=f"Scratchpad '{req.name}' not found.")
    await pad.close()
    scratchpad_runtime.remove(req.name)
    return {"status": "closed", "name": req.name}


@router.post("/cleanup")
async def scratchpad_cleanup(req: ScratchpadPadRequest):
    pad = scratchpad_runtime.get(req.name)
    if not pad:
        raise HTTPException(status_code=404, detail=f"Scratchpad '{req.name}' not found.")
    await pad.cleanup()
    scratchpad_runtime.remove(req.name)
    return {"status": "cleaned", "name": req.name}


@router.get("/list")
async def scratchpad_list():
    return {"pads": scratchpad_runtime.list_pads()}
