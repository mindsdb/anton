"""FastAPI app exposing Anton over HTTP.

Currently exposes the OpenAI Responses API (/v1/responses), matching the
shape served by anton_servicesrepo/scratchpad_service so a single client
can target either backend. /v1/chat/completions and conversation CRUD will
be added later — start by getting antontron talking to a local anton.

Build the app with `create_app(settings)` rather than importing a module-level
`app`, so the host process owns settings resolution.
"""

from __future__ import annotations

import json
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from anton import __version__
from anton.config.settings import AntonSettings
from anton.server import session_manager
from anton.server.formatter import format_responses_stream
from anton.server.models import (
    ResponseObject,
    ResponseOutput,
    ResponseOutputContent,
    ResponseStatus,
    ResponsesRequest,
)


def create_app(settings: AntonSettings) -> FastAPI:
    """Build a FastAPI instance bound to the given AntonSettings."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        session_manager.configure(settings)
        yield
        await session_manager.close_all()

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

    return app
