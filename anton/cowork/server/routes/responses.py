"""POST /v1/responses — OpenAI Responses API.

Streaming SSE by default; pass {"stream": false} for one-shot JSON.
Cowork extensions: project (name) + attachment_ids fields on the request.
The returned `response.created` event carries the conversation_id the
frontend uses as its task id.
"""

from __future__ import annotations

import json
import logging
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..anton_api import conversation_manager
from ..anton_api.formatter import format_responses_stream
from ..anton_api.models import (
    Message,
    ResponseObject,
    ResponseOutput,
    ResponseOutputContent,
    ResponseStatus,
    ResponsesRequest,
)
from .attachments import assign_attachments, attachment_context
from .settings import get_config_status


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["responses"])


def _resolve_input(req: ResponsesRequest) -> str:
    if isinstance(req.input, str):
        return req.input
    user_messages = [m for m in req.input if isinstance(m, Message) and m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message in input")
    content = user_messages[-1].content
    if not isinstance(content, str):
        raise HTTPException(status_code=400, detail="Only string user content is supported")
    return content


def _assembled_user_input(content: str, attachment_ids: list[str]) -> str:
    context = attachment_context(attachment_ids)
    if not context:
        return content
    return f"{content}\n\n{context}"


@router.post("/responses")
async def create_response(req: ResponsesRequest):
    if not conversation_manager.is_anton_available():
        raise HTTPException(
            status_code=503,
            detail="Anton is not installed in this desktop environment.",
        )

    config = get_config_status()
    if not config["config_ready"]:
        raise HTTPException(
            status_code=400,
            detail=config["config_error"] or "Anton is not configured.",
        )

    user_text = _resolve_input(req)
    final_input = _assembled_user_input(user_text, req.attachment_ids)
    if req.attachment_ids and req.conversation:
        assign_attachments(req.attachment_ids, req.conversation)

    if req.stream:
        async def stream():
            assigned_for_new = False
            cid: str | None = None
            # Per-turn event capture for the cowork sidecar. The sink is
            # called by `format_responses_stream` for every event before
            # serialisation so we record exactly the SSE-shape dicts the
            # client adapter consumes (no parsing back from the wire).
            recorded_events: list[dict] = []
            started_at_ms: int | None = None

            def _record(event_type: str, data: dict) -> None:
                nonlocal started_at_ms
                if started_at_ms is None:
                    started_at_ms = int(time.time() * 1000)
                # Defensive copy so later mutations of `data` (rare but
                # cheap to guard against) don't churn what we'll persist.
                recorded_events.append({**data})

            try:
                event_stream, cid = await conversation_manager.chat_stream(
                    final_input,
                    conversation_id=req.conversation,
                    project=req.project,
                    model=req.model if req.model and req.model != "anton" else None,
                )
                # If this is a new conversation, retroactively assign attachments
                # to the conversation id we just minted.
                if req.attachment_ids and not req.conversation and not assigned_for_new:
                    assign_attachments(req.attachment_ids, cid)
                    assigned_for_new = True

                async for chunk in format_responses_stream(
                    event_stream,
                    model=req.model or "anton",
                    conversation_id=cid,
                    event_sink=_record,
                ):
                    yield chunk
            except conversation_manager.AntonConfigurationError as exc:
                logger.warning("Anton configuration error: %s", exc)
                yield (
                    "event: response.failed\n"
                    f"data: {json.dumps({'type': 'response.failed', 'code': 'config_required', 'error': str(exc)})}\n\n"
                )
            except conversation_manager.AntonRuntimeError as exc:
                logger.error("Anton runtime error: %s", exc)
                yield (
                    "event: response.failed\n"
                    f"data: {json.dumps({'type': 'response.failed', 'code': 'anton_error', 'error': str(exc)})}\n\n"
                )
            except Exception:
                logger.exception("response stream failed")
                yield (
                    "event: response.failed\n"
                    f"data: {json.dumps({'type': 'response.failed', 'code': 'server_error', 'error': 'Internal server error'})}\n\n"
                )
            finally:
                # Persist whatever we captured so reopening the
                # conversation rebuilds the Thinking block + scratchpad
                # tabs. Runs even on failed/aborted streams so partial
                # turns are still recoverable.
                if cid and recorded_events:
                    try:
                        conversation_manager.record_turn_events(
                            cid, started_at_ms, recorded_events,
                        )
                    except Exception:
                        logger.debug("Could not record turn events", exc_info=True)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            },
        )

    # Non-streaming: collect text and return a single ResponseObject.
    from anton.core.llm.provider import StreamTextDelta

    collected: list[str] = []
    try:
        event_stream, cid = await conversation_manager.chat_stream(
            final_input,
            conversation_id=req.conversation,
            project=req.project,
            model=req.model if req.model and req.model != "anton" else None,
        )
        if req.attachment_ids and not req.conversation:
            assign_attachments(req.attachment_ids, cid)
        async for event in event_stream:
            if isinstance(event, StreamTextDelta):
                collected.append(event.text)
    except conversation_manager.AntonConfigurationError as exc:
        logger.warning("Anton configuration error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except conversation_manager.AntonRuntimeError as exc:
        logger.error("Anton runtime error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return ResponseObject(
        model=req.model or "anton",
        status=ResponseStatus.completed,
        output=[ResponseOutput(
            status=ResponseStatus.completed,
            content=[ResponseOutputContent(text="".join(collected))],
        )],
    )
