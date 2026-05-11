"""Cowork-side endpoints that back the `data-vault-form` workflow.

Form submissions flow through `POST /v1/datavault/submissions`. The
endpoint stages the values in the in-memory submission store and
returns a `submission_id` the cowork frontend uses when sending its
chat continuation back to Anton. Anton's tool then redeems the
submission id for the actual values just-in-time, never letting
credential material round-trip through chat history.

Phase 1 deliberately keeps this endpoint minimal — no validation
yet, no orchestration. The connection-test logic still lives behind
the existing `/v1/datasources/validate` + `/v1/datasources` routes;
the agent reaches them via the existing publisher / connector
tools. We add staging here so the chat continuation can stay free
of secrets.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..anton_api import datavault_agent, datavault_submissions

router = APIRouter()
logger = logging.getLogger(__name__)


class SubmitFormRequest(BaseModel):
    form_id: str
    conversation_id: Optional[str] = None
    values: dict[str, Any] = {}
    skipped: list[str] = []
    # Anton's `request_credentials` tool emits the spec; the form
    # panel echoes it back on submit so the agent endpoint knows what
    # engine to validate against, what auth_method to use, etc.
    # Without this we'd have to look up the form by id from a
    # short-lived store; passing it through is cheaper and simpler.
    form_spec: Optional[dict[str, Any]] = None


@router.post("/submissions")
async def submit_form(req: SubmitFormRequest):
    """Stage values + run the cowork-side agent that processes the
    submission. Returns a Response-API SSE stream the existing client
    adapter consumes natively — text deltas + a `data-vault-form-patch`
    block + `response.completed` with a status field.

    Field VALUES never echo back through this response; only
    references (form_id, slug) and human-readable summaries.
    """
    if not req.form_id:
        raise HTTPException(status_code=400, detail="form_id is required")

    submission_id = datavault_submissions.stage_submission(
        form_id=req.form_id,
        conversation_id=req.conversation_id,
        values=req.values or {},
        skipped=req.skipped or [],
    )

    spec = req.form_spec or {"form_id": req.form_id}
    # Defensive fill — request_credentials marks engine as required
    # but if Anton ever emits a form without it, the agent surfaces
    # a clear "unknown engine" message instead of crashing.
    spec.setdefault("form_id", req.form_id)

    async def stream():
        try:
            async for chunk in datavault_agent.process_submission_stream(
                submission_id=submission_id,
                form_spec=spec,
                conversation_id=req.conversation_id,
            ):
                yield chunk
        except Exception:
            logger.exception("datavault stream failed")
            import json as _json
            yield (
                "event: response.failed\n"
                f"data: {_json.dumps({'type': 'response.failed', 'error': 'Internal server error'})}\n\n"
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


@router.get("/submissions/{submission_id}")
async def peek_submission(submission_id: str):
    """Return the metadata + field NAMES for a staged submission. The
    actual values are intentionally redacted for any client-facing
    request — only the agent-side tool may read them, via a different
    helper that doesn't go through the HTTP layer.
    """
    entry = datavault_submissions.get_submission(submission_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Submission not found")
    return {
        "submission_id": entry["submission_id"],
        "form_id": entry["form_id"],
        "conversation_id": entry.get("conversation_id"),
        "field_names": list(entry.get("values", {}).keys()),
        "skipped": entry.get("skipped", []),
        "created_at": entry.get("created_at"),
        "status": entry.get("status"),
    }
