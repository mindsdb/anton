"""
Chat completion endpoint — inference-only passthrough.

All models are resolved to provider configs and handed to InferenceService.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session
from starlette.responses import JSONResponse

from minds.api.v1.deps import get_limits_service
from minds.common.guards import ResourceType, require_usage_available
from minds.common.logger import get_logger
from minds.common.statsig import is_langfuse_enabled
from minds.db.pg_session import get_session
from minds.handlers.chat_completions_request_handler import (
    chat_completions_request_handler,
)
from minds.requests.chat_completions_request import ChatCompletionsRequest
from minds.requests.context import extract_context_from_request
from minds.services.limits import LimitsService

logger = get_logger(__name__)
router = APIRouter()


@router.options("/completions")
async def options_handler():
    """Handle CORS preflight requests."""
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )


@router.post("/completions")
async def chat_completions(
    chat_completions_request: ChatCompletionsRequest,
    request: Request,
    session: Session = Depends(get_session),
    limits_service: LimitsService = Depends(get_limits_service),
):
    """
    Handle chat completions (passthrough inference).

    Args:
        chat_completions_request: The chat request.
        request: The HTTP request (for context extraction).
        session: Database session.
        limits_service: Usage limits service.

    Returns:
        Streaming or JSON response from the provider.
    """
    context = extract_context_from_request(request=request)
    logger.debug(f"Chat completions request for model: {chat_completions_request.model}")

    try:
        await require_usage_available(limits_service, ResourceType.TOKENS)

        langfuse_enabled = is_langfuse_enabled(context=context)
        trace_propagation_kwargs: dict = {}

        if not langfuse_enabled:
            handler = chat_completions_request_handler.__wrapped__
            instrument = False
        else:
            handler = chat_completions_request_handler
            instrument = True
            if context.langfuse_trace_id:
                trace_propagation_kwargs["langfuse_trace_id"] = context.langfuse_trace_id
                if context.langfuse_parent_observation_id:
                    trace_propagation_kwargs["langfuse_parent_observation_id"] = context.langfuse_parent_observation_id

        response = await handler(
            context=context,
            session=session,
            chat_completions_request=chat_completions_request,
            instrument=instrument,
            **trace_propagation_kwargs,
        )

        return response
    except Exception as e:
        logger.error(f"Error processing chat completions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
