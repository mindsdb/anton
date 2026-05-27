"""
Chat completion endpoints for API v1.

This module contains endpoints for handling chat completion requests,
including both streaming and non-streaming responses.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session
from starlette.responses import JSONResponse

from minds.api.v1.deps import get_limits_service, get_mindsdb_client
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

# Set up logging
logger = get_logger(__name__)

# Create router for chat completion endpoints
router = APIRouter()


@router.options("/completions")
async def options_handler():
    """
    Handle CORS preflight requests for chat completions endpoints.

    Returns:
        JSONResponse: CORS headers for preflight requests
    """
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
    mindsdb_client=Depends(get_mindsdb_client),
    session: Session = Depends(get_session),
    limits_service: LimitsService = Depends(get_limits_service),
):
    """
    Handle chat completions for documents (API v1).

    This endpoint provides OpenAI-compatible chat completions with support for
    both streaming and non-streaming responses. It integrates with MindsDB for
    AI model management and includes comprehensive observability.

    Args:
        chat_completions_request (ChatCompletionsRequest): The request containing
            chat messages and other parameters.
        request (Request): The FastAPI request object to extract context.

    Returns:
        StreamingResponse | JSONResponse: A streaming response if stream=True,
            otherwise a JSON response containing chat completion messages.

    Raises:
        HTTPException: 429 if usage limit exceeded.
        HTTPException: 500 if there's an error processing the request.
    """
    # Extract user context from request
    context = extract_context_from_request(request=request)
    logger.debug(f"🔄 Context: {context.model_dump()}")

    # Check usage limits before processing
    await require_usage_available(limits_service, ResourceType.TOKENS)
    langfuse_enabled = is_langfuse_enabled(context=context)
    logger.debug(f"🔄 [{context.request_id}] Langfuse is enabled: {langfuse_enabled}")

    try:
        logger.debug(f"🔄 [{context.request_id}] Starting chat completions v1")

        # Distributed-trace propagation kwargs are only understood by the
        # @observe-wrapped handler (langfuse strips them before calling the
        # function). The unwrapped handler used when Langfuse is disabled would
        # raise TypeError if handed them, so only attach them in that branch.
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
                    trace_propagation_kwargs["langfuse_parent_observation_id"] = (
                        context.langfuse_parent_observation_id
                    )

        response = await handler(
            context=context,
            session=session,
            mindsdb_client=mindsdb_client,
            chat_completions_request=chat_completions_request,
            instrument=instrument,
            limits_service=limits_service,
            **trace_propagation_kwargs,
        )

        return response
    except Exception as e:
        logger.error(f"❌ [{context.request_id}] Error processing chat completions request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
