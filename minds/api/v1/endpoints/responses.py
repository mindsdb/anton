"""
Responses API endpoints for API v1.

This module contains endpoints for handling OpenAI-compatible Responses API requests,
including both streaming and non-streaming responses.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session
from starlette.responses import JSONResponse

from minds.api.v1.deps import get_conversations_service, get_limits_service, get_mindsdb_client
from minds.common.guards import ResourceType, require_usage_available
from minds.common.logger import get_logger
from minds.common.statsig import is_langfuse_enabled
from minds.db.pg_session import get_session
from minds.handlers.responses_request_handler import (
    ConversationMindMismatchError,
    responses_request_handler,
)
from minds.requests.context import extract_context_from_request
from minds.requests.responses_request import ResponsesRequest
from minds.services.conversations import ConversationsService
from minds.services.limits import LimitsService

logger = get_logger(__name__)

router = APIRouter()


@router.options("/")
async def options_handler():
    """
    Handle CORS preflight requests for Responses API endpoints.

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


@router.post("/")
async def responses(
    responses_request: ResponsesRequest,
    request: Request,
    mindsdb_client=Depends(get_mindsdb_client),
    session: Session = Depends(get_session),
    conversations_service: ConversationsService = Depends(get_conversations_service),
    limits_service: LimitsService = Depends(get_limits_service),
):
    """
    Handle Responses API requests (API v1).

    This endpoint provides OpenAI-compatible Responses API with support for
    both streaming and non-streaming responses. It integrates with MindsDB for
    AI model management and includes comprehensive observability.

    Args:
        responses_request (ResponsesRequest): The request containing chat messages and other parameters.
        request (Request): The FastAPI request object to extract context.
        mindsdb_client (Server): The MindsDB client for MindsDB operations.
        session (Session): The SQLAlchemy session for database operations.
        conversations_service (ConversationsService): The conversation service for database operations.

    Returns:
        StreamingResponse | JSONResponse: A streaming response if stream=True,
            otherwise a JSON response containing Responses API messages.

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
        logger.debug(f"🔄 [{context.request_id}] Starting Responses API v1")

        # Distributed-trace propagation kwargs are only understood by the
        # @observe-wrapped handler (langfuse strips them before calling the
        # function). The unwrapped handler used when Langfuse is disabled would
        # raise TypeError if handed them, so only attach them in that branch.
        trace_propagation_kwargs: dict = {}
        if not langfuse_enabled:
            handler = responses_request_handler.__wrapped__
            instrument = False
        else:
            handler = responses_request_handler
            instrument = True
            if context.langfuse_trace_id:
                trace_propagation_kwargs["langfuse_trace_id"] = context.langfuse_trace_id
                if context.langfuse_parent_observation_id:
                    trace_propagation_kwargs["langfuse_parent_observation_id"] = context.langfuse_parent_observation_id

        response = await handler(
            context=context,
            session=session,
            mindsdb_client=mindsdb_client,
            responses_request=responses_request,
            conversation_service=conversations_service,
            instrument=instrument,
            limits_service=limits_service,
            **trace_propagation_kwargs,
        )

        return response
    except ConversationMindMismatchError as e:
        logger.warning(f"❌ [{context.request_id}] Conversation mind mismatch: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"❌ [{context.request_id}] Error processing Responses API request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
