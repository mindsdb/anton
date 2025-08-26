"""
Chat completion endpoints for API v1.

This module contains endpoints for handling chat completion requests,
including both streaming and non-streaming responses.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from langfuse import observe
from sqlmodel import Session
from starlette.responses import JSONResponse

from minds.common.logger import setup_logging
from minds.db.pg_session import get_session
from minds.handlers.chat_completions_request_handler import (
    chat_completions_request_handler,
)
from minds.requests.chat_completions_request import ChatCompletionsRequest
from minds.requests.context import Context, extract_context_from_request
from minds.requests.utils import setup_langfuse_observation
from minds.client.mindsdb import create_mindsdb_client_from_request

# Set up logging
logger = setup_logging()

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
@observe(name="Chat Completions v1", as_type="generation")
async def chat_completions(
    chat_completions_request: ChatCompletionsRequest,
    request: Request,
    session: Session = Depends(get_session)
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
        HTTPException: 500 if there's an error processing the request.
    """
    # Extract user context from request
    user = extract_context_from_request(request)
    
    logger.debug(f"🔄 Context: {user.model_dump()}")

    # Set up Langfuse observation
    request_id = setup_langfuse_observation(context=user)

    mindsdb_client = create_mindsdb_client_from_request(request)

    try:
        logger.debug(f"🔄 [{request_id}] Starting chat completions v1")

        response = await chat_completions_request_handler(
            request_id=request_id,
            session=session,
            mindsdb_client=mindsdb_client,
            chat_completions_request=chat_completions_request,
        )

        session.commit()
    except Exception as e:
        logger.error(f"❌ [{request_id}] Error processing chat completions request: {str(e)}", exc_info=True)
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        session.close()

    return response
