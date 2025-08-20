import traceback

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langfuse.decorators import observe
from starlette.responses import JSONResponse

from minds.common.logger import setup_logging
from minds.db.pg_session import get_session
from minds.handlers.chat_completions_request_handler import (
    chat_completions_request_handler,
)
from minds.requests.chat_completions_request import ChatCompletionsRequest
from minds.requests.context import extract_context_from_request
from minds.requests.utils import setup_langfuse_observation

# Set up logging
logger = setup_logging()

app = FastAPI()

# Define allowed origins - make it accept any origin
ALLOWED_ORIGINS = ["*"]  # Allow any origin

# Add CORS middleware first
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # Set to False when using "*"
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Create router after middleware
router = APIRouter()  # Remove the prefix here


@router.get("/healthz")
async def healthz():
    return {"status": "ok"}


@router.options("/chat/completions")
@router.options("/v1/chat/completions")
async def options_handler():
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )


@router.post("/chat/completions")
@router.post("/v1/chat/completions")
@observe(name="Chat Completions")
async def chat_completions(
    chat_completions_request: ChatCompletionsRequest, request: Request
):
    """
    Endpoint to handle chat completions for documents.

    Args:
        chat_completions_request (DocumentsChatCompletionsRequest): The request containing chat messages and other parameters.
        request (Request): The FastAPI request object to extract context.

    Returns:
        StreamingResponse: A streaming response containing chat completion messages.
    """

    # Use the utility function to create context from a request
    context = extract_context_from_request(request=request)
    logger.debug(f"🔄 Context: {context.model_dump()}")

    # Set up Langfuse observation
    request_id = setup_langfuse_observation(context=context)

    # Create a new session
    session = get_session()

    try:
        logger.debug(f"🔄 [{request_id}] Starting chat with documents ")

        response = await chat_completions_request_handler(
            request_id=request_id,
            session=session,
            chat_completions_request=chat_completions_request,
        )

        session.commit()
    except Exception as e:
        logger.error(
            f"❌ [{request_id}] Error processing chat with documents request: {str(e)}"
        )
        logger.error(traceback.format_exc())
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

    return response


# Include router last
app.include_router(router)
