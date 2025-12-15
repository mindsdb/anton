from langfuse import observe
from mindsdb_sdk.server import Server
from sqlmodel import Session
from starlette.responses import JSONResponse, StreamingResponse

from minds.common.logger import setup_logging
from minds.handlers.responses_handler import ResponsesHandler
from minds.requests.context import Context
from minds.requests.langfuse_tracing import get_langfuse_trace_id, setup_langfuse_observation
from minds.requests.responses_request import ResponsesRequest
from minds.requests.stream import (
    process_non_streaming_producer,
    process_streaming_producer,
)

logger = setup_logging()


@observe(name="Responses Handler v1", as_type="generation")
async def responses_request_handler(
    session: Session,
    context: Context,
    mindsdb_client: Server,
    responses_request: ResponsesRequest,
    instrument: bool = True,
) -> StreamingResponse | JSONResponse:
    """
    Handle OpenAI-compatible Responses API requests.

    Args:
        session (Session): The SQLAlchemy session for database operations.
        context (Context): The context of the request.
        mindsdb_client (Server): The MindsDB client for database operations.
        responses_request (ResponsesRequest): The request object containing Responses API parameters.
        instrument (bool): Whether to instrument the PydanticAIAgent.
    Returns:
        Union[StreamingResponse, JSONResponse]: A streaming response if the request is for streaming,
            otherwise a JSON response.
    """
    # Set up Langfuse observation
    setup_langfuse_observation(context=context)
    request_id = get_langfuse_trace_id() or str(context.request_id)

    logger.debug(f"🔄[{request_id}] Responses Request: {responses_request.model_dump()}")

    stream = responses_request.stream if responses_request.stream is not None else False
    logger.debug(f"🔄[{request_id}] Stream: {stream}")

    conversation = responses_request.conversation
    logger.debug(f"🔄[{request_id}] Conversation: {conversation}")

    input = responses_request.input
    logger.debug(f"🔄[{request_id}] Input: {input}")

    model = responses_request.model
    logger.debug(f"🔄[{request_id}] Model: {model}")

    # metadata = responses_request.metadata
    # logger.debug(f"🔄[{request_id}] Metadata: {metadata}")

    responses_handler = ResponsesHandler(
        session=session,
        context=context,
        mindsdb_client=mindsdb_client,
        conversation=conversation,
        input=input,
        model=model,
        stream=stream,
        instrument=instrument,
    )

    if stream:
        logger.debug(f"🔄[{request_id}] Responses API request is streaming.")
        response = await process_streaming_producer(
            producer=lambda streamer: responses_handler.responses(streamer=streamer),
            request_id=request_id,
            model=model,
        )
    else:
        logger.debug(f"🔄[{request_id}] Responses API request is non-streaming.")
        response = await process_non_streaming_producer(
            producer=lambda streamer: responses_handler.responses(streamer=streamer),
            request_id=request_id,
            model=model,
        )

    return response
