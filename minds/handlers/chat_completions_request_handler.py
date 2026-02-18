from langfuse import observe
from mindsdb_sdk.server import Server
from pydantic_ai.usage import RunUsage
from sqlmodel import Session
from starlette.responses import JSONResponse, StreamingResponse

from minds.common.logger import setup_logging
from minds.handlers.chat_completions_handler import ChatCompletionsHandler
from minds.model.chat_completion import ChatCompletion
from minds.requests.chat_completions_request import ChatCompletionsRequest
from minds.requests.context import Context
from minds.requests.langfuse_tracing import get_langfuse_trace_id, setup_langfuse_observation
from minds.requests.stream import (
    format_messages_for_non_streaming_chat_completions_api,
    format_messages_for_streaming_chat_completions_api,
    process_non_streaming_producer,
    process_streaming_producer,
)

# Set up logging
logger = setup_logging()


@observe(name="Chat Completions Handler v1", as_type="generation")
async def chat_completions_request_handler(
    session: Session,
    context: Context,
    mindsdb_client: Server,
    chat_completions_request: ChatCompletionsRequest,
    instrument: bool = True,
) -> StreamingResponse | JSONResponse:
    """
    Handle chat completions requests.

    Args:
            request_id (str): The unique identifier for the request.
            session (Session): The SQLAlchemy session for database operations.
            mindsdb_client (Server): The MindsDB client for database operations.
            chat_completions_request (ChatCompletionsRequest): The request object containing chat completion parameters.
            instrument (bool): Whether to instrument the PydanticAIAgent.
    Returns:
            Union[StreamingResponse, JSONResponse]: A streaming response if the request is for streaming,
                otherwise a JSON response.
    """

    # Set up Langfuse observation
    setup_langfuse_observation(context=context)
    request_id = get_langfuse_trace_id() or str(context.request_id)

    logger.debug(f"🔄[{request_id}] Chat Completion Request: {chat_completions_request.model_dump()}")

    stream = chat_completions_request.stream if chat_completions_request.stream is not None else False
    logger.debug(f"🔄[{request_id}] Stream: {stream}")

    messages = chat_completions_request.messages
    logger.debug(f"🔄[{request_id}] Messages: {messages}")

    model = chat_completions_request.model
    logger.debug(f"🔄[{request_id}] Model: {model}")

    metadata = chat_completions_request.metadata
    logger.debug(f"🔄[{request_id}] Metadata: {metadata}")

    async def save_chat_completion(usage: RunUsage):
        """Save a ChatCompletion record with token usage to the database."""
        chat_completion = ChatCompletion(
            organization_id=context.organization_id,
            user_id=context.user_id,
            model_name=model,
            request_id=request_id,
            langfuse_trace_id=get_langfuse_trace_id(),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
        session.add(chat_completion)
        session.commit()
        logger.debug(
            f"🔄[{request_id}] Saved ChatCompletion usage: {usage.input_tokens} in / {usage.output_tokens} out"
        )

    chat_completions_handler = ChatCompletionsHandler(
        session=session,
        context=context,
        mindsdb_client=mindsdb_client,
        messages=messages,
        model=model,
        stream=stream,
        metadata=metadata,
        instrument=instrument,
        on_complete_callback=save_chat_completion,
    )

    if stream:
        logger.debug(f"🔄[{request_id}] Chat completions request is streaming.")
        response = await process_streaming_producer(
            producer=lambda streamer: chat_completions_handler.chat_completions(streamer=streamer),
            request_id=request_id,
            format_func=format_messages_for_streaming_chat_completions_api,
            model=model,
        )
    else:
        logger.debug(f"🔄[{request_id}] Chat completions request is non-streaming.")
        response = await process_non_streaming_producer(
            producer=lambda streamer: chat_completions_handler.chat_completions(streamer=streamer),
            request_id=request_id,
            model=model,
            format_func=format_messages_for_non_streaming_chat_completions_api,
        )

    return response
