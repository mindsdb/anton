from mindsdb_sdk.server import Server
from sqlmodel import Session
from starlette.responses import JSONResponse, StreamingResponse

from minds.common.logger import get_logger
from minds.handlers.openai_request_handler import OpenAIRequestHandler
from minds.requests.chat_completions_request import ChatCompletionsRequest
from minds.requests.context import Context
from minds.requests.langfuse_tracing import (
    capture_langfuse_generation_context,
    get_langfuse_trace_id,
    lazy_observe,
    setup_langfuse_observation,
)
from minds.requests.stream import (
    format_messages_for_non_streaming_chat_completions_api,
    format_messages_for_streaming_chat_completions_api,
    process_non_streaming_producer,
    process_streaming_producer,
)
from minds.services.limits import LimitsService

# Set up logging
logger = get_logger(__name__)


# ``capture_input=False`` / ``capture_output=False``:
# - ``input``  defaults to the function's args (Session, Context, MindsDB
#   client, ...) which don't serialize to anything an eval can replay.
#   The handler attaches the actual request payload (model / messages /
#   tools / temperature / max_tokens) via ``update_generation_usage(input=...)``
#   inside ``proxy_chat_completions`` instead.
# - ``output`` defaults to the function's return value (a ``JSONResponse``
#   or ``StreamingResponse``), which is a Starlette object whose dict
#   shape (status_code / body / raw_headers) is useless for replays and
#   silently clobbers the assistant message dict we attach via
#   ``update_generation_usage(output=...)``. Disabling auto-capture lets
#   our explicit value land on the span.
@lazy_observe(
    name="Chat Completions Handler v1",
    as_type="generation",
    capture_input=False,
    capture_output=False,
)
async def chat_completions_request_handler(
    session: Session,
    context: Context,
    mindsdb_client: Server,
    chat_completions_request: ChatCompletionsRequest,
    instrument: bool = True,
    limits_service: LimitsService | None = None,
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

    # Capture the @observe trace context so the handler can attach a child
    # generation carrying token usage from streaming code paths that run
    # AFTER this decorated function returns (the StreamingResponse body
    # iterator is consumed by the ASGI server only after the route handler
    # returns, by which point the @observe span has already closed).
    langfuse_trace_context = capture_langfuse_generation_context()

    logger.debug(f"🔄[{request_id}] Chat Completion Request: {chat_completions_request.model_dump()}")

    stream = chat_completions_request.stream if chat_completions_request.stream is not None else False
    logger.debug(f"🔄[{request_id}] Stream: {stream}")

    messages = chat_completions_request.messages
    logger.debug(f"🔄[{request_id}] Messages: {messages}")

    model = chat_completions_request.model
    logger.debug(f"🔄[{request_id}] Model: {model}")

    metadata = chat_completions_request.metadata
    logger.debug(f"🔄[{request_id}] Metadata: {metadata}")

    tools = chat_completions_request.tools
    tool_choice = chat_completions_request.tool_choice
    temperature = chat_completions_request.temperature
    # max_completion_tokens takes precedence over max_tokens (OpenAI convention)
    max_tokens = chat_completions_request.max_completion_tokens or chat_completions_request.max_tokens

    chat_completions_handler = await OpenAIRequestHandler.create(
        session=session,
        context=context,
        mindsdb_client=mindsdb_client,
        messages=messages,
        model=model,
        stream=stream,
        metadata=metadata,
        instrument=instrument,
        request_id=request_id,
        langfuse_trace_id=get_langfuse_trace_id(),
        langfuse_trace_context=langfuse_trace_context,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        max_tokens=max_tokens,
        limits_service=limits_service,
    )

    # Passthrough models bypass the normal Mind pipeline
    if chat_completions_handler.is_passthrough:
        logger.debug(f"[{request_id}] Passthrough shortcut — proxying directly to upstream LLM.")
        return await chat_completions_handler.proxy_chat_completions()

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
