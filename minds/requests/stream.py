import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any
from uuid import UUID

from langfuse import get_client
from pydantic import BaseModel
from starlette.responses import JSONResponse, StreamingResponse

from minds.common.logger import setup_logging
from minds.schemas.chat import ChatCompletion, ChatCompletionChunk, Choice, Message, Role, StreamChoice, StreamMessage
from minds.schemas.responses import (
    Response,
    ResponseDelta,
    ResponseOutput,
    ResponseOutputContent,
    ResponseStatus,
    StreamingResponseEvent,
)
from minds.schemas.responses import (
    StreamingResponse as StreamingResponseSchema,
)

# Set up logging
logger = setup_logging()


def create_stream_message(role: Role, content: Any, request_id: str) -> StreamMessage:
    """
    Create a StreamMessage with the given role and content.
    Args:
            role (Role): The role of the message (e.g
                    Role.user, Role.assistant, Role.system).
            content (Any): The content of the message, can be a string or any other type.
            request_id (str): The unique ID for the request.

    Returns:
            StreamMessage: A StreamMessage object with the specified role and content.
    """

    logger.debug(f"Creating stream message ({request_id}) for role '{role}': {content}")
    return StreamMessage(id=request_id, role=role, content=content)


class MessageStreamer:
    async def push(self, role: Role, content: Any) -> None:
        """
        Push a message to the streamer.

        Args:
                role (Role): The role of the message (e.g., Role.user, Role.assistant).
                content (Any): The content of the message, can be a string or any other type.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class Streamer(MessageStreamer):
    """Push-based emitter backed by an asyncio.Queue for streaming consumption."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._queue: asyncio.Queue[StreamMessage | None] = asyncio.Queue()

    async def push(self, role: Role, content: Any) -> None:
        message = create_stream_message(role=role, content=content, request_id=self.request_id)
        await self._queue.put(message)
        # Give event loop a chance to process the queue
        await asyncio.sleep(0)

    async def close(self) -> None:
        await self._queue.put(None)

    async def __aiter__(self):
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item


class StreamerCollector(MessageStreamer):
    """Push-based emitter that collects messages into a list (for non-streaming)."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.messages: list[StreamMessage] = []

    async def push(self, role: Role, content: Any) -> None:
        message = create_stream_message(role=role, content=content, request_id=self.request_id)
        self.messages.append(message)

    @staticmethod
    async def close() -> None:
        # No-op for collector
        return None


def extract_sql_query_from_thoughts(content: Any) -> str | None:
    """
    Extract the SQL query from the content if it exists.

    Args:
        content (Any): The content of the message, can be a string or any other type.

    Returns:
        str | None: The SQL query if it exists, otherwise None.
    """
    if isinstance(content, dict) and content.get("type") == "sql_query":
        return content.get("query")
    return None


async def format_messages_for_streaming_chat_completions_api(
    message_generator: AsyncGenerator[StreamMessage, Any],
    model: str,
    on_complete_callback: Callable[[list[ChatCompletionChunk]], Awaitable[None]] | None = None,
    **kwargs,
) -> AsyncGenerator[str, None]:
    """
    Format messages for streaming for chat completions API as SSE events.

    Args:
            message_generator: An async generator that yields Message objects
            model (str): The model name to include in the SSE event.
            on_complete_callback (Callable[[list[ChatCompletionChunk]], Awaitable[None]] | None): Optional callback
                    function (ignored for chat completions API).
    Returns:
            An async generator yielding SSE formatted events
    """
    async_index = 0
    last_message_id: str | None = None

    # Emit each incoming message immediately. After the producer finishes,
    # emit a small final chunk that indicates finish_reason="stop" so clients
    # can detect completion without requiring us to peek the next message.
    assistant_messages = []
    async for msg in message_generator:
        # Properly serialize BaseModel content
        content = msg.content
        if isinstance(content, BaseModel):
            content = content.model_dump()

        # Extract the SQL query from the content if it exists
        sql_query = extract_sql_query_from_thoughts(content)
        if sql_query:
            content = sql_query

        chunk = ChatCompletionChunk(
            id=msg.id,
            model=model,
            choices=[
                StreamChoice(
                    index=async_index,
                    delta=Message(role=msg.role, content=content),
                )
            ],
        )

        last_message_id = msg.id
        async_index += 1
        if msg.role == Role.assistant:
            assistant_messages.append(chunk)
        # Yield immediately (no peek)
        yield f"event: completion\ndata: {chunk.model_dump_json()}\n\n"

    # After the producer finished, emit a final small chunk marking completion
    # with finish_reason="stop".
    if last_message_id is not None:
        final_chunk = ChatCompletionChunk(
            id=last_message_id,
            model=model,
            choices=[
                StreamChoice(
                    index=async_index - 1,
                    delta=Message(role=Role.assistant, content=""),
                    finish_reason="stop",
                )
            ],
        )
        yield f"event: completion\ndata: {final_chunk.model_dump_json()}\n\n"


async def format_messages_for_streaming_responses_api(
    message_generator: AsyncGenerator[StreamMessage, Any],
    model: str,
    on_complete_callback: Callable[[str], Awaitable[None]],
    message_id: UUID,
    **kwargs,
) -> AsyncGenerator[str, None]:
    """
    Format messages for streaming for responses API as SSE events.

    Args:
        message_generator: An async generator that yields StreamMessage objects
        model (str): The model name to include in the SSE event.
        on_complete_callback (Callable[[str], Awaitable[None]]): A callback function to call after
                the messages are processed.
        message_id: UUID, The ID of the message to include in the response.

    Returns:
        An async generator yielding SSE formatted events
    """
    async_index = 1

    # First emit a chunk with status set to 'created' with no output
    created_chunk = StreamingResponseSchema(
        type=StreamingResponseEvent.created.value,
        sequence_number=0,
        response=Response(
            model=model,
            status=ResponseStatus.in_progress.value,
        ),
    )
    yield f"event: {created_chunk.type.value}\ndata: {created_chunk.model_dump_json()}\n\n"

    # Emit each incoming message immediately.
    assistant_content = ""
    sql_query = None
    async for msg in message_generator:
        # Properly serialize BaseModel content
        content = msg.content
        if isinstance(content, BaseModel):
            content = content.model_dump()

        # Emit thoughts (system messages) separately as an in-progress chunks
        if msg.role == Role.system:
            # Extract the SQL query from the system message if it exists
            sql_query = extract_sql_query_from_thoughts(content)
            if sql_query:
                content = sql_query

            chunk = StreamingResponseSchema(
                type=StreamingResponseEvent.in_progress.value,
                sequence_number=0,
                response=Response(
                    model=model,
                    status=ResponseStatus.in_progress.value,
                    output=[
                        ResponseOutput(
                            id=str(message_id),
                            status=ResponseStatus.in_progress.value,
                            role=Role.system.value,
                            content=[ResponseOutputContent(text=content)],
                        )
                    ],
                ),
            )

        # Emit assistant messages as delta chunks
        if msg.role == Role.assistant:
            chunk = StreamingResponseSchema(
                type=StreamingResponseEvent.output_text_delta.value,
                sequence_number=async_index,
                response=ResponseDelta(
                    item_id=str(message_id),
                    delta=content,
                ),
            )

            assistant_content += content

        yield f"event: {chunk.type.value}\ndata: {chunk.model_dump_json()}\n\n"
        async_index += 1

    # After the producer finished, emit a final small chunk marking completion
    # with status set to 'completed'.
    # The response ID should be the same as first (created) chunk.
    completed_chunk = StreamingResponseSchema(
        type=StreamingResponseEvent.completed.value,
        sequence_number=async_index,
        response=Response(
            id=created_chunk.response.id,
            model=model,
            status=ResponseStatus.completed.value,
            output=[
                ResponseOutput(
                    id=str(message_id),
                    status=ResponseStatus.completed.value,
                    role=Role.assistant.value,
                    content=[ResponseOutputContent(text=assistant_content)],
                )
            ],
        ),
    )
    yield f"event: {completed_chunk.type.value}\ndata: {completed_chunk.model_dump_json()}\n\n"

    if on_complete_callback:
        await on_complete_callback(assistant_content, sql_query)


async def process_streaming_producer(
    producer: Callable[[Any], Awaitable[None]],
    request_id: str,
    format_func: Callable[..., AsyncGenerator[str, None]],
    message_id: UUID | None = None,
    **format_kwargs,
) -> StreamingResponse:
    """
    Run a push-based producer that emits StreamMessage objects to an streamer, and
    return a StreamingResponse that forwards those as SSE.

    The producer must accept a single argument `streamer` exposing `push(StreamMessage)` and `close()`.

    Args:
        producer (Callable[[Any], Awaitable[None]]): An async function that takes a
                `MessageStreamer` instance and emits `StreamMessage` objects.
        request_id (str): The unique ID for the request, used to identify the stream.
        format_func (Callable[..., AsyncGenerator[str, None]]): The format function to use for formatting messages.
        message_id (UUID | None): Optional ID of the message to include in the response. Required for responses API.
        **format_kwargs: Additional keyword arguments to pass to the format function.

    Returns:
        StreamingResponse: A response that streams the messages as Server-Sent Events (SSE).
    """
    trace_id = get_client().get_current_trace_id()
    observation_id = get_client().get_current_observation_id()

    logger.debug(f"Trace ID - Streaming Producer: {trace_id}")
    logger.debug(f"Observation ID - Streaming Producer: {observation_id}")

    async def stream():
        streamer = Streamer(request_id=request_id)

        async def run_producer():
            try:
                await producer(streamer)
            finally:
                # Ensure the sentinel is always sent so the consumer loop terminates
                await streamer.close()

        task = asyncio.create_task(run_producer())
        async for message in streamer:
            logger.debug(f"Message: {message}")
            yield message

        await task

    # Conditionally include message_id in format_func call only if provided
    format_args = {"message_generator": stream(), **format_kwargs}
    if message_id is not None:
        format_args["message_id"] = message_id

    return StreamingResponse(
        format_func(**format_args),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )


async def format_messages_for_non_streaming_chat_completions_api(
    messages: list[StreamMessage],
    model: str,
    on_complete_callback: Callable[[ChatCompletion], Awaitable[None]] | None = None,
    **kwargs,
) -> JSONResponse:
    """
    Format messages for non-streaming for chat completions API.

    Args:
        messages (list[StreamMessage]): A list of StreamMessage objects.
        model (str): The model name to include in the response.
        on_complete_callback (Callable[[ChatCompletion], Awaitable[None]] | None): Optional callback
            function (ignored for chat completions API). Required for responses API.

    Returns:
            JSONResponse: A response that contains the ChatCompletion built from the messages.
    """
    message_id = messages[-1].id if messages else ""
    choices = []
    for index, search_message in enumerate(messages):
        # Skip system messages (thoughts) in the final response.
        if search_message.role != Role.system:
            # Properly serialize BaseModel content
            content = search_message.content
            if isinstance(content, BaseModel):
                content = content.model_dump()

            # Extract the SQL query from the content if it exists
            sql_query = extract_sql_query_from_thoughts(content)
            if sql_query:
                content = sql_query

            choice = Choice(
                index=index,
                message=Message(role=search_message.role, content=content),
                finish_reason="stop",  # Set finish_reason to "stop" for successful completion
            )
            choices.append(choice)

    response = ChatCompletion(id=message_id, model=model, choices=choices)
    # Callback is ignored for chat completions API
    return JSONResponse(response.model_dump())


async def format_messages_for_non_streaming_responses_api(
    messages: list[StreamMessage],
    model: str,
    on_complete_callback: Callable[[str], Awaitable[None]],
    message_id: UUID,
    **kwargs,
) -> JSONResponse:
    """
    Format messages for non-streaming for responses API.

    Args:
        messages (list[StreamMessage]): A list of StreamMessage objects.
        model (str): The model name to include in the response.
        on_complete_callback (Callable[[str], Awaitable[None]]): A callback function to call after the
                messages are processed.
        message_id: UUID, The ID of the message to include in the response.

    Returns:
        JSONResponse: A Response object built from the messages.
    """
    output = []
    sql_query = None
    for _, search_message in enumerate(messages):
        # Skip system messages (thoughts) in the final response,
        # but extract the SQL query.
        if search_message.role == Role.system:
            sql_query = extract_sql_query_from_thoughts(search_message.content)
            if sql_query:
                content = sql_query

        if search_message.role != Role.system:
            # Properly serialize BaseModel content
            content = search_message.content
            if isinstance(content, BaseModel):
                content = content.model_dump()

            output.append(
                ResponseOutput(
                    id=str(message_id),
                    status=ResponseStatus.completed.value,
                    content=[ResponseOutputContent(text=content)],
                )
            )

    response = Response(
        model=model,
        output=output,
        status=ResponseStatus.completed.value,
    )

    if on_complete_callback and response.output and response.output[0].content:
        await on_complete_callback(response.output[0].content[0].text, sql_query)

    return JSONResponse(response.model_dump())


async def process_non_streaming_producer(
    producer: Callable[[Any], Awaitable[None]],
    request_id: str,
    format_func: Callable[..., Awaitable[JSONResponse]],
    message_id: UUID | None = None,
    **format_kwargs,
) -> JSONResponse:
    """
    Run a push-based producer and return a non-streaming JSON response built
    from all emitted messages.

    The producer must accept a single argument `collector` exposing `push(StreamMessage)` and `close()`.

    Args:
        producer (Callable[[Any], Awaitable[None]]): An async function that takes a
                `StreamerCollector` instance and emits `StreamMessage` objects.
        request_id (str): The unique ID for the request, used to identify the stream.
        format_func (Callable[..., Awaitable[JSONResponse]]): The format function to use for formatting messages.
        message_id (UUID | None): Optional ID of the message to include in the response. Required for responses API.
        **format_kwargs: Additional keyword arguments to pass to the format function.

    Returns:
        JSONResponse: A response that contains the formatted messages.
    """
    collector = StreamerCollector(request_id=request_id)
    await producer(collector)

    # Conditionally include message_id in format_func call only if provided
    format_args = {"messages": collector.messages, **format_kwargs}
    if message_id is not None:
        format_args["message_id"] = message_id

    return await format_func(**format_args)
