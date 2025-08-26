import asyncio
import json
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from langfuse import get_client, observe
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse, StreamingResponse

from minds.common.logger import setup_logging
from minds.schemas.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    Message,
    Role,
    StreamChoice,
)

# Set up logging
logger = setup_logging()


def sanitize_content_for_observation(content: Any) -> str:
    """
    Sanitize content for Langfuse observation to prevent base64 data URI parsing errors.

    Args:
        content: Any content that needs to be serialized for observation

    Returns:
        str: A safely serialized string representation of the content
    """
    try:
        if content is None:
            return ""
        elif isinstance(content, str):
            return content
        elif isinstance(content, list | dict):
            # For complex objects like citations, convert to JSON string
            # This prevents Langfuse from trying to interpret them as media content
            return json.dumps(content, default=lambda x: x.model_dump() if hasattr(x, "model_dump") else str(x))
        else:
            # For other types, convert to string
            return str(content)
    except Exception as e:
        logger.warning(f"Failed to serialize content for observation: {e}")
        return f"<serialization_error: {type(content).__name__}>"


class StreamMessage(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    role: Role
    content: str | list[Any] | None = None


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


async def format_messages_for_streaming(
    message_generator: AsyncGenerator[StreamMessage, Any], model: str
) -> AsyncGenerator[str, None]:
    """
    Format messages for streaming as SSE events.

    Args:
            message_generator: An async generator that yields Message objects
            model (str): The model name to include in the SSE event.

    Returns:
            An async generator yielding SSE formatted events
    """
    async_index = 0
    async for search_message in message_generator:
        # Sanitize content for Langfuse observation to prevent parsing errors
        sanitized_content = sanitize_content_for_observation(search_message.content)

        # Create chunk with original content for streaming
        chunk = ChatCompletionChunk(
            id=search_message.id,
            model=model,
            choices=[
                StreamChoice(
                    index=async_index,
                    delta=Message(role=search_message.role, content=search_message.content),
                )
            ],
        )
        async_index += 1

        # Log sanitized content to avoid issues with complex objects
        logger.debug(f"Streaming message {async_index}: {search_message.role} - {sanitized_content[:100]}...")

        yield f"data: {chunk.model_dump_json()}\n\n"


@observe(name="Process Streaming Producer")
async def process_streaming_producer(
    producer: Callable[[Any], Awaitable[None]],
    request_id: str,
    trace_name: str,
    model: str,
) -> StreamingResponse:
    """
    Run a push-based producer that emits StreamMessage objects to an streamer, and
    return a StreamingResponse that forwards those as SSE.

    The producer must accept a single argument `streamer` exposing `emit(StreamMessage)` and `close()`.

    Args:
            producer (Callable[[Any], Awaitable[None]]): An async function that takes a
                    `MessageStreamer` instance and emits `StreamMessage` objects.
            request_id (str): The unique ID for the request, used to identify the stream.
            trace_name (str): The name of the original trace.
            model (str): The model name to include in the SSE event.
    Returns:
            StreamingResponse: A response that streams the messages as Server-Sent Events (SSE).
    """
    trace_id = get_client().get_current_trace_id()
    observation_id = get_client().get_current_observation_id()

    logger.debug(f"Trace ID - Streaming Producer: {trace_id}")
    logger.debug(f"Observation ID - Streaming Producer: {observation_id}")

    @observe()
    async def stream():
        streamer = Streamer(request_id=request_id)

        @observe(name=trace_name)
        async def run_producer():
            try:
                await producer(
                    streamer,
                )
            finally:
                # Ensure the sentinel is always sent so the consumer loop terminates
                await streamer.close()

        task = asyncio.create_task(
            run_producer(
                langfuse_trace_id=trace_id,
                langfuse_parent_observation_id=observation_id,
            )
        )
        async for message in streamer:
            logger.debug(f"Message: {message}")
            yield message
        await task

    return StreamingResponse(
        format_messages_for_streaming(
            message_generator=stream(
                langfuse_trace_id=trace_id,
                langfuse_parent_observation_id=observation_id,
            ),
            model=model,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )


async def _build_json_response_from_messages(messages: list[StreamMessage], model: str) -> JSONResponse:
    """
    Build a JSON response from a list of StreamMessage objects.

    Args:
            messages (list[StreamMessage]): A list of StreamMessage objects.
            model (str): The model name to include in the response.
    Returns:
            JSONResponse: A response that contains the ChatCompletion built from the messages.
    """
    message_id = messages[-1].id if messages else ""
    choices = []
    for index, search_message in enumerate(messages):
        choice = Choice(
            index=index,
            message=Message(role=search_message.role, content=search_message.content),
        )
        choices.append(choice)

    response = ChatCompletion(id=message_id, model=model, choices=choices)
    return JSONResponse(response.model_dump())


@observe(name="Process Non-Streaming Producer", as_type="generation")
async def process_non_streaming_producer(
    producer: Callable[[Any], Awaitable[None]],
    request_id: str,
    model: str,
) -> JSONResponse:
    """
    Run a push-based producer and return a non-streaming JSON ChatCompletion built
    from all emitted messages.

    The producer must accept a single argument `collector` exposing `push(StreamMessage)` and `close()`.
    Args:
            producer (Callable[[Any], Awaitable[None]]): An async function that takes a
                    `StreamerCollector` instance and emits `StreamMessage` objects.
            request_id (str): The unique ID for the request, used to identify the stream.
            model (str): The model name to include in the response.
    Returns:
            JSONResponse: A response that contains the ChatCompletion built from the messages.
    """
    collector = StreamerCollector(request_id=request_id)
    await producer(collector)
    return await _build_json_response_from_messages(messages=collector.messages, model=model)
