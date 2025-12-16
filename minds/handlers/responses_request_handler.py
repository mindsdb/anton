from langfuse import observe
from mindsdb_sdk.server import Server
from sqlmodel import Session
from starlette.responses import JSONResponse, StreamingResponse

from minds.common.logger import setup_logging
from minds.handlers.chat_completions_handler import ChatCompletionsHandler
from minds.requests.context import Context
from minds.requests.langfuse_tracing import get_langfuse_trace_id, setup_langfuse_observation
from minds.requests.responses_request import ResponsesRequest
from minds.requests.stream import (
    process_non_streaming_producer,
    process_streaming_producer,
)
from minds.schemas.chat import ChatCompletion, ChatCompletionChunk, Role
from minds.schemas.conversations import ConversationCreateRequest, ConversationItem
from minds.services.conversations import ConversationsService

logger = setup_logging()


@observe(name="Responses Handler v1", as_type="generation")
async def responses_request_handler(
    session: Session,
    context: Context,
    mindsdb_client: Server,
    responses_request: ResponsesRequest,
    conversation_service: ConversationsService,
    instrument: bool = True,
) -> StreamingResponse | JSONResponse:
    """
    Handle OpenAI-compatible Responses API requests.

    Args:
        session (Session): The SQLAlchemy session for database operations.
        context (Context): The context of the request.
        mindsdb_client (Server): The MindsDB client for database operations.
        responses_request (ResponsesRequest): The request object containing Responses API parameters.
        conversation_service (ConversationsService): The conversation service for database operations.
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

    conversation_id = conversation

    # If no conversation is provided, create a new conversation
    # Add items included as messages of that conversation
    if not conversation_id:
        conversation_items = []

        if input:
            if isinstance(input, str):
                conversation_items.append(
                    ConversationItem(
                        role=Role.user,
                        content=input,
                    )
                )
            elif isinstance(input, list):
                for message in input:
                    conversation_items.append(
                        ConversationItem(
                            role=message.role,
                            content=message.content
                        )
                    )

        new_conversation = await conversation_service.create_conversation(
            ConversationCreateRequest(
                items=conversation_items
            )
        )
        conversation_id = new_conversation.id

    # Get the conversation along with it's messages
    # This will reflect the conversation ID provided or the new one created (with an updated list of messages)
    conversation = await conversation_service.get_conversation_model_with_messages(conversation_id)

    # Convert the Message object to chat completions compatible Message (Role and Content) objects
    messages = []
    for message in conversation.messages:
        messages.append(message.to_chat_message())

    # Use the chat completions handler (as a wrapper) to handle the responses request
    chat_completions_handler = ChatCompletionsHandler(
        session=session,
        context=context,
        mindsdb_client=mindsdb_client,
        messages=messages,
        model=model,
        stream=stream,
        instrument=instrument
    )


    async def save_assistant_response(message: list[ChatCompletionChunk] | ChatCompletion):
        """
        Save the assistant response to the database.

        Args:
            message_chunks (list[StreamMessage]): The list of message chunks to save.
        """
        if isinstance(message, ChatCompletion):
            content = message.choices[0].message.content
        elif isinstance(message, list) and len(message) > 0 and isinstance(message[0], ChatCompletionChunk):
            content = ""
            for message_chunk in message:
                if message_chunk.choices and message_chunk.choices[0].delta.content:
                    content += message_chunk.choices[0].delta.content or ""

        await conversation_service.add_message_to_conversation(conversation_id=conversation_id, role=Role.assistant, content=content)


    if stream:
        logger.debug(f"🔄[{request_id}] Responses API request is streaming.")
        response = await process_streaming_producer(
            producer=lambda streamer: chat_completions_handler.chat_completions(streamer=streamer),
            request_id=request_id,
            model=model,
            on_complete_callback=save_assistant_response,
        )
    else:
        logger.debug(f"🔄[{request_id}] Responses API request is non-streaming.")
        response = await process_non_streaming_producer(
            producer=lambda streamer: chat_completions_handler.chat_completions(streamer=streamer),
            request_id=request_id,
            model=model,
            on_complete_callback=save_assistant_response,
        )

    return response
