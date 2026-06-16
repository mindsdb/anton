"""
Responses API handler — stateful conversation management for inference-only.

Handles OpenAI-compatible Responses API requests by managing conversation state
and delegating inference to the OpenAIRequestHandler.
"""

from sqlmodel import Session
from starlette.responses import JSONResponse, StreamingResponse

from minds.common.logger import get_logger
from minds.handlers.openai_request_handler import OpenAIRequestHandler
from minds.requests.context import Context
from minds.requests.langfuse_tracing import (
    capture_langfuse_generation_context,
    get_langfuse_trace_id,
    lazy_observe,
    setup_langfuse_observation,
)
from minds.requests.responses_request import ResponsesRequest
from minds.requests.streamers import SimpleStreamer
from minds.schemas.chat import Message, Role
from minds.schemas.conversations import ConversationCreateRequest, ConversationItem, ConversationMetadata
from minds.services.conversations import ConversationsService
from minds.services.limits import LimitsService

logger = get_logger(__name__)


@lazy_observe(name="Responses Handler v1", as_type="generation")
async def responses_request_handler(
    session: Session,
    context: Context,
    responses_request: ResponsesRequest,
    conversation_service: ConversationsService,
    instrument: bool = True,
    limits_service: LimitsService | None = None,
) -> StreamingResponse | JSONResponse:
    """
    Handle OpenAI-compatible Responses API requests.

    Manages conversation state and delegates inference to the OpenAIRequestHandler
    for passthrough model resolution and inference.

    Args:
        session (Session): The SQLAlchemy session for database operations.
        context (Context): The context of the request.
        responses_request (ResponsesRequest): The request object containing Responses API parameters.
        conversation_service (ConversationsService): The conversation service for database operations.
        instrument (bool): Whether to instrument the request.
        limits_service (LimitsService): The limits service for usage enforcement.

    Returns:
        StreamingResponse | JSONResponse: A streaming response if the request is for streaming,
            otherwise a JSON response.
    """
    # Set up Langfuse observation
    setup_langfuse_observation(context=context)
    request_id = get_langfuse_trace_id() or str(context.request_id)

    # Capture the @observe trace context so streaming code paths can attach
    # a child generation carrying token usage after this decorated handler
    # returns (the StreamingResponse body iterator outlives @observe).
    langfuse_trace_context = capture_langfuse_generation_context()

    logger.debug(f"[{request_id}] Responses Request: {responses_request.model_dump()}")

    stream = responses_request.stream if responses_request.stream is not None else False
    conversation_id = responses_request.conversation
    input_data = responses_request.input
    model = responses_request.model
    metadata = responses_request.metadata

    try:
        # If no conversation is provided, create a new one
        if not conversation_id:
            conversation_items = []

            if input_data:
                if isinstance(input_data, str):
                    conversation_items.append(
                        ConversationItem(
                            role=Role.user,
                            content=input_data,
                        )
                    )
                elif isinstance(input_data, list):
                    for message in input_data:
                        conversation_items.append(ConversationItem(role=message.role, content=message.content))

            new_conversation = await conversation_service.create_conversation(
                ConversationCreateRequest(
                    metadata=ConversationMetadata(model_name=model),
                    items=conversation_items,
                ),
            )
            conversation_id = new_conversation.id

        # If a conversation is provided, add the input as a new message
        else:
            # Verify the conversation exists and the caller has access.
            await conversation_service.get_conversation(conversation_id)

            if input_data:
                if isinstance(input_data, str):
                    await conversation_service.create_conversation_message(
                        conversation_id=conversation_id,
                        role=Role.user,
                        content=input_data,
                    )
                elif isinstance(input_data, list):
                    for message in input_data:
                        await conversation_service.create_conversation_message(
                            conversation_id=conversation_id,
                            role=message.role,
                            content=message.content,
                        )

        # Get all conversation messages
        conversation_messages = await conversation_service.get_conversation_messages(conversation_id)

        # Convert to chat completions compatible Message objects
        messages = []
        for message in conversation_messages:
            messages.append(Message(role=message.role, content=message.content.text))

        # Create the inference handler
        responses_handler = await OpenAIRequestHandler.create(
            session=session,
            context=context,
            messages=messages,
            model=model,
            stream=stream,
            metadata=metadata,
            instrument=instrument,
            request_id=request_id,
            langfuse_trace_id=get_langfuse_trace_id(),
            langfuse_trace_context=langfuse_trace_context,
            reasoning_effort=responses_request.reasoning.effort if responses_request.reasoning else None,
            limits_service=limits_service,
        )

        # Create a message placeholder for the assistant response
        message = await conversation_service.create_conversation_message_placeholder(
            conversation_id=conversation_id,
            role=Role.assistant,
        )

        # Use the responses method from the handler which manages the conversation state
        response = await responses_handler.responses(
            streamer=SimpleStreamer(),
            message=message,
        )

        return response

    except Exception:
        raise
