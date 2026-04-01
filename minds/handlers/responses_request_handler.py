from langfuse import observe
from mindsdb_sdk.server import Server
from sqlmodel import Session
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, StreamingResponse

from minds.agents.helpers import is_anton_agent
from minds.common.logger import setup_logging
from minds.db.pg_session import get_open_session
from minds.handlers.openai_request_handler import OpenAIRequestHandler
from minds.requests.context import Context
from minds.requests.langfuse_tracing import get_langfuse_trace_id, setup_langfuse_observation
from minds.requests.responses_request import ResponsesRequest
from minds.requests.stream import (
    format_messages_for_non_streaming_responses_api,
    format_messages_for_streaming_responses_api,
    process_non_streaming_producer,
    process_streaming_producer,
)
from minds.schemas.chat import Message, Role
from minds.schemas.conversations import ConversationCreateRequest, ConversationItem, ConversationMetadata
from minds.services.conversations import ConversationsService
from minds.services.limits import LimitsService
from minds.services.minds import MindsService

logger = setup_logging()


class ConversationMindMismatchError(Exception):
    """Exception for when a conversation is not associated with the current mind."""

    pass


@observe(name="Responses Handler v1", as_type="generation")
async def responses_request_handler(
    session: Session,
    context: Context,
    mindsdb_client: Server,
    responses_request: ResponsesRequest,
    conversation_service: ConversationsService,
    instrument: bool = True,
    limits_service: LimitsService | None = None,
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

    metadata = responses_request.metadata
    logger.debug(f"🔄[{request_id}] Metadata: {metadata}")

    conversation_id = conversation

    owned_session: Session | None = None
    minds_service = MindsService(
        session=session,
        mindsdb_client=mindsdb_client,
        user_id=context.user_id,
        organization_id=context.organization_id,
    )

    # If streaming is enabled, create a dedicated session NOT managed by Depends(get_session)
    # and a dedicated conversations service
    # This is done because the message is initially created (flushed) as a placeholder and when it is completed,
    # the message content is updated and committed.
    # The message events are also flushed and only committed along with the message content.
    # If Depends(get_session) is used here, the session will be closed after the request is completed,
    # but while the stream is still in progress.
    # This will cause the message events flushed to fail because it is not part of the same session.
    if stream:
        # If the Mind is using Anton, events need to be stored in the database.
        mind = await minds_service.get_mind_model(model)
        if is_anton_agent(mind):
            # This session will be closed after the response is completed using a BackgroundTask
            # (Line 205 at the time of writing).
            owned_session = get_open_session()
            session = owned_session
            conversation_service = ConversationsService(
                session=session,
                mindsdb_client=mindsdb_client,
                user_id=context.user_id,
                organization_id=context.organization_id,
            )
            event_callback = conversation_service.create_conversation_message_event
        else:
            event_callback = None

    try:
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
                        conversation_items.append(ConversationItem(role=message.role, content=message.content))

            new_conversation = await conversation_service.create_conversation(
                ConversationCreateRequest(
                    metadata=ConversationMetadata(model_name=model),
                    items=conversation_items,
                ),
                minds_service,
            )
            conversation_id = new_conversation.id

        # If a conversation is provided, add the input as a new message to the conversation
        else:
            # First check if the conversation exists and it is associated with the current mind
            conversation = await conversation_service.get_conversation(conversation_id)
            if conversation.metadata.model_name != model:
                raise ConversationMindMismatchError(
                    f"Conversation {conversation_id} is not associated with the current mind {model}"
                )

            if input:
                if isinstance(input, str):
                    await conversation_service.create_conversation_message(
                        conversation_id=conversation_id,
                        role=Role.user,
                        content=input,
                    )
                elif isinstance(input, list):
                    for message in input:
                        await conversation_service.create_conversation_message(
                            conversation_id=conversation_id,
                            role=message.role,
                            content=message.content,
                        )

        # Get the conversation along with it's messages
        # This will reflect the conversation ID provided or the new one created (with an updated list of messages)
        conversation_messages = await conversation_service.get_conversation_messages(conversation_id)

        # Convert the Message object to chat completions compatible Message (Role and Content) objects
        messages = []
        for message in conversation_messages:
            messages.append(Message(role=message.role, content=message.content.text))

        # Use the chat completions handler (as a wrapper) to handle the responses request
        responses_handler = await OpenAIRequestHandler.create(
            session=session,
            context=context,
            mindsdb_client=mindsdb_client,
            messages=messages,
            model=model,
            stream=stream,
            metadata=metadata,
            instrument=instrument,
            limits_service=limits_service,
        )

        # Create a message placeholder for the assistant response
        # This is done to get the message ID to include in the response
        message = await conversation_service.create_conversation_message_placeholder(
            conversation_id=conversation_id,
            role=Role.assistant,
        )
        message_id = message.id

        if stream:
            logger.debug(f"🔄[{request_id}] Responses API request is streaming.")
            response = await process_streaming_producer(
                producer=lambda streamer: responses_handler.responses(streamer=streamer, message=message),
                request_id=request_id,
                format_func=format_messages_for_streaming_responses_api,
                model=model,
                message_id=message_id,
                event_callback=event_callback,
            )
            # If session is owned, close it after the response is completed.
            if owned_session is not None:
                response.background = BackgroundTask(owned_session.close)
        else:
            logger.debug(f"🔄[{request_id}] Responses API request is non-streaming.")
            response = await process_non_streaming_producer(
                producer=lambda streamer: responses_handler.responses(streamer=streamer, message=message),
                request_id=request_id,
                format_func=format_messages_for_non_streaming_responses_api,
                model=model,
                message_id=message_id,
            )

        return response
    except Exception:
        if owned_session is not None:
            owned_session.close()
        raise
