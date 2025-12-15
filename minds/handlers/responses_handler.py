from bisect import insort_right
from mindsdb_sdk.server import Server
from sqlmodel import Session

from minds.common.logger import setup_logging
from minds.handlers.chat_completions_handler import ChatCompletionsHandler
from minds.services.conversations import ConversationsService
from minds.requests.context import Context
from minds.requests.stream import MessageStreamer
from minds.schemas.chat import Message, Role
from minds.schemas.conversations import ConversationCreateRequest, ConversationItem

logger = setup_logging()


class ResponsesHandler:
    """
    Handler for OpenAI compatible responses requests.
    """

    def __init__(
        self,
        session: Session,
        context: Context,
        mindsdb_client: Server,
        model: str,
        stream: bool,
        input: str | list[Message] | None = None,
        # TODO: The OpenAI API also supports a conversation object?
        conversation: str | None = None,
        # TODO: Metadata has not been added? Is it needed?
        instrument: bool = True,
    ):
        """
        Initialize the ResponsesHandler.

        Args:
            session (Session): The SQLAlchemy session for database operations.
            context (Context): The context of the request.
            mindsdb_client (Server): The MindsDB client for database operations.
            model (str): The model to use for responses.
            stream (bool): Whether to stream the response.
            input (str | list[Message] | None): The input for the responses request.
            conversation (str | None): The conversation ID for the responses request.
            instrument (bool): Whether to instrument the PydanticAIAgent.
        """
        self.session = session
        self.context = context
        self.mindsdb_client = mindsdb_client
        self.model = model
        self.stream = stream
        self.input = input
        self.conversation = conversation
        self.instrument = instrument

    async def responses(self, streamer: MessageStreamer) -> str | None:
        """
        OpenAI compatible responses handler.

        Args:
            streamer (MessageStreamer): The streamer to push messages to.

        Returns:
            str | None: A string response or None if no response is generated.
        """
        conversation_id = self.conversation

        conversation_service = ConversationsService(
            session=self.session,
            user_id=self.context.user_id,
            tenant_id=self.context.tenant_id,
        )

        # If no conversation is provided, create a new conversation
        # Add items included as messages of that conversation
        if not self.conversation:
            conversation_items = []

            if self.input:
                if isinstance(self.input, str):
                    conversation_items.append(
                        ConversationItem(
                            role=Role.user,
                            content=self.input,
                        )
                    )
                elif isinstance(self.input, list):
                    for message in self.input:
                        conversation_items.append(
                            ConversationItem(
                                role=message.user,
                                content=message.content
                            )
                        )

            conversation = await conversation_service.create_conversation(
                ConversationCreateRequest(
                    items=conversation_items
                )
            )
            conversation_id = conversation.id

        # Get the conversation along with it's messages
        # This will reflect the conversation ID provided or the new one created (with an updated list of messages)
        if self.conversation:
            conversation = await conversation_service.get_conversation_model_with_messages(conversation_id)

        # Convert the Message object to chat completions compatible Message (Role and Content) objects
        messages = []
        for message in conversation.messages:
            messages.append(message.to_chat_message())

        # Instantiate the chat completions handler
        chat_completions_handler = ChatCompletionsHandler(
            self.session,
            self.context,
            self.mindsdb_client,
            messages,
            self.model,
            self.stream,
            instrument=self.instrument
        )

        response = await chat_completions_handler.chat_completions(streamer)
        return response
