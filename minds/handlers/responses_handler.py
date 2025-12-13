from mindsdb_sdk.server import Server
from sqlmodel import Session

from minds.common.logger import setup_logging
from minds.services.conversations import ConversationService
from minds.requests.context import Context
from minds.requests.stream import MessageStreamer
from minds.schemas.chat import Message

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
        conversation_service: ConversationService,
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
        pass