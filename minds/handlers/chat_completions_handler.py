from langfuse import observe
from mindsdb_sdk.server import Server
from sqlmodel import Session

from minds.agent.database_agent import DatabaseAgent
from minds.agent.database_toolkit import DatabaseToolkit
from minds.common.logger import setup_logging
from minds.model.mind_datasource import DataCatalogStatus
from minds.requests.context import Context
from minds.requests.stream import MessageStreamer
from minds.services.minds import MindsService
from minds.schemas.chat import Message, Role

# Set up logging
logger = setup_logging()


class ChatCompletionsHandler:
    def __init__(
        self,
        session: Session,
        context: Context,
        mindsdb_client: Server,
        messages: list[Message],
        model: str,
        stream: bool,
    ):
        """
        Initialize the ChatCompletionsHandler with a list of messages.

        Args:
                session (Session): The SQLAlchemy session for database operations.
                mindsdb_client (Server): The MindsDB client for database operations.
                messages (List[Message]): List of messages to handle.
                model (str): The model to use for chat completions.
                stream (bool): Whether to stream the response.
        """
        self.session = session
        self.context = context
        self.mindsdb_client = mindsdb_client
        self.messages = messages
        self.model = model
        self.stream = stream

    @observe(name="Chat Completions Handler")
    async def chat_completions(self, streamer: MessageStreamer) -> str | None:
        """
        OpenAI compatible chat completions handler.

        Args:
            streamer (MessageStreamer): The streamer to push messages to.

        Returns:
            str | None: A string response or None if no response is generated.
        """
        minds_service = MindsService(session=self.session, mindsdb_client=self.mindsdb_client, user_id=self.context.user_id, tenant_id=self.context.tenant_id)
        mind = await minds_service.get_mind_model(self.model)

        # If the Mind has datasources that are currently loading, inform the user
        # and complete the request
        if any(relationship.status in [DataCatalogStatus.LOADING, DataCatalogStatus.PENDING] for relationship in mind.mind_datasources):
            await streamer.push(
                role=Role.assistant,
                content="The Mind is not ready yet. Please try again later.",
            )
            return None

        database_toolkit = DatabaseToolkit(mind=mind, mindsdb_client=self.mindsdb_client)
        database_agent = DatabaseAgent(mind=mind, database_toolkit=database_toolkit)

        async for chunk in database_agent.get_completion(self.messages, stream=self.stream):
            await streamer.push(role=Role.assistant, content=chunk)
