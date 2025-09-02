from langfuse import observe
from mindsdb_sdk.server import Server
from sqlalchemy.orm import selectinload
from sqlmodel import Session, and_, select

from minds.agent.database_agent import DatabaseAgent
from minds.agent.database_toolkit import DatabaseToolkit
from minds.common.logger import setup_logging
from minds.model.mind import Mind
from minds.model.mind_datasource import MindDatasource
from minds.requests.context import Context
from minds.requests.stream import MessageStreamer
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
        Dummy chat completions method.

        Args:
            streamer (MessageStreamer): The streamer to push messages to.
        Returns:
            str | None: A string response or None if no response is generated.
        """
        statement = (
            select(Mind)
            .options(selectinload(Mind.mind_datasources).selectinload(MindDatasource.datasource))
            .where(and_(Mind.name == self.model, Mind.user_id == self.context.user_id, Mind.is_active))
        )
        mind = self.session.exec(statement).first()

        database_toolkit = DatabaseToolkit(mind=mind, mindsdb_client=self.mindsdb_client)
        database_agent = DatabaseAgent(mind=mind, database_toolkit=database_toolkit)

        async for chunk in database_agent.get_completion(self.messages):
            await streamer.push(role=Role.assistant, content=chunk)
