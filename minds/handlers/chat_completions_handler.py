from typing import List

from langfuse import observe
from sqlmodel import Session
from mindsdb_sdk.server import Server

from minds.common.logger import setup_logging
from minds.requests.schemas import Message, Role
from minds.requests.stream import MessageStreamer


# Set up logging
logger = setup_logging()


class ChatCompletionsHandler:
    def __init__(
        self,
        session: Session,
        mindsdb_client: Server,
        messages: List[Message],
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

        await streamer.push(role=Role.system, content=f"Using model: {self.model}")
        logger.info(f"Using model: {self.model}")
        await streamer.push(role=Role.system, content="Messages received:")
        for message in self.messages:
            await streamer.push(role=message.role, content=f"\t{message.content}")
            logger.info(f"Message received: {message.role} {message.content}")

        await streamer.push(
            role=Role.system, content="This is a dummy chat completion response."
        )
        logger.info("This is a dummy chat completion response.")

        # Example MindsDB usage
        try:
            # Get available models from MindsDB
            models = self.mindsdb_client.models.list()
            await streamer.push(
                role=Role.system, content=f"Available MindsDB models: {len(models)}"
            )
            logger.info(f"Found {len(models)} MindsDB models.")

            # Example: Get databases
            databases = self.mindsdb_client.databases.list()
            await streamer.push(
                role=Role.system, content=f"Available databases: {len(databases)}"
            )
            logger.info(f"Found {len(databases)} databases.")

        except Exception as e:
            await streamer.push(
                role=Role.system, content=f"Error accessing MindsDB: {str(e)}"
            )
            logger.error(f"MindsDB error: {str(e)}")

        # Use MindsDB session for chat completions
        try:
            # Convert messages to the format expected by MindsDB
            messages_text = "\n".join(
                [f"{msg.role}: {msg.content}" for msg in self.messages]
            )

            # Example: Use MindsDB for chat completion
            # This is a placeholder - you'll need to implement the actual MindsDB chat completion logic
            await streamer.push(
                role=Role.assistant,
                content=f"Processing with MindsDB session. Messages: {messages_text}",
            )

            # For now, return a simple response
            result = f"Processed {len(self.messages)} messages with MindsDB session"
            await streamer.push(role=Role.assistant, content=result)

        except Exception as e:
            error_msg = f"Error in MindsDB chat completion: {str(e)}"
            await streamer.push(role=Role.assistant, content=error_msg)
            logger.error(error_msg)
            result = error_msg

        return result
