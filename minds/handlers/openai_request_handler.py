from mindsdb_sdk.server import Server
from sqlmodel import Session

from minds.agents.agent_controller import AgentController
from minds.agents.base import AgentRunContext
from minds.common.logger import setup_logging
from minds.common.settings.app_settings import get_app_settings
from minds.model.chat_completion import ChatCompletion
from minds.model.mind_datasource import DataCatalogStatus
from minds.requests.chat_completions_request import ChatCompletionRequestMetadata
from minds.requests.context import Context
from minds.requests.stream import MessageStreamer
from minds.schemas.chat import Message, Role
from minds.services.conversations import ConversationsService
from minds.services.minds import MindsService

logger = setup_logging()
settings = get_app_settings()


class OpenAIRequestHandler:
    def __init__(
        self,
        session: Session,
        context: Context,
        mindsdb_client: Server,
        messages: list[Message],
        model: str,
        stream: bool,
        metadata: ChatCompletionRequestMetadata | None = None,
        instrument: bool = True,
        request_id: str | None = None,
        langfuse_trace_id: str | None = None,
    ):
        """
        Initialize the ChatCompletionsHandler with a list of messages.

        Args:
                session (Session): The SQLAlchemy session for database operations.
                mindsdb_client (Server): The MindsDB client for database operations.
                messages (List[Message]): List of messages to handle.
                model (str): The model to use for chat completions.
                stream (bool): Whether to stream the response.
                instrument (bool): Whether to instrument the PydanticAIAgent.
                request_id: The request ID.
                langfuse_trace_id: The Langfuse trace ID.
        """
        self.session = session
        self.context = context
        self.mindsdb_client = mindsdb_client
        self.messages = messages
        self.model = model
        self.stream = stream
        self.metadata = metadata
        self.instrument = instrument
        self.request_id = request_id
        self.langfuse_trace_id = langfuse_trace_id

        self.mind_ready = True
        self.agent = None

    @classmethod
    async def create(
        cls,
        session: Session,
        context: Context,
        mindsdb_client: Server,
        messages: list[Message],
        model: str,
        stream: bool,
        metadata: ChatCompletionRequestMetadata | None = None,
        instrument: bool = True,
        request_id: str | None = None,
        langfuse_trace_id: str | None = None,
    ) -> "OpenAIRequestHandler":
        """
        Async factory method to create a OpenAIRequestHandler instance.

        Args:
            session (Session): The SQLAlchemy session for database operations.
            context (Context): The context of the request.
            mindsdb_client (Server): The MindsDB client for database operations.
            messages (List[Message]): List of messages to handle.
            model (str): The model to use for chat completions.
            stream (bool): Whether to stream the response.
            metadata (ChatCompletionRequestMetadata): The metadata for the chat completion request.
            instrument (bool): Whether to instrument the PydanticAIAgent.
            request_id: The request ID.
            langfuse_trace_id: The Langfuse trace ID.
        """
        handler = cls(
            session=session,
            context=context,
            mindsdb_client=mindsdb_client,
            messages=messages,
            model=model,
            stream=stream,
            metadata=metadata,
            instrument=instrument,
            request_id=request_id,
            langfuse_trace_id=langfuse_trace_id,
        )

        minds_service = MindsService(
            session=session,
            mindsdb_client=mindsdb_client,
            user_id=context.user_id,
            organization_id=context.organization_id,
        )
        mind = await minds_service.get_mind_model(model)

        # If the Mind has datasources that are currently loading, inform the user
        # and complete the request
        statuses = []
        for relationship in mind.mind_datasources:
            status = await relationship.status
            statuses.append(status)
        if any(status in [DataCatalogStatus.LOADING, DataCatalogStatus.PENDING] for status in statuses):
            handler.mind_ready = False

        agent_controller = AgentController()
        agent = agent_controller.get_agent(
            agent_name=mind.parameters.get("agent_name") or settings.agents.default_agent,
            mind=mind,
            mindsdb_client=mindsdb_client,
            run_context=AgentRunContext(
                metadata=metadata,
                instrument=instrument,
            ),
        )
        handler.agent = agent

        return handler

    async def chat_completions(self, streamer: MessageStreamer):
        """
        OpenAI compatible chat completions API handler.

        Args:
            streamer (MessageStreamer): The streamer to push messages to.

        Returns:
            None: No return value.
        """
        if not self.mind_ready:
            await streamer.push(role=Role.assistant, content="The Mind is not ready yet. Please try again later.")
            return

        _ = await self.agent.run(messages=self.messages, streamer=streamer, stream=self.stream)

        usage = await self.agent.get_last_run_usage()

        chat_completion = ChatCompletion(
            organization_id=self.context.organization_id,
            user_id=self.context.user_id,
            model_name=self.model,
            request_id=self.request_id,
            langfuse_trace_id=self.langfuse_trace_id,
            input_tokens=usage[0] if usage else 0,
            output_tokens=usage[1] if usage else 0,
        )
        self.session.add(chat_completion)
        self.session.commit()
        logger.debug(
            f"🔄[{self.request_id}] Saved ChatCompletion usage: "
            f"{usage[0] if usage else 0} in / {usage[1] if usage else 0} out"
        )

    async def responses(self, streamer: MessageStreamer, message: Message):
        """
        OpenAI compatible responses API handler.

        Args:
            streamer (MessageStreamer): The streamer to push messages to.
            message (Message): The message to update.
        """
        if not self.mind_ready:
            await streamer.push(role=Role.assistant, content="The Mind is not ready yet. Please try again later.")
            return

        conversation_service = ConversationsService(
            session=self.session,
            mindsdb_client=self.mindsdb_client,
            user_id=self.context.user_id,
            organization_id=self.context.organization_id,
        )

        response = await self.agent.run(messages=self.messages, streamer=streamer, stream=self.stream)

        usage = await self.agent.get_last_run_usage()

        await conversation_service.update_conversation_message_content(
            message=message,
            content=response.answer,
            sql_query=response.sql,
            model_name=self.model,
            request_id=self.request_id,
            langfuse_trace_id=self.langfuse_trace_id,
            input_tokens=usage[0] if usage else 0,
            output_tokens=usage[1] if usage else 0,
        )
