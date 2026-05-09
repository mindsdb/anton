from mindsdb_sdk.server import Server
from sqlmodel import Session
from starlette.responses import JSONResponse, StreamingResponse

from minds.agents.agent_controller import AgentController
from minds.agents.base import AgentRunContext
from minds.agents.helpers import get_agent
from minds.agents.passthrough_agent.agent import PassthroughAgent
from minds.common.logger import setup_logging
from minds.common.passthrough_config import is_passthrough_model, resolve_passthrough_model
from minds.common.settings.app_settings import get_app_settings
from minds.model.chat_completion import ChatCompletion
from minds.model.mind_datasource import DataCatalogStatus
from minds.requests.chat_completions_request import ChatCompletionRequestMetadata
from minds.requests.context import Context
from minds.requests.langfuse_tracing import (
    capture_langfuse_generation_context,
    update_generation_usage,
)
from minds.requests.stream import MessageStreamer
from minds.schemas.chat import Message, Role
from minds.services.conversations import ConversationsService
from minds.services.limits import LimitsService
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
        langfuse_trace_context: dict | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        limits_service: LimitsService | None = None,
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
                tools: List of tool definitions for function calling.
                tool_choice: Controls which tool the model calls.
                temperature: Sampling temperature.
                max_tokens: Maximum number of tokens to generate.
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
        # Captured @observe trace context, used to attach a child generation
        # carrying token usage when the surrounding @observe scope has already
        # closed by the time usage is known (every streaming path).
        self.langfuse_trace_context = langfuse_trace_context
        self.tools = tools
        self.tool_choice = tool_choice
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.limits_service = limits_service

        self.mind_ready = True
        self.agent = None
        self.is_passthrough = False

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
        langfuse_trace_context: dict | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        limits_service: LimitsService | None = None,
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
            tools: List of tool definitions for function calling.
            tool_choice: Controls which tool the model calls.
            temperature: Sampling temperature.
            max_tokens: Maximum number of tokens to generate.
            limits_service: The limits service for usage checking.
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
            langfuse_trace_context=langfuse_trace_context,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            limits_service=limits_service,
        )

        # Passthrough models bypass Mind lookup entirely
        if is_passthrough_model(model):
            config = resolve_passthrough_model(model)
            handler.agent = PassthroughAgent(config=config, instrument=instrument)
            handler.is_passthrough = True
            logger.debug(
                f"[{request_id}] Passthrough model {model!r} → {config.label or config.api_kind}:{config.model_name}"
            )
            return handler

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
            agent_name=get_agent(mind).value,
            mind=mind,
            mindsdb_client=mindsdb_client,
            context=context,
        )
        handler.agent = agent

        return handler

    def _save_usage(
        self,
        usage: tuple[int, int] | None,
        *,
        langfuse_trace_context: dict | None = None,
    ) -> None:
        """Persist token usage to the database AND record it on the Langfuse generation.

        ``langfuse_trace_context`` selects the Langfuse update mode:
        - ``None`` → caller is inside the @observe scope; the helper updates
          the current generation in place.
        - dict   → caller is outside the @observe scope (streaming body
          iterator runs after the handler returns); the helper attaches a
          child generation observation to the captured trace.
        """
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
            f"[{self.request_id}] Saved ChatCompletion usage: "
            f"{usage[0] if usage else 0} in / {usage[1] if usage else 0} out"
        )

        update_generation_usage(
            usage=usage,
            model=self.model,
            trace_context=langfuse_trace_context,
        )

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

        _ = await self.agent.run(
            messages=self.messages,
            streamer=streamer,
            stream=self.stream,
            run_context=AgentRunContext(metadata=self.metadata, instrument=self.instrument),
        )

        usage = await self.agent.get_last_run_usage()
        # Streaming path runs in a producer task whose lifetime outlives the
        # @observe-decorated handler — pass the captured trace context so the
        # Langfuse update attaches a child generation. Non-streaming runs
        # synchronously inside the @observe scope; in that case the caller
        # supplies langfuse_trace_context=None and the helper updates the
        # current generation directly.
        self._save_usage(usage, langfuse_trace_context=self.langfuse_trace_context if self.stream else None)

    async def proxy_chat_completions(self) -> StreamingResponse | JSONResponse:
        """Passthrough proxy — returns the upstream response directly."""
        agent: PassthroughAgent = self.agent
        response = await agent.proxy(
            messages=self.messages,
            stream=self.stream,
            request_id=self.request_id or "",
            tools=self.tools,
            tool_choice=self.tool_choice,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if isinstance(response, StreamingResponse):
            # Wrap the streaming body to save usage after the last chunk.
            original_body = response.body_iterator

            # Capture the @observe trace context NOW, while we are still
            # synchronously inside chat_completions_request_handler. The body
            # iterator below runs from the ASGI server *after* this method
            # returns, by which point the @observe span has closed.
            captured_ctx = self.langfuse_trace_context or capture_langfuse_generation_context()

            async def _wrapped_body():
                async for chunk in original_body:
                    yield chunk
                # After stream completes, save usage.
                usage = await agent.get_last_run_usage()
                self._save_usage(usage, langfuse_trace_context=captured_ctx)

            return StreamingResponse(
                _wrapped_body(),
                media_type="text/event-stream",
            )
        else:
            # Non-streaming: usage is already set on the agent and we are
            # still inside the @observe scope — update the current generation
            # in place (langfuse_trace_context=None).
            usage = await agent.get_last_run_usage()
            self._save_usage(usage, langfuse_trace_context=None)
            return response

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

        response = await self.agent.run(
            messages=self.messages,
            streamer=streamer,
            stream=self.stream,
            run_context=AgentRunContext(
                metadata=self.metadata,
                instrument=self.instrument,
                conversation_id=message.conversation_id,
                message_id=message.id,
            ),
        )

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

        # Mirror the Langfuse update done by _save_usage on the chat_completions
        # path. Streaming runs from the producer task — outside the @observe
        # scope — so we pass the captured trace context. Non-streaming uses
        # in-scope mode (None) and updates the current generation directly.
        update_generation_usage(
            usage=usage,
            model=self.model,
            trace_context=self.langfuse_trace_context if self.stream else None,
            output=response.answer if response is not None else None,
        )
