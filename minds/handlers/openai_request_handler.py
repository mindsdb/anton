import json
from typing import Any

from mindsdb_sdk.server import Server
from sqlmodel import Session
from starlette.responses import JSONResponse, StreamingResponse

from minds.agents.agent_controller import AgentController
from minds.agents.base import AgentRunContext
from minds.agents.helpers import get_agent
from minds.agents.passthrough_agent.agent import PassthroughAgent
from minds.common.logger import get_logger
from minds.common.passthrough_config import is_passthrough_model, resolve_passthrough_model
from minds.common.settings.app_settings import get_app_settings
from minds.model.chat_completion import ChatCompletion
from minds.model.mind_datasource import DataCatalogStatus
from minds.requests.chat_completions_request import ChatCompletionRequestMetadata
from minds.requests.context import Context
from minds.requests.langfuse_tracing import (
    record_tool_call_spans,
    update_generation_usage,
)
from minds.requests.stream import MessageStreamer
from minds.schemas.chat import Message, Role
from minds.services.conversations import ConversationsService
from minds.services.limits import LimitsService
from minds.services.minds import MindsService

logger = get_logger(__name__)
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
        input_payload: Any = None,
        output_payload: Any = None,
        extra_metadata: dict | None = None,
        tool_calls_for_spans: list[dict] | None = None,
    ) -> None:
        """Persist token usage to the database AND record it on the Langfuse generation.

        ``langfuse_trace_context`` selects the Langfuse update mode:
        - ``None`` → caller is inside the @observe scope; the helper updates
          the current generation in place.
        - dict   → caller is outside the @observe scope (streaming body
          iterator runs after the handler returns); the helper attaches a
          child generation observation to the captured trace.

        For passthrough requests, Langfuse records the **concrete** upstream
        model (``cfg.model_name`` like ``claude-sonnet-4-6``) rather than the
        alias (``_sonnet_``) so its cost rollup lands against an actual entry
        in its model registry. The alias plus provider context goes onto
        ``metadata`` so analytics can still slice by the alias surface. The
        DB ``ChatCompletion`` row keeps ``self.model`` (the alias) because
        that's what the client called us with — useful for attribution back
        to the alias.

        ``input_payload`` / ``output_payload`` flow through to Langfuse so
        the trace is eval-replayable: the prompt + tools (input) and the
        assistant message + tool_calls (output) are visible without
        re-fetching from the provider. ``extra_metadata`` is merged into
        the per-passthrough metadata blob — used today to carry
        ``server_artifacts`` (server-side web_search/fetch/reasoning items
        the streaming converter captured). ``tool_calls_for_spans`` triggers
        one Langfuse child span per call so tool selection + arguments
        become filterable in the UI.
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

        model_for_langfuse = self.model
        metadata: dict | None = None
        if self.is_passthrough and isinstance(self.agent, PassthroughAgent):
            cfg = self.agent.config
            model_for_langfuse = cfg.model_name
            # Typed Pydantic metadata model — typos surface at construction,
            # ``exclude_none`` drops reasoning_effort for aliases that have
            # no reasoning-level concept (Anthropic, Gemini, Fireworks).
            metadata = cfg.to_observability_metadata().to_metadata()
            logger.debug(
                f"[{self.request_id}] Passthrough Langfuse metadata: "
                f"model={model_for_langfuse} alias={cfg.alias} "
                f"provider={cfg.label} reasoning_effort={cfg.reasoning_effort}"
            )
        if extra_metadata:
            metadata = {**(metadata or {}), **extra_metadata}

        # Whole-turn eval support: only when the caller adopted an upstream trace
        # (Langfuse-Trace-Id present) do we stamp trace-level I/O. The trace input
        # is the caller-supplied turn input (idempotent across the turn's calls);
        # the trace output is this call's output (last call of the turn wins, so
        # the final answer ends up as the trace output). For requests that own
        # their trace we leave trace I/O untouched (unchanged behaviour).
        adopting_upstream_trace = bool(self.context.langfuse_trace_id)
        trace_input = self.context.langfuse_trace_input if adopting_upstream_trace else None
        trace_output = output_payload if adopting_upstream_trace else None

        update_generation_usage(
            usage=usage,
            model=model_for_langfuse,
            trace_context=langfuse_trace_context,
            metadata=metadata,
            input=input_payload,
            output=output_payload,
            trace_input=trace_input,
            trace_output=trace_output,
        )

        if tool_calls_for_spans:
            # Tool-call child spans share the same trace context as the
            # parent generation update so they nest correctly in the
            # Langfuse UI. ``metadata`` carries alias/provider so a
            # "show me every web_search call we made via latest:sonnet"
            # query is one filter away.
            record_tool_call_spans(
                tool_calls=tool_calls_for_spans,
                trace_context=langfuse_trace_context,
                metadata=metadata,
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
        # Snapshot the request shape now so it can be attached to the
        # Langfuse generation as ``input`` regardless of streaming mode.
        input_payload = self._build_passthrough_input_payload()

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

            # The trace context was already captured at the top of
            # chat_completions_request_handler (the only caller of this method)
            # while inside the @observe scope, and threaded in here. The body
            # iterator below runs from the ASGI server *after* that scope has
            # closed, so it relies on this captured value.
            captured_ctx = self.langfuse_trace_context

            async def _wrapped_body():
                async for chunk in original_body:
                    yield chunk
                # After stream completes, save usage AND the assistant
                # message + server artifacts the converter accumulated.
                usage = await agent.get_last_run_usage()
                output_payload = agent.get_last_run_output()
                server_artifacts = agent.get_last_run_server_artifacts()
                self._save_usage(
                    usage,
                    langfuse_trace_context=captured_ctx,
                    input_payload=input_payload,
                    output_payload=output_payload,
                    extra_metadata=(
                        {"server_artifacts": server_artifacts}
                        if isinstance(server_artifacts, list) and server_artifacts
                        else None
                    ),
                    tool_calls_for_spans=output_payload.get("tool_calls") if isinstance(output_payload, dict) else None,
                )

            return StreamingResponse(
                _wrapped_body(),
                media_type="text/event-stream",
            )

        # Non-streaming branch. ``response`` may be a successful 200
        # JSONResponse (whose ``content`` we want as the Langfuse output)
        # or an upstream-error JSONResponse the provider proxy synthesized
        # (5xx/4xx with an ``error`` blob) — in that case we still want a
        # Langfuse generation, just one whose output is the error and whose
        # usage is zero, so failed requests don't disappear from traces.
        usage = await agent.get_last_run_usage()
        is_error = getattr(response, "status_code", 200) >= 400
        if is_error:
            output_payload = _extract_jsonresponse_content(response)
            extra: dict[str, Any] = {"status_code": response.status_code, "level": "ERROR"}
            self._save_usage(
                usage or (0, 0),
                langfuse_trace_context=None,
                input_payload=input_payload,
                output_payload=output_payload,
                extra_metadata=extra,
            )
            return response

        output_payload = agent.get_last_run_output()
        server_artifacts = agent.get_last_run_server_artifacts()
        self._save_usage(
            usage,
            langfuse_trace_context=None,
            input_payload=input_payload,
            output_payload=output_payload,
            extra_metadata=(
                {"server_artifacts": server_artifacts}
                if isinstance(server_artifacts, list) and server_artifacts
                else None
            ),
            tool_calls_for_spans=output_payload.get("tool_calls") if isinstance(output_payload, dict) else None,
        )
        return response

    def _build_passthrough_input_payload(self) -> dict[str, Any]:
        """Snapshot the inbound request as a JSON-safe dict for Langfuse ``input``.

        ``messages`` is dumped via Pydantic ``model_dump`` so the trace
        captures the exact shape the agent saw; ``tools`` / ``tool_choice``
        / ``temperature`` / ``max_tokens`` are passed through as-is so an
        eval replay can reconstruct the upstream call without re-deriving
        them from the route layer.
        """
        return {
            "model": self.model,
            "stream": self.stream,
            "messages": [m.model_dump() for m in self.messages],
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

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
        # When the caller adopted an upstream trace, also stamp trace-level I/O
        # for whole-turn evals (see _save_usage for the rationale).
        adopting_upstream_trace = bool(self.context.langfuse_trace_id)
        answer = response.answer if response is not None else None
        update_generation_usage(
            usage=usage,
            model=self.model,
            trace_context=self.langfuse_trace_context if self.stream else None,
            output=answer,
            trace_input=self.context.langfuse_trace_input if adopting_upstream_trace else None,
            trace_output=answer if adopting_upstream_trace else None,
        )


def _extract_jsonresponse_content(response: JSONResponse) -> Any:
    """Decode the JSON body of a Starlette ``JSONResponse`` back to a Python value.

    Provider proxies synthesize an error ``JSONResponse`` (4xx/5xx) before
    we get to record it on Langfuse. ``response.body`` is the already-
    serialized bytes — round-trip through ``json.loads`` so the trace
    stores the structured ``{"error": ...}`` blob the client sees rather
    than an opaque bytes literal. Falls back to a stringified body on
    decode failure so the trace still gets something useful.
    """
    try:
        return json.loads(response.body)
    except (TypeError, ValueError, AttributeError):
        return {"raw": str(getattr(response, "body", ""))}
