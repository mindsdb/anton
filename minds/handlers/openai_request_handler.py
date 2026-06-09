import json
from typing import Any

from sqlmodel import Session
from starlette.responses import JSONResponse, StreamingResponse

from minds.common.logger import get_logger
from minds.common.settings.app_settings import get_app_settings
from minds.inference.model_resolver import ModelResolver
from minds.inference.service import InferenceResult, InferenceService
from minds.model.chat_completion import ChatCompletion
from minds.requests.chat_completions_request import ChatCompletionRequestMetadata
from minds.requests.context import Context
from minds.requests.langfuse_tracing import (
    record_tool_call_spans,
    update_generation_usage,
)
from minds.requests.stream import MessageStreamer
from minds.schemas.chat import Message
from minds.services.conversations import ConversationsService
from minds.services.limits import LimitsService

logger = get_logger(__name__)
settings = get_app_settings()


class OpenAIRequestHandler:
    def __init__(
        self,
        session: Session,
        context: Context,
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
        """Initialize the chat completions handler for passthrough inference."""
        self.session = session
        self.context = context
        self.messages = messages
        self.model = model
        self.stream = stream
        self.metadata = metadata
        self.instrument = instrument
        self.request_id = request_id
        self.langfuse_trace_id = langfuse_trace_id
        self.langfuse_trace_context = langfuse_trace_context
        self.tools = tools
        self.tool_choice = tool_choice
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.limits_service = limits_service

        self.inference_service: InferenceService | None = None

    @classmethod
    async def create(
        cls,
        session: Session,
        context: Context,
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
        **langfuse_kwargs,
    ) -> "OpenAIRequestHandler":
        """Async factory method to create an OpenAIRequestHandler for passthrough inference."""
        handler = cls(
            session=session,
            context=context,
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

        # All models are passthrough — resolve via InferenceService
        handler.inference_service = InferenceService(model_resolver=ModelResolver(get_app_settings()))
        logger.debug(f"[{request_id}] Passthrough inference for model {model!r}")
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
        result: InferenceResult | None = None,
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
        if result:
            cfg = result.config
            model_for_langfuse = cfg.model_name
            # Typed Pydantic metadata model — typos surface at construction,
            # ``exclude_none`` drops reasoning_effort for aliases that have
            # no reasoning-level concept (Anthropic, Gemini, Fireworks).
            metadata = cfg.to_observability_metadata().to_metadata()
            logger.debug(
                f"[{self.request_id}] Langfuse metadata: "
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

    async def proxy_chat_completions(self) -> StreamingResponse | JSONResponse:
        """Passthrough proxy — returns the upstream response directly."""
        # Snapshot the request shape now so it can be attached to the
        # Langfuse generation as ``input`` regardless of streaming mode.
        input_payload = self._build_passthrough_input_payload()

        response, result = await self.inference_service.inference(
            model_name=self.model,
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
                # After stream completes, read the usage_box to get final token counts
                # and full assistant message. usage_box is a mutable reference shared
                # with the streaming generator — the adapter populates it after the
                # last chunk is yielded, so we can read the final values here.
                usage_box = result.usage_box
                if usage_box is not None:
                    usage = usage_box.value
                    output_payload = usage_box.output_payload
                    server_artifacts = list(usage_box.server_artifacts)
                else:
                    # Fallback for non-streaming (should not happen in practice)
                    usage = result.usage
                    output_payload = result.output
                    server_artifacts = result.artifacts
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
                    result=result,
                )

            return StreamingResponse(
                _wrapped_body(),
                media_type="text/event-stream",
            )

        # Non-streaming branch. ``response`` may be a successful 200
        # JSONResponse (whose ``content`` we want as the Langfuse output)
        # or an upstream-error JSONResponse the provider synthesized
        # (5xx/4xx with an ``error`` blob) — in that case we still want a
        # Langfuse generation, just one whose output is the error and whose
        # usage is zero, so failed requests don't disappear from traces.
        usage = result.usage
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
                result=result,
            )
            return response

        output_payload = result.output
        server_artifacts = result.artifacts
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
            result=result,
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
        OpenAI compatible responses API handler — passthrough inference with conversation state.

        Args:
            streamer (MessageStreamer): The streamer to push messages to.
            message (Message): The message to update.
        """
        # Snapshot the request shape for Langfuse
        input_payload = self._build_passthrough_input_payload()

        response, result = await self.inference_service.inference(
            model_name=self.model,
            messages=self.messages,
            stream=self.stream,
            request_id=self.request_id or "",
            tools=self.tools,
            tool_choice=self.tool_choice,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        conversation_service = ConversationsService(
            session=self.session,
            user_id=self.context.user_id,
            organization_id=self.context.organization_id,
        )

        if isinstance(response, StreamingResponse):
            # Wrap the streaming body to save message state after the last chunk
            original_body = response.body_iterator
            captured_ctx = self.langfuse_trace_context

            async def _wrapped_body():
                async for chunk in original_body:
                    yield chunk
                # After stream completes, extract final output and save message
                usage_box = result.usage_box
                output_payload = usage_box.output_payload if usage_box is not None else result.output

                # Extract assistant response from output
                answer = None
                if isinstance(output_payload, dict) and "content" in output_payload:
                    content = output_payload["content"]
                    if isinstance(content, list) and content:
                        answer = content[0].get("text", "")
                    elif isinstance(content, str):
                        answer = content

                if answer:
                    await conversation_service.update_message_content(
                        message=message,
                        content=answer,
                        model_name=self.model,
                        request_id=self.request_id,
                        langfuse_trace_id=self.langfuse_trace_id,
                        input_tokens=usage_box.value[0] if usage_box and usage_box.value else 0,
                        output_tokens=usage_box.value[1] if usage_box and usage_box.value else 0,
                    )

                # Record usage on Langfuse
                usage = usage_box.value if usage_box else result.usage
                adopting_upstream_trace = bool(self.context.langfuse_trace_id)
                update_generation_usage(
                    usage=usage,
                    model=self.model,
                    trace_context=captured_ctx,
                    input=input_payload,
                    output=output_payload,
                    trace_input=self.context.langfuse_trace_input if adopting_upstream_trace else None,
                    trace_output=output_payload if adopting_upstream_trace else None,
                )

            return StreamingResponse(_wrapped_body(), media_type="text/event-stream")

        # Non-streaming: update message immediately
        output_payload = result.output
        answer = None
        if isinstance(output_payload, dict) and "content" in output_payload:
            content = output_payload["content"]
            if isinstance(content, list) and content:
                answer = content[0].get("text", "")
            elif isinstance(content, str):
                answer = content

        usage = result.usage
        if answer:
            await conversation_service.update_message_content(
                message=message,
                content=answer,
                model_name=self.model,
                request_id=self.request_id,
                langfuse_trace_id=self.langfuse_trace_id,
                input_tokens=usage[0] if usage else 0,
                output_tokens=usage[1] if usage else 0,
            )

        # Record usage on Langfuse
        adopting_upstream_trace = bool(self.context.langfuse_trace_id)
        update_generation_usage(
            usage=usage,
            model=self.model,
            trace_context=None,
            input=input_payload,
            output=output_payload,
            trace_input=self.context.langfuse_trace_input if adopting_upstream_trace else None,
            trace_output=output_payload if adopting_upstream_trace else None,
        )

        return response


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
