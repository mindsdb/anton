"""
Chat completions handler — inference-only passthrough.

All requests are resolved to provider configs via InferenceService.
"""

from sqlmodel import Session
from starlette.responses import JSONResponse, StreamingResponse

from minds.common.logger import get_logger
from minds.handlers.openai_request_handler import OpenAIRequestHandler
from minds.requests.chat_completions_request import ChatCompletionsRequest
from minds.requests.context import Context
from minds.requests.langfuse_tracing import (
    capture_langfuse_generation_context,
    get_langfuse_trace_id,
    lazy_observe,
    setup_langfuse_observation,
)
from minds.services.limits import LimitsService

logger = get_logger(__name__)


@lazy_observe(
    name="Chat Completions Handler v1",
    as_type="generation",
    capture_input=False,
    capture_output=False,
)
async def chat_completions_request_handler(
    session: Session,
    context: Context,
    chat_completions_request: ChatCompletionsRequest,
    instrument: bool = True,
    limits_service: LimitsService | None = None,
    **langfuse_kwargs,
) -> StreamingResponse | JSONResponse:
    """
    Handle passthrough chat completions requests.

    Routes all models through InferenceService to the configured providers.
    """
    setup_langfuse_observation(context=context)
    langfuse_trace_id = get_langfuse_trace_id()
    request_id = langfuse_trace_id or str(context.request_id)

    langfuse_trace_context = capture_langfuse_generation_context()

    logger.debug(f"Chat completions for model: {chat_completions_request.model}")

    handler = await OpenAIRequestHandler.create(
        session=session,
        context=context,
        messages=chat_completions_request.messages,
        model=chat_completions_request.model,
        stream=chat_completions_request.stream or False,
        metadata=chat_completions_request.metadata,
        instrument=instrument,
        request_id=request_id,
        langfuse_trace_id=langfuse_trace_id,
        langfuse_trace_context=langfuse_trace_context,
        tools=chat_completions_request.tools,
        tool_choice=chat_completions_request.tool_choice,
        temperature=chat_completions_request.temperature,
        max_tokens=chat_completions_request.max_completion_tokens or chat_completions_request.max_tokens,
        reasoning_effort=chat_completions_request.reasoning_effort,
        limits_service=limits_service,
        **langfuse_kwargs,
    )

    return await handler.proxy_chat_completions()
