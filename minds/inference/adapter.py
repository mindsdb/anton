"""Abstract interface for LLM provider adapters.

Each adapter translates OpenAI-format requests to a provider's native format,
executes the request, and translates the response back to OpenAI format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from starlette.responses import JSONResponse, StreamingResponse

if TYPE_CHECKING:
    from minds.inference.types import PassthroughModelConfig
    from minds.inference.types import UsageBox
    from minds.schemas.chat import Message


class ProviderAdapter(ABC):
    """Abstract interface for LLM provider adapters.

    Adapters handle:
    1. Request translation (OpenAI format → provider format)
    2. Provider API calls
    3. Response translation (provider format → OpenAI format)
    4. State tracking (usage, output, artifacts)

    Implementations MUST:
    - Accept OpenAI-format messages and tools
    - Return OpenAI-format responses (StreamingResponse or JSONResponse)
    - Track usage and output for retrieval after complete() returns
    - Support both streaming and non-streaming modes
    """

    @abstractmethod
    async def complete(
        self,
        config: PassthroughModelConfig,
        messages: list[Message],
        stream: bool,
        request_id: str,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> StreamingResponse | JSONResponse:
        """Execute an LLM inference request.

        Args:
            config: Resolved provider configuration (api_kind, model_name, api_key, etc.)
            messages: OpenAI-format message list
            stream: Whether to return streaming response
            request_id: Unique request identifier for logging
            tools: OpenAI-format function-calling tools (optional)
            tool_choice: Tool selection constraint (optional)
            temperature: Sampling temperature (optional)
            max_tokens: Maximum output tokens (optional)

        Returns:
            StreamingResponse (if stream=True) or JSONResponse (if stream=False).
            Response body is always OpenAI chat.completion format.

        Side effects:
            Populates internal state for get_last_usage(), get_last_output(),
            get_last_artifacts(). For streaming responses, these are available
            only after the stream completes.
        """
        pass

    @abstractmethod
    async def get_last_usage(self) -> tuple[int, int] | None:
        """Return token usage from the most recent complete() call.

        Returns:
            (input_tokens, output_tokens) tuple, or None if no completed request.
            For streaming responses, only available after stream completes.
        """
        pass

    @abstractmethod
    def get_last_output(self) -> dict[str, Any] | None:
        """Return the OpenAI-format assistant message from the most recent complete() call.

        Format: {"role": "assistant", "content": "...", "tool_calls": [...]}

        Returns:
            dict or None if no completed request or request had an error.
            For streaming responses, only available after stream completes.
        """
        pass

    @abstractmethod
    def get_last_artifacts(self) -> list[dict[str, Any]]:
        """Return server artifacts from the most recent complete() call.

        Server artifacts are provider-internal outputs not sent to the client:
        - Web search results (Anthropic, OpenAI)
        - Reasoning chains (OpenAI o1-preview)
        - Function call results
        - etc.

        These are recorded on the Langfuse trace for evaluation and debugging.

        Returns:
            List of dicts (possibly empty), never None.
            For streaming responses, only available after stream completes.
        """
        pass

    @abstractmethod
    def get_usage_box(self) -> UsageBox | None:
        """Return the UsageBox from the most recent complete() call.

        For streaming responses, the box is populated as the stream is drained.
        Handlers should check this after streaming completes to access updated
        usage, output, and artifacts.

        Returns:
            The UsageBox or None if no request has been made.
        """
        pass
