"""Inference orchestration service.

The InferenceService is the primary entry point for executing LLM inference
requests. It coordinates model resolution, adapter selection, and state capture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from starlette.responses import JSONResponse, StreamingResponse

from minds.inference.adapter import ProviderAdapter
from minds.inference.model_resolver import ModelResolver
from minds.inference.providers.anthropic_adapter import AnthropicAdapter
from minds.inference.providers.fireworks_adapter import FireworksAdapter
from minds.inference.providers.gemini_adapter import GeminiAdapter
from minds.inference.providers.openai_adapter import OpenAIAdapter
from minds.inference.types import ApiKind, UsageBox

if TYPE_CHECKING:
    from typing import Any

    from minds.inference.types import PassthroughModelConfig
    from minds.schemas.chat import Message
    from minds.schemas.passthrough import PassthroughModelStatsigConfig


@dataclass(frozen=True)
class InferenceResult:
    """Result of a single inference execution.

    Contains the resolved configuration (for observability), token usage,
    the assistant's message, and any server artifacts from the provider.

    For streaming responses, usage, output, and artifacts may be empty when
    this result is created. The handler should check again after the stream
    is fully drained via the usage_box reference.
    """

    config: PassthroughModelConfig
    usage: tuple[int, int] | None
    output: dict[str, Any] | None
    artifacts: list[dict[str, Any]]
    usage_box: UsageBox | None = None


class InferenceService:
    """Orchestrates LLM inference across multiple providers.

    Resolves model aliases, selects the appropriate adapter, delegates to
    the provider, and captures state for observability.
    """

    def __init__(self, model_resolver: ModelResolver) -> None:
        """Initialize the service with a model resolver.

        Args:
            model_resolver: Resolver for mapping model aliases to configs.
        """
        self.model_resolver = model_resolver

    async def inference(
        self,
        model_name: str,
        messages: list[Message],
        stream: bool,
        request_id: str,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        policy: PassthroughModelStatsigConfig | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[StreamingResponse | JSONResponse, InferenceResult]:
        """Execute an inference request.

        Resolves the model alias to a provider config, creates the appropriate
        adapter, delegates to the provider, and captures the response along with
        usage metadata.

        Args:
            model_name: Passthrough model alias (e.g., ``latest:sonnet``).
            messages: Chat messages to send to the model.
            stream: Whether to stream the response.
            request_id: Unique request identifier for tracing.
            tools: Optional list of tool definitions.
            tool_choice: Optional tool choice constraint.
            temperature: Optional temperature override.
            max_tokens: Optional max tokens override.
            policy: Per-user passthrough routing policy from Statsig (alias
                overrides, allow-list, search settings, effort overrides).
                ``None`` = no policy.
            reasoning_effort: Client-requested effort level; validated against
                the resolved concrete model (400 on mismatch).

        Returns:
            Tuple of (HTTP response, InferenceResult with captured state).

        Raises:
            HTTPException: If model alias is unknown or provider unconfigured.
            ValueError: If api_kind is not recognized.
        """
        # Step 1: Resolve model alias to config (stamping per-user policy)
        config = self.model_resolver.resolve(model_name, policy=policy, reasoning_effort=reasoning_effort)

        # Step 2: Create fresh adapter for this request
        adapter = self._create_adapter(config.api_kind)

        # Step 3: Delegate to adapter
        response = await adapter.complete(
            config=config,
            messages=messages,
            stream=stream,
            request_id=request_id,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Step 4: Capture state from adapter
        usage = adapter.get_last_usage()
        output = adapter.get_last_output()
        artifacts = adapter.get_last_artifacts()
        usage_box = adapter.get_usage_box()

        # Step 5: Create result
        result = InferenceResult(
            config=config,
            usage=usage,
            output=output,
            artifacts=artifacts,
            usage_box=usage_box,
        )

        # Step 6: Return response and result
        return response, result

    def _create_adapter(self, api_kind: ApiKind) -> ProviderAdapter:
        """Create a fresh adapter for the given API kind.

        Args:
            api_kind: The upstream provider transport type.

        Returns:
            A new adapter instance (OpenAIAdapter, AnthropicAdapter,
            GeminiAdapter, or FireworksAdapter).

        Raises:
            ValueError: If api_kind is not recognized.
        """
        if api_kind == ApiKind.OPENAI_RESPONSES:
            return OpenAIAdapter()
        elif api_kind == ApiKind.ANTHROPIC_MESSAGES:
            return AnthropicAdapter()
        elif api_kind == ApiKind.GEMINI_NATIVE:
            return GeminiAdapter()
        elif api_kind == ApiKind.FIREWORKS:
            return FireworksAdapter()
        else:
            raise ValueError(f"Unknown api_kind: {api_kind}")
