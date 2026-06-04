from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from anton.core.llm.provider import LLMResponse, ProviderConnectionInfo, ToolCall, Usage


def make_mock_llm() -> AsyncMock:
    """Return an AsyncMock LLM client with coding_provider configured for sync use.

    ``AsyncMock`` makes all child attributes ``AsyncMock`` too, which means
    methods we call synchronously on the provider would otherwise return
    coroutines.  This helper fixes that for both providers — ``coding_provider``
    (whose ``export_connection_info()`` is read in ``ChatSession.__init__``) and
    ``planning_provider`` (whose ``native_web_tools()`` is read in the same
    constructor to resolve the per-session web tool routing).
    """
    mock = AsyncMock()
    mock.coding_provider = MagicMock()
    mock.coding_provider.export_connection_info = MagicMock(
        return_value=ProviderConnectionInfo(provider="anthropic", api_key="test")
    )
    mock.coding_model = "claude-sonnet-4-6"
    mock.planning_provider = MagicMock()
    # Default test posture: no native web tools — fallback tools also off
    # unless a specific test configures otherwise via ChatSessionConfig.
    mock.planning_provider.native_web_tools = MagicMock(return_value=set())
    return mock


@pytest.fixture()
def make_llm_response():
    def _factory(
        content: str = "",
        tool_calls: list[ToolCall] | None = None,
        input_tokens: int = 10,
        output_tokens: int = 20,
        stop_reason: str | None = "end_turn",
    ) -> LLMResponse:
        return LLMResponse(
            content=content,
            tool_calls=tool_calls or [],
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            stop_reason=stop_reason,
        )

    return _factory
