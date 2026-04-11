from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    context_pressure: float = 0.0


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None


@dataclass
class StreamTextDelta:
    text: str


@dataclass
class StreamToolUseStart:
    id: str
    name: str


@dataclass
class StreamToolUseDelta:
    id: str
    json_delta: str


@dataclass
class StreamToolUseEnd:
    id: str


@dataclass
class StreamComplete:
    response: LLMResponse


@dataclass
class StreamTaskProgress:
    """Progress event from agent task execution (planning, building, executing)."""

    phase: str
    message: str
    eta_seconds: float | None = None


@dataclass
class StreamToolResult:
    """Tool result that should be displayed to the user (e.g. scratchpad dump)."""

    content: str


@dataclass
class StreamContextCompacted:
    """Notification that context was compacted to free up space."""

    message: str


StreamEvent = (
    StreamTextDelta
    | StreamToolUseStart
    | StreamToolUseDelta
    | StreamToolUseEnd
    | StreamComplete
    | StreamTaskProgress
    | StreamToolResult
    | StreamContextCompacted
)


_CONTEXT_WINDOWS: list[tuple[str, int]] = [
    # Anton defaults (exact model IDs first)
    ("claude-sonnet-4-6", 200_000),
    ("claude-haiku-4-5-20251001", 200_000),
    # Claude families
    ("claude-opus-4", 200_000),
    ("claude-sonnet-4", 200_000),
    ("claude-haiku-4", 200_000),
    ("claude-3", 200_000),
    ("claude-", 200_000),
    # OpenAI families
    ("gpt-5", 400_000),
    ("gpt-4.1", 1_000_000),
    ("gpt-4o", 128_000),
    ("gpt-4", 128_000),
    ("o3", 200_000),
    ("o1", 200_000),
]
_DEFAULT_CONTEXT_WINDOW = 128_000


def compute_context_pressure(model: str, input_tokens: int) -> float:
    """Return input_tokens / context_window as a 0.0–1.0 float."""
    window = _DEFAULT_CONTEXT_WINDOW
    for prefix, size in _CONTEXT_WINDOWS:
        if model.startswith(prefix):
            window = size
            break
    return min(input_tokens / window, 1.0)


class ContextOverflowError(Exception):
    """Raised when the LLM rejects a request due to context length exceeded."""

    def __init__(self, message: str, input_tokens: int = 0, limit: int = 0):
        super().__init__(message)
        self.input_tokens = input_tokens
        self.limit = limit


class TokenLimitExceeded(Exception):
    """Raised when the LLM returns 429 due to billing/token limits."""


class LLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse: ...

    async def stream(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamEvent]:
        """Stream LLM responses. Default falls back to complete()."""
        response = await self.complete(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
        )
        if response.content:
            yield StreamTextDelta(text=response.content)
        yield StreamComplete(response=response)
