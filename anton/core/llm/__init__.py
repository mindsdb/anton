"""Reusable LLM abstractions and providers.

This package is the canonical home for Anton's LLM layer.
"""

from .anthropic import AnthropicProvider
from .client import LLMClient
from .openai import OpenAIProvider, build_chat_completion_kwargs
from .prompts import CHAT_SYSTEM_PROMPT, LEARNING_EXTRACT_PROMPT, build_visualizations_prompt
from .provider import (
    ContextOverflowError,
    LLMProvider,
    LLMResponse,
    StreamComplete,
    StreamContextCompacted,
    StreamEvent,
    StreamTaskProgress,
    StreamTextDelta,
    StreamToolResult,
    StreamToolUseDelta,
    StreamToolUseEnd,
    StreamToolUseStart,
    ToolCall,
    Usage,
    compute_context_pressure,
)

__all__ = [
    "AnthropicProvider",
    "LLMClient",
    "OpenAIProvider",
    "build_chat_completion_kwargs",
    "CHAT_SYSTEM_PROMPT",
    "LEARNING_EXTRACT_PROMPT",
    "build_visualizations_prompt",
    "ContextOverflowError",
    "LLMProvider",
    "LLMResponse",
    "StreamComplete",
    "StreamContextCompacted",
    "StreamEvent",
    "StreamTaskProgress",
    "StreamTextDelta",
    "StreamToolResult",
    "StreamToolUseDelta",
    "StreamToolUseEnd",
    "StreamToolUseStart",
    "ToolCall",
    "Usage",
    "compute_context_pressure",
]
