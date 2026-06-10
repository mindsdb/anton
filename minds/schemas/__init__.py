"""
Schema package for the Minds API (inference-only).

This package contains all Pydantic models for request/response validation
organized by domain.
"""

from .chat import ChatCompletion, ChatCompletionChunk, Choice, Message, Role, StreamChoice, Usage
from .limits import (
    LimitsConfig,
    MindLimitsConfig,
    ResourceUsageConfig,
)

__all__ = [
    # Chat schemas
    "Role",
    "Message",
    "ChatCompletion",
    "ChatCompletionChunk",
    "Choice",
    "StreamChoice",
    "Usage",
    # Limits schemas
    "LimitsConfig",
    "ResourceUsageConfig",
    "MindLimitsConfig",
]
