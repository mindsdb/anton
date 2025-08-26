"""
Schema package for the Minds API.

This package contains all Pydantic models for request/response validation
organized by domain.
"""

from .chat import ChatCompletion, ChatCompletionChunk, Choice, Message, Role, StreamChoice, Usage
from .minds import (
    AddDatasourceRequest,
    DeleteMindRequest,
    MindCreateRequest,
    MindDatasourceResponse,
    MindResponse,
    MindUpdateRequest,
)

__all__ = [
    # Chat schemas
    "Role", "Message", "ChatCompletion", "ChatCompletionChunk",
    "Choice", "StreamChoice", "Usage",
    
    # Mind schemas  
    "MindCreateRequest", "MindUpdateRequest", "MindResponse",
    "AddDatasourceRequest", "DeleteMindRequest", "MindDatasourceResponse",
]