from abc import ABC, abstractmethod
from typing import Any

from mindsdb_sdk.server import Server
from pydantic import BaseModel, Field

from minds.agents.base_response import AgentResponse
from minds.model.mind import Mind
from minds.requests.chat_completions_request import ChatCompletionRequestMetadata
from minds.requests.stream import MessageStreamer
from minds.schemas.chat import Message


class AgentRunContext(BaseModel):
    metadata: ChatCompletionRequestMetadata | None = Field(default=None)
    instrument: bool = Field(default=True)


class BaseAgentConfig(BaseModel):
    instrument: bool = Field(default=True)


class BaseAgent(ABC):
    """
    This is the base agent class that all agents should inherit from.

    Args:
        mind: The mind to use for the agent.
    """

    def __init__(self, mind: Mind, mindsdb_client: Server | None = None, config: Any | None = None):
        self.mind = mind
        self.mindsdb_client = mindsdb_client
        self.config = config

    @classmethod
    def build_config(cls, run_context: AgentRunContext) -> BaseAgentConfig:
        return BaseAgentConfig(instrument=run_context.instrument)

    @abstractmethod
    async def run(self, messages: list[Message], streamer: MessageStreamer, stream: bool = True) -> AgentResponse: ...

    @abstractmethod
    async def get_last_run_usage(self) -> tuple[int, int] | None: ...
