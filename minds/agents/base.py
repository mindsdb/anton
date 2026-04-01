from abc import ABC, abstractmethod
from uuid import UUID

from mindsdb_sdk.server import Server
from pydantic import BaseModel, Field

from minds.agents.base_response import AgentResponse
from minds.model.mind import Mind
from minds.requests.chat_completions_request import ChatCompletionRequestMetadata
from minds.requests.context import Context
from minds.requests.stream import MessageStreamer
from minds.schemas.chat import Message


class AgentRunContext(BaseModel):
    metadata: ChatCompletionRequestMetadata | None = Field(default=None)
    instrument: bool = Field(default=True)

    # Applies only when calling the Conversations API
    conversation_id: UUID | None = Field(default=None)
    message_id: UUID | None = Field(default=None)


class BaseAgent(ABC):
    """
    This is the base agent class that all agents should inherit from.

    Args:
        mind: The mind to use for the agent.
    """

    def __init__(
        self,
        mind: Mind,
        mindsdb_client: Server | None = None,
        context: Context | None = None,
    ):
        self.mind = mind
        self.mindsdb_client = mindsdb_client
        self.context = context

    async def run(
        self,
        messages: list[Message],
        streamer: MessageStreamer,
        run_context: AgentRunContext,
        stream: bool,
    ) -> AgentResponse:
        # Check usage limits before processing

        return await self._run(messages=messages, streamer=streamer, stream=stream, run_context=run_context)

    @abstractmethod
    async def _run(
        self,
        messages: list[Message],
        streamer: MessageStreamer,
        run_context: AgentRunContext,
        stream: bool,
    ) -> AgentResponse: ...

    @abstractmethod
    async def get_last_run_usage(self) -> tuple[int, int] | None: ...
