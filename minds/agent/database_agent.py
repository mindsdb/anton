import textwrap
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime

from pydantic_ai import Agent as PydanticAIAgent
from pydantic_ai import RunContext

from minds.agent.database_toolkit import DatabaseToolkit
from minds.agent.llm import get_llm_config
from minds.agent.prompt_templates import CHART_GENERATION_INSTRUCTIONS
from minds.common.logger import setup_logging
from minds.model.mind import Mind
from minds.requests.stream import MessageStreamer
from minds.schemas.chat import Message, Role

logger = setup_logging()


@dataclass
class DatabaseDeps:
    """Dependencies for the database agent."""

    toolkit: DatabaseToolkit
    conversation_context: str | None = None
    streamer: MessageStreamer | None = None


@dataclass
class DatabaseAgentConfig:
    """Config for the database agent."""

    enable_charting: bool = False
    instrument: bool = True


class DatabaseAgent:
    """Experimental agent implementation using Pydantic instead of Langchain.

    This class is intended to replace LangchainAgent but maintains the same interface
    to support a smooth transition away from the Langchain dependency.
    """

    def __init__(
        self,
        mind: Mind,
        database_toolkit: DatabaseToolkit,
        config: DatabaseAgentConfig | None = None,
    ):
        """Initialize the PydanticAgent.

        Args:
            mind: The database mind record.
            database_toolkit: DatabaseToolkit instance for executing database operations.
            config: Config for the database agent.
        """

        self.mind = mind
        self.deps = DatabaseDeps(toolkit=database_toolkit)
        if not config:
            config = DatabaseAgentConfig()
        self.config = config
        self.last_run_usage = None

        PydanticAIAgent.instrument_all(instrument=self.config.instrument)

        self._pydantic_agent = self._setup_agent()

    def _setup_agent(self) -> PydanticAIAgent:
        """Set up and configure the Pydantic AI agent with database tools.

        Returns:
            PydanticAIAgent: Configured agent with database tools.
        """
        llm_model = get_llm_config(self.mind.provider, self.mind.model_name)

        system_prompt = self._get_system_prompt()

        agent = PydanticAIAgent(
            model=llm_model,
            system_prompt=system_prompt,
            deps_type=DatabaseDeps,
        )

        @agent.tool
        async def generate_and_execute_sql(ctx: RunContext[DatabaseDeps]) -> str:
            """Generate and execute SQL to answer the user's question.

            This tool does not require any input parameters. The user's question is already available
            through the context dependencies and will be used to generate appropriate SQL.
            Do not pass any parameters to this tool.

            Returns:
                The database response, formatted as human readable text.
            """
            return await ctx.deps.toolkit.generate_and_execute_sql(
                ctx.deps.conversation_context,
                ctx.deps.streamer,
            )

        return agent

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the database agent.

        Returns:
            The system prompt for the database agent.
        """

        prompt = textwrap.dedent("""
            You are a helpful database assistant created by MindsDB that can query databases and provide insights.
            When querying data, automatically order results by the variable most relevant to the question, in the
            direction that provides the most actionable insights, unless otherwise specified.
        """).strip()

        # Add Mind-specific system prompt if available
        if self.mind.parameters.get("system_prompt") or self.mind.parameters.get("prompt_template"):
            prompt += "\n\n" + (
                self.mind.parameters.get("system_prompt") or self.mind.parameters.get("prompt_template")
            )

        # Add charting instructions if enabled
        if self.config and getattr(self.config, "enable_charting", False):
            prompt += "\n\n" + CHART_GENERATION_INSTRUCTIONS

        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")

        prompt += f"\n\nCurrent date: {current_date}\nCurrent time: {current_time}"

        return prompt

    def _build_conversation_context(self, messages: list[Message]) -> str:
        """Build a conversation context string from messages.

        Args:
            messages: List of conversation messages.

        Returns:
            Formatted conversation context string.
        """
        if not messages:
            return ""

        if len(messages) == 1:
            return messages[0].content if messages[0].content else ""

        # Build conversation context from multiple messages
        context_parts = []
        for msg in messages:
            role = msg.role.name if msg.role else "user"
            content = msg.content if msg.content else ""

            if role == "user":
                context_parts.append(f"User: {content}")
            elif role == "assistant":
                context_parts.append(f"Assistant: {content}")
            elif role == "system":
                context_parts.append(f"System: {content}")

        conversation_context = "\n\n".join(context_parts)

        # Add instruction for the agent to understand this is a conversation
        if len(messages) > 1:
            conversation_context = (
                "This is a conversation history. Please respond to the most recent user message "
                f"while considering the full context:\n\n{conversation_context}"
                "IMPORTANT: Use the prior conversation only for context and intent. "
                "Do not copy or reuse previous answers, even if the same or similar questions appear, "
                "as the underlying data, schema, or results may have changed."
            )

        return conversation_context

    async def get_completion(self, messages: list[Message], stream: bool = False) -> AsyncGenerator[str, None]:
        """Get completion from the Pydantic-based agent.

        Args:
            messages: List of message dictionaries.
            stream: Whether to stream the response.

        Returns:
            ChatCompletionCustom object for non-streaming or a generator for streaming.
        """
        # Use the preconfigured agent with tools
        agent = self._pydantic_agent

        # Build conversation context string from all messages
        conversation_context = self._build_conversation_context(messages)

        # Store the complete conversation context for the tool context
        self.deps.conversation_context = conversation_context

        if stream:
            async with agent.run_stream(conversation_context, deps=self.deps) as result:
                async for chunk in result.stream_text(delta=True):
                    yield chunk
        else:
            result = await agent.run(conversation_context, deps=self.deps)
            yield result.output

    async def run_completion(self, messages: list[Message], streamer: MessageStreamer, stream: bool = False):
        """Run completion and push results to the streamer.
        The streamer will also be added to the dependencies to allow tools to push messages (thoughts).

        Args:
            messages: List of message dictionaries.
            streamer: MessageStreamer instance to push messages to.
            stream: Whether to stream the response.
        """
        # Use the preconfigured agent with tools
        agent = self._pydantic_agent

        # Build conversation context string from all messages
        conversation_context = self._build_conversation_context(messages)

        # Store the complete conversation context for the tool context
        self.deps.conversation_context = conversation_context
        self.deps.streamer = streamer

        if stream:
            async with agent.run_stream(conversation_context, deps=self.deps) as result:
                async for chunk in result.stream_text(delta=True):
                    await streamer.push(role=Role.assistant, content=chunk)
                self.last_run_usage = result.usage()
        else:
            result = await agent.run(conversation_context, deps=self.deps)
            await streamer.push(role=Role.assistant, content=result.output)
            self.last_run_usage = result.usage()
