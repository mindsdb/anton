import textwrap
from dataclasses import dataclass
from datetime import datetime

from mindsdb_sdk.server import Server
from pydantic_ai import Agent as PydanticAIAgent
from pydantic_ai import RunContext

from minds.agents.base import AgentRunContext, BaseAgent
from minds.agents.base_response import AgentResponse
from minds.agents.database_agent.database_toolkit import DatabaseToolkit
from minds.agents.database_agent.prompt_templates import CHART_GENERATION_INSTRUCTIONS
from minds.agents.llm import get_llm_config
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


class DatabaseAgent(BaseAgent):
    """Experimental agent implementation using Pydantic instead of Langchain.

    This class is intended to replace LangchainAgent but maintains the same interface
    to support a smooth transition away from the Langchain dependency.
    """

    def __init__(
        self,
        mind: Mind,
        mindsdb_client: Server,
    ):
        """Initialize the PydanticAgent.

        Args:
            mind: The database mind record.
            database_toolkit: DatabaseToolkit instance for executing database operations.
            config: Config for the database agent.
        """
        super().__init__(mind=mind, mindsdb_client=mindsdb_client)

        database_toolkit = DatabaseToolkit(mind=mind, mindsdb_client=mindsdb_client)
        self.deps = DatabaseDeps(toolkit=database_toolkit)

    def _setup_agent(self, enable_charting: bool) -> PydanticAIAgent:
        """Set up and configure the Pydantic AI agent with database tools.

        Args:
            enable_charting: Whether to enable charting.

        Returns:
            PydanticAIAgent: Configured agent with database tools.
        """
        llm_model = get_llm_config(self.mind.provider, self.mind.model_name)

        system_prompt = self._get_system_prompt(enable_charting=enable_charting)

        agent = PydanticAIAgent(
            model=llm_model,
            system_prompt=system_prompt,
            deps_type=DatabaseDeps,
            output_type=AgentResponse,
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

    def _get_system_prompt(self, enable_charting: bool) -> str:
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
        if enable_charting:
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

    async def _run(
        self,
        messages: list[Message],
        streamer: MessageStreamer,
        run_context: AgentRunContext,
        stream: bool,
    ) -> AgentResponse:
        """Run completion and push results to the streamer.
        The streamer will also be added to the dependencies to allow tools to push messages (thoughts).

        Args:
            messages: List of message dictionaries.
            streamer: MessageStreamer instance to push messages to.
            run_context: The run context for the agent.
            stream: Whether to stream the response.
        """
        PydanticAIAgent.instrument_all(instrument=run_context.instrument)

        # Use the preconfigured agent with tools
        enable_charting = run_context.metadata.enable_charting if run_context.metadata else False
        agent = self._setup_agent(enable_charting=enable_charting)

        # Build conversation context string from all messages
        conversation_context = self._build_conversation_context(messages)

        # Store the complete conversation context for the tool context
        self.deps.conversation_context = conversation_context
        self.deps.streamer = streamer

        result = None
        if stream:
            async with agent.run_stream(conversation_context, deps=self.deps) as stream:
                # stream_output() returns the complete answer generated so far,
                # To return the delta, we need to remove the previous answer chunk from the current answer chunk.
                previous_answer_chunk = ""
                async for chunk in stream.stream_output():
                    result = chunk
                    answer_chunk = chunk.answer
                    delta_answer_chunk = answer_chunk.removeprefix(previous_answer_chunk)
                    if delta_answer_chunk:
                        await streamer.push(role=Role.assistant, content=delta_answer_chunk)
                    previous_answer_chunk = answer_chunk

                self.last_run_usage = stream.usage()

        else:
            result = await agent.run(conversation_context, deps=self.deps)
            await streamer.push(role=Role.assistant, content=result.output)
            self.last_run_usage = result.usage()

        return result

    async def get_last_run_usage(self) -> tuple[int, int] | None:
        """Get the last run usage for the database agent.

        Returns:
            The last run usage for the database agent.
        """
        input_tokens = self.last_run_usage.input_tokens if self.last_run_usage else None
        output_tokens = self.last_run_usage.output_tokens if self.last_run_usage else None
        logger.debug(f"Last run usage: {input_tokens} in / {output_tokens} out")

        if input_tokens is None or output_tokens is None:
            return None

        return (input_tokens, output_tokens)
