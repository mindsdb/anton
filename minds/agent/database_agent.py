from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from pydantic_ai import Agent as PydanticAIAgent
from pydantic_ai import RunContext

from minds.agent.database_toolkit import DatabaseToolkit
from minds.agent.llm import get_llm_config
from minds.common.logger import setup_logging
from minds.model.mind import Mind


logger = setup_logging()

PydanticAIAgent.instrument_all()


@dataclass
class DatabaseDeps:
    """Dependencies for the database agent."""
    toolkit: DatabaseToolkit
    conversation_context: Optional[str] = None


class DatabaseAgent:
    """Experimental agent implementation using Pydantic instead of Langchain.

    This class is intended to replace LangchainAgent but maintains the same interface
    to support a smooth transition away from the Langchain dependency.
    """

    def __init__(
        self, 
        mind: Mind, 
        database_toolkit: DatabaseToolkit,
        ):
        """Initialize the PydanticAgent.

        Args:
            mind: The database mind record.
            database_toolkit: DatabaseToolkit instance for executing database operations.
        """
        self.mind = mind
        self.deps = DatabaseDeps(toolkit=database_toolkit)

        self._pydantic_agent = self._setup_agent()

    def _setup_agent(self) -> PydanticAIAgent:
        """Set up and configure the Pydantic AI agent with database tools.

        Returns:
            PydanticAIAgent: Configured agent with database tools.
        """
        llm_model = get_llm_config(self.mind.provider, self.mind.model_name)

        base_prompt = "You are a helpful database assistant created by MindsDB that can query databases and provide insights."

        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")

        system_prompt = f"{base_prompt}\n\nCurrent date: {current_date}\nCurrent time: {current_time}"
        
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
            )

        return agent

    def _build_conversation_context(self, messages: List[Dict]) -> str:
        """Build a conversation context string from messages.

        Args:
            messages: List of conversation messages

        Returns:
            Formatted conversation context string
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

            if role == 'user':
                context_parts.append(f"User: {content}")
            elif role == 'assistant':
                context_parts.append(f"Assistant: {content}")
            elif role == 'system':
                context_parts.append(f"System: {content}")

        conversation_context = "\n\n".join(context_parts)

        # Add instruction for the agent to understand this is a conversation
        if len(messages) > 1:
            conversation_context = f"This is a conversation history. Please respond to the most recent user message while considering the full context:\n\n{conversation_context}"

        return conversation_context

    async def get_completion(self, messages):
        """Get completion from the Pydantic-based agent.

        Args:
            messages: List of message dictionaries.

        Returns:
            ChatCompletionCustom object for non-streaming or a generator for streaming.
        """
        # Use the preconfigured agent with tools
        agent = self._pydantic_agent

        # Build conversation context string from all messages
        conversation_context = self._build_conversation_context(messages)

        # Store the complete conversation context for the tool context
        self.deps.conversation_context = conversation_context

        async with agent.run_stream(
            conversation_context, deps=self.deps
        ) as result:
            async for chunk in result.stream_text(delta=True):
                yield chunk
