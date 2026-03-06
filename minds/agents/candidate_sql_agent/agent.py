from mindsdb_sdk.server import Server
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

from minds.agents.base import BaseAgent, BaseAgentConfig
from minds.agents.base_response import AgentResponse
from minds.agents.candidate_sql_agent.controller_agents.agents import (
    FeedbackAgentDeps,
    LightweightRouterAgentDeps,
    answer_feedback_agent,
    feedback_agent,
    lightweight_router_agent,
)
from minds.agents.candidate_sql_agent.settings import CandidateSQLAgentSettings
from minds.agents.candidate_sql_agent.text_to_sql_agents.agents import TextToSQLPipeline
from minds.agents.helpers import is_native_query_mode_enabled, model_for
from minds.common.logger import setup_logging
from minds.model.mind import Mind
from minds.requests.stream import MessageStreamer
from minds.schemas.chat import Message, Role

logger = setup_logging()
agent_settings = CandidateSQLAgentSettings()


class CandidateSQLAgent(BaseAgent):
    def __init__(self, mind: Mind, mindsdb_client: Server, config: BaseAgentConfig | None = None):
        super().__init__(mind=mind, mindsdb_client=mindsdb_client, config=config)

        Agent.instrument_all(instrument=self.config.instrument)

    async def run(self, messages: list[Message], streamer: MessageStreamer, stream: bool = False) -> AgentResponse:
        """Run completion and push results to the streamer.
        The streamer will also be added to the dependencies to allow tools to push messages (thoughts).

        Args:
            messages: List of message dictionaries.
            streamer: MessageStreamer instance to push messages to.
            stream: Whether to stream the response.
        """
        prompt = messages.pop(-1).content
        message_history = self._convert_to_pydantic_ai_messages(messages)

        native_query_mode_enabled = is_native_query_mode_enabled(self.mind, agent_settings)

        # Use lightweight router agent (table names only, no column details)
        lightweight_router_deps = LightweightRouterAgentDeps(
            mind=self.mind,
            mindsdb_client=self.mindsdb_client,
            is_native_query_mode_enabled=native_query_mode_enabled,
        )

        logger.info(f"Running lightweight router agent for mind '{self.mind.name}' to determine if SQL is needed")
        router_result = await lightweight_router_agent.run(
            prompt,
            message_history=message_history,
            deps=lightweight_router_deps,
            model=model_for(self.mind),
        )

        try:
            if router_result.output.handoff:
                logger.info(
                    f"Lightweight router determined SQL is needed for mind '{self.mind.name}'. "
                    "Proceeding directly to SQL pipeline."
                )
                text_to_sql_pipeline = TextToSQLPipeline(
                    mind=self.mind,
                    mindsdb_client=self.mindsdb_client,
                    is_native_query_mode_enabled=native_query_mode_enabled,
                )
                # Text-to-SQL handoff
                text_to_sql_result = await text_to_sql_pipeline.run(
                    prompt=prompt,
                    message_history=message_history,
                    streamer=streamer,
                )

                complete_answer = ""
                async with answer_feedback_agent.run_stream(
                    f"Question: {prompt}\nExecution result: {text_to_sql_result.execution_result}",
                    message_history=message_history,
                    model=model_for(self.mind),
                ) as result:
                    previous_feedback_chunk = ""
                    previous_next_steps_chunk = ""
                    async for chunk in result.stream_output():
                        feedback_chunk = chunk.feedback
                        delta_feedback_chunk = feedback_chunk.removeprefix(previous_feedback_chunk)
                        complete_answer += delta_feedback_chunk
                        if delta_feedback_chunk:
                            await streamer.push(role=Role.assistant, content=delta_feedback_chunk)
                        previous_feedback_chunk = feedback_chunk

                        if chunk.next_steps:
                            # Add a newline between feedback and the first next step
                            if previous_next_steps_chunk == "":
                                await streamer.push(role=Role.assistant, content="\n\n")
                                complete_answer += "\n\n"

                            next_steps_chunk = chunk.next_steps
                            delta_next_steps_chunk = next_steps_chunk.removeprefix(previous_next_steps_chunk)
                            if delta_next_steps_chunk:
                                await streamer.push(role=Role.assistant, content=delta_next_steps_chunk)
                                complete_answer += delta_next_steps_chunk
                            previous_next_steps_chunk = next_steps_chunk

                await streamer.push(role=Role.assistant, content="\n\n")
                complete_answer += "\n\n"
                await streamer.push(role=Role.assistant, content=text_to_sql_result.execution_result)
                complete_answer += text_to_sql_result.execution_result

                return AgentResponse(
                    sql=text_to_sql_result.final_query,
                    answer=complete_answer,
                    notes=[],
                )
            else:
                logger.info(
                    f"Lightweight router determined feedback response is sufficient for mind '{self.mind.name}'. "
                    "Skipping SQL pipeline."
                )
                await streamer.push(role=Role.assistant, content=router_result.output.feedback)
                return AgentResponse(
                    sql="",
                    answer=router_result.output.feedback,
                    notes=[],
                )
        except Exception as e:
            feedback_agent_deps = FeedbackAgentDeps(
                mind=self.mind,
                mindsdb_client=self.mindsdb_client,
                is_native_query_mode_enabled=native_query_mode_enabled,
            )
            feedback_result = await feedback_agent.run(
                f"Question: {prompt}\nError: {e}",
                message_history=message_history,
                deps=feedback_agent_deps,
                model=model_for(self.mind),
            )
            await streamer.push(role=Role.assistant, content=feedback_result.output)
            return AgentResponse(
                sql="",
                answer=feedback_result.output,
                notes=[],
            )

    def _convert_to_pydantic_ai_messages(self, messages: list[Message]) -> list[ModelMessage]:
        """Convert a list of Message objects to a list of PydanticAI objects."""
        pydantic_ai_messages = []
        for message in messages:
            role = message.role.name
            content = message.content or ""
            if role == "system":
                pydantic_ai_messages.append(ModelRequest(parts=[SystemPromptPart(content=content)]))
            elif role == "user":
                pydantic_ai_messages.append(ModelRequest(parts=[UserPromptPart(content=content)]))
            elif role == "assistant":
                if isinstance(content, BaseModel) and hasattr(content, "sql_query"):
                    parts = [TextPart(content=content.text), TextPart(content=content.sql_query)]
                else:
                    parts = [TextPart(content=content)]
                pydantic_ai_messages.append(ModelResponse(parts=parts))
            else:
                raise ValueError(f"Unsupported role: {role}")

        return pydantic_ai_messages

    async def get_last_run_usage(self) -> tuple[int, int] | None:
        """Get the last run usage for the candidate SQL agent.

        Returns:
            The last run usage for the candidate SQL agent.
        """
        # TODO: Implement this
        return None
