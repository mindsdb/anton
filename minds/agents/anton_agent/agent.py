import json
from pathlib import Path

from pydantic import BaseModel
from mindsdb_sdk.server import Server

from minds.agents.anton_agent.anton.anton import Anton
from minds.agents.anton_agent.anton.llm.anthropic import AnthropicProvider
from minds.agents.anton_agent.anton.llm.openai import OpenAIProvider
from minds.agents.anton_agent.anton.llm.provider import LLMProvider
from minds.agents.anton_agent.anton.llm.structured import generate_object
from minds.agents.anton_agent.anton.prompts import (
    QUERY_CLASSIFICATION_PROMPT,
    REMOVE_VISUALIZATIONS_BIAS_PROMPT,
    VISUALIZATIONS_LITE_PROMPT,
    VISUALIZATIONS_PROMPT,
)
from minds.agents.anton_agent.settings import AntonAgentSettings
from minds.agents.anton_agent.stream_event_formatter import AntonStreamEventFormatter
from minds.agents.base import AgentRunContext, BaseAgent
from minds.agents.base_response import AgentResponse
from minds.agents.helpers import mind_layer
from minds.common.logger import get_logger
from minds.common.settings.app_settings import get_app_settings
from minds.db.pg_session import get_engine, get_session_factory
from minds.model.mind import Mind
from minds.requests.stream import MessageStreamer
from minds.schemas.chat import Message, Role
from minds.services.conversations import ConversationsService
from minds.services.memory import MemoryRepository, MemoryService

logger = get_logger(__name__)


def _make_provider(provider_name: str, api_key: str) -> LLMProvider:
    """Instantiate an LLM provider by name."""
    if provider_name == "openai":
        return OpenAIProvider(api_key=api_key)
    return AnthropicProvider(api_key=api_key)


class QueryClassification(BaseModel):
    """Classification of a user query to determine intent and verification criteria."""
    needs_dashboard: bool
    dashboard_type: str  # trend, comparison, distribution, overview, none
    complexity: str  # simple, moderate, complex
    key_metrics: list[str]
    task_summary: str
    # Task completion verification fields
    success_criteria: list[str] = []
    expected_artifacts: list[str] = []
    requires_data_query: bool = False
    is_multi_step: bool = False


_DEFAULT_CLASSIFICATION = QueryClassification(
    needs_dashboard=False,
    dashboard_type="none",
    complexity="simple",
    key_metrics=[],
    task_summary="",
    success_criteria=[],
    expected_artifacts=[],
    requires_data_query=False,
    is_multi_step=False,
)


async def classify_query(messages: list[dict], llm_provider: LLMProvider, model: str) -> QueryClassification:
    """Classify user query into a structured Pydantic object, using conversation history for context."""
    try:
        return await generate_object(
            QueryClassification,
            llm_provider=llm_provider,
            model=model,
            system=QUERY_CLASSIFICATION_PROMPT,
            messages=messages,
        )
    except Exception:
        last_content = messages[-1].get("content", "") if messages else ""
        logger.warning("Query classification failed — defaulting to no dashboard", exc_info=True)
        return _DEFAULT_CLASSIFICATION.model_copy(update={"task_summary": last_content[:100]})

agent_settings = AntonAgentSettings()


class AntonAgent(BaseAgent):
    """
    The Anton agent is a general-purpose agent that can be used to answer questions about the data.
    """

    def __init__(self, mind: Mind, mindsdb_client: Server):
        super().__init__(mind=mind, mindsdb_client=mindsdb_client)

        # Create a unique workspace directory per org/user
        self.workspace_dir = Path(agent_settings.root_workspace_dir) / str(mind.organization_id) / str(mind.user_id)
        # TODO: This is temporary. Implement a proper usage tracking system.
        self._usage = (0, 0)

    async def _run(
        self,
        messages: list[Message],
        streamer: MessageStreamer,
        run_context: AgentRunContext,
        stream: bool,
    ) -> AgentResponse:
        app_settings = get_app_settings()

        # 1. Build datasource info from mind.mind_datasources
        datasource_info = []
        ds_list = []
        for mind_datasource in self.mind.mind_datasources:
            datasource = mind_datasource.datasource
            datasource_info.append(f"- {datasource.name}: {datasource.engine}")
            ds_list.append({"name": datasource.name, "engine": datasource.engine})

        default_ds = ds_list[0]["name"] if ds_list else ""

        # 2. Build runtime_context with datasource list
        output_dir = self.workspace_dir / agent_settings.output_dir
        if run_context and run_context.conversation_id and run_context.message_id:
            # When using a remote back-end such as Docker,
            # this output directory will relate to the file system of that remote back-end.
            output_dir = output_dir / str(run_context.conversation_id) / str(run_context.message_id)

        runtime_context_parts = []
        if datasource_info:
            runtime_context_parts.append(
                "You have access to the following data sources provided by MindsDB:\n"
                + "\n".join(datasource_info)
                + "\n\n"
                "To query data, use query_minds_data() in the scratchpad.\n"
                'Example: query_minds_data("SELECT * FROM users LIMIT 5")\n'
                'Optional: query_minds_data("SELECT ...", datasource="other_ds")\n'
                "Write SQL appropriate for each datasource's engine."
            )

        # 3. Resolve LLM config: mind.provider/model_name or mind.parameters overrides
        params = self.mind.parameters or {}

        # The planning model will be inferred from the mind.provider and mind.model_name.
        # The coding model will be inferred from mind.parameters.coding_provider and mind.parameters.coding_model.
        provider = self.mind.provider or app_settings.default_models.default_provider
        planning_provider = provider
        if planning_provider == "anthropic":
            planning_model = self.mind.model_name or app_settings.default_models.anthropic_model
            planning_api_key = app_settings.anthropic.api_key
        elif provider == "openai":
            planning_model = self.mind.model_name or app_settings.default_models.openai_model
            planning_api_key = app_settings.openai.api_key
        else:
            raise ValueError(f"Unknown provider: {provider}")

        coding_provider = params.get("coding_provider") or app_settings.default_models.default_coding_provider
        if coding_provider == "anthropic":
            coding_model = params.get("coding_model") or app_settings.default_models.anthropic_coding_model
            coding_api_key = app_settings.anthropic.api_key
        elif coding_provider == "openai":
            coding_model = params.get("coding_model") or app_settings.default_models.openai_coding_model
            coding_api_key = app_settings.openai.api_key
        else:
            raise ValueError(f"Unknown coding provider: {coding_provider}")

        # Classify the query using the coding model to decide visualization prompt
        enable_charting = run_context.metadata.enable_charting if run_context.metadata else False

        # Include recent conversation history for context (e.g. "show me a chart of that")
        classification_messages = []
        recent_history = messages[-5:]  # last few turns for context
        for msg in recent_history[:-1]:
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            content = msg.content or ""
            classification_messages.append({"role": role, "content": content[:500]})
        # Current user message in full, with charting context if enabled
        user_content = messages[-1].content if messages else ""
        if enable_charting:
            user_content += "\n\n[System note: Proactive Dashboards is enabled — the user has opted in " \
                "to automatic visualizations. Bias toward needs_dashboard=true when the query " \
                "involves data analysis, even if no chart is explicitly requested.]"
        classification_messages.append({"role": "user", "content": user_content})

        coding_llm = _make_provider(coding_provider, coding_api_key)
        classification = await classify_query(classification_messages, coding_llm, coding_model)
        logger.info(
            "Query classified: needs_dashboard=%s, type=%s, complexity=%s, summary=%s",
            classification.needs_dashboard,
            classification.dashboard_type,
            classification.complexity,
            classification.task_summary,
        )

        if classification.needs_dashboard or enable_charting:
            visualizations_prompt = VISUALIZATIONS_PROMPT.format(
                output_dir=output_dir,
                output_file_name=agent_settings.output_file_name,
            )
            if not enable_charting:
                visualizations_prompt += "\n" + REMOVE_VISUALIZATIONS_BIAS_PROMPT
        else:
            visualizations_prompt = VISUALIZATIONS_LITE_PROMPT.format(
                output_dir=output_dir,
                output_file_name=agent_settings.output_file_name,
            )
            visualizations_prompt += "\n" + REMOVE_VISUALIZATIONS_BIAS_PROMPT

        if classification.needs_dashboard:
            task_context = (
                f"\nQUERY INTENT (from classification):\n"
                f"- Dashboard type: {classification.dashboard_type}\n"
                f"- Complexity: {classification.complexity}\n"
                f"- Key metrics: {', '.join(classification.key_metrics)}\n"
                f"- Task summary: {classification.task_summary}\n"
            )
            visualizations_prompt += task_context

        runtime_context_parts.append(visualizations_prompt)

        # Add the system prompt from the mind
        mind_system_prompt = mind_layer(self.mind)
        if mind_system_prompt:
            runtime_context_parts.append(mind_system_prompt)
        runtime_context = "\n".join(runtime_context_parts)

        # 4. Resolve backend
        backend = params.get("backend") or agent_settings.backend

        # 5. Build scratchpad env vars for query_minds_data()
        extra_env = {}
        # TODO: Is this the best way to pass this information to Anton?
        # These are mandatory variables for Anton to work.
        extra_env["ANTON_MINDS_URL"] = agent_settings.minds_internal_url
        extra_env["ANTON_MINDS_USER_ID"] = str(self.mind.user_id)
        extra_env["ANTON_MINDS_ORG_ID"] = str(self.mind.organization_id)
        extra_env["ANTON_MINDS_CONVERSATION_ID"] = str(run_context.conversation_id)
        extra_env["ANTON_MINDS_SSL_VERIFY"] = "false"
        if ds_list:
            extra_env["ANTON_MINDS_DATASOURCE"] = default_ds
            extra_env["ANTON_MINDS_DATASOURCES_JSON"] = json.dumps(ds_list)

        # For the get_llm() function to work
        extra_env["ANTON_SCRATCHPAD_PROVIDER"] = coding_provider
        extra_env["ANTON_SCRATCHPAD_MODEL"] = coding_model
        extra_env["ANTON_SCRATCHPAD_API_KEY"] = coding_api_key

        # Scratchpad persistence settings
        extra_env["ANTON_SCRATCHPAD_PERSIST_SESSION"] = str(agent_settings.scratchpad_persist_session)
        extra_env["ANTON_SCRATCHPAD_SESSION_PATH"] = agent_settings.scratchpad_session_path

        # 6. Build history and prompt
        prompt = messages[-1].content
        history = [message.model_dump() for message in messages[:-1]]

        # 7. Build mind memory service and conversation service
        # (optional — skipped if mind.id is unavailable)
        shared_memory: MemoryService | None = None
        conversations_service: ConversationsService | None = None
        db_session = None
        if self.mind.id is not None:
            try:
                engine = get_engine(app_settings.database.uri)
                session_factory = get_session_factory(engine)
                db_session = session_factory()

                try:
                    repo = MemoryRepository(session=db_session, mind_id=self.mind.id)
                    shared_memory = MemoryService(
                        repo=repo,
                        token_budget=agent_settings.shared_memory_token_budget,
                        max_topics=agent_settings.shared_memory_max_topics,
                    )
                except Exception:
                    logger.exception("Failed to initialise mind memory service — continuing without it")
                try:
                    conversations_service = ConversationsService(
                        session=db_session,
                        mindsdb_client=self.mindsdb_client,
                        user_id=self.mind.user_id,
                        organization_id=self.mind.organization_id,
                    )
                except Exception:
                    logger.exception("Failed to initialise conversation service — continuing without it")
            except Exception:
                logger.exception("Failed to initialise both memory and conversation services")
                if db_session is not None:
                    db_session.close()
                    db_session = None

        # Get all events created as part of the conversation so far
        events: list[dict] = []
        if conversations_service is not None and run_context.conversation_id:
            messages = await conversations_service.get_conversation_messages(
                run_context.conversation_id,
                with_events=True,
            )
            for message in messages:
                for event in message.events:
                    events.append(event)

        # 8. Create Anton instance with all params
        mind_workspace = self.workspace_dir / str(self.mind.id)
        self.agent = Anton(
            workspace_dir=str(mind_workspace),
            runtime_context=runtime_context,
            history=history,
            backend=backend,
            planning_provider=planning_provider,
            planning_model=planning_model,
            planning_api_key=planning_api_key,
            coding_provider=coding_provider,
            coding_model=coding_model,
            coding_api_key=coding_api_key,
            extra_env=extra_env,
            shared_memory=shared_memory,
            events=events,
            classification=classification,
        )

        formatter = AntonStreamEventFormatter()
        complete_answer = ""
        try:
            async for event in self.agent.chat_stream(prompt):
                for role_str, content in formatter.on_event(event):
                    role = Role(role_str)
                    await streamer.push(role=role, content=content)
                    if role == Role.assistant:
                        complete_answer += content
                    else:
                        # Discard intermediate assistant text from prior thinking
                        # phases (e.g. "Let me try again..." between failed
                        # scratchpad retries). Only the final segment is the answer.
                        complete_answer = ""
        except Exception:
            logger.exception("Error during Anton agent execution")
            raise
        finally:
            await self.agent.close()
            if db_session is not None:
                db_session.close()

        return AgentResponse(
            sql="",
            answer=complete_answer,
        )

    async def get_last_run_usage(self) -> tuple[int, int] | None:
        return self._usage
