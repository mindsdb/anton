import json
from pathlib import Path

from mindsdb_sdk.server import Server

from minds.agents.anton_agent.anton.anton import Anton
from minds.agents.anton_agent.anton.prompts import REMOVE_VISUALIZATIONS_BIAS_PROMPT, VISUALIZATIONS_PROMPT
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
from minds.services.memory import MemoryRepository, MemoryService

logger = get_logger(__name__)

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

        # Add the visualizations prompt
        visualizations_prompt = VISUALIZATIONS_PROMPT.format(
            output_dir=output_dir,
            output_file_name=agent_settings.output_file_name,
        )
        # enable_charting means something slightly different here in comparison to the other agents.
        # If it is set to True, Anton will be biased towards generating visualizations for each turn.
        # If it is set to False, Anton will only generate visualizations when explicitly requested by the user.
        enable_charting = run_context.metadata.enable_charting if run_context.metadata else False
        if not enable_charting:
            visualizations_prompt += "\n" + REMOVE_VISUALIZATIONS_BIAS_PROMPT
        runtime_context_parts.append(visualizations_prompt)

        # Add the system prompt from the mind
        mind_system_prompt = mind_layer(self.mind)
        if mind_system_prompt:
            runtime_context_parts.append(mind_system_prompt)
        runtime_context = "\n".join(runtime_context_parts)

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

        # 6. Build history and prompt
        prompt = messages[-1].content
        history = [message.model_dump() for message in messages[:-1]]

        # 7. Build mind memory service (optional — skipped if mind.id is unavailable)
        shared_memory: MemoryService | None = None
        db_session = None
        if self.mind.id is not None:
            try:
                engine = get_engine(app_settings.database.uri)
                session_factory = get_session_factory(engine)
                db_session = session_factory()
                repo = MemoryRepository(session=db_session, mind_id=self.mind.id)
                shared_memory = MemoryService(
                    repo=repo,
                    token_budget=agent_settings.shared_memory_token_budget,
                    max_topics=agent_settings.shared_memory_max_topics,
                )
            except Exception:
                logger.exception("Failed to initialise mind memory service — continuing without it")
                if db_session is not None:
                    db_session.close()
                    db_session = None
                shared_memory = None

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
