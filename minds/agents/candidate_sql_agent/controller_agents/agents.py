from dataclasses import dataclass

from mindsdb_sdk.server import Server
from pydantic_ai import Agent, RunContext

from minds.agents.candidate_sql_agent.controller_agents.instructions_templates import (
    ANSWER_FEEDBACK_INSTRUCTIONS,
    FEEDBACK_INSTRUCTIONS_TEMPLATE,
    LIGHTWEIGHT_ROUTER_INSTRUCTIONS_TEMPLATE,
)
from minds.agents.candidate_sql_agent.controller_agents.models import AnswerFeedbackAgentResult, RouterAgentResult
from minds.agents.candidate_sql_agent.settings import CandidateSQLAgentSettings
from minds.agents.helpers import current_date_time_layer, mind_layer
from minds.cache import data_catalog_cache
from minds.common.logger import setup_logging
from minds.common.settings.app_settings import get_app_settings
from minds.model.mind import Mind

logger = setup_logging()
settings = get_app_settings()
agent_settings = CandidateSQLAgentSettings()


# =========================
# Deps for Controller Agents
# =========================


@dataclass
class LightweightRouterAgentDeps:
    mind: Mind
    mindsdb_client: Server | None = None
    is_native_query_mode_enabled: bool = False


@dataclass
class FeedbackAgentDeps:
    mind: Mind
    mindsdb_client: Server | None = None
    is_native_query_mode_enabled: bool = False


# ============================================================
# Controller Agents
# ============================================================


def _compact_table_list_summary(catalog, max_tables: int = 50) -> str:
    tables = [t.table.name for t in catalog.mind_datasource.mind_datasource_tables]
    total = len(tables)
    shown = tables[:max_tables]
    suffix = f" ... (+{total - max_tables} more)" if total > max_tables else ""
    ds_name = catalog.mind_datasource.datasource.name
    return f"Data Source: {ds_name}\nTotal Tables: {total}\nTables: {', '.join(shown)}{suffix}"


lightweight_router_agent: Agent[LightweightRouterAgentDeps, RouterAgentResult] = Agent(
    model=None,
    output_type=RouterAgentResult,
)


@lightweight_router_agent.instructions
async def lightweight_router_instructions(ctx: RunContext[LightweightRouterAgentDeps]) -> str:
    """Lightweight router that uses only table names (no column details) to decide routing."""
    data_catalogs = await data_catalog_cache.load(ctx.deps.mind)
    logger.info(f"Lightweight router agent loaded {len(data_catalogs)} catalog(s) using cache")

    # Use compact table list summary to keep context small and complete
    table_list_context = "\n\n".join([_compact_table_list_summary(c) for c in data_catalogs])

    p = current_date_time_layer()
    p += LIGHTWEIGHT_ROUTER_INSTRUCTIONS_TEMPLATE.format(
        table_list_context=table_list_context,
    )
    mp = mind_layer(ctx.deps.mind)
    if mp:
        p += "\n" + mp
    return p


feedback_agent: Agent[FeedbackAgentDeps, str] = Agent(
    model=None,
    output_type=str,
)


@feedback_agent.instructions
async def feedback_instructions(ctx: RunContext[FeedbackAgentDeps]) -> str:
    data_catalogs = await data_catalog_cache.load(ctx.deps.mind)
    logger.info(f"Feedback agent loaded {len(data_catalogs)} catalog(s) using cache")

    # Calculate token budget per catalog to prevent context length exceeded
    tokens_per_catalog = (
        agent_settings.max_catalog_tokens_orchestrator // len(data_catalogs)
        if data_catalogs
        else agent_settings.max_catalog_tokens_orchestrator
    )
    data_catalog_context = "\n\n".join(
        [
            c.to_context_str(
                max_tokens=tokens_per_catalog,
                include_statistics=False,
                include_datasource_name=ctx.deps.is_native_query_mode_enabled,
            )
            for c in data_catalogs
        ]
    )

    p = current_date_time_layer()
    p += FEEDBACK_INSTRUCTIONS_TEMPLATE.format(
        data_catalog_context=data_catalog_context,
    )
    return p


answer_feedback_agent: Agent[None, AnswerFeedbackAgentResult] = Agent(
    model=None,
    output_type=AnswerFeedbackAgentResult,
    instructions=ANSWER_FEEDBACK_INSTRUCTIONS,
)
