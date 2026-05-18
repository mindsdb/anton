import json
import re
from dataclasses import dataclass

import pandas as pd
from mindsdb_sdk.server import Server
from mindsdb_sql_parser import parse_sql
from mindsdb_sql_parser.ast import Select
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import RunUsage, UsageLimits

from minds.agents.candidate_sql_agent.candidate_generator_agent.agent import CandidateGenerator, SQLCandidate
from minds.agents.candidate_sql_agent.error_classifier import classify_error, format_error_guidance
from minds.agents.candidate_sql_agent.linker_agent.agent import LinkedSchema, SchemaLinker
from minds.agents.candidate_sql_agent.selection_agent.agent import QueryComplexityClassifier, SelectionAgent
from minds.agents.candidate_sql_agent.settings import CandidateSQLAgentSettings
from minds.agents.candidate_sql_agent.text_to_sql_agents.instructions_templates import (
    MINDSDB_SQL_INSTRUCTIONS,
    NATIVE_BIGQUERY_SQL_ERROR_INSTRUCTIONS,
    NATIVE_BIGQUERY_SQL_INSTRUCTIONS,
    NATIVE_MSSQL_SQL_ERROR_INSTRUCTIONS,
    NATIVE_MSSQL_SQL_INSTRUCTIONS,
    NATIVE_SNOWFLAKE_SQL_ERROR_INSTRUCTIONS,
    NATIVE_SNOWFLAKE_SQL_INSTRUCTIONS,
)
from minds.agents.candidate_sql_agent.text_to_sql_agents.instructions_templates import (
    QUERY_PLANNING_INSTRUCTIONS as QUERY_PLANNING_INSTRUCTIONS_TEMPLATE,
)
from minds.agents.candidate_sql_agent.text_to_sql_agents.instructions_templates import (
    QUERY_PLANNING_RETRY_INSTRUCTIONS as QUERY_PLANNING_RETRY_INSTRUCTIONS_TEMPLATE,
)
from minds.agents.candidate_sql_agent.text_to_sql_agents.instructions_templates import (
    SQL_GENERATION_INSTRUCTIONS as QUERY_GENERATION_INSTRUCTIONS_TEMPLATE,
)
from minds.agents.candidate_sql_agent.text_to_sql_agents.instructions_templates import (
    SQL_GENERATION_RETRY_INSTRUCTIONS as QUERY_GENERATION_RETRY_INSTRUCTIONS_TEMPLATE,
)
from minds.agents.candidate_sql_agent.text_to_sql_agents.models import (
    AcquiredKnowledge,
    AcquiredKnowledgeItem,
    DataCatalogSubset,
    QueryAttempt,
    QueryPlan,
    QueryPlanStep,
    QueryPlanStepType,
    SQLQuery,
    TextToSQLToolResult,
)
from minds.agents.exceptions import DataCatalogValidationError, QueryGenerationError, QueryPlanningError
from minds.agents.helpers import charting_layer, current_date_time_layer, mind_layer, model_for
from minds.cache import data_catalog_cache
from minds.common.logger import get_logger
from minds.common.settings.app_settings import get_app_settings
from minds.model.data_catalog import DataCatalog
from minds.model.mind import Mind
from minds.requests.stream import MessageStreamer
from minds.schemas.chat import Role

logger = get_logger(__name__)
settings = get_app_settings()
agent_settings = CandidateSQLAgentSettings()


class TextToSQLPipelineError(Exception):
    """
    Exception for errors in the Text-to-SQL pipeline.
    """

    pass


# =====================================
# Deps for Text-to-SQL Pipeline Agents
# =====================================


@dataclass
class PlanningAgentDeps:
    mind: Mind
    data_catalog_context: str


@dataclass
class PlanningAgentRetryDeps(PlanningAgentDeps):
    failed_query_plan: str
    error_message: str


@dataclass
class SQLGenAgentDeps:
    mind: Mind
    data_catalog_subset_context: str
    acquired_knowledge: str
    is_native_query_mode_enabled: bool
    native_engine: str | None


@dataclass
class SQLGenRetryAgentDeps(SQLGenAgentDeps):
    failed_query: str
    error_message: str
    previous_attempts: list[QueryAttempt] = None


@dataclass
class RefinementAgentDeps:
    mind: Mind
    data_catalog_context: str
    acquired_knowledge: str
    remaining_steps: str


@dataclass
class SummarizeAgentDeps:
    mind: Mind
    prompt: str
    step_description: str
    acquired_knowledge: str


# ============================================================
# Text-to-SQL Pipeline Agents (programmatic handoff pattern)
# ============================================================

planning_agent: Agent[PlanningAgentDeps, QueryPlan] = Agent(
    model=None,
    output_type=QueryPlan,
)


@planning_agent.instructions
async def planning_instructions(ctx: RunContext[PlanningAgentDeps]) -> str:
    p = current_date_time_layer()
    p += QUERY_PLANNING_INSTRUCTIONS_TEMPLATE.format(
        data_catalog_context=ctx.deps.data_catalog_context,
    )
    mp = mind_layer(ctx.deps.mind)
    if mp:
        p += "\n" + mp
    logger.debug(f"Planning instructions: {p}")
    return p


planning_retry_agent: Agent[PlanningAgentRetryDeps, QueryPlan] = Agent(
    model=None,
    output_type=QueryPlan,
)


@planning_retry_agent.instructions
async def planning_retry_instructions(ctx: RunContext[PlanningAgentRetryDeps]) -> str:
    p = current_date_time_layer()
    p += QUERY_PLANNING_RETRY_INSTRUCTIONS_TEMPLATE.format(
        data_catalog_context=ctx.deps.data_catalog_context,
        failed_query_plan=ctx.deps.failed_query_plan,
        error_message=ctx.deps.error_message,
    )
    mp = mind_layer(ctx.deps.mind)
    if mp:
        p += "\n" + mp
    logger.debug(f"Planning retry instructions: {p}")
    return p


sql_gen_agent: Agent[SQLGenAgentDeps, SQLQuery] = Agent(
    model=None,
    output_type=SQLQuery,
)


@sql_gen_agent.instructions
async def sql_gen_instructions(ctx: RunContext[SQLGenAgentDeps]) -> str:
    p = current_date_time_layer()
    p += QUERY_GENERATION_INSTRUCTIONS_TEMPLATE.format(
        data_catalog_subset_context=ctx.deps.data_catalog_subset_context,
        acquired_knowledge=ctx.deps.acquired_knowledge,
    )

    # Native query mode instructions depend on the engine
    if ctx.deps.is_native_query_mode_enabled:
        engine = (ctx.deps.native_engine or "").strip().lower()
        if engine == "bigquery":
            p += "\n\n" + NATIVE_BIGQUERY_SQL_INSTRUCTIONS
        elif engine == "snowflake":
            p += "\n\n" + NATIVE_SNOWFLAKE_SQL_INSTRUCTIONS
        elif engine == "mssql":
            p += "\n\n" + NATIVE_MSSQL_SQL_INSTRUCTIONS
        else:
            logger.warning(f"Unsupported native engine '{ctx.deps.native_engine}'. Falling back to MindsDB SQL.")
            p += "\n\n" + MINDSDB_SQL_INSTRUCTIONS
    else:
        p += "\n\n" + MINDSDB_SQL_INSTRUCTIONS

    p += "\n" + charting_layer()
    mp = mind_layer(ctx.deps.mind)
    if mp:
        p += "\n" + mp
    logger.debug(f"SQL gen instructions: {p}")
    return p


sql_gen_retry_agent: Agent[SQLGenRetryAgentDeps, SQLQuery] = Agent(
    model=None,
    output_type=SQLQuery,
)


@sql_gen_retry_agent.instructions
async def sql_retry_instructions(ctx: RunContext[SQLGenRetryAgentDeps]) -> str:
    # Format previous attempts for context
    if ctx.deps.previous_attempts:
        attempts_list = []
        for i, attempt in enumerate(ctx.deps.previous_attempts, 1):
            attempt_str = f"Attempt {i}:\nSQL: {attempt.query}"
            if attempt.error:
                attempt_str += f"\nError: {attempt.error}"
            attempts_list.append(attempt_str)

    # Classify error and get targeted guidance
    classified_error = classify_error(ctx.deps.error_message)
    error_guidance = format_error_guidance(classified_error)
    logger.info(f"Error classified as {classified_error.category.value}: {classified_error.subcategory}")

    p = current_date_time_layer()
    p += QUERY_GENERATION_RETRY_INSTRUCTIONS_TEMPLATE.format(
        data_catalog_subset_context=ctx.deps.data_catalog_subset_context,
        acquired_knowledge=ctx.deps.acquired_knowledge,
        failed_query=ctx.deps.failed_query,
        error_message=ctx.deps.error_message,
        error_guidance=error_guidance,
    )

    # Native query mode instructions depend on the engine
    if ctx.deps.is_native_query_mode_enabled:
        engine = (ctx.deps.native_engine or "").strip().lower()
        if engine == "bigquery":
            p += "\n\n" + NATIVE_BIGQUERY_SQL_INSTRUCTIONS
            p += "\n\n" + NATIVE_BIGQUERY_SQL_ERROR_INSTRUCTIONS
        elif engine == "snowflake":
            p += "\n\n" + NATIVE_SNOWFLAKE_SQL_INSTRUCTIONS
            p += "\n\n" + NATIVE_SNOWFLAKE_SQL_ERROR_INSTRUCTIONS
        elif engine == "mssql":
            p += "\n\n" + NATIVE_MSSQL_SQL_INSTRUCTIONS
            p += "\n\n" + NATIVE_MSSQL_SQL_ERROR_INSTRUCTIONS
        else:
            logger.warning(f"Unsupported native engine '{ctx.deps.native_engine}'. Falling back to MindsDB SQL.")
            p += "\n\n" + MINDSDB_SQL_INSTRUCTIONS
            # TODO Add mindsdb error instructions
    else:
        p += "\n\n" + MINDSDB_SQL_INSTRUCTIONS

    p += "\n" + charting_layer()
    mp = mind_layer(ctx.deps.mind)
    if mp:
        p += "\n" + mp
    logger.debug(f"SQL retry instructions: {p}")
    return p


summarize_agent: Agent[SummarizeAgentDeps, str] = Agent(
    model=None,
    output_type=str,
)


@summarize_agent.instructions
async def summarize_instructions(ctx: RunContext[SummarizeAgentDeps]) -> str:
    p = current_date_time_layer()
    # TODO: Move this into instructions templates, double check if it is true for native queries
    # especiall the functions part as SUM etc
    p += f"""
You are a summarization assistant. Your task is to explain the acquired knowledge from SQL queries in clear, " \
    "natural language for the user.

**USER'S QUESTION:**
{ctx.deps.prompt}

**STEP DESCRIPTION:**
{ctx.deps.step_description}

**ACQUIRED KNOWLEDGE FROM SQL QUERIES:**
{ctx.deps.acquired_knowledge}

**INSTRUCTIONS:**
- Explain the SQL query results in natural language
- Do NOT perform any calculations, formatting, or transformations - all data operations were already done in SQL
- Simply explain what was found in the SQL queries
- Be concise and clear
- If the acquired knowledge contains query results, explain them in a readable way
- If multiple queries were run, synthesize the information into a coherent explanation

**CRITICAL RESTRICTIONS:**
- **NEVER calculate values** - all calculations (SUM, AVG, COUNT, etc.) were already done in SQL
- **NEVER format data** - all formatting (ROUND, DATE_FORMAT, CONCAT, etc.) was already done in SQL
- **NEVER transform data** - all transformations (CAST, CASE, string manipulation, etc.) were already done in SQL
- **ONLY explain** - your job is to take the already-formatted SQL results and explain them in natural language
- Present the data as-is from SQL results and explain what it means
- Answer the user's question based on the SQL results without modifying the data
"""
    mp = mind_layer(ctx.deps.mind)
    if mp:
        p += "\n" + mp
    return p


# ============================================================
# Text-to-SQL Pipeline (programmatic handoff)
# ============================================================


class TextToSQLPipeline:
    """Programmatic-handoff text_to_sql pipeline.

    This pipeline returns evidence (SQL + result). The orchestrator agent is responsible
    for turning that evidence into a human-readable final answer.
    """

    def __init__(
        self,
        mind: Mind,
        mindsdb_client: Server,
        is_native_query_mode_enabled: bool = False,
    ):
        self.mind = mind
        self.mindsdb_client = mindsdb_client
        self.is_native_query_mode_enabled = is_native_query_mode_enabled

        self._native_datasource_name: str | None = (
            self.mind.mind_datasources[0].datasource.name if is_native_query_mode_enabled else None
        )

    def _set_native_datasource_from_linked_schema(self, linked_schema: LinkedSchema | None) -> None:
        if not linked_schema or not linked_schema.tables:
            return
        datasources = {table.split(".", 1)[0] for table in linked_schema.tables if "." in table}
        if len(datasources) == 1:
            self._native_datasource_name = next(iter(datasources))

    def _get_native_datasource_engine_for(self, datasource_name: str | None) -> str | None:
        ds_name = (datasource_name or "").strip().lower()
        if not ds_name or not self.mind.mind_datasources:
            return None
        for md in self.mind.mind_datasources:
            ds = md.datasource
            if ds and (ds.name or "").strip().lower() == ds_name:
                return ds.engine
        return None

    def _strip_datasource_prefix_for_native(self, query: str, datasource_name: str) -> str:
        if not query or not datasource_name:
            return query
        pattern = re.compile(
            rf"(?i)\b{re.escape(datasource_name)}\s*\.",
            re.IGNORECASE,
        )
        return pattern.sub("", query)

    def _resolve_step_native_datasource(
        self,
        step: QueryPlanStep,
        pruned_data_catalogs: list[DataCatalog],
    ) -> str | None:
        available_ds = {
            (c.mind_datasource.datasource.name or "").strip()
            for c in pruned_data_catalogs
            if c.mind_datasource and c.mind_datasource.datasource
        }
        available_ds = {n for n in available_ds if n}
        available_ds_lower = {n.lower(): n for n in available_ds}

        resolved: str | None = None

        # 1) Explicit datasource selection from the plan — native mode requires exactly one
        selected_ds = [ds.strip() for ds in (step.data_catalog_subset.datasources or []) if ds.strip()]
        if len(selected_ds) == 1:
            resolved = available_ds_lower.get(selected_ds[0].lower())
        elif len(selected_ds) > 1:
            return None  # Cross-datasource step — native mode not applicable

        # 2) Infer from table tokens — only valid when all tables share one datasource
        if not resolved:
            datasources_in_tables = {
                token.split(".", 1)[0].strip()
                for token in (step.data_catalog_subset.tables or [])
                if token and "." in token and token.split(".", 1)[0].strip()
            }
            if len(datasources_in_tables) == 1:
                top_ds = next(iter(datasources_in_tables))
                resolved = available_ds_lower.get(top_ds.lower(), top_ds)
            elif len(datasources_in_tables) > 1:
                return None  # Cross-datasource step — native mode not applicable

        # 3) Single available datasource in pruned catalogs
        if not resolved and len(available_ds) == 1:
            resolved = next(iter(available_ds))

        # 4) Global fallback
        if not resolved:
            resolved = self._native_datasource_name

        # Avoid routing to `mindsdb` — use first non-mindsdb datasource instead
        if resolved and resolved.lower() == "mindsdb":
            resolved = next(
                (
                    (md.datasource.name or "").strip()
                    for md in self.mind.mind_datasources
                    if md.datasource and (md.datasource.name or "").strip().lower() != "mindsdb"
                ),
                None,
            )

        return resolved

    async def run(
        self,
        prompt: str,
        message_history: list[ModelMessage],
        streamer: MessageStreamer,
    ) -> TextToSQLToolResult:
        logger.info(f"Starting text-to-SQL pipeline for prompt: {prompt[:100]}...")
        usage = RunUsage()
        usage_limits = UsageLimits(request_limit=agent_settings.max_request_limit)

        logger.info(f"Loading data catalogs for mind '{self.mind.name}'")

        data_catalogs = await data_catalog_cache.load(self.mind)
        logger.info(f"Loaded {len(data_catalogs)} data catalog(s) using cache")

        # Use compact mode (no statistics) to reduce token usage
        # Approximate token budget: 128K total - 20K for conversation - 10K for response = ~98K for catalog

        # Calculate approximate size of catalogs to determine if we need summary mode
        total_tables = sum(len(c.mind_datasource.mind_datasource_tables) for c in data_catalogs)
        estimated_chars = (
            total_tables * agent_settings.avg_columns_per_table_estimate * agent_settings.chars_per_column_line_estimate
        )
        estimated_tokens = estimated_chars // 4

        logger.info(f"Data catalog size estimate: {total_tables} tables, ~{estimated_tokens} tokens")

        # If catalog is very large, use summary mode - provides table list only
        # and relies on pruning for detailed schema
        if (
            total_tables > agent_settings.large_catalog_table_threshold
            or estimated_tokens > agent_settings.large_catalog_token_threshold
        ):
            logger.info(f"Large catalog detected ({total_tables} tables). Using summary mode - table list only.")
            data_catalog_context_str = "\n\n".join(
                [
                    c.get_table_list_summary(
                        include_datasource_name=not self.is_native_query_mode_enabled,
                    )
                    for c in data_catalogs
                ]
            )
            data_catalog_context_str += (
                "\n\nIMPORTANT: This is a large catalog. The planning step MUST use catalog"
                " pruning to select specific tables before generating SQL. Detailed schema will be provided only for "
                "pruned tables."
            )
        else:
            # Normal mode - include all table details without statistics
            tokens_per_catalog = (
                agent_settings.max_catalog_tokens_pipeline // len(data_catalogs)
                if data_catalogs
                else agent_settings.max_catalog_tokens_pipeline
            )
            data_catalog_context_str = "\n\n".join(
                [
                    c.to_context_str(
                        max_tokens=tokens_per_catalog,
                        include_statistics=False,
                        include_datasource_name=not self.is_native_query_mode_enabled,
                    )
                    for c in data_catalogs
                ]
            )

        # Schema Linking: Identify relevant tables/columns before planning
        # This reduces context size and focuses the LLM on relevant schema elements
        linked_schema: LinkedSchema | None = None
        if agent_settings.enable_schema_linking:
            logger.info("Running schema linking to identify relevant schema elements...")
            await streamer.push(role=Role.thought_planning, content="Identifying relevant schema elements...")

            try:
                schema_linker = SchemaLinker(self.mind)
                linked_schema = await schema_linker.link(prompt, data_catalogs)
                logger.info(
                    f"Schema linking complete: {len(linked_schema.tables)} tables, "
                    f"{sum(len(cols) for cols in linked_schema.columns.values())} columns"
                )
                self._set_native_datasource_from_linked_schema(linked_schema)
                await streamer.push(
                    role=Role.thought_planning,
                    content=f"Schema linking: {len(linked_schema.tables)} relevant tables identified",
                )

                # Replace full catalog context with filtered schema
                filtered_context = schema_linker.filter_catalogs_by_linked_schema(data_catalogs, linked_schema)
                if filtered_context.strip():
                    data_catalog_context_str = filtered_context
                    logger.info("Using filtered schema context from schema linking")
            except Exception as e:
                logger.warning(f"Schema linking failed, using full catalog: {e}")
                await streamer.push(
                    role=Role.thought_planning,
                    content=f"Schema linking skipped: {str(e)[:50]}...",
                )

        await streamer.push(role=Role.thought_planning, content="Planning query steps...")

        # (1) Plan
        logger.info("Planning query steps...")
        query_plan = await self._plan_with_retry(
            prompt=prompt,
            message_history=message_history,
            data_catalog_context_str=data_catalog_context_str,
            data_catalogs=data_catalogs,
            streamer=streamer,
            usage=usage,
            usage_limits=usage_limits,
        )
        logger.info(f"Query plan created with {len(query_plan.steps)} step(s)")
        await streamer.push(
            role=Role.thought_planning,
            content=f"Query plan created with {len(query_plan.steps)} step(s)",
        )
        for i, step in enumerate(query_plan.steps, 1):
            logger.info(f"  Step {i} ({step.type.value}): {step.description}")
            await streamer.push(
                role=Role.thought_planning,
                content=f"  Step {i} ({step.type.value}): {step.description}",
            )

        # (2) Execute loop with handoffs
        logger.info(f"Starting execution of query plan with {len(query_plan.steps)} initial step(s)")
        final_query, execution_result, steps_executed, acquired_knowledge = await self._execute(
            query_plan=query_plan,
            prompt=prompt,
            message_history=message_history,
            data_catalogs=data_catalogs,
            data_catalog_context_str=data_catalog_context_str,
            streamer=streamer,
            usage=usage,
            usage_limits=usage_limits,
            linked_schema=linked_schema,
        )

        logger.info(f"Pipeline execution completed. Total steps executed: {steps_executed}")
        if final_query:
            logger.info(
                f"Final query length: {len(final_query)} characters, Result length: {len(execution_result)} characters"
            )

        return TextToSQLToolResult(
            final_query=final_query,
            execution_result=execution_result,
            steps_executed=steps_executed,
            acquired_knowledge=acquired_knowledge.to_string() if acquired_knowledge else None,
        )

    async def _plan_with_retry(
        self,
        prompt: str,
        message_history: list[ModelMessage],
        data_catalog_context_str: str,
        data_catalogs: list[DataCatalog],
        streamer: MessageStreamer,
        usage: RunUsage,
        usage_limits: UsageLimits,
    ) -> QueryPlan:
        last_error: Exception | None = None
        last_plan = None

        for attempt in range(agent_settings.max_planning_retries):
            logger.info(f"Planning attempt {attempt + 1} of {agent_settings.max_planning_retries}")
            await streamer.push(
                role=Role.thought_planning,
                content=f"Planning attempt {attempt + 1} of {agent_settings.max_planning_retries}",
            )
            try:
                if attempt == 0:
                    logger.info("Planning query steps (initial attempt)")
                    plan_res = await planning_agent.run(
                        prompt,
                        message_history=message_history,
                        deps=PlanningAgentDeps(mind=self.mind, data_catalog_context=data_catalog_context_str),
                        model=model_for(self.mind),
                        usage=usage,
                        usage_limits=usage_limits,
                    )
                else:
                    logger.info(f"Planning query steps (retry attempt {attempt + 1})")
                    plan_res = await planning_retry_agent.run(
                        prompt,
                        message_history=message_history,
                        deps=PlanningAgentRetryDeps(
                            mind=self.mind,
                            data_catalog_context=data_catalog_context_str,
                            failed_query_plan=last_plan.to_string() if last_plan else "",
                            error_message=str(last_error),
                        ),
                        model=model_for(self.mind),
                        usage=usage,
                        usage_limits=usage_limits,
                    )

                last_plan = plan_res.output
                self._validate_query_plan(last_plan)

                # Validate that tables referenced in plan steps exist in the data catalog
                try:
                    plan_table_names = self._extract_plan_table_names(last_plan)
                    if plan_table_names:
                        logger.info(f"Validating {len(plan_table_names)} table(s) referenced in plan steps")
                        await streamer.push(
                            role=Role.thought_planning,
                            content=f"Validating {len(plan_table_names)} table(s) referenced in plan steps",
                        )
                        self._validate_tables_exist_in_catalog(plan_table_names, data_catalogs)
                        logger.info("Table validation passed")
                except DataCatalogValidationError as e:
                    logger.warning(f"Table validation failed: {str(e)}")
                    await streamer.push(
                        role=Role.thought_planning,
                        content=f"Table validation failed: {str(e)}",
                    )
                    # Convert to QueryPlanningError to trigger retry mechanism
                    raise QueryPlanningError(
                        f"Data catalog validation failed: {str(e)}. "
                        "The plan references tables that don't exist in the catalog. "
                        "Please adjust the plan to use only tables that are available in the data catalog."
                    ) from e
                return last_plan

            except Exception as e:
                last_error = e
                logger.warning(f"Planning attempt {attempt + 1} failed: {str(e)}")
                await streamer.push(
                    role=Role.thought_planning,
                    content=f"Planning attempt {attempt + 1} failed: {str(e)}",
                )
                if attempt == agent_settings.max_planning_retries - 1:
                    raise QueryPlanningError(
                        f"Planning failed after {agent_settings.max_planning_retries} attempts. "
                        f"Final error: {last_error}"
                    ) from e

    def _validate_query_plan(self, query_plan: QueryPlan) -> None:
        if not query_plan.steps:
            raise QueryPlanningError("Query plan must have at least one step.")

        if query_plan.steps[-1].type != QueryPlanStepType.FINAL:
            raise QueryPlanningError("Last step must be a final step.")

        final_step_count = sum(1 for step in query_plan.steps if step.type == QueryPlanStepType.FINAL)
        if final_step_count != 1:
            raise QueryPlanningError(
                f"Query plan must have exactly one final step, but found {final_step_count}. "
                "Only the last step should be marked as 'final'."
            )

    def _extract_plan_table_names(self, query_plan: QueryPlan) -> list[str]:
        """
        Extract and deduplicate fully-qualified table names from all plan steps.

        Args:
            query_plan: The query plan to extract table names from

        Returns:
            List of unique fully-qualified table names (format: datasource.table)
        """
        table_names = set()
        for step in query_plan.steps:
            if step.data_catalog_subset.tables:
                table_names.update(step.data_catalog_subset.tables)
        return sorted(list(table_names))

    def _validate_tables_exist_in_catalog(self, table_names: list[str], data_catalogs: list[DataCatalog]) -> None:
        """
        Validate that all specified tables exist in the data catalog.

        Args:
            table_names: List of fully-qualified table names (format: datasource.table)
            data_catalogs: List of DataCatalog objects to check against

        Raises:
            DataCatalogValidationError: If any tables don't exist in the catalog
        """
        # System tables that may not be in catalog but are valid to use
        IGNORE_TABLES = {
            "information_schema.meta_columns",
            "information_schema.meta_tables",
            "information_schema.meta_column_statistics",
        }

        # Build a map of datasource -> set of available table names
        datasource_to_tables: dict[str, set[str]] = {}
        for catalog in data_catalogs:
            datasource_name = (catalog.mind_datasource.datasource.name or "").strip().lower()
            table_names_in_catalog = {
                (mdt.table.name or "").strip().lower()
                for mdt in catalog.mind_datasource.mind_datasource_tables
                if mdt.table and mdt.table.name
            }
            datasource_to_tables[datasource_name] = table_names_in_catalog

        missing_tables = []
        tables_to_validate = []

        # Parse and validate each table
        for fully_qualified_name in table_names:
            # Skip system tables in ignore list
            if fully_qualified_name in IGNORE_TABLES:
                continue

            # Parse datasource and table name (split on first dot)
            if "." not in fully_qualified_name:
                # No datasource prefix - can't validate
                missing_tables.append((fully_qualified_name, None, "Table name missing datasource prefix"))
                continue

            parts = fully_qualified_name.split(".", 1)
            datasource_name = parts[0].strip().lower()
            table_name = parts[1].strip().lower()

            tables_to_validate.append((fully_qualified_name, datasource_name, table_name))

        # Check if tables exist in catalog
        for fully_qualified_name, datasource_name, table_name in tables_to_validate:
            if datasource_name not in datasource_to_tables:
                missing_tables.append(
                    (
                        fully_qualified_name,
                        datasource_name,
                        f"Datasource '{datasource_name}' not found in catalog",
                    )
                )
                continue

            available_tables = datasource_to_tables[datasource_name]
            if table_name not in available_tables:
                missing_tables.append((fully_qualified_name, datasource_name, None))

        # If any tables are missing, raise error with detailed message
        if missing_tables:
            error_parts = ["The following tables referenced in the plan do not exist in the data catalog:"]
            for fully_qualified_name, _datasource_name, reason in missing_tables:
                if reason:
                    error_parts.append(f"  - {fully_qualified_name}: {reason}")
                else:
                    error_parts.append(f"  - {fully_qualified_name}")

            # Add available tables for each datasource that had missing tables
            datasources_with_errors = {ds for _, ds, _ in missing_tables if ds}
            for datasource_name in sorted(datasources_with_errors):
                if datasource_name in datasource_to_tables:
                    available = sorted(datasource_to_tables[datasource_name])
                    if available:
                        error_parts.append(f"\nAvailable tables in '{datasource_name}': {', '.join(available)}")

            error_message = "\n".join(error_parts)
            raise DataCatalogValidationError(error_message)

    async def _execute(
        self,
        query_plan: QueryPlan,
        prompt: str,
        message_history: list[ModelMessage],
        data_catalogs: list,
        data_catalog_context_str: str,
        streamer: MessageStreamer,
        usage: RunUsage,
        usage_limits: UsageLimits,
        linked_schema: LinkedSchema | None = None,
    ) -> tuple[str, str, int, AcquiredKnowledge]:
        steps = query_plan.steps
        acquired_knowledge = AcquiredKnowledge()
        step_count = 0
        max_steps = agent_settings.max_query_plan_steps

        await streamer.push(
            role=Role.thought_execution,
            content=f"Executing plan with {len(steps)} step(s)...",
        )

        while steps:
            step = steps.pop(0)
            step_count += 1

            # Enforce maximum step limit to prevent runaway plans
            if step_count > max_steps:
                logger.warning(f"Reached maximum step limit of {max_steps}. Stopping execution.")
                await streamer.push(
                    role=Role.thought_execution,
                    content=f"Reached maximum step limit of {max_steps}. "
                    "Stopping execution to prevent excessive resource usage.",
                )
                return (
                    "",
                    f"Query plan exceeded maximum step limit of {max_steps}. Please simplify your query.",
                    step_count,
                    acquired_knowledge,
                )

            step_type_label = step.type.value
            logger.info(f"Executing step {step_count} ({step_type_label}): {step.description}")

            await streamer.push(
                role=Role.thought_execution_step_start,
                content=f"Executing {step_type_label} step {step_count}: {step.description}",
            )
            await streamer.push(
                role=Role.thought_execution_step,
                content=f"Data catalog subset for step {step_count}\n{step.data_catalog_subset.to_string()}",
            )

            # (2a) SQL gen/repair + execute handoff
            logger.info("Pruning data catalogs for step...")
            await streamer.push(
                role=Role.thought_execution_step,
                content="Pruning data catalogs for step...",
            )
            pruned_data_catalogs = await self._prune_data_catalogs_for_step(
                data_catalogs=data_catalogs,
                data_catalog_subset=step.data_catalog_subset,
            )
            step_native_datasource_name: str | None = None
            if self.is_native_query_mode_enabled:
                step_native_datasource_name = self._resolve_step_native_datasource(step, pruned_data_catalogs)
            step_native_mode = step_native_datasource_name is not None
            logger.info(
                "Step native mode resolution: enabled=%s, datasource=%s",
                step_native_mode,
                step_native_datasource_name or "none",
            )
            # Apply token limits to prevent context length exceeded during SQL generation
            tokens_per_catalog = (
                agent_settings.max_catalog_tokens_orchestrator // len(pruned_data_catalogs)
                if pruned_data_catalogs
                else agent_settings.max_catalog_tokens_orchestrator
            )
            data_catalog_subset_context = "\n\n".join(
                [
                    c.to_context_str(
                        max_tokens=tokens_per_catalog,
                        include_statistics=False,
                        include_datasource_name=not step_native_mode,
                    )
                    for c in pruned_data_catalogs
                ]
            )

            logger.info("Generating and executing SQL...")
            await streamer.push(
                role=Role.thought_execution_step,
                content="Generating and executing SQL...",
            )

            # final step => return query and execution result or summarize from knowledge
            if step.type == QueryPlanStepType.FINAL:
                final_action = step.final_action or "query"

                if final_action == "summarize":
                    logger.info("Final step action is 'summarize' - summarizing from acquired knowledge")
                    await streamer.push(
                        role=Role.thought_execution_step,
                        content="Summarizing answer from acquired knowledge...",
                    )
                    summary = await self._summarize_from_knowledge(
                        prompt=prompt,
                        step_description=step.description,
                        message_history=message_history,
                        acquired_knowledge=acquired_knowledge,
                        streamer=streamer,
                        usage=usage,
                        usage_limits=usage_limits,
                    )
                    return "", summary, step_count, acquired_knowledge
                else:
                    # Default: execute SQL query
                    query, result_df, chart_intent = await self._execute_final_step(
                        step_description=step.description,
                        message_history=message_history,
                        data_catalog_subset_context=data_catalog_subset_context,
                        acquired_knowledge=acquired_knowledge,
                        streamer=streamer,
                        usage=usage,
                        usage_limits=usage_limits,
                        native_datasource_name=step_native_datasource_name,
                        is_native_query_mode_enabled=step_native_mode,
                    )

                    execution_result = self._generate_markdown_table(
                        result_df,
                        override_max_rows=agent_settings.max_display_rows_to_user,
                        markdown_prefix=(
                            "**Tip:** This table is truncated for readability. Expand to view the full data.\n"
                            if agent_settings.max_display_rows_to_user < len(result_df)
                            else ""
                        ),
                    )

                    # Append chart intent if provided
                    if chart_intent:
                        chart_json = json.dumps(chart_intent, indent=2)
                        execution_result += f"\n\n```chart\n{chart_json}\n```"

                    return query, execution_result, step_count, acquired_knowledge

            # exploratory => add knowledge
            else:
                # Use multi-path generation if enabled and query is complex
                use_multi_path = agent_settings.enable_multi_path_generation
                if use_multi_path and linked_schema is not None:
                    complexity_classifier = QueryComplexityClassifier()
                    is_simple = complexity_classifier.is_simple(step.description, len(linked_schema.tables))
                    if is_simple:
                        logger.info("Simple query detected, skipping multi-path generation")
                        use_multi_path = False

                if use_multi_path:
                    await self._execute_multi_path_step(
                        step_description=step.description,
                        data_catalog_subset_context=data_catalog_subset_context,
                        acquired_knowledge=acquired_knowledge,
                        streamer=streamer,
                        linked_schema=linked_schema,
                        native_datasource_name=step_native_datasource_name,
                        is_native_query_mode_enabled=step_native_mode,
                    )
                else:
                    await self._execute_exploratory_step(
                        step_description=step.description,
                        message_history=message_history,
                        data_catalog_subset_context=data_catalog_subset_context,
                        acquired_knowledge=acquired_knowledge,
                        streamer=streamer,
                        usage=usage,
                        usage_limits=usage_limits,
                        native_datasource_name=step_native_datasource_name,
                        is_native_query_mode_enabled=step_native_mode,
                    )
                logger.info(
                    f"Step {step_count} completed. Information has been added to the acquired knowledge.",
                )
                await streamer.push(
                    role=Role.thought_execution_step_end,
                    content=f"Step {step_count} completed. Information has been added to the acquired knowledge.",
                )

            # NOTE: Refinement loop removed per SOTA research findings.
            # Research shows refinement loops amplify errors and increase latency.
            # Execute plan as generated without mid-execution modifications.

        logger.warning(f"Execution completed but no FINAL step was produced. Total steps executed: {step_count}")
        return "", "No FINAL step produced by the plan.", step_count, acquired_knowledge

    async def _prune_data_catalogs_for_step(
        self,
        data_catalogs: list[DataCatalog],
        data_catalog_subset: DataCatalogSubset,
    ) -> list[DataCatalog]:
        selected_ds = set(data_catalog_subset.datasources or [])
        selected_tables = set(data_catalog_subset.tables or [])
        logger.info(f"Pruning selected {len(selected_ds)} datasource(s) and {len(selected_tables)} table(s)")

        (
            selected_ds_lower,
            selected_fq,
            selected_tables_only,
            selected_tables_by_ds,
        ) = self._normalize_selected_tables(selected_ds, selected_tables)

        selected_ds_lower = self._resolve_selected_datasources(
            selected_ds_lower,
            selected_tables_by_ds,
            data_catalogs,
        )

        filtered_data_catalogs = []
        for data_catalog in data_catalogs:
            namespace = data_catalog.mind_datasource.datasource.name

            if selected_ds and selected_ds_lower:
                ns_lower = namespace.lower() if namespace else None
                if ns_lower not in selected_ds_lower:
                    continue

            if selected_tables:
                tables = []
                ns_lower = namespace.lower() if namespace else None
                for tbl in data_catalog.mind_datasource.mind_datasource_tables:
                    tbl_name = tbl.table.name
                    tbl_lower = tbl_name.lower() if tbl_name else ""
                    fq_lower = f"{ns_lower}.{tbl_lower}" if ns_lower else tbl_lower

                    # Match qualified table tokens (datasource.table), or unqualified table names
                    if fq_lower in selected_fq:
                        tables.append(tbl)
                        continue
                    if tbl_lower in selected_tables_only:
                        tables.append(tbl)
                        continue
                    if ns_lower and ns_lower in selected_tables_by_ds and tbl_lower in selected_tables_by_ds[ns_lower]:
                        tables.append(tbl)
                        continue

                if not tables:
                    continue

                # We can't use deepcopy because SQLAlchemy ORM objects can't be deepcopied
                # Instead, create a wrapper that filters the tables when generating context
                class FilteredDataCatalog:
                    """Wrapper around DataCatalog that filters mind_datasource_tables for context generation."""

                    def __init__(self, catalog, filtered_tables):
                        self._catalog = catalog
                        self._filtered_tables = filtered_tables
                        self.modified_at = catalog.modified_at
                        self.mind_datasource = catalog.mind_datasource

                    def to_context_str(
                        self,
                        max_tokens: int | None = None,
                        include_statistics: bool = True,
                        table_names: list[str] | None = None,
                        include_datasource_name: bool = True,
                    ) -> str:
                        """Generate context string with filtered tables."""
                        # Temporarily replace the tables list, generate context, then restore
                        original_tables = list(self._catalog.mind_datasource.mind_datasource_tables)
                        try:
                            self._catalog.mind_datasource.mind_datasource_tables = self._filtered_tables
                            return self._catalog.to_context_str(
                                max_tokens=max_tokens,
                                include_statistics=include_statistics,
                                table_names=table_names,
                                include_datasource_name=include_datasource_name,
                            )
                        finally:
                            # Restore original list to avoid side effects
                            self._catalog.mind_datasource.mind_datasource_tables = original_tables

                filtered_data_catalogs.append(FilteredDataCatalog(data_catalog, tables))
            else:
                # No filtering needed, just use the original catalog
                filtered_data_catalogs.append(data_catalog)

        logger.info(f"Data catalog pruning completed. Filtered to {len(filtered_data_catalogs)} catalog(s)")

        # Fallback: If pruning resulted in empty catalog, return catalogs for selected datasources only
        # This prevents context overflow while still providing relevant schema information
        if not filtered_data_catalogs and data_catalogs:
            logger.warning("Pruning resulted in empty catalog. Falling back to catalogs for selected datasources only.")
            # Filter to only datasources that were selected in the plan
            if selected_ds:
                fallback_catalogs = [
                    dc
                    for dc in data_catalogs
                    if (dc.mind_datasource.datasource.name or "").lower() in selected_ds_lower
                ]
                if fallback_catalogs:
                    logger.info(f"Fallback: Using {len(fallback_catalogs)} catalog(s) for selected datasources")
                    return fallback_catalogs

            # If no datasources were selected or none match, return empty list
            # This will cause SQL generation to fail fast with a clear error rather than
            # making uneducated guesses with all catalogs (which causes context overflow)
            logger.error(
                f"Pruning failed: Selected tables {selected_tables} not found in any catalog. "
                f"Available datasources: {[dc.mind_datasource.datasource.name for dc in data_catalogs]}"
            )
            return []

        return filtered_data_catalogs

    @staticmethod
    def _normalize_selected_tables(
        selected_ds: set[str],
        selected_tables: set[str],
    ) -> tuple[set[str], set[str], set[str], dict[str, set[str]]]:
        # TODO: Check for 3-part names
        # Normalize selected table list to handle case and qualification differences
        selected_ds_lower = {ds.lower() for ds in selected_ds if ds}
        selected_fq: set[str] = set()
        selected_tables_only: set[str] = set()
        selected_tables_by_ds: dict[str, set[str]] = {}

        for token in selected_tables:
            if not token:
                continue
            parts = token.split(".")
            if len(parts) >= 2:
                ds = parts[0].lower()
                tbl = parts[-1].lower()
                # TODO: Preserve schema in 3-part names (datasource.schema.table) to avoid collisions.
                selected_fq.add(f"{ds}.{tbl}")
                selected_tables_by_ds.setdefault(ds, set()).add(tbl)
            else:
                selected_tables_only.add(token.lower())

        return selected_ds_lower, selected_fq, selected_tables_only, selected_tables_by_ds

    @staticmethod
    def _resolve_selected_datasources(
        selected_ds_lower: set[str],
        selected_tables_by_ds: dict[str, set[str]],
        data_catalogs: list[DataCatalog],
    ) -> set[str]:
        # Determine which datasources actually exist in the catalog
        available_ds_lower = {(dc.mind_datasource.datasource.name or "").lower() for dc in data_catalogs}
        # If selected datasources don't match any catalog, fall back to datasources
        # inferred from qualified table tokens (e.g., spider2_lite_patents.PUBLICATIONS).
        inferred_ds_lower = set(selected_tables_by_ds.keys())
        if selected_ds_lower and not (selected_ds_lower & available_ds_lower) and inferred_ds_lower:
            logger.warning(
                "Selected datasources do not match any catalog; falling back to datasources inferred from tables."
            )
            selected_ds_lower = inferred_ds_lower
        # If no datasources explicitly selected but tables are qualified, use those.
        if not selected_ds_lower and inferred_ds_lower:
            selected_ds_lower = inferred_ds_lower
        return selected_ds_lower

    async def _execute_exploratory_step(
        self,
        step_description: str,
        message_history: list[ModelMessage],
        data_catalog_subset_context: str,
        acquired_knowledge: AcquiredKnowledge,
        streamer: MessageStreamer,
        usage: RunUsage,
        usage_limits: UsageLimits,
        native_datasource_name: str | None = None,
        is_native_query_mode_enabled: bool | None = None,
    ) -> None:
        """Execute an exploratory step: generate SQL, execute it, and add to acquired knowledge."""
        last_error: Exception | None = None
        last_query = ""
        query_attempts: list[QueryAttempt] = []
        native_mode = (
            self.is_native_query_mode_enabled if is_native_query_mode_enabled is None else is_native_query_mode_enabled
        )
        native_engine = self._get_native_datasource_engine_for(native_datasource_name)

        for attempt in range(agent_settings.max_sql_retries):
            logger.info(f"SQL generation attempt {attempt + 1} of {agent_settings.max_sql_retries}")
            await streamer.push(
                role=Role.thought_execution_step,
                content=f"SQL generation attempt {attempt + 1} of {agent_settings.max_sql_retries}",
            )
            try:
                if attempt == 0 or attempt == agent_settings.max_sql_retries - 1:
                    res = await sql_gen_agent.run(
                        step_description,
                        message_history=message_history,
                        deps=SQLGenAgentDeps(
                            mind=self.mind,
                            data_catalog_subset_context=data_catalog_subset_context,
                            acquired_knowledge=acquired_knowledge.to_string(),
                            is_native_query_mode_enabled=native_mode,
                            native_engine=native_engine,
                        ),
                        model=model_for(self.mind),
                        usage=usage,
                        usage_limits=usage_limits,
                    )
                else:
                    logger.info(f"Generating corrected SQL query (retry attempt {attempt + 1})")
                    await streamer.push(
                        role=Role.thought_execution_step,
                        content=f"Generating corrected SQL query (retry attempt {attempt + 1})",
                    )
                    res = await sql_gen_retry_agent.run(
                        step_description,
                        message_history=message_history,
                        deps=SQLGenRetryAgentDeps(
                            mind=self.mind,
                            data_catalog_subset_context=data_catalog_subset_context,
                            acquired_knowledge=acquired_knowledge.to_string(),
                            failed_query=last_query,
                            error_message=str(last_error),
                            previous_attempts=query_attempts,
                            is_native_query_mode_enabled=native_mode,
                            native_engine=native_engine,
                        ),
                        model=model_for(self.mind),
                        usage=usage,
                        usage_limits=usage_limits,
                    )

                query = res.output.query
                last_query = query
                logger.info(f"SQL query generated (length: {len(query)} characters)")

                logger.info(f"SQL query generated: {query}")
                await streamer.push(
                    role=Role.thought_execution_step_sql,
                    content=f"SQL query generated:\n\n``` {query}\n```",
                )

                # If native query mode is enabled, wrap the query in a native query block
                if native_mode:
                    datasource_name = native_datasource_name or self._native_datasource_name
                    if not datasource_name:
                        raise QueryGenerationError("Native query mode enabled but no datasource available")
                    query = self._strip_datasource_prefix_for_native(query, datasource_name)
                    query = f"SELECT * FROM {datasource_name} ({query})"
                    logger.info(f"Native query mode enabled. Wrapped query: {query}")

                sanitized = self._sanitize_and_validate_sql_mindsdb(query)
                logger.info("SQL query validated and sanitized")
                logger.info("Final SQL to execute: %s", sanitized)

                result_df = self._execute_sql(sanitized)
                logger.info(f"SQL execution successful. Number of rows: {len(result_df)}")
                await streamer.push(
                    role=Role.thought_execution_step,
                    content=f"SQL execution successful. Number of rows: {len(result_df)}",
                )

                # Record successful attempt
                query_attempts.append(QueryAttempt(query=sanitized, result=self._generate_markdown_table(result_df)))
                break

            except Exception as e:
                last_error = e
                logger.warning(f"SQL attempt {attempt + 1} failed: {str(e)}")
                await streamer.push(
                    role=Role.thought_execution_step,
                    content=f"SQL attempt {attempt + 1} failed: {str(e)[:100]}...",
                )

                # Record failed attempt
                query_attempts.append(QueryAttempt(query=last_query or "", error=str(e)))

        # Create acquired knowledge item with all query attempts
        knowledge_item = AcquiredKnowledgeItem(
            step=step_description,
            attempts=query_attempts,
        )
        acquired_knowledge.add_item(knowledge_item)

    async def _execute_multi_path_step(
        self,
        step_description: str,
        data_catalog_subset_context: str,
        acquired_knowledge: AcquiredKnowledge,
        streamer: MessageStreamer,
        linked_schema: LinkedSchema | None = None,
        native_datasource_name: str | None = None,
        is_native_query_mode_enabled: bool | None = None,
    ) -> None:
        """
        Execute a step using multi-path candidate generation.

        Generates multiple SQL candidates using different strategies,
        executes them, and selects the best one.
        """
        logger.info("Using multi-path SQL generation")
        await streamer.push(
            role=Role.thought_execution_step,
            content="Generating multiple SQL candidates...",
        )

        # Generate candidates with MindsDB client for preflight scoring
        candidate_generator = CandidateGenerator(self.mind, mindsdb_client=self.mindsdb_client)

        # Use linked schema if available, otherwise create a minimal one
        if linked_schema is None:
            linked_schema = LinkedSchema(
                tables=[],
                columns={},
                joins=[],
                reasoning="No schema linking performed",
            )

        native_mode = (
            self.is_native_query_mode_enabled if is_native_query_mode_enabled is None else is_native_query_mode_enabled
        )
        native_engine = self._get_native_datasource_engine_for(native_datasource_name)

        try:
            candidates = await candidate_generator.generate(
                question=step_description,
                linked_schema=linked_schema,
                schema_context=data_catalog_subset_context,
                engine=native_engine,
                is_native_query_mode=native_mode,
            )
        except Exception as e:
            logger.error(f"Multi-path generation failed: {e}")
            raise QueryGenerationError(f"Failed to generate SQL candidates: {e}") from e

        logger.info(f"Generated {len(candidates)} SQL candidates")
        await streamer.push(
            role=Role.thought_execution_step,
            content=f"Generated {len(candidates)} SQL candidates",
        )

        # Try to execute each candidate with preflight scoring
        # Early exit when perfect score (2) is found
        query_attempts: list[QueryAttempt] = []
        best_candidate: SQLCandidate | None = None

        for candidate in candidates:
            # First validate columns against linked schema (catches hallucination)
            if linked_schema and linked_schema.columns:
                try:
                    is_valid, invalid_cols = candidate_generator.validate_columns(
                        candidate.query,
                        linked_schema,
                        use_parser=not native_mode,
                        dialect=native_engine,
                    )
                except Exception as e:
                    error = f"Column validation failed: {str(e)}"
                    # Score 0: invalid candidate due to validation error (no preflight/execution)
                    candidate.execution_error = error
                    candidate.preflight_score = 0
                    logger.warning(f"Candidate {candidate.strategy} column validation failed: {e}")
                    query_attempts.append(QueryAttempt(query=candidate.query, error=error))
                    continue
                if not is_valid:
                    error = f"Hallucinated columns detected: {', '.join(invalid_cols[:5])}"
                    # Score 0: invalid candidate due to hallucinated columns (no preflight/execution)
                    candidate.execution_error = error
                    candidate.preflight_score = 0
                    logger.warning(f"Candidate {candidate.strategy} has invalid columns: {invalid_cols}")
                    query_attempts.append(QueryAttempt(query=candidate.query, error=error))
                    continue

            # Preflight score: 0=failed, 1=explain ok, 2=explain+exec ok
            candidate_query = candidate.query
            if native_mode:
                datasource_name = native_datasource_name or self._native_datasource_name
                if not datasource_name:
                    raise QueryGenerationError("Native query mode enabled but no datasource available")
                candidate_query = self._strip_datasource_prefix_for_native(candidate_query, datasource_name)
                candidate_query = f"SELECT * FROM {datasource_name} ({candidate_query})"
                logger.info(f"Native query mode enabled. Wrapped query: {candidate_query}")

            score, sanitized, error, preflight_result_df = candidate_generator.preflight_score(
                candidate_query,
                sanitize_fn=self._sanitize_and_validate_sql_mindsdb,
                engine=native_engine,
                is_native_query_mode=native_mode,
            )
            candidate.preflight_score = score

            if score == 0:
                candidate.execution_error = error
                logger.warning(f"Candidate {candidate.strategy} preflight failed: {error[:100]}")
                query_attempts.append(QueryAttempt(query=candidate.query, error=error))
                continue

            candidate.executed = True
            candidate.execution_result = self._generate_markdown_table(preflight_result_df)

            logger.info(f"Candidate {candidate.strategy} executed: {len(preflight_result_df)} rows (score=1)")
            await streamer.push(
                role=Role.thought_execution_step,
                content=f"Candidate {candidate.strategy}: {len(preflight_result_df)} rows ✓",
            )

            query_attempts.append(QueryAttempt(query=sanitized, result=candidate.execution_result))

            # First successful candidate wins — exit early
            best_candidate = candidate
            logger.info(f"Successful candidate found: {candidate.strategy}, skipping remaining")
            break

        # Select best candidate (may already be set from early exit)
        if best_candidate is None:
            executed_candidates = [c for c in candidates if c.executed and not c.execution_error]

            if executed_candidates:
                if len(executed_candidates) == 1:
                    best_candidate = executed_candidates[0]
                    logger.info(f"Single successful candidate: {best_candidate.strategy}")
                else:
                    # Multiple candidates - use selection agent with preflight scores
                    selection_agent = SelectionAgent(self.mind)
                    best_candidate = await selection_agent.select(
                        candidates=executed_candidates,
                        question=step_description,
                        schema_context=data_catalog_subset_context,
                    )
                    logger.info(f"Selected best candidate: {best_candidate.strategy}")

        if best_candidate:
            await streamer.push(
                role=Role.thought_execution_step_sql,
                content=f"Selected SQL ({best_candidate.strategy}):\n\n```sql\n{best_candidate.query}\n```",
            )
        else:
            logger.error("All candidates failed execution")
            await streamer.push(
                role=Role.thought_execution_step,
                content="All SQL candidates failed execution",
            )

        # Record all attempts in acquired knowledge
        knowledge_item = AcquiredKnowledgeItem(
            step=step_description,
            attempts=query_attempts,
        )
        acquired_knowledge.add_item(knowledge_item)

    async def _execute_final_step(
        self,
        step_description: str,
        message_history: list[ModelMessage],
        data_catalog_subset_context: str,
        acquired_knowledge: AcquiredKnowledge,
        streamer: MessageStreamer,
        usage: RunUsage,
        usage_limits: UsageLimits,
        native_datasource_name: str | None = None,
        is_native_query_mode_enabled: bool | None = None,
    ) -> tuple[str, pd.DataFrame, dict | None]:
        """Execute the final step: generate SQL, execute it, and return the result with optional chart intent."""
        last_error: Exception | None = None
        last_query = ""
        query_attempts: list[QueryAttempt] = []
        native_mode = (
            self.is_native_query_mode_enabled if is_native_query_mode_enabled is None else is_native_query_mode_enabled
        )
        native_engine = self._get_native_datasource_engine_for(native_datasource_name)

        for attempt in range(agent_settings.max_sql_retries):
            logger.info(f"SQL generation attempt {attempt + 1} of {agent_settings.max_sql_retries}")
            await streamer.push(
                role=Role.thought_execution_step,
                content=f"SQL generation attempt {attempt + 1} of {agent_settings.max_sql_retries}",
            )
            try:
                if attempt == 0 or attempt == agent_settings.max_sql_retries - 1:
                    res = await sql_gen_agent.run(
                        step_description,
                        message_history=message_history,
                        deps=SQLGenAgentDeps(
                            mind=self.mind,
                            data_catalog_subset_context=data_catalog_subset_context,
                            acquired_knowledge=acquired_knowledge.to_string(),
                            is_native_query_mode_enabled=native_mode,
                            native_engine=native_engine,
                        ),
                        model=model_for(self.mind),
                        usage=usage,
                        usage_limits=usage_limits,
                        output_type=SQLQuery,
                    )
                else:
                    logger.info(f"Generating corrected SQL query (retry attempt {attempt + 1})")
                    await streamer.push(
                        role=Role.thought_execution_step,
                        content=f"Generating corrected SQL query (retry attempt {attempt + 1})",
                    )
                    res = await sql_gen_retry_agent.run(
                        step_description,
                        message_history=message_history,
                        deps=SQLGenRetryAgentDeps(
                            mind=self.mind,
                            data_catalog_subset_context=data_catalog_subset_context,
                            acquired_knowledge=acquired_knowledge.to_string(),
                            failed_query=last_query,
                            error_message=str(last_error),
                            previous_attempts=query_attempts,
                            is_native_query_mode_enabled=native_mode,
                            native_engine=native_engine,
                        ),
                        model=model_for(self.mind),
                        usage=usage,
                        usage_limits=usage_limits,
                        output_type=SQLQuery,
                    )

                query = res.output.query
                last_query = query
                logger.info(f"SQL query generated (length: {len(query)} characters)")

                logger.info(f"SQL query generated: {query}")
                await streamer.push(
                    role=Role.thought_execution_step_sql,
                    content=f"SQL query generated:\n\n``` {query}\n```",
                )

                # If native query mode is enabled, wrap the query in a native query block
                if native_mode:
                    datasource_name = native_datasource_name or self._native_datasource_name
                    if not datasource_name:
                        raise QueryGenerationError("Native query mode enabled but no datasource available")
                    query = self._strip_datasource_prefix_for_native(query, datasource_name)
                    query = f"SELECT * FROM {datasource_name} ({query})"
                    logger.info(f"Native query mode enabled. Wrapped query: {query}")

                sanitized = self._sanitize_and_validate_sql_mindsdb(query)
                logger.info("SQL query validated and sanitized")
                logger.info("Final SQL to execute: %s", sanitized)

                result_df = self._execute_sql(sanitized)
                logger.info(f"SQL execution successful. Number of rows: {len(result_df)}")
                await streamer.push(
                    role=Role.thought_execution_step,
                    content=f"SQL execution successful. Number of rows: {len(result_df)}",
                )

                # Extract chart intent if provided
                chart_intent = res.output.chart_intent
                if chart_intent:
                    logger.info(f"Chart intent provided: {chart_intent.get('type', 'unknown')} chart")

                return query, result_df, chart_intent

            except Exception as e:
                last_error = e
                logger.warning(f"SQL attempt {attempt + 1} failed: {str(e)}")
                await streamer.push(
                    role=Role.thought_execution_step,
                    content=f"SQL attempt {attempt + 1} failed: {str(e)[:100]}...",
                )

                # Record failed attempt
                query_attempts.append(QueryAttempt(query=last_query or "", error=str(e)))

                if attempt == agent_settings.max_sql_retries - 1:
                    logger.error(f"SQL failed after {agent_settings.max_sql_retries} attempts. Final error: {e}")
                    await streamer.push(
                        role=Role.thought_execution_step,
                        content=f"SQL failed after {agent_settings.max_sql_retries} attempts. "
                        f"Final error: {str(e)[:100]}...",
                    )
                    raise TextToSQLPipelineError(
                        f"SQL failed after {agent_settings.max_sql_retries} attempts. Final error: {e}"
                    ) from e

    async def _summarize_from_knowledge(
        self,
        prompt: str,
        step_description: str,
        message_history: list[ModelMessage],
        acquired_knowledge: AcquiredKnowledge,
        streamer: MessageStreamer,
        usage: RunUsage,
        usage_limits: UsageLimits,
    ) -> str:
        """Summarize the answer from acquired knowledge without executing SQL."""
        logger.info("Summarizing answer from acquired knowledge")

        summarize_deps = SummarizeAgentDeps(
            mind=self.mind,
            prompt=prompt,
            step_description=step_description,
            acquired_knowledge=acquired_knowledge.to_string(),
        )

        result = await summarize_agent.run(
            prompt,
            message_history=message_history,
            deps=summarize_deps,
            model=model_for(self.mind),
            usage=usage,
            usage_limits=usage_limits,
        )

        summary = result.output
        logger.info(f"Summary generated (length: {len(summary)} characters)")
        await streamer.push(
            role=Role.thought_execution_step,
            content="Summary generated from acquired knowledge",
        )

        return summary

    def _sanitize_and_validate_sql_mindsdb(self, sql: str) -> str:
        if not sql or not isinstance(sql, str):
            raise QueryGenerationError("Empty SQL generated")

        if sql.strip().lower().startswith(("error:", "sorry,")):
            raise QueryGenerationError("LLM returned an error string instead of SQL")

        # Allow double quotes for native dialect queries (e.g., Snowflake case-sensitive identifiers)
        # Native dialect format: raw SQL only (no datasource wrapper)
        if not self.is_native_query_mode_enabled:
            stripped_query = re.sub(r"'([^']|'')*'", "", sql)
            if re.search(r'"[^"]*"', stripped_query):
                raise QueryGenerationError(
                    "Double quotes are not allowed outside of string literals. Use backticks instead."
                )

        if re.search(
            r"\b(INSERT|UPDATE|DELETE|CREATE|DROP|TRUNCATE|ALTER|MERGE|GRANT|REVOKE|DESCRIBE|SHOW)\b",
            sql,
            re.IGNORECASE,
        ):
            raise QueryGenerationError("Only SELECT queries are allowed. DESCRIBE and SHOW commands are not supported.")

        if re.search(r"\bLOG10\s*\(", sql, re.IGNORECASE):
            raise QueryGenerationError(
                "LOG10 function is not supported in Snowflake. Use LN(x)/LN(10) instead for base-10 logarithm."
            )

        if sql.count(";") > 1 or (sql.count(";") == 1 and not sql.strip().endswith(";")):
            raise QueryGenerationError("Multiple SQL statements are not allowed. Generate a single SELECT query only.")

        if parse_sql is not None and Select is not None:
            try:
                ast = parse_sql(sql)
            except Exception as e:
                raise QueryGenerationError(f"Unparseable SQL: {e}") from e
            if not isinstance(ast, Select):
                raise QueryGenerationError("Only SELECT queries are allowed")

        return sql.strip().rstrip(";")

    def _execute_sql(self, query: str) -> pd.DataFrame:
        try:
            query_result = self.mindsdb_client.query(query)
            df = query_result.fetch()
            logger.info("\\n%s", df)
            return df

        except Exception as e:
            logger.error(f"Error executing SQL: {e}", exc_info=True)
            raise

    def _generate_markdown_table(
        self,
        df: pd.DataFrame,
        override_max_rows: int | None = None,
        markdown_prefix: str = "",
        markdown_suffix: str = "",
    ) -> str:
        if df.empty:
            return (
                f"{markdown_prefix}| result |\n"
                f"{markdown_suffix}|--------|\n"
                f"{markdown_suffix}| (empty) |{markdown_suffix}"
            )

        row_count = len(df)
        max_rows = override_max_rows or min(agent_settings.max_display_rows_to_agent, len(df))

        df_display = df.head(max_rows).copy()

        # Format numeric columns to prevent scientific notation
        # This logic is less costly than using format_numeric_columns from minds.common.utilities
        markdown_table = df_display.to_markdown(
            index=False,
            tablefmt="github",
            floatfmt=",.1f",  # commas + 1 decimal, avoids scientific notation
            intfmt=",d",  # commas for ints
            missingval="",  # blank instead of "nan"
        )

        if row_count > max_rows:
            markdown_table += f"\n\n[Showing {max_rows} of {row_count} rows]\n\n"

        return f"{markdown_prefix}{markdown_table}{markdown_suffix}"
