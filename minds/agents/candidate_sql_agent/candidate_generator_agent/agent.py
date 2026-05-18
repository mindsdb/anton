"""
Multi-path SQL candidate generator.

Generates multiple SQL candidates using different reasoning strategies:
- Divide and Conquer: Break question into subproblems
- Query Plan CoT: Generate execution plan then convert to SQL
- Direct Generation: Simple direct SQL generation

Each path runs independently and may produce different SQL approaches.
"""

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mindsdb_sql_parser import parse_sql
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from minds.agents.candidate_sql_agent.candidate_generator_agent.instructions_templates import (
    DIRECT_SYSTEM_PROMPT_BIGQUERY,
    DIRECT_SYSTEM_PROMPT_MINDSDB,
    DIRECT_SYSTEM_PROMPT_MSSQL,
    DIRECT_SYSTEM_PROMPT_SNOWFLAKE,
    DIVIDE_CONQUER_SYSTEM_PROMPT_BIGQUERY,
    DIVIDE_CONQUER_SYSTEM_PROMPT_MINDSDB,
    DIVIDE_CONQUER_SYSTEM_PROMPT_MSSQL,
    DIVIDE_CONQUER_SYSTEM_PROMPT_SNOWFLAKE,
    QUERY_PLAN_SYSTEM_PROMPT_BIGQUERY,
    QUERY_PLAN_SYSTEM_PROMPT_MINDSDB,
    QUERY_PLAN_SYSTEM_PROMPT_MSSQL,
    QUERY_PLAN_SYSTEM_PROMPT_SNOWFLAKE,
)
from minds.agents.candidate_sql_agent.linker_agent.agent import LinkedSchema
from minds.agents.candidate_sql_agent.settings import CandidateSQLAgentSettings
from minds.common.logger import get_logger
from minds.common.settings.app_settings import get_app_settings

logger = get_logger(__name__)
settings = get_app_settings()
agent_settings = CandidateSQLAgentSettings()

if TYPE_CHECKING:
    from minds.model.mind import Mind


@dataclass
class SQLCandidate:
    """A SQL candidate with metadata about its generation."""

    query: str
    strategy: str  # "divide_conquer", "query_plan", "direct"
    reasoning: str = ""
    executed: bool = False
    execution_result: str | None = None
    execution_error: str | None = None
    preflight_score: int = 0  # 0=failed, 1=executed successfully


class DivideConquerOutput(BaseModel):
    """Output from divide and conquer strategy."""

    tables_needed: list[str] = Field(description="Tables required for the query")
    columns_to_select: list[str] = Field(description="Columns to SELECT")
    join_conditions: list[str] = Field(description="JOIN conditions if any")
    where_filters: list[str] = Field(description="WHERE clause filters")
    group_by: list[str] = Field(description="GROUP BY columns if any")
    order_by: str = Field(default="", description="ORDER BY clause")
    limit: int | None = Field(default=None, description="LIMIT value if any")
    final_sql: str = Field(description="The complete SQL query")


class QueryPlanOutput(BaseModel):
    """Output from query plan strategy."""

    scan_tables: list[str] = Field(description="Tables to scan")
    filter_conditions: list[str] = Field(description="Filter conditions to apply")
    join_operations: list[str] = Field(description="Join operations")
    aggregations: list[str] = Field(description="Aggregation operations")
    sort_order: str = Field(default="", description="Sort specification")
    row_limit: int | None = Field(default=None, description="Row limit")
    final_sql: str = Field(description="The complete SQL query")


class DirectOutput(BaseModel):
    """Output from direct generation strategy."""

    query: str = Field(description="The SQL query")


@dataclass
class CandidateGeneratorDeps:
    """Dependencies for candidate generator agents."""

    mind: "Mind"
    schema_context: str
    question: str


divide_conquer_agent = Agent(
    model=None,  # Set at runtime
    system_prompt=DIVIDE_CONQUER_SYSTEM_PROMPT_SNOWFLAKE,
    output_type=DivideConquerOutput,
    retries=0,
)

query_plan_agent = Agent(
    model=None,  # Set at runtime
    system_prompt=QUERY_PLAN_SYSTEM_PROMPT_SNOWFLAKE,
    output_type=QueryPlanOutput,
    retries=0,
)

direct_agent = Agent(
    model=None,  # Set at runtime
    system_prompt=DIRECT_SYSTEM_PROMPT_SNOWFLAKE,
    output_type=DirectOutput,
    retries=0,
)


def _select_system_prompts(*, native_mode: bool, engine: str | None) -> tuple[str, str, str]:
    engine_lower = (engine or "").lower()
    if native_mode and engine_lower == "bigquery":
        return (
            DIVIDE_CONQUER_SYSTEM_PROMPT_BIGQUERY,
            QUERY_PLAN_SYSTEM_PROMPT_BIGQUERY,
            DIRECT_SYSTEM_PROMPT_BIGQUERY,
        )
    if native_mode and engine_lower == "snowflake":
        return (
            DIVIDE_CONQUER_SYSTEM_PROMPT_SNOWFLAKE,
            QUERY_PLAN_SYSTEM_PROMPT_SNOWFLAKE,
            DIRECT_SYSTEM_PROMPT_SNOWFLAKE,
        )
    if native_mode and engine_lower == "mssql":
        return (
            DIVIDE_CONQUER_SYSTEM_PROMPT_MSSQL,
            QUERY_PLAN_SYSTEM_PROMPT_MSSQL,
            DIRECT_SYSTEM_PROMPT_MSSQL,
        )
    return (
        DIVIDE_CONQUER_SYSTEM_PROMPT_MINDSDB,
        QUERY_PLAN_SYSTEM_PROMPT_MINDSDB,
        DIRECT_SYSTEM_PROMPT_MINDSDB,
    )


class CandidateGenerator:
    """
    Generates multiple SQL candidates using different reasoning strategies.

    This implements the multi-path generation approach from CHASE-SQL research,
    where different reasoning strategies produce diverse SQL candidates that
    can be compared and selected.
    """

    def __init__(self, mind: "Mind", mindsdb_client=None):
        self.mind = mind
        self.mindsdb_client = mindsdb_client

    async def _run_agent_with_retry(self, agent: Agent, user_prompt: str, model):
        last_error: Exception | None = None
        for attempt in range(agent_settings.max_candidate_retries):
            try:
                return await agent.run(user_prompt, model=model)
            except ModelHTTPError as e:
                last_error = e
                status_code = getattr(e, "status_code", None)
                if status_code != 500 or attempt == agent_settings.max_candidate_retries - 1:
                    raise
                logger.warning(
                    f"Model 500 from {getattr(e, 'model_name', 'unknown')} "
                    f"(attempt {attempt + 1}/{agent_settings.max_candidate_retries}). Retrying..."
                )
                await asyncio.sleep(min(2**attempt, 8))
            except Exception as e:
                last_error = e
                raise
        if last_error:
            raise last_error

    def validate_columns(
        self,
        sql: str,
        linked_schema: LinkedSchema,
        *,
        use_parser: bool = True,
        dialect: str | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Validate that column references in SQL exist in the linked schema.

        Returns (is_valid, invalid_columns).
        This catches column hallucination before execution.
        """

        def _strip_identifier(token: str) -> str:
            token = token.strip()
            if (token.startswith("`") and token.endswith("`")) or (token.startswith('"') and token.endswith('"')):
                token = token[1:-1]
            return token.strip().lower()

        def _alias_name(alias_expr: exp.Expression | None) -> str | None:
            if alias_expr is None:
                return None
            if isinstance(alias_expr, exp.TableAlias):
                if isinstance(alias_expr.this, exp.Identifier):
                    return alias_expr.this.name
                if hasattr(alias_expr, "name"):
                    return alias_expr.name
                return None
            if isinstance(alias_expr, exp.Identifier):
                return alias_expr.name
            if isinstance(alias_expr, str):
                return alias_expr
            return None

        def _parse_with_sqlglot(sql_text: str) -> exp.Expression:
            dialect_name = dialect.lower() if dialect else None
            try:
                if dialect_name:
                    return parse_one(sql_text, read=dialect_name)
                return parse_one(sql_text)
            except ParseError as e:
                raise ValueError(f"SQL parsing failed for validation: {e}") from e

        # Build set of valid column names from linked schema
        valid_columns: set[str] = set()
        column_to_tables: dict[str, set[str]] = {}
        table_basenames: set[str] = set()
        datasource_names: set[str] = set()
        schema_names: set[str] = set()
        for table, cols in linked_schema.columns.items():
            parts = [p for p in table.split(".") if p]
            if parts:
                datasource_names.add(parts[0].lower())
            if len(parts) >= 3:
                schema_names.add(parts[1].lower())
            table_name = parts[-1] if parts else table
            table_basenames.add(table_name.lower())
            for col in cols:
                col_lower = str(col).lower()
                valid_columns.add(col_lower)
                valid_columns.add(f"{table_name.lower()}.{col_lower}")
                column_to_tables.setdefault(col_lower, set()).add(table_name.lower())

        def _collect_refs_sqlglot(sql_text: str) -> tuple[dict[str, str], list[tuple[str, str]], list[str]]:
            alias_map: dict[str, str] = {}
            dot_refs: list[tuple[str, str]] = []
            unqualified_tokens: list[str] = []

            parsed = _parse_with_sqlglot(sql_text)
            for table_expr in parsed.find_all(exp.Table):
                table_name = table_expr.name
                if not table_name:
                    continue
                base = _strip_identifier(table_name)
                if base:
                    alias_map[base] = base
                alias = _alias_name(table_expr.args.get("alias"))
                if alias:
                    alias_clean = _strip_identifier(alias)
                    if alias_clean:
                        alias_map[alias_clean] = base

            for col_expr in parsed.find_all(exp.Column):
                col_name = col_expr.name
                if not col_name:
                    continue
                table_name = col_expr.table
                if table_name:
                    dot_refs.append((table_name, col_name))
                else:
                    unqualified_tokens.append(col_name)

            return alias_map, dot_refs, unqualified_tokens

        def _collect_refs_mindsdb(sql_text: str) -> tuple[dict[str, str], list[tuple[str, str]], list[str]]:
            from mindsdb_sql_parser.ast import Identifier

            from minds.common.mindsdb import query_traversal

            alias_map: dict[str, str] = {}
            dot_refs: list[tuple[str, str]] = []
            unqualified_tokens: list[str] = []

            def _alias_from_node(node) -> str | None:
                for attr in ("alias", "alias_name", "as_name", "table_alias", "alias_identifier"):
                    val = getattr(node, attr, None)
                    if val is None:
                        continue
                    if isinstance(val, Identifier):
                        parts = [p for p in getattr(val, "parts", []) if p]
                        if parts:
                            return parts[-1]
                    if isinstance(val, str):
                        return val
                    name = getattr(val, "name", None)
                    if name:
                        return name
                return None

            parsed = parse_sql(sql_text)

            def _collect_identifiers(node, is_table, **_kwargs):
                if isinstance(node, Identifier):
                    parts = [p for p in node.parts if p]
                    if is_table:
                        if parts:
                            base = _strip_identifier(parts[-1])
                            if base:
                                alias_map[base] = base
                            alias = _alias_from_node(node)
                            if alias:
                                alias_clean = _strip_identifier(alias)
                                if alias_clean:
                                    alias_map[alias_clean] = base
                        return
                    if len(parts) >= 2:
                        dot_refs.append((parts[-2], parts[-1]))
                    elif len(parts) == 1:
                        unqualified_tokens.append(parts[0])

            query_traversal(parsed, _collect_identifiers)
            return alias_map, dot_refs, unqualified_tokens

        if use_parser:
            alias_map, dot_refs, unqualified_tokens = _collect_refs_mindsdb(sql)
        else:
            alias_map, dot_refs, unqualified_tokens = _collect_refs_sqlglot(sql)

        invalid: list[str] = []

        for table_ref, col_ref in dot_refs:
            table_clean = _strip_identifier(table_ref)
            col_clean = _strip_identifier(col_ref)
            if not table_clean or not col_clean:
                continue

            # Skip datasource/schema qualifiers
            if table_clean in datasource_names or table_clean in schema_names:
                continue

            # Resolve aliases to base table name if available
            table_base = alias_map.get(table_clean, table_clean)
            # If we can resolve the table, require the column to exist on that table
            if table_base in table_basenames:
                if f"{table_base}.{col_clean}" not in valid_columns:
                    invalid.append(f"{table_ref}.{col_ref}")
                continue
            # Unknown table: fall back to global column existence check
            if col_clean not in valid_columns:
                invalid.append(f"{table_ref}.{col_ref}")

        return (len(invalid) == 0, invalid)

    def preflight_score(
        self,
        sql: str,
        sanitize_fn=None,
        *,
        linked_schema: LinkedSchema | None = None,
        engine: str | None = None,
        is_native_query_mode: bool = False,
    ) -> tuple[int, str, str, object | None]:
        """
        Score a SQL candidate using preflight checks.

        Returns (score, sanitized_sql, error, exec_result):
        - Score 1: Query executed successfully
        - Score 0: Failed (invalid syntax, hallucinated columns, or execution error)
        """
        # 1. Sanitize
        try:
            sanitized = sanitize_fn(sql) if sanitize_fn else sql
        except Exception as e:
            return (0, sql, str(e), None)

        # 2. Parse + schema validation (non-native only — native queries are validated by execution)
        if not is_native_query_mode:
            try:
                parse_sql(sanitized)
            except Exception as e:
                return (0, sanitized, f"SQL parsing failed: {e}", None)

        if linked_schema and linked_schema.columns:
            try:
                is_valid, invalid_cols = self.validate_columns(
                    sanitized,
                    linked_schema,
                    use_parser=not is_native_query_mode,
                    dialect=engine,
                )
            except Exception as e:
                return (0, sanitized, f"Column validation failed: {e}", None)
            if not is_valid:
                return (0, sanitized, f"Hallucinated columns detected: {', '.join(invalid_cols[:5])}", None)

        try:
            exec_result = self.mindsdb_client.query(sanitized).fetch()
            return (1, sanitized, "", exec_result)
        except Exception as e:
            return (0, sanitized, str(e), None)

    async def generate(
        self,
        question: str,
        linked_schema: LinkedSchema,
        schema_context: str,
        engine: str | None = None,
        is_native_query_mode: bool = False,
    ) -> list[SQLCandidate]:
        """
        Generate SQL candidates via multiple paths.

        Args:
            question: The natural language question
            linked_schema: Schema elements identified by schema linker
            schema_context: Formatted schema context string

        Returns:
            List of SQLCandidate objects from different strategies
        """
        from minds.agents.helpers import model_for

        model = model_for(self.mind)

        user_prompt = f"""Question: {question}

Schema:
{schema_context}

Generate SQL to answer this question."""

        divide_prompt, plan_prompt, direct_prompt = _select_system_prompts(
            native_mode=is_native_query_mode,
            engine=engine,
        )

        tasks = [
            self._divide_and_conquer(user_prompt, model, agent=divide_conquer_agent, system_prompt=divide_prompt),
            self._query_plan_cot(user_prompt, model, agent=query_plan_agent, system_prompt=plan_prompt),
            self._direct_generation(user_prompt, model, agent=direct_agent, system_prompt=direct_prompt),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates = []
        strategy_names = ["divide_conquer", "query_plan", "direct"]

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Strategy {strategy_names[i]} failed: {result}")
                continue
            candidates.append(result)

        if not candidates:
            logger.error("All candidate generation strategies failed")
            raise RuntimeError("Failed to generate any SQL candidates")

        logger.info(f"Generated {len(candidates)} SQL candidates")
        return candidates

    async def _divide_and_conquer(self, user_prompt: str, model, *, agent: Agent, system_prompt: str) -> SQLCandidate:
        """Generate SQL using divide and conquer strategy."""
        logger.debug("Running divide and conquer strategy")

        agent.system_prompt = system_prompt
        result = await self._run_agent_with_retry(agent, user_prompt, model)

        reasoning = (
            f"Tables: {result.output.tables_needed}\n"
            f"Columns: {result.output.columns_to_select}\n"
            f"Joins: {result.output.join_conditions}\n"
            f"Filters: {result.output.where_filters}\n"
            f"Group by: {result.output.group_by}"
        )

        return SQLCandidate(
            query=result.output.final_sql,
            strategy="divide_conquer",
            reasoning=reasoning,
        )

    async def _query_plan_cot(self, user_prompt: str, model, *, agent: Agent, system_prompt: str) -> SQLCandidate:
        """Generate SQL using query plan chain-of-thought strategy."""
        logger.debug("Running query plan CoT strategy")

        agent.system_prompt = system_prompt
        result = await self._run_agent_with_retry(agent, user_prompt, model)

        reasoning = (
            f"Scan: {result.output.scan_tables}\n"
            f"Filter: {result.output.filter_conditions}\n"
            f"Join: {result.output.join_operations}\n"
            f"Aggregate: {result.output.aggregations}"
        )

        return SQLCandidate(
            query=result.output.final_sql,
            strategy="query_plan",
            reasoning=reasoning,
        )

    async def _direct_generation(self, user_prompt: str, model, *, agent: Agent, system_prompt: str) -> SQLCandidate:
        """Generate SQL directly without explicit reasoning steps."""
        logger.debug("Running direct generation strategy")

        agent.system_prompt = system_prompt
        result = await self._run_agent_with_retry(agent, user_prompt, model)

        return SQLCandidate(
            query=result.output.query,
            strategy="direct",
            reasoning="Direct generation without explicit reasoning",
        )

    async def generate_with_execution(
        self,
        question: str,
        linked_schema: LinkedSchema,
        schema_context: str,
        execute_fn,
        max_fix_attempts: int = 2,
    ) -> list[SQLCandidate]:
        """
        Generate candidates and attempt to execute each.

        Failed candidates get fix attempts before being marked as failed.

        Args:
            question: The natural language question
            linked_schema: Schema elements identified by schema linker
            schema_context: Formatted schema context string
            execute_fn: Function to execute SQL (returns result or raises exception)
            max_fix_attempts: Max attempts to fix each failed candidate

        Returns:
            List of SQLCandidate objects with execution results
        """
        candidates = await self.generate(question, linked_schema, schema_context)

        for candidate in candidates:
            await self._try_execute_candidate(candidate, execute_fn, question, schema_context, max_fix_attempts)

        return candidates

    async def _try_execute_candidate(
        self,
        candidate: SQLCandidate,
        execute_fn,
        question: str,
        schema_context: str,
        max_fix_attempts: int,
    ) -> None:
        """Try to execute a candidate, with fix attempts on failure."""
        current_sql = candidate.query

        for attempt in range(max_fix_attempts + 1):
            try:
                result = execute_fn(current_sql)
                if asyncio.iscoroutine(result):
                    result = await result
                candidate.executed = True
                candidate.execution_result = str(result)
                candidate.query = current_sql
                logger.info(f"Candidate {candidate.strategy} executed successfully")
                return
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Candidate {candidate.strategy} attempt {attempt + 1} failed: {error_msg[:100]}")

                if attempt < max_fix_attempts:
                    current_sql = await self._fix_sql(current_sql, error_msg, question, schema_context)
                else:
                    candidate.execution_error = error_msg

    async def _fix_sql(self, sql: str, error: str, question: str, schema_context: str) -> str:
        """Attempt to fix SQL based on error message."""
        from minds.agents.helpers import model_for

        fix_prompt = f"""Fix this SQL query based on the error.

Question: {question}

Schema:
{schema_context}

Failed SQL:
{sql}

Error:
{error}

Return only the corrected SQL query."""

        try:
            result = await direct_agent.run(
                fix_prompt,
                model=model_for(self.mind),
            )
            return result.output.query
        except Exception as e:
            logger.warning(f"Failed to fix SQL: {e}")
            return sql
