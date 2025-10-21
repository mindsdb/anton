import copy
import re
from datetime import datetime

from mindsdb_sdk.server import Server
from mindsdb_sql_parser import parse_sql
from mindsdb_sql_parser.ast import Select
from pydantic_ai import Agent as PydanticAIAgent

from minds.agent.exceptions import QueryGenerationError
from minds.agent.llm import get_llm_config
from minds.agent.prompt_templates import (
    PLANNING_PROMPT_TEMPLATE,
    RETRY_PROMPT_TEMPLATE,
    get_prompt_template_for_engines,
)
from minds.cache import data_catalog_cache
from minds.common.logger import setup_logging
from minds.common.vars import (
    MAX_COLUMN_WIDTH,
    MAX_DISPLAY_ROWS,
    MAX_SQL_RETRIES,
)
from minds.model.data_catalog import DataCatalog
from minds.model.database_agent import QueryGenerationResult, QueryGenerationResultRetry, QueryPlanResult
from minds.model.mind import Mind

logger = setup_logging()


class DatabaseToolkit:
    """Toolkit for database operations that works with Pydantic AI agents.

    This toolkit provides tools for interacting with databases through MindsDB,
    including listing databases and executing SQL queries.
    """

    def __init__(
        self,
        mind: Mind,
        mindsdb_client: Server,
    ):
        """Initialize the DatabaseToolkit with MindsDB connection.

        Args:
            mind: MindsDB mind instance from mindsdb_sdk.connect()
            mindsdb_client: MindsDB server instance from mindsdb_sdk.connect()
        """
        self.mind = mind
        self.mindsdb_client = mindsdb_client

    async def generate_and_execute_sql(self, conversation_context: str) -> str:
        return await self._generate_and_execute_with_retry(conversation_context)

    async def _generate_and_execute_with_retry(self, conversation_context: str) -> str:
        """Generate and execute SQL with LLM-driven retry logic."""
        last_error = None
        last_query = ""

        for attempt in range(MAX_SQL_RETRIES):
            logger.info(f"Attempt {attempt + 1} of {MAX_SQL_RETRIES}")

            try:
                query = None
                # Initial attempt will have no error context
                if attempt == 0:
                    query = await self.generate_sql(conversation_context)
                # Middle attempts: use correction with error context
                elif attempt < MAX_SQL_RETRIES - 1:
                    query = await self._generate_corrected_sql(conversation_context, last_query, str(last_error))
                # Final attempt: generate corrected SQL from scratch without error context
                else:
                    logger.info("The final attempt will exclude error context and try from scratch.")
                    query = await self.generate_sql(conversation_context)

                last_query = query
                sanitized_query = self._sanitize_and_validate_sql_mindsdb(query)

                return await self.execute_sql(sanitized_query, raise_on_error=True)
            except Exception as e:
                last_error = e

                if attempt == MAX_SQL_RETRIES - 1:
                    # Final attempt failed
                    return (
                        f"Sorry, I'm having an issue querying the data I need. "
                        f"Tried {MAX_SQL_RETRIES} times. Final error: {str(e)}"
                    )

                logger.info(f"SQL attempt {attempt + 1} failed, retrying: {str(e)}")

    def _sanitize_and_validate_sql_mindsdb(self, sql: str) -> str:
        """Validate that SQL is a single SELECT and add LIMIT if missing. Uses mindsdb_sql_parser when available."""
        if not sql or not isinstance(sql, str):
            raise QueryGenerationError("Empty SQL generated")
        # Guard against propagating error strings
        if sql.strip().lower().startswith(("error:", "sorry,")):
            raise QueryGenerationError("LLM returned an error string instead of SQL")

        # Check for double quotes outside string literals
        # Strip out all single-quoted string literals first
        stripped_query = re.sub(r"'([^']|'')*'", "", sql)
        # Then look for any double-quoted fragments outside of string literals
        if re.search(r'"[^"]*"', stripped_query):
            raise QueryGenerationError(
                "Double quotes are not allowed outside of string literals. "
                "If the column names contain spaces or special characters, or there is a need to enforce "
                "case sensitivity, use backticks instead of double quotes."
            )

        # Disallow dangerous statements quickly
        if re.search(r"\b(INSERT|UPDATE|DELETE|CREATE|DROP|TRUNCATE|ALTER|MERGE|GRANT|REVOKE)\b", sql, re.IGNORECASE):
            raise QueryGenerationError("Only SELECT queries are allowed")

        # Validate syntax and ensure it's a SELECT using the parser if available
        if parse_sql is not None and Select is not None:
            try:
                ast = parse_sql(sql)
            except Exception as e:
                raise QueryGenerationError(f"Unparseable SQL: {e}") from e
            if not isinstance(ast, Select):
                raise QueryGenerationError("Only SELECT queries are allowed")

        # Strip trailing semicolon and ensure LIMIT exists (best-effort)
        sanitized = sql.strip().rstrip(";")

        return sanitized

    def _filter_catalogs_with_plan(self, data_catalogs: list[DataCatalog], plan: QueryPlanResult):
        """Filter catalogs according to the plan selection (datasources and tables)."""
        if not plan or (not plan.selected_datasources and not plan.selected_tables):
            return data_catalogs

        filtered = []
        selected_ds = set(plan.selected_datasources or [])
        selected_tables = set(plan.selected_tables or [])

        for catalog in data_catalogs:
            namespace = catalog.mind_datasource.datasource.name

            if selected_ds and namespace and namespace not in selected_ds:
                continue

            # If table filters were specified, reduce tables map
            if selected_tables:
                tables = []
                for tbl in catalog.mind_datasource.mind_datasource_tables:
                    tbl_name = tbl.table.name
                    fq = f"{namespace}.{tbl_name}" if namespace else tbl_name
                    if not selected_tables or fq in selected_tables:
                        tables.append(tbl)
                # If nothing matched and there was a hard table filter, skip catalog
                if selected_tables and not tables:
                    continue
                catalog.mind_datasource.mind_datasource_tables = tables

            filtered.append(catalog)

        return filtered

    async def _plan_selection(
        self, conversation_context: str, data_catalogs: list[DataCatalog]
    ) -> QueryPlanResult | None:
        """Run a planning agent to decide engine/datasources/tables from full catalogs."""
        # Convert catalogs to context string
        catalog_contexts = [catalog.to_context_str() for catalog in data_catalogs]
        current_date = datetime.now().strftime("%Y-%m-%d")
        catalog_contexts.append(f"Current date: {current_date}")
        catalog_context = "\n\n".join(catalog_contexts)

        # Get LLM config
        llm_config = get_llm_config(self.mind.provider, self.mind.model_name)

        planning_prompt = PLANNING_PROMPT_TEMPLATE.format(
            context=catalog_context, conversation_context=conversation_context
        )

        planning_agent = PydanticAIAgent(
            model=llm_config,
            system_prompt=planning_prompt,
            output_type=QueryPlanResult,
        )
        try:
            plan_res = await planning_agent.run(conversation_context)
            plan = plan_res.output
            logger.info(f"Planning step result: {plan}")
            return plan
        except Exception as e:
            logger.warning(f"Planning step failed, proceeding without filtering: {e}")
            return None

    async def _generate_corrected_sql(self, conversation_context: str, failed_query: str, error_message: str) -> str:
        """Use LLM to analyze error and generate corrected SQL following existing patterns."""
        logger.info(f"Attempting LLM-driven SQL correction for error: {error_message}")

        # Load data catalogs (same as generate_sql)
        data_catalogs = data_catalog_cache.load(self.mind)

        if not data_catalogs:
            logger.warning(f"No data catalogs found for agent {self.mind.name} during retry")
            raise Exception("No database context available for retry")

        # Run planning step to narrow down relevant engines/datasources/tables
        plan = await self._plan_selection(conversation_context, data_catalogs)
        if plan and plan.error:
            logger.info(f"Planning step returned error: {plan.error}. Proceeding with full catalogs.")
            plan = None

        # Filter catalogs by plan (if available)
        data_catalogs_filtered = (
            self._filter_catalogs_with_plan(copy.deepcopy(data_catalogs), plan) if plan else data_catalogs
        )

        # Extract engine types from data catalogs (same as generate_sql)
        engines = {catalog.mind_datasource.datasource.engine for catalog in data_catalogs_filtered}
        logger.info(f"Engine types detected for retry in agent '{self.mind.name}': {engines}")

        # Convert data catalogs to context strings (same as generate_sql)
        catalog_contexts = [catalog.to_context_str() for catalog in data_catalogs_filtered]

        # Add current date to the context (same as generate_sql)
        current_date = datetime.now().strftime("%Y-%m-%d")
        date_context = f"Current date: {current_date}"
        catalog_contexts.append(date_context)

        catalog_context = "\n\n".join(catalog_contexts)

        # Log the retry context
        logger.info(f"Retry context for agent '{self.mind.name}' ({len(catalog_context)} chars)")

        # Use same LLM config pattern as generate_sql
        try:
            llm_config = get_llm_config(self.mind.provider, self.mind.model_name)
        except Exception as e:
            logger.error(f"Error getting LLM config: {str(e)}")
            raise Exception("Error getting LLM config") from e

        # Create retry prompt with error context - use same template selection logic
        retry_prompt = RETRY_PROMPT_TEMPLATE.format(
            conversation_context=conversation_context,
            context=catalog_context,
            failed_query=failed_query,
            error_message=error_message,
        )

        # Log the retry prompt
        logger.info(f"Retry prompt for SQL correction ({len(retry_prompt)} chars):")
        # logger.info(f"Retry prompt:\n{retry_prompt}")

        # Create correction agent using same pattern as generate_sql
        correction_agent = PydanticAIAgent(
            model=llm_config,
            system_prompt=retry_prompt,
            output_type=QueryGenerationResultRetry,
        )

        # Generate corrected SQL
        sql_result = await correction_agent.run(conversation_context)
        if sql_result.output.error:
            error_msg = f"Error generating SQL query: {str(sql_result.output.error)}"
            logger.error(error_msg, exc_info=True)
            raise QueryGenerationError(f"SQL correction failed: {sql_result.output.error}")

        corrected_query = sql_result.output.query
        logger.info(f"Generated corrected SQL: {corrected_query}")

        return corrected_query

    async def generate_sql(self, conversation_context: str) -> str:
        """
        Generate SQL based on user input and database context.

        Args:
            conversation_context: The complete conversation context

        Returns:
            A SQL query string that addresses the user's request
        """
        # Load data catalogs for this agent
        data_catalogs = data_catalog_cache.load(self.mind)

        if not data_catalogs:
            logger.warning(f"No data catalogs found for agent {self.mind.name}")
            raise QueryGenerationError("No database context available. Unable to generate SQL query.")

        # Run planning step to narrow down relevant engines/datasources/tables
        plan = await self._plan_selection(conversation_context, data_catalogs)
        if plan and plan.error:
            logger.info(f"Planning step returned error: {plan.error}. Proceeding with full catalogs.")
            plan = None

        # Filter catalogs by plan (if available)
        data_catalogs_filtered = (
            self._filter_catalogs_with_plan(copy.deepcopy(data_catalogs), plan) if plan else data_catalogs
        )

        # Extract engine types from filtered data catalogs
        engines = {catalog.mind_datasource.datasource.engine for catalog in data_catalogs_filtered}
        logger.info(f"Engine types detected for agent '{self.mind.name}' after planning: {engines}")

        # Convert data catalogs to context strings
        catalog_contexts = [catalog.to_context_str() for catalog in data_catalogs_filtered]

        # Add current date to the context
        current_date = datetime.now().strftime("%Y-%m-%d")
        date_context = f"Current date: {current_date}"
        catalog_contexts.append(date_context)

        # Compose a prompt for SQL generation using the data cavtalog context
        catalog_context = "\n\n".join(catalog_contexts)

        # Log the exact context string being sent to the LLM
        logger.info(f"Data catalog context for agent '{self.mind.name}' ({len(catalog_context)} chars):")
        # logger.info(f"Context string:\n{catalog_context}")

        llm_config = get_llm_config(self.mind.provider, self.mind.model_name)

        # Get the appropriate prompt template based on engines
        prompt_template = get_prompt_template_for_engines(engines)
        generation_prompt = prompt_template.format(context=catalog_context)

        # Log the final system prompt being sent to the LLM
        logger.info(f"Final system prompt for SQL generation ({len(generation_prompt)} chars):")
        # logger.info(f"System prompt:\n{generation_prompt}")

        generation_agent = PydanticAIAgent(
            model=llm_config,
            system_prompt=generation_prompt,
            output_type=QueryGenerationResult,
        )
        sql_result = await generation_agent.run(conversation_context)
        if sql_result.output.error:
            error_msg = f"Error generating SQL query: {str(sql_result.output.error)}"
            logger.error(error_msg, exc_info=True)
            raise QueryGenerationError(sql_result.output.error)
        return sql_result.output.query

    async def execute_sql(self, query: str, raise_on_error: bool = False) -> str:
        """Execute an SQL query against MindsDB.

        Args:
            query: The SQL query to execute
            raise_on_error: If True, re-raise exceptions instead of returning error strings

        Returns:
            The query results as a formatted string table
        """
        try:
            logger.info(f"Executing SQL query: {query}")

            # Execute the query using the SDK
            query_result = self.mindsdb_client.query(query)

            # Fetch results as pandas DataFrame
            df = query_result.fetch()

            # If DataFrame is empty, return message
            if df.empty:
                return f"Query executed successfully: {query}\nNo results returned."

            # Format the result as a nicely formatted table
            result = f"Query executed: {query}\n\n"

            # Get row and column counts
            row_count = len(df)
            col_count = len(df.columns)
            result += f"Results: {row_count} rows x {col_count} columns\n\n"

            # Truncate large DataFrames and format nicely
            # Use constants for max rows and column width
            max_rows = min(MAX_DISPLAY_ROWS, row_count)

            # Format the DataFrame with truncation options
            table_str = df.to_string(
                index=False,  # Don't show row indices
                max_rows=max_rows,  # Limit number of rows
                max_cols=None,  # Show all columns
                max_colwidth=MAX_COLUMN_WIDTH,  # Truncate wide column values
                justify="left",  # Left align text
                na_rep="NULL",  # Show NULL values clearly
            )

            result += table_str

            # Add note if truncated
            if row_count > max_rows:
                result += f"\n\n[Showing {max_rows} of {row_count} rows]"

            return result

        except Exception as e:
            error_msg = f"Error executing SQL query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if raise_on_error:
                raise e
            return f"Error executing query: {str(e)}"
