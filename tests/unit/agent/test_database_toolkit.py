"""
Unit tests for DatabaseToolkit class.

Tests the database toolkit functionality including:
- Toolkit initialization
- SQL generation and execution
- Error handling and retry logic
- Data catalog filtering
- Query planning
- SQL sanitization and validation
"""

import sys
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

import pandas as pd
import pytest

from minds.agent.database_toolkit import DatabaseToolkit
from minds.agent.exceptions import QueryGenerationError
from minds.model.data_catalog import DataCatalog
from minds.model.database_agent import QueryGenerationResult, QueryGenerationResultRetry, QueryPlanResult
from minds.model.datasource import Datasource
from minds.model.mind import Mind
from minds.model.mind_datasource import MindDatasource

# Mock langfuse before importing any modules that use it
if "langfuse" not in sys.modules:
    mock_langfuse = Mock()
    mock_langfuse.observe = lambda f=None, **_: (lambda *a, **k: f(*a, **k)) if f else (lambda x: x)
    sys.modules["langfuse"] = mock_langfuse


class TestDatabaseToolkit:
    """Test cases for DatabaseToolkit class."""

    @pytest.fixture
    def mock_mind(self):
        """Create a mock Mind instance for testing."""
        mind = Mock(spec=Mind)
        mind.name = "test-mind"
        mind.provider = "openai"
        mind.model_name = "gpt-3.5-turbo"
        mind.user_id = "test-user"
        mind.tenant_id = "test-tenant"
        return mind

    @pytest.fixture
    def mock_mindsdb_client(self):
        """Create a mock MindsDB client for testing."""
        client = Mock()
        client.query = Mock()
        return client

    @pytest.fixture
    def mock_datasource(self):
        """Create a mock Datasource instance for testing."""
        datasource = Mock(spec=Datasource)
        datasource.id = UUID("12345678-1234-5678-1234-567812345678")
        # The DatabaseToolkit uses the nested mind_datasource.datasource.name
        # as the canonical namespace. Use the same value as integration_name in
        # the tests to keep fixtures consistent.
        datasource.name = "test-integration"
        datasource.engine = "postgres"
        datasource.created_at = datetime.now()
        datasource.modified_at = datetime.now()
        datasource.deleted_at = None
        datasource.tenant_id = "test-tenant"
        return datasource

    @pytest.fixture
    def mock_mind_datasource(self, mock_datasource):
        """Create a mock MindDatasource instance for testing."""
        mind_datasource = Mock(spec=MindDatasource)
        mind_datasource.id = UUID("87654321-4321-8765-4321-876543218765")
        mind_datasource.mind_id = UUID("11111111-2222-3333-4444-555555555555")
        mind_datasource.datasource_id = mock_datasource.id
        mind_datasource.datasource = mock_datasource
        mind_datasource.created_at = datetime.now()
        mind_datasource.modified_at = datetime.now()
        mind_datasource.deleted_at = None
        mind_datasource.tenant_id = "test-tenant"
        mind_datasource.status = "active"
        # Provide a default list of mind_datasource_tables compatible with
        # the DatabaseToolkit expectations. Each entry should have a `.table`
        # attribute with a `.name`.
        mock_table = Mock()
        mock_table.table = Mock()
        mock_table.table.name = "users"
        mind_datasource.mind_datasource_tables = [mock_table]
        return mind_datasource

    @pytest.fixture
    def mock_data_catalog(self, mock_mind_datasource):
        """Create a mock DataCatalog instance for testing."""

        class DummyCatalog:
            def __init__(self, mind_datasource):
                self.mind_datasource = mind_datasource
                self.integration_name = "test-integration"
                self.datasource_name = "test-datasource"
                self.tables = {"users": {"id": "int", "name": "string"}}

            def to_context_str(self):
                return "Mock catalog context"

        return DummyCatalog(mock_mind_datasource)

    @pytest.fixture
    def database_toolkit(self, mock_mind, mock_mindsdb_client):
        """Create a DatabaseToolkit instance for testing."""
        return DatabaseToolkit(mind=mock_mind, mindsdb_client=mock_mindsdb_client)

    @pytest.fixture
    def mock_streamer(self):
        """Mock MessageStreamer for tests that need it."""
        m = Mock()
        m.push = AsyncMock()
        return m

    def test_database_toolkit_initialization(self, mock_mind, mock_mindsdb_client):
        """Test DatabaseToolkit initialization."""
        toolkit = DatabaseToolkit(mind=mock_mind, mindsdb_client=mock_mindsdb_client)

        assert toolkit.mind == mock_mind
        assert toolkit.mindsdb_client == mock_mindsdb_client

    @pytest.mark.asyncio
    async def test_generate_and_execute_sql_success(self, database_toolkit, mock_streamer):
        """Test successful SQL generation and execution."""
        conversation_context = "Show me all users"
        expected_result = "Query executed successfully"

        with patch.object(database_toolkit, "_generate_and_execute_with_retry") as mock_retry:
            mock_retry.return_value = expected_result

            result = await database_toolkit.generate_and_execute_sql(conversation_context, mock_streamer)

            assert result == expected_result
            mock_retry.assert_called_once_with(conversation_context, mock_streamer)

    @pytest.mark.asyncio
    async def test_generate_and_execute_with_retry_success_first_attempt(self, database_toolkit, mock_streamer):
        """Test successful SQL generation on first attempt."""
        conversation_context = "Show me all users"
        expected_sql = "SELECT * FROM users"
        expected_result = "Query executed successfully"

        with (
            patch.object(database_toolkit, "generate_sql") as mock_generate,
            patch.object(database_toolkit, "_sanitize_and_validate_sql_mindsdb") as mock_sanitize,
            patch.object(database_toolkit, "execute_sql") as mock_execute,
        ):
            mock_generate.return_value = expected_sql
            mock_sanitize.return_value = expected_sql
            mock_execute.return_value = expected_result

            result = await database_toolkit._generate_and_execute_with_retry(conversation_context, mock_streamer)

            assert result == expected_result
            mock_generate.assert_called_once_with(conversation_context)
            mock_sanitize.assert_called_once_with(expected_sql)
            mock_execute.assert_called_once_with(expected_sql, raise_on_error=True)
            assert mock_streamer.push.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_and_execute_with_retry_success_after_retry(self, database_toolkit, mock_streamer):
        """Test successful SQL generation after retry."""
        conversation_context = "Show me all users"
        failed_sql = "SELECT * FROM non_existent_table"
        corrected_sql = "SELECT * FROM users"
        expected_result = "Query executed successfully"

        with (
            patch.object(database_toolkit, "generate_sql") as mock_generate,
            patch.object(database_toolkit, "_generate_corrected_sql") as mock_correct,
            patch.object(database_toolkit, "_sanitize_and_validate_sql_mindsdb") as mock_sanitize,
            patch.object(database_toolkit, "execute_sql") as mock_execute,
        ):
            # First attempt: generate_sql returns failed_sql, sanitize fails, execute fails
            # Second attempt: _generate_corrected_sql returns corrected_sql, sanitize succeeds
            mock_generate.side_effect = [failed_sql, corrected_sql]  # First and final attempts
            mock_correct.return_value = corrected_sql  # Middle attempts
            mock_sanitize.side_effect = [QueryGenerationError("Table not found"), None, None]
            mock_execute.side_effect = [Exception("Table not found"), expected_result]

            result = await database_toolkit._generate_and_execute_with_retry(conversation_context, mock_streamer)

            assert result == expected_result
            assert mock_generate.call_count == 1  # Only first attempt
            assert mock_correct.call_count == 2  # Second and third attempts
            assert mock_sanitize.call_count == 3  # First, second, and third attempts
            assert mock_execute.call_count == 2  # Failed and successful attempts

    @pytest.mark.asyncio
    async def test_generate_and_execute_with_retry_max_retries_exceeded(self, database_toolkit, mock_streamer):
        """Test SQL generation when max retries are exceeded."""
        conversation_context = "Show me all users"
        error_message = "Persistent error"

        with (
            patch.object(database_toolkit, "generate_sql") as mock_generate,
            patch.object(database_toolkit, "_generate_corrected_sql") as mock_correct,
            patch.object(database_toolkit, "_sanitize_and_validate_sql_mindsdb") as mock_sanitize,
            patch.object(database_toolkit, "execute_sql") as mock_execute,
        ):
            # All attempts fail
            mock_generate.side_effect = Exception(error_message)
            mock_correct.side_effect = Exception(error_message)
            mock_sanitize.side_effect = QueryGenerationError(error_message)
            mock_execute.side_effect = Exception(error_message)

            result = await database_toolkit._generate_and_execute_with_retry(conversation_context, mock_streamer)

            expected_error = (
                f"Sorry, I'm having an issue querying the data I need. Tried 4 times. Final error: {error_message}"
            )
            assert result == expected_error

    def test_sanitize_and_validate_sql_mindsdb_valid_sql(self, database_toolkit):
        """Test SQL sanitization with valid SQL."""
        sql = "SELECT * FROM users WHERE id = 1"
        result = database_toolkit._sanitize_and_validate_sql_mindsdb(sql)
        assert result == sql

    def test_sanitize_and_validate_sql_mindsdb_empty_sql(self, database_toolkit):
        """Test SQL sanitization with empty SQL."""
        with pytest.raises(QueryGenerationError, match="Empty SQL generated"):
            database_toolkit._sanitize_and_validate_sql_mindsdb("")

    def test_sanitize_and_validate_sql_mindsdb_none_sql(self, database_toolkit):
        """Test SQL sanitization with None SQL."""
        with pytest.raises(QueryGenerationError, match="Empty SQL generated"):
            database_toolkit._sanitize_and_validate_sql_mindsdb(None)

    def test_sanitize_and_validate_sql_mindsdb_error_string(self, database_toolkit):
        """Test SQL sanitization with error string."""
        with pytest.raises(QueryGenerationError, match="LLM returned an error string instead of SQL"):
            database_toolkit._sanitize_and_validate_sql_mindsdb("Error: Something went wrong")

    def test_sanitize_and_validate_sql_mindsdb_double_quotes(self, database_toolkit):
        """Test SQL sanitization with double quotes."""
        with pytest.raises(QueryGenerationError, match="Double quotes are not allowed outside of string literals"):
            database_toolkit._sanitize_and_validate_sql_mindsdb('SELECT "column" FROM table')

    def test_sanitize_and_validate_sql_mindsdb_dangerous_statements(self, database_toolkit):
        """Test SQL sanitization with dangerous statements."""
        dangerous_queries = [
            "INSERT INTO users VALUES (1, 'test')",
            "UPDATE users SET name = 'test'",
            "DELETE FROM users",
            "CREATE TABLE test (id INT)",
            "DROP TABLE users",
            "TRUNCATE TABLE users",
            "ALTER TABLE users ADD COLUMN test VARCHAR(255)",
        ]

        for query in dangerous_queries:
            with pytest.raises(QueryGenerationError, match="Only SELECT queries are allowed"):
                database_toolkit._sanitize_and_validate_sql_mindsdb(query)

    def test_sanitize_and_validate_sql_mindsdb_invalid_syntax(self, database_toolkit):
        """Test SQL sanitization with invalid syntax."""
        with patch("minds.agent.database_toolkit.parse_sql") as mock_parse:
            mock_parse.side_effect = Exception("Syntax error")

            with pytest.raises(QueryGenerationError, match="Unparseable SQL"):
                database_toolkit._sanitize_and_validate_sql_mindsdb("INVALID SQL")

    def test_sanitize_and_validate_sql_mindsdb_non_select_statement(self, database_toolkit):
        """Test SQL sanitization with non-SELECT statement."""
        with patch("minds.agent.database_toolkit.parse_sql") as mock_parse:
            # Create a mock AST that is not a Select statement
            mock_ast = Mock()
            mock_ast.__class__ = type("ShowStatement", (), {})  # Not a Select class
            mock_parse.return_value = mock_ast

            with pytest.raises(QueryGenerationError, match="Only SELECT queries are allowed"):
                database_toolkit._sanitize_and_validate_sql_mindsdb("SHOW TABLES")

    def test_sanitize_and_validate_sql_mindsdb_removes_trailing_semicolon(self, database_toolkit):
        """Test SQL sanitization removes trailing semicolon."""
        sql = "SELECT * FROM users;"
        result = database_toolkit._sanitize_and_validate_sql_mindsdb(sql)
        assert result == "SELECT * FROM users"

    def test_filter_catalogs_with_plan_no_plan(self, database_toolkit, mock_data_catalog):
        """Test catalog filtering with no plan."""
        catalogs = [mock_data_catalog]
        result = database_toolkit._filter_catalogs_with_plan(catalogs, None)
        assert result == catalogs

    def test_filter_catalogs_with_plan_empty_plan(self, database_toolkit, mock_data_catalog):
        """Test catalog filtering with empty plan."""
        catalogs = [mock_data_catalog]
        plan = QueryPlanResult()
        result = database_toolkit._filter_catalogs_with_plan(catalogs, plan)
        assert result == catalogs

    def test_filter_catalogs_with_plan_selected_datasources(self, database_toolkit, mock_data_catalog):
        """Test catalog filtering with selected datasources."""
        catalogs = [mock_data_catalog]
        plan = QueryPlanResult(selected_datasources=["test-integration"])
        result = database_toolkit._filter_catalogs_with_plan(catalogs, plan)
        assert result == catalogs

    def test_filter_catalogs_with_plan_excluded_datasources(self, database_toolkit, mock_data_catalog):
        """Test catalog filtering with excluded datasources."""
        catalogs = [mock_data_catalog]
        plan = QueryPlanResult(selected_datasources=["other-integration"])
        result = database_toolkit._filter_catalogs_with_plan(catalogs, plan)
        assert result == []

    def test_filter_catalogs_with_plan_selected_tables(self, database_toolkit, mock_data_catalog):
        """Test catalog filtering with selected tables."""
        catalogs = [mock_data_catalog]
        plan = QueryPlanResult(selected_tables=["test-integration.users"])
        result = database_toolkit._filter_catalogs_with_plan(catalogs, plan)
        assert result == catalogs

    def test_filter_catalogs_with_plan_excluded_tables(self, database_toolkit, mock_data_catalog):
        """Test catalog filtering with excluded tables."""
        catalogs = [mock_data_catalog]
        plan = QueryPlanResult(selected_tables=["test-integration.other_table"])
        result = database_toolkit._filter_catalogs_with_plan(catalogs, plan)
        assert result == []

    @pytest.mark.asyncio
    async def test_plan_selection_success(self, database_toolkit, mock_data_catalog):
        """Test successful query planning."""
        conversation_context = "Show me all users"
        catalogs = [mock_data_catalog]
        expected_plan = QueryPlanResult(
            preferred_engine="postgres",
            selected_datasources=["test-integration"],
            selected_tables=["test-integration.users"],
            rationale="Users table contains the requested data",
        )

        with (
            patch("minds.agent.database_toolkit.get_llm_config") as mock_get_llm,
            patch("minds.agent.database_toolkit.PydanticAIAgent") as mock_agent_class,
        ):
            mock_llm = Mock()
            mock_get_llm.return_value = mock_llm

            mock_agent = Mock()
            mock_result = Mock()
            mock_result.output = expected_plan
            mock_agent.run = AsyncMock(return_value=mock_result)
            mock_agent_class.return_value = mock_agent

            result = await database_toolkit._plan_selection(conversation_context, catalogs)

            assert result == expected_plan
            mock_get_llm.assert_called_once_with(database_toolkit.mind.provider, database_toolkit.mind.model_name)

    @pytest.mark.asyncio
    async def test_plan_selection_failure(self, database_toolkit, mock_data_catalog):
        """Test query planning failure."""
        conversation_context = "Show me all users"
        catalogs = [mock_data_catalog]

        with (
            patch("minds.agent.database_toolkit.get_llm_config") as mock_get_llm,
            patch("minds.agent.database_toolkit.PydanticAIAgent") as mock_agent_class,
        ):
            mock_llm = Mock()
            mock_get_llm.return_value = mock_llm

            mock_agent = Mock()
            mock_agent.run = AsyncMock(side_effect=Exception("Planning failed"))
            mock_agent_class.return_value = mock_agent

            result = await database_toolkit._plan_selection(conversation_context, catalogs)

            assert result is None

    @pytest.mark.asyncio
    async def test_generate_corrected_sql_success(self, database_toolkit, mock_data_catalog):
        """Test successful SQL correction."""
        conversation_context = "Show me all users"
        failed_query = "SELECT * FROM non_existent_table"
        error_message = "Table not found"
        expected_corrected_sql = "SELECT * FROM users"

        with (
            patch("minds.agent.database_toolkit.data_catalog_cache") as mock_cache,
            patch("minds.agent.database_toolkit.get_llm_config") as mock_get_llm,
            patch("minds.agent.database_toolkit.PydanticAIAgent") as mock_agent_class,
            patch.object(database_toolkit, "_plan_selection", return_value=None),
        ):
            mock_cache.load.return_value = [mock_data_catalog]
            mock_llm = Mock()
            mock_get_llm.return_value = mock_llm

            mock_agent = Mock()
            mock_result = Mock()
            mock_result.output = QueryGenerationResultRetry(query=expected_corrected_sql)
            mock_agent.run = AsyncMock(return_value=mock_result)
            mock_agent_class.return_value = mock_agent

            result = await database_toolkit._generate_corrected_sql(conversation_context, failed_query, error_message)

            assert result == expected_corrected_sql

    @pytest.mark.asyncio
    async def test_generate_corrected_sql_no_catalogs(self, database_toolkit):
        """Test SQL correction with no data catalogs."""
        conversation_context = "Show me all users"
        failed_query = "SELECT * FROM non_existent_table"
        error_message = "Table not found"

        with patch("minds.agent.database_toolkit.data_catalog_cache") as mock_cache:
            mock_cache.load.return_value = []

            with pytest.raises(Exception, match="No database context available for retry"):
                await database_toolkit._generate_corrected_sql(conversation_context, failed_query, error_message)

    @pytest.mark.asyncio
    async def test_generate_corrected_sql_llm_error(self, database_toolkit, mock_data_catalog):
        """Test SQL correction with LLM error."""
        conversation_context = "Show me all users"
        failed_query = "SELECT * FROM non_existent_table"
        error_message = "Table not found"

        with (
            patch("minds.agent.database_toolkit.data_catalog_cache") as mock_cache,
            patch("minds.agent.database_toolkit.get_llm_config") as mock_get_llm,
        ):
            mock_cache.load.return_value = [mock_data_catalog]
            mock_get_llm.side_effect = Exception("LLM config error")

            with pytest.raises(Exception, match="LLM config error"):
                await database_toolkit._generate_corrected_sql(conversation_context, failed_query, error_message)

    @pytest.mark.asyncio
    async def test_generate_sql_success(self, database_toolkit, mock_data_catalog):
        """Test successful SQL generation."""
        conversation_context = "Show me all users"
        expected_sql = "SELECT * FROM users"

        with (
            patch("minds.agent.database_toolkit.data_catalog_cache") as mock_cache,
            patch("minds.agent.database_toolkit.get_llm_config") as mock_get_llm,
            patch("minds.agent.database_toolkit.PydanticAIAgent") as mock_agent_class,
            patch("minds.agent.database_toolkit.get_prompt_template_for_engines") as mock_get_template,
            patch.object(database_toolkit, "_plan_selection", return_value=None),
        ):
            mock_cache.load.return_value = [mock_data_catalog]
            mock_llm = Mock()
            mock_get_llm.return_value = mock_llm
            mock_get_template.return_value = "Mock prompt template"

            mock_agent = Mock()
            mock_result = Mock()
            mock_result.output = QueryGenerationResult(query=expected_sql)
            mock_agent.run = AsyncMock(return_value=mock_result)
            mock_agent_class.return_value = mock_agent

            result = await database_toolkit.generate_sql(conversation_context)

            assert result == expected_sql

    @pytest.mark.asyncio
    async def test_generate_sql_no_catalogs(self, database_toolkit):
        """Test SQL generation with no data catalogs."""
        conversation_context = "Show me all users"

        with patch("minds.agent.database_toolkit.data_catalog_cache") as mock_cache:
            mock_cache.load.return_value = []

            with pytest.raises(QueryGenerationError, match="No database context available"):
                await database_toolkit.generate_sql(conversation_context)

    @pytest.mark.asyncio
    async def test_generate_sql_with_error(self, database_toolkit, mock_data_catalog):
        """Test SQL generation with error in result."""
        conversation_context = "Show me all users"
        error_message = "Failed to generate SQL"

        with (
            patch("minds.agent.database_toolkit.data_catalog_cache") as mock_cache,
            patch("minds.agent.database_toolkit.get_llm_config") as mock_get_llm,
            patch("minds.agent.database_toolkit.PydanticAIAgent") as mock_agent_class,
            patch("minds.agent.database_toolkit.get_prompt_template_for_engines") as mock_get_template,
            patch.object(database_toolkit, "_plan_selection", return_value=None),
        ):
            mock_cache.load.return_value = [mock_data_catalog]
            mock_llm = Mock()
            mock_get_llm.return_value = mock_llm
            mock_get_template.return_value = "Mock prompt template"

            mock_agent = Mock()
            mock_result = Mock()
            mock_result.output = QueryGenerationResult(error=error_message)
            mock_agent.run = AsyncMock(return_value=mock_result)
            mock_agent_class.return_value = mock_agent

            with pytest.raises(QueryGenerationError, match=error_message):
                await database_toolkit.generate_sql(conversation_context)

    @pytest.mark.asyncio
    async def test_execute_sql_success(self, database_toolkit):
        """Test successful SQL execution."""
        query = "SELECT * FROM users"
        mock_df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})

        mock_query_result = Mock()
        mock_query_result.fetch.return_value = mock_df
        database_toolkit.mindsdb_client.query.return_value = mock_query_result

        result = await database_toolkit.execute_sql(query)

        assert "Query executed: SELECT * FROM users" in result
        assert "Results: 2 rows x 2 columns" in result
        assert "Alice" in result
        assert "Bob" in result
        database_toolkit.mindsdb_client.query.assert_called_once_with(query)

    @pytest.mark.asyncio
    async def test_execute_sql_empty_results(self, database_toolkit):
        """Test SQL execution with empty results."""
        query = "SELECT * FROM users WHERE id = 999"
        mock_df = pd.DataFrame()

        mock_query_result = Mock()
        mock_query_result.fetch.return_value = mock_df
        database_toolkit.mindsdb_client.query.return_value = mock_query_result

        result = await database_toolkit.execute_sql(query)

        assert "Query executed successfully: SELECT * FROM users WHERE id = 999" in result
        assert "No results returned." in result

    @pytest.mark.asyncio
    async def test_execute_sql_with_error(self, database_toolkit):
        """Test SQL execution with error."""
        query = "SELECT * FROM non_existent_table"
        error_message = "Table not found"

        database_toolkit.mindsdb_client.query.side_effect = Exception(error_message)

        result = await database_toolkit.execute_sql(query)

        assert f"Error executing query: {error_message}" in result

    @pytest.mark.asyncio
    async def test_execute_sql_with_error_raise_on_error(self, database_toolkit):
        """Test SQL execution with error and raise_on_error=True."""
        query = "SELECT * FROM non_existent_table"
        error_message = "Table not found"

        database_toolkit.mindsdb_client.query.side_effect = Exception(error_message)

        with pytest.raises(Exception, match=error_message):
            await database_toolkit.execute_sql(query, raise_on_error=True)

    @pytest.mark.asyncio
    async def test_execute_sql_large_results_truncation(self, database_toolkit):
        """Test SQL execution with large results that get truncated."""
        query = "SELECT * FROM large_table"
        # Create a large DataFrame
        large_data = {"id": list(range(1000)), "name": [f"User{i}" for i in range(1000)]}
        mock_df = pd.DataFrame(large_data)

        mock_query_result = Mock()
        mock_query_result.fetch.return_value = mock_df
        database_toolkit.mindsdb_client.query.return_value = mock_query_result

        result = await database_toolkit.execute_sql(query)

        assert "Query executed: SELECT * FROM large_table" in result
        assert "Results: 1000 rows x 2 columns" in result
        # Should show truncation note
        assert "Showing" in result and "of 1000 rows" in result

    def test_sanitize_and_validate_sql_mindsdb_with_string_literals(self, database_toolkit):
        """Test SQL sanitization with string literals containing double quotes."""
        sql = "SELECT name FROM users WHERE description = 'He said \"Hello\"'"
        result = database_toolkit._sanitize_and_validate_sql_mindsdb(sql)
        assert result == sql

    def test_sanitize_and_validate_sql_mindsdb_with_backticks(self, database_toolkit):
        """Test SQL sanitization with backticks for column names."""
        sql = "SELECT `user_id`, `full name` FROM users"
        result = database_toolkit._sanitize_and_validate_sql_mindsdb(sql)
        assert result == sql

    def test_sanitize_and_validate_sql_mindsdb_case_insensitive_dangerous_keywords(self, database_toolkit):
        """Test SQL sanitization with case-insensitive dangerous keywords."""
        dangerous_queries = [
            "insert into users values (1, 'test')",
            "UPDATE users SET name = 'test'",
            "delete from users",
            "create table test (id int)",
            "drop table users",
            "truncate table users",
            "alter table users add column test varchar(255)",
        ]

        for query in dangerous_queries:
            with pytest.raises(QueryGenerationError, match="Only SELECT queries are allowed"):
                database_toolkit._sanitize_and_validate_sql_mindsdb(query)

    @pytest.mark.asyncio
    async def test_generate_sql_with_planning_error(self, database_toolkit, mock_data_catalog):
        """Test SQL generation when planning returns an error."""
        conversation_context = "Show me all users"
        expected_sql = "SELECT * FROM users"

        with (
            patch("minds.agent.database_toolkit.data_catalog_cache") as mock_cache,
            patch.object(database_toolkit, "_plan_selection") as mock_plan,
            patch("minds.agent.database_toolkit.get_llm_config") as mock_get_llm,
            patch("minds.agent.database_toolkit.PydanticAIAgent") as mock_agent_class,
            patch("minds.agent.database_toolkit.get_prompt_template_for_engines") as mock_get_template,
        ):
            mock_cache.load.return_value = [mock_data_catalog]
            # Planning returns error
            error_plan = QueryPlanResult(error="Planning failed")
            mock_plan.return_value = error_plan

            mock_llm = Mock()
            mock_get_llm.return_value = mock_llm
            mock_get_template.return_value = "Mock prompt template"

            mock_agent = Mock()
            mock_result = Mock()
            mock_result.output = QueryGenerationResult(query=expected_sql)
            mock_agent.run = AsyncMock(return_value=mock_result)
            mock_agent_class.return_value = mock_agent

            result = await database_toolkit.generate_sql(conversation_context)

            assert result == expected_sql
            # Should proceed with full catalogs when planning fails
            mock_plan.assert_called_once()

    def test_filter_catalogs_with_plan_namespace_fallback(self, database_toolkit):
        """Test catalog filtering with namespace fallback to datasource_name."""
        # Create a catalog without integration_name but with datasource_name
        catalog = Mock(spec=DataCatalog)
        catalog.integration_name = None
        catalog.datasource_name = "fallback-datasource"
        # Provide nested mind_datasource.datasource.name to be used by toolkit
        md = Mock()
        ds = Mock()
        ds.name = "fallback-datasource"
        md.datasource = ds
        md.mind_datasource_tables = []
        catalog.mind_datasource = md
        catalog.tables = {"users": {"id": "int", "name": "string"}}

        catalogs = [catalog]
        plan = QueryPlanResult(selected_datasources=["fallback-datasource"])
        result = database_toolkit._filter_catalogs_with_plan(catalogs, plan)
        assert result == catalogs

    def test_filter_catalogs_with_plan_no_namespace(self, database_toolkit):
        """Test catalog filtering with no namespace available."""
        catalog = Mock(spec=DataCatalog)
        catalog.integration_name = None
        catalog.datasource_name = None
        catalog.tables = {"users": {"id": "int", "name": "string"}}
        # Provide a mind_datasource with a datasource.name = None so the toolkit
        # can safely access nested attributes.
        md = Mock()
        ds = Mock()
        ds.name = None
        md.datasource = ds
        md.mind_datasource_tables = []
        catalog.mind_datasource = md

        catalogs = [catalog]
        plan = QueryPlanResult(selected_datasources=["some-datasource"])
        result = database_toolkit._filter_catalogs_with_plan(catalogs, plan)
        # When namespace is None, the catalog is not filtered out
        assert result == [catalog]

    def test_filter_catalogs_with_plan_table_filtering(self, database_toolkit):
        """Test catalog filtering with table filtering."""
        catalog = Mock(spec=DataCatalog)
        catalog.integration_name = "test-integration"
        # Provide nested mind_datasource with table objects as expected by toolkit
        md = Mock()
        md.datasource = Mock()
        md.datasource.name = "test-integration"
        tbl_users = Mock()
        tbl_users.table = Mock()
        tbl_users.table.name = "users"
        tbl_orders = Mock()
        tbl_orders.table = Mock()
        tbl_orders.table.name = "orders"
        md.mind_datasource_tables = [tbl_users, tbl_orders]
        catalog.mind_datasource = md

        catalogs = [catalog]
        plan = QueryPlanResult(selected_tables=["test-integration.users"])
        result = database_toolkit._filter_catalogs_with_plan(catalogs, plan)

        # Should filter tables to only include users (within nested structure)
        assert len(result) == 1
        assert any(getattr(t.table, "name", None) == "users" for t in result[0].mind_datasource.mind_datasource_tables)
        assert all(getattr(t.table, "name", None) != "orders" for t in result[0].mind_datasource.mind_datasource_tables)

    def test_filter_catalogs_with_plan_no_matching_tables(self, database_toolkit):
        """Test catalog filtering when no tables match the filter."""
        catalog = Mock(spec=DataCatalog)
        catalog.integration_name = "test-integration"
        md = Mock()
        md.datasource = Mock()
        md.datasource.name = "test-integration"
        tbl_users = Mock()
        tbl_users.table = Mock()
        tbl_users.table.name = "users"
        md.mind_datasource_tables = [tbl_users]
        catalog.mind_datasource = md

        catalogs = [catalog]
        plan = QueryPlanResult(selected_tables=["test-integration.nonexistent"])
        result = database_toolkit._filter_catalogs_with_plan(catalogs, plan)

        # Should exclude catalog when no tables match
        assert result == []
