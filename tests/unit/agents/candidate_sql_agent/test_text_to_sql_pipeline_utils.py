"""
Deterministic unit tests for TextToSQLPipeline helper methods.

We avoid hitting LLMs/models by testing pure helpers directly and patching SQL parser.
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from minds.agents.candidate_sql_agent.linker_agent.agent import LinkedSchema
from minds.agents.candidate_sql_agent.text_to_sql_agents.agents import TextToSQLPipeline
from minds.agents.exceptions import QueryGenerationError


class TestTextToSQLPipelineSanitizeValidate:
    def _pipeline(self, native: bool):
        mind = Mock()
        mind.mind_datasources = [Mock(datasource=Mock(name="ds0"))]
        return TextToSQLPipeline(mind=mind, mindsdb_client=Mock(), is_native_query_mode_enabled=native)

    def test_sanitize_empty_sql_raises(self):
        pipeline = self._pipeline(native=False)
        with pytest.raises(QueryGenerationError, match="Empty SQL generated"):
            pipeline._sanitize_and_validate_sql_mindsdb(None)

    def test_sanitize_error_string_raises(self):
        pipeline = self._pipeline(native=False)
        with pytest.raises(QueryGenerationError, match="error string"):
            pipeline._sanitize_and_validate_sql_mindsdb("Error: nope")

    def test_sanitize_disallows_non_select_statements(self):
        pipeline = self._pipeline(native=False)
        with pytest.raises(QueryGenerationError, match="Only SELECT queries are allowed"):
            pipeline._sanitize_and_validate_sql_mindsdb("SHOW TABLES")

    def test_sanitize_disallows_log10(self):
        pipeline = self._pipeline(native=False)
        with pytest.raises(QueryGenerationError, match="LOG10 function is not supported"):
            pipeline._sanitize_and_validate_sql_mindsdb("SELECT LOG10(x) FROM t")

    def test_sanitize_disallows_multiple_statements(self):
        pipeline = self._pipeline(native=False)
        with pytest.raises(QueryGenerationError, match="Multiple SQL statements are not allowed"):
            pipeline._sanitize_and_validate_sql_mindsdb("SELECT 1; SELECT 2;")

    def test_sanitize_strips_trailing_semicolon(self):
        pipeline = self._pipeline(native=False)

        class DummySelect:
            pass

        with (
            patch("minds.agents.candidate_sql_agent.text_to_sql_agents.agents.parse_sql", return_value=DummySelect()),
            patch("minds.agents.candidate_sql_agent.text_to_sql_agents.agents.Select", DummySelect),
        ):
            assert pipeline._sanitize_and_validate_sql_mindsdb("SELECT 1;") == "SELECT 1"

    def test_sanitize_double_quotes_rejected_in_mindsdb_mode(self):
        pipeline = self._pipeline(native=False)
        with pytest.raises(QueryGenerationError, match="Double quotes are not allowed"):
            pipeline._sanitize_and_validate_sql_mindsdb('SELECT "col" FROM t')

    def test_sanitize_double_quotes_allowed_in_native_mode(self):
        pipeline = self._pipeline(native=True)

        class DummySelect:
            pass

        with (
            patch("minds.agents.candidate_sql_agent.text_to_sql_agents.agents.parse_sql", return_value=DummySelect()),
            patch("minds.agents.candidate_sql_agent.text_to_sql_agents.agents.Select", DummySelect),
        ):
            assert (
                pipeline._sanitize_and_validate_sql_mindsdb('SELECT o."user_id" FROM ORDERS o;')
                == 'SELECT o."user_id" FROM ORDERS o'
            )

    def test_sanitize_unparseable_sql_raises(self):
        pipeline = self._pipeline(native=False)
        with (
            patch(
                "minds.agents.candidate_sql_agent.text_to_sql_agents.agents.parse_sql",
                side_effect=Exception("bad"),
            ),
            pytest.raises(QueryGenerationError, match="Unparseable SQL"),
        ):
            pipeline._sanitize_and_validate_sql_mindsdb("SELECT ???")


class TestTextToSQLPipelineNativeHelpers:
    def test_set_native_datasource_from_linked_schema_chooses_most_frequent(self):
        pipeline = TextToSQLPipeline(
            mind=Mock(mind_datasources=[Mock(datasource=Mock(name="fallback"))]),
            mindsdb_client=Mock(),
            is_native_query_mode_enabled=True,
        )
        linked = LinkedSchema(tables=["a.t1", "a.t2", "b.t3"], columns={}, joins=[])
        pipeline._set_native_datasource_from_linked_schema(linked)
        assert pipeline._get_native_datasource_name() == "a"

    def test_strip_datasource_prefix_for_native(self):
        pipeline = TextToSQLPipeline(
            mind=Mock(mind_datasources=[Mock(datasource=Mock(name="fallback"))]),
            mindsdb_client=Mock(),
            is_native_query_mode_enabled=True,
        )
        q = "SELECT * FROM DS.table JOIN ds.other o ON 1=1"
        assert pipeline._strip_datasource_prefix_for_native(q, "ds") == "SELECT * FROM table JOIN other o ON 1=1"


class TestTextToSQLPipelineMarkdownTable:
    def test_generate_markdown_table_empty_df_includes_empty_row(self):
        pipeline = TextToSQLPipeline(mind=Mock(), mindsdb_client=Mock(), is_native_query_mode_enabled=False)
        df = pd.DataFrame()
        md = pipeline._generate_markdown_table(df)
        assert "| result |" in md
        assert "(empty)" in md

    def test_generate_markdown_table_truncates_with_note(self):
        pipeline = TextToSQLPipeline(mind=Mock(), mindsdb_client=Mock(), is_native_query_mode_enabled=False)
        df = pd.DataFrame({"id": list(range(25))})
        md = pipeline._generate_markdown_table(df, override_max_rows=5)
        assert "[Showing 5 of 25 rows]" in md
