"""
Unit tests for CandidateGenerator helpers (deterministic behavior).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic_ai.exceptions import ModelHTTPError

from minds.agents.candidate_sql_agent.candidate_generator_agent.agent import (
    CandidateGenerator,
    DirectOutput,
    DivideConquerOutput,
    QueryPlanOutput,
)
from minds.agents.candidate_sql_agent.candidate_generator_agent.instructions_templates import (
    DIRECT_SYSTEM_PROMPT_BIGQUERY,
    DIRECT_SYSTEM_PROMPT_MINDSDB,
    DIRECT_SYSTEM_PROMPT_SNOWFLAKE,
    DIVIDE_CONQUER_SYSTEM_PROMPT_BIGQUERY,
    DIVIDE_CONQUER_SYSTEM_PROMPT_MINDSDB,
    DIVIDE_CONQUER_SYSTEM_PROMPT_SNOWFLAKE,
    QUERY_PLAN_SYSTEM_PROMPT_BIGQUERY,
    QUERY_PLAN_SYSTEM_PROMPT_MINDSDB,
    QUERY_PLAN_SYSTEM_PROMPT_SNOWFLAKE,
)
from minds.agents.candidate_sql_agent.linker_agent.agent import LinkedSchema


class TestCandidateGeneratorValidateColumns:
    def test_validate_columns_mindsdb_and_sqlglot_consistency(self):
        gen = CandidateGenerator(mind=Mock(), mindsdb_client=None)
        linked = LinkedSchema(
            tables=["ds.orders"],
            columns={"ds.orders": ["id", "user_id"]},
            joins=[],
        )
        sql = "SELECT o.user_id FROM ds.orders o"
        ok_mindsdb, invalid_mindsdb = gen.validate_columns(sql, linked, use_parser=True)
        ok_sqlglot, invalid_sqlglot = gen.validate_columns(sql, linked, use_parser=False, dialect="snowflake")
        assert ok_mindsdb is True
        assert ok_sqlglot is True
        assert invalid_mindsdb == []
        assert invalid_sqlglot == []

    def test_validate_columns_accepts_valid_qualified_columns_with_alias(self):
        gen = CandidateGenerator(mind=Mock(), mindsdb_client=None)
        linked = LinkedSchema(
            tables=["ds.sch.orders"],
            columns={"ds.sch.orders": ["id", "user_id"]},
            joins=[],
        )
        sql = "SELECT o.user_id FROM ds.sch.orders o"
        ok, invalid = gen.validate_columns(sql, linked, use_parser=True)
        assert ok is True
        assert invalid == []


class TestCandidateGeneratorPromptSelection:
    @pytest.mark.anyio
    async def test_generate_uses_mindsdb_prompt_when_non_native(self):
        gen = CandidateGenerator(mind=Mock(), mindsdb_client=None)
        prompts: list[str] = []

        async def _fake_run(_self, agent, _user_prompt, _model):
            prompts.append(agent.system_prompt)
            if agent.output_type is DivideConquerOutput:
                output = DivideConquerOutput(
                    tables_needed=[],
                    columns_to_select=[],
                    join_conditions=[],
                    where_filters=[],
                    group_by=[],
                    order_by="",
                    limit=None,
                    final_sql="SELECT 1",
                )
            elif agent.output_type is QueryPlanOutput:
                output = QueryPlanOutput(
                    scan_tables=[],
                    filter_conditions=[],
                    join_operations=[],
                    aggregations=[],
                    sort_order="",
                    row_limit=None,
                    final_sql="SELECT 1",
                )
            else:
                output = DirectOutput(query="SELECT 1")
            return SimpleNamespace(output=output)

        with patch.object(CandidateGenerator, "_run_agent_with_retry", new=_fake_run):
            await gen.generate(
                question="q",
                linked_schema=LinkedSchema(tables=[], columns={}, joins=[]),
                schema_context="schema",
                engine=None,
                is_native_query_mode=False,
            )

        assert prompts
        assert all("datasource.table" in p for p in prompts)

    @pytest.mark.anyio
    async def test_generate_uses_bigquery_prompt_when_native_bigquery(self):
        gen = CandidateGenerator(mind=Mock(), mindsdb_client=None)
        prompts: list[str] = []

        async def _fake_run(_self, agent, _user_prompt, _model):
            prompts.append(agent.system_prompt)
            if agent.output_type is DivideConquerOutput:
                output = DivideConquerOutput(
                    tables_needed=[],
                    columns_to_select=[],
                    join_conditions=[],
                    where_filters=[],
                    group_by=[],
                    order_by="",
                    limit=None,
                    final_sql="SELECT 1",
                )
            elif agent.output_type is QueryPlanOutput:
                output = QueryPlanOutput(
                    scan_tables=[],
                    filter_conditions=[],
                    join_operations=[],
                    aggregations=[],
                    sort_order="",
                    row_limit=None,
                    final_sql="SELECT 1",
                )
            else:
                output = DirectOutput(query="SELECT 1")
            return SimpleNamespace(output=output)

        with patch.object(CandidateGenerator, "_run_agent_with_retry", new=_fake_run):
            await gen.generate(
                question="q",
                linked_schema=LinkedSchema(tables=[], columns={}, joins=[]),
                schema_context="schema",
                engine="bigquery",
                is_native_query_mode=True,
            )

        assert prompts
        assert all("BigQuery Standard SQL" in p for p in prompts)

    @pytest.mark.anyio
    async def test_generate_uses_snowflake_prompt_when_native_snowflake(self):
        gen = CandidateGenerator(mind=Mock(), mindsdb_client=None)
        prompts: list[str] = []

        async def _fake_run(_self, agent, _user_prompt, _model):
            prompts.append(agent.system_prompt)
            if agent.output_type is DivideConquerOutput:
                output = DivideConquerOutput(
                    tables_needed=[],
                    columns_to_select=[],
                    join_conditions=[],
                    where_filters=[],
                    group_by=[],
                    order_by="",
                    limit=None,
                    final_sql="SELECT 1",
                )
            elif agent.output_type is QueryPlanOutput:
                output = QueryPlanOutput(
                    scan_tables=[],
                    filter_conditions=[],
                    join_operations=[],
                    aggregations=[],
                    sort_order="",
                    row_limit=None,
                    final_sql="SELECT 1",
                )
            else:
                output = DirectOutput(query="SELECT 1")
            return SimpleNamespace(output=output)

        with patch.object(CandidateGenerator, "_run_agent_with_retry", new=_fake_run):
            await gen.generate(
                question="q",
                linked_schema=LinkedSchema(tables=[], columns={}, joins=[]),
                schema_context="schema",
                engine="snowflake",
                is_native_query_mode=True,
            )

        assert prompts
        assert any("SNOWFLAKE" in p for p in prompts)

    @pytest.mark.anyio
    async def test_generate_falls_back_to_mindsdb_prompt_when_native_unknown_engine(self):
        gen = CandidateGenerator(mind=Mock(), mindsdb_client=None)
        prompts: list[str] = []

        async def _fake_run(_self, agent, _user_prompt, _model):
            prompts.append(agent.system_prompt)
            if agent.output_type is DivideConquerOutput:
                output = DivideConquerOutput(
                    tables_needed=[],
                    columns_to_select=[],
                    join_conditions=[],
                    where_filters=[],
                    group_by=[],
                    order_by="",
                    limit=None,
                    final_sql="SELECT 1",
                )
            elif agent.output_type is QueryPlanOutput:
                output = QueryPlanOutput(
                    scan_tables=[],
                    filter_conditions=[],
                    join_operations=[],
                    aggregations=[],
                    sort_order="",
                    row_limit=None,
                    final_sql="SELECT 1",
                )
            else:
                output = DirectOutput(query="SELECT 1")
            return SimpleNamespace(output=output)

        with patch.object(CandidateGenerator, "_run_agent_with_retry", new=_fake_run):
            await gen.generate(
                question="q",
                linked_schema=LinkedSchema(tables=[], columns={}, joins=[]),
                schema_context="schema",
                engine="unknown_engine",
                is_native_query_mode=True,
            )

        assert prompts
        assert all("datasource.table" in p for p in prompts)

    def test_validate_columns_flags_hallucinated_columns(self):
        gen = CandidateGenerator(mind=Mock(), mindsdb_client=None)
        linked = LinkedSchema(
            tables=["ds.orders"],
            columns={"ds.orders": ["id", "user_id"]},
            joins=[],
        )
        sql = "SELECT o.not_a_col FROM ds.orders o"
        ok, invalid = gen.validate_columns(sql, linked, use_parser=True)
        assert ok is False
        assert any("not_a_col" in x for x in invalid)

    def test_validate_columns_flags_wrong_table_qualified_column(self):
        gen = CandidateGenerator(mind=Mock(), mindsdb_client=None)
        linked = LinkedSchema(
            tables=["ds.orders", "ds.users"],
            columns={"ds.orders": ["id", "user_id"], "ds.users": ["id", "email"]},
            joins=[],
        )
        sql = "SELECT u.user_id FROM ds.users u"
        ok, invalid = gen.validate_columns(sql, linked, use_parser=True)
        assert ok is False
        assert any("user_id" in x for x in invalid)

    def test_validate_columns_ignores_datasource_and_schema_qualifiers(self):
        gen = CandidateGenerator(mind=Mock(), mindsdb_client=None)
        linked = LinkedSchema(
            tables=["ds.sch.orders"],
            columns={"ds.sch.orders": ["id", "user_id"]},
            joins=[],
        )
        # `ds.user_id` is not a real column ref; it should be ignored as a qualifier and not flagged.
        sql = "SELECT ds.user_id FROM ds.sch.orders"
        ok, invalid = gen.validate_columns(sql, linked, use_parser=True)
        assert ok is True
        assert invalid == []

    def test_validate_columns_sqlglot_flags_hallucination(self):
        gen = CandidateGenerator(mind=Mock(), mindsdb_client=None)
        linked = LinkedSchema(
            tables=["ds.orders"],
            columns={"ds.orders": ["id", "user_id"]},
            joins=[],
        )
        sql = "SELECT o.not_a_col FROM ds.orders o"
        ok, invalid = gen.validate_columns(sql, linked, dialect="snowflake")
        assert ok is False
        assert any("not_a_col" in x for x in invalid)

    def test_validate_columns_sqlglot_bigquery_backticks(self):
        gen = CandidateGenerator(mind=Mock(), mindsdb_client=None)
        linked = LinkedSchema(
            tables=["ds.sch.orders"],
            columns={"ds.sch.orders": ["id", "user_id"]},
            joins=[],
        )
        sql = "SELECT o.user_id FROM `ds.sch.orders` AS o"
        ok, invalid = gen.validate_columns(sql, linked, dialect="bigquery")
        assert ok is True
        assert invalid == []


class TestCandidateGeneratorPreflightScore:
    def test_preflight_score_sanitize_failure_returns_zero(self):
        gen = CandidateGenerator(mind=Mock(), mindsdb_client=Mock())

        def sanitize(_sql: str) -> str:
            raise ValueError("bad sql")

        score, sanitized, error, exec_result = gen.preflight_score("SELECT 1", sanitize_fn=sanitize)
        assert score == 0
        assert sanitized == "SELECT 1"
        assert error == "bad sql"
        assert exec_result is None

    def test_preflight_score_exec_success_returns_one(self):
        client = Mock()
        exec_result = Mock()
        exec_result.fetch.return_value = Mock()
        client.query.return_value = exec_result

        gen = CandidateGenerator(mind=Mock(), mindsdb_client=client)
        score, sanitized, error, exec_df = gen.preflight_score("SELECT 1")
        assert score == 1
        assert sanitized == "SELECT 1"
        assert error == ""
        assert exec_df is not None

    def test_preflight_score_exec_failure_returns_zero(self):
        client = Mock()
        exec_result = Mock()
        exec_result.fetch.side_effect = RuntimeError("exec failed")
        client.query.return_value = exec_result

        gen = CandidateGenerator(mind=Mock(), mindsdb_client=client)
        score, sanitized, error, exec_df = gen.preflight_score("SELECT 1")
        assert score == 0
        assert sanitized == "SELECT 1"
        assert "exec failed" in error
        assert exec_df is None

    def test_preflight_score_native_exec_success_returns_one(self):
        client = Mock()
        exec_result = Mock()
        exec_result.fetch.return_value = Mock()
        client.query.return_value = exec_result

        gen = CandidateGenerator(mind=Mock(), mindsdb_client=client)
        sql = "SELECT * FROM snowflake_ds (SELECT 1)"
        score, sanitized, error, exec_df = gen.preflight_score(
            sql,
            is_native_query_mode=True,
            engine="snowflake",
        )
        assert score == 1
        assert sanitized == sql
        assert error == ""
        assert exec_df is not None
        client.query.assert_called_once_with(sql)


class TestCandidateGeneratorAsyncPaths:
    @pytest.mark.anyio
    async def test_run_agent_with_retry_retries_on_500(self):
        gen = CandidateGenerator(mind=Mock(), mindsdb_client=Mock())
        agent = Mock()
        agent.run = AsyncMock(
            side_effect=[
                ModelHTTPError(status_code=500, model_name="x"),
                Mock(output=Mock()),
            ]
        )

        with (
            patch("minds.agents.candidate_sql_agent.candidate_generator_agent.agent.asyncio.sleep", new=AsyncMock()),
            patch(
                "minds.agents.candidate_sql_agent.candidate_generator_agent.agent.agent_settings.max_candidate_retries",
                2,
            ),
        ):
            res = await gen._run_agent_with_retry(agent, user_prompt="p", model=Mock())

        assert agent.run.await_count == 2
        assert res.output is not None

    @pytest.mark.anyio
    async def test_run_agent_with_retry_raises_on_non_500(self):
        gen = CandidateGenerator(mind=Mock(), mindsdb_client=Mock())
        agent = Mock()
        agent.run = AsyncMock(side_effect=ModelHTTPError(status_code=429, model_name="x"))

        with (
            patch(
                "minds.agents.candidate_sql_agent.candidate_generator_agent.agent.agent_settings.max_candidate_retries",
                2,
            ),
            pytest.raises(ModelHTTPError),
        ):
            await gen._run_agent_with_retry(agent, user_prompt="p", model=Mock())

    @pytest.mark.anyio
    async def test_generate_filters_failed_strategies(self):
        gen = CandidateGenerator(mind=Mock(), mindsdb_client=None)

        ok = Mock()
        bad = RuntimeError("boom")

        with (
            patch.object(gen, "_divide_and_conquer", new=AsyncMock(return_value=ok)),
            patch.object(gen, "_query_plan_cot", new=AsyncMock(side_effect=bad)),
            patch.object(gen, "_direct_generation", new=AsyncMock(return_value=ok)),
        ):
            res = await gen.generate(
                question="q", linked_schema=LinkedSchema(tables=[], columns={}, joins=[]), schema_context="ctx"
            )

        assert len(res) == 2

    @pytest.mark.anyio
    async def test_generate_raises_when_all_strategies_fail(self):
        gen = CandidateGenerator(mind=Mock(), mindsdb_client=None)

        with (
            patch.object(gen, "_divide_and_conquer", new=AsyncMock(side_effect=RuntimeError("a"))),
            patch.object(gen, "_query_plan_cot", new=AsyncMock(side_effect=RuntimeError("b"))),
            patch.object(gen, "_direct_generation", new=AsyncMock(side_effect=RuntimeError("c"))),
            pytest.raises(RuntimeError, match="generate any SQL candidates"),
        ):
            await gen.generate(
                question="q", linked_schema=LinkedSchema(tables=[], columns={}, joins=[]), schema_context="ctx"
            )

    @pytest.mark.anyio
    async def test_generate_with_execution_fixes_on_failure(self):
        gen = CandidateGenerator(mind=Mock(), mindsdb_client=None)
        linked = LinkedSchema(tables=[], columns={}, joins=[])

        c = Mock()
        c.query = "SELECT bad"
        c.strategy = "direct"
        c.executed = False
        c.execution_error = None
        c.execution_result = None

        async def _fix_sql(_sql, _err, *_a, **_k):
            return "SELECT good"

        def _execute_fn(sql: str):
            if "bad" in sql:
                raise RuntimeError("syntax")
            return "ok"

        with (
            patch.object(gen, "generate", new=AsyncMock(return_value=[c])),
            patch.object(gen, "_fix_sql", new=AsyncMock(side_effect=_fix_sql)),
        ):
            res = await gen.generate_with_execution(
                question="q",
                linked_schema=linked,
                schema_context="ctx",
                execute_fn=_execute_fn,
                max_fix_attempts=1,
            )

        assert res[0].executed is True
        assert res[0].execution_error is None
        assert res[0].query == "SELECT good"


class TestInstructionTemplates:
    """Verify that combined prompts contain the right base and dialect content."""

    # ── Each prompt contains its strategy base ────────────────────────────────

    def test_divide_conquer_prompts_contain_base(self):
        for prompt in (
            DIVIDE_CONQUER_SYSTEM_PROMPT_SNOWFLAKE,
            DIVIDE_CONQUER_SYSTEM_PROMPT_MINDSDB,
            DIVIDE_CONQUER_SYSTEM_PROMPT_BIGQUERY,
        ):
            assert "Break down questions into subproblems" in prompt
            assert "Identify required tables" in prompt
            assert "Identify GROUP BY" in prompt

    def test_query_plan_prompts_contain_base(self):
        for prompt in (
            QUERY_PLAN_SYSTEM_PROMPT_SNOWFLAKE,
            QUERY_PLAN_SYSTEM_PROMPT_MINDSDB,
            QUERY_PLAN_SYSTEM_PROMPT_BIGQUERY,
        ):
            assert "execution plan" in prompt
            assert "SCAN" in prompt
            assert "AGGREGATE" in prompt

    def test_direct_prompts_contain_base(self):
        for prompt in (
            DIRECT_SYSTEM_PROMPT_SNOWFLAKE,
            DIRECT_SYSTEM_PROMPT_MINDSDB,
            DIRECT_SYSTEM_PROMPT_BIGQUERY,
        ):
            assert "Generate SQL to answer the question directly" in prompt
            assert "simple and focused" in prompt

    def test_snowflake_prompts_contain_dialect_rules(self):
        for prompt in (
            DIVIDE_CONQUER_SYSTEM_PROMPT_SNOWFLAKE,
            QUERY_PLAN_SYSTEM_PROMPT_SNOWFLAKE,
            DIRECT_SYSTEM_PROMPT_SNOWFLAKE,
        ):
            assert "native Snowflake SQL" in prompt
            assert "UPPERCASE" in prompt
            assert "double-quoted" in prompt
            assert "datasource wrapper" not in prompt.lower() or "Do NOT wrap" in prompt

    def test_mindsdb_prompts_contain_dialect_rules(self):
        for prompt in (
            DIVIDE_CONQUER_SYSTEM_PROMPT_MINDSDB,
            QUERY_PLAN_SYSTEM_PROMPT_MINDSDB,
            DIRECT_SYSTEM_PROMPT_MINDSDB,
        ):
            assert "MindsDB SQL" in prompt
            assert "datasource.table" in prompt
            assert "native wrappers" in prompt

    def test_bigquery_prompts_contain_dialect_rules(self):
        for prompt in (
            DIVIDE_CONQUER_SYSTEM_PROMPT_BIGQUERY,
            QUERY_PLAN_SYSTEM_PROMPT_BIGQUERY,
            DIRECT_SYSTEM_PROMPT_BIGQUERY,
        ):
            assert "BigQuery Standard SQL" in prompt
            assert "backticks" in prompt
            assert "UNNEST" in prompt

    def test_all_prompts_contain_schema_only_rule(self):
        all_prompts = [
            DIVIDE_CONQUER_SYSTEM_PROMPT_SNOWFLAKE,
            DIVIDE_CONQUER_SYSTEM_PROMPT_MINDSDB,
            DIVIDE_CONQUER_SYSTEM_PROMPT_BIGQUERY,
            QUERY_PLAN_SYSTEM_PROMPT_SNOWFLAKE,
            QUERY_PLAN_SYSTEM_PROMPT_MINDSDB,
            QUERY_PLAN_SYSTEM_PROMPT_BIGQUERY,
            DIRECT_SYSTEM_PROMPT_SNOWFLAKE,
            DIRECT_SYSTEM_PROMPT_MINDSDB,
            DIRECT_SYSTEM_PROMPT_BIGQUERY,
        ]
        for prompt in all_prompts:
            assert "never invent names" in prompt

    def test_snowflake_prompts_do_not_contain_other_dialects(self):
        for prompt in (
            DIVIDE_CONQUER_SYSTEM_PROMPT_SNOWFLAKE,
            QUERY_PLAN_SYSTEM_PROMPT_SNOWFLAKE,
            DIRECT_SYSTEM_PROMPT_SNOWFLAKE,
        ):
            assert "MindsDB SQL" not in prompt
            assert "BigQuery" not in prompt

    def test_mindsdb_prompts_do_not_contain_other_dialects(self):
        for prompt in (
            DIVIDE_CONQUER_SYSTEM_PROMPT_MINDSDB,
            QUERY_PLAN_SYSTEM_PROMPT_MINDSDB,
            DIRECT_SYSTEM_PROMPT_MINDSDB,
        ):
            assert "Snowflake" not in prompt
            assert "BigQuery" not in prompt

    def test_bigquery_prompts_do_not_contain_other_dialects(self):
        for prompt in (
            DIVIDE_CONQUER_SYSTEM_PROMPT_BIGQUERY,
            QUERY_PLAN_SYSTEM_PROMPT_BIGQUERY,
            DIRECT_SYSTEM_PROMPT_BIGQUERY,
        ):
            assert "Snowflake" not in prompt
            assert "MindsDB SQL" not in prompt
