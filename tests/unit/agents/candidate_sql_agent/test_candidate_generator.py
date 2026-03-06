"""
Unit tests for CandidateGenerator helpers (deterministic behavior).
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic_ai.exceptions import ModelHTTPError

from minds.agents.candidate_sql_agent.candidate_generator_agent.agent import CandidateGenerator
from minds.agents.candidate_sql_agent.linker_agent.agent import LinkedSchema


class TestCandidateGeneratorValidateColumns:
    def test_validate_columns_accepts_valid_qualified_columns_with_alias(self):
        gen = CandidateGenerator(mind=Mock(), mindsdb_client=None)
        linked = LinkedSchema(
            tables=["ds.sch.orders"],
            columns={"ds.sch.orders": ["id", "user_id"]},
            joins=[],
        )
        sql = "SELECT o.user_id FROM ds.sch.orders o"
        ok, invalid = gen.validate_columns(sql, linked)
        assert ok is True
        assert invalid == []

    def test_validate_columns_flags_hallucinated_columns(self):
        gen = CandidateGenerator(mind=Mock(), mindsdb_client=None)
        linked = LinkedSchema(
            tables=["ds.orders"],
            columns={"ds.orders": ["id", "user_id"]},
            joins=[],
        )
        sql = "SELECT o.not_a_col FROM ds.orders o"
        ok, invalid = gen.validate_columns(sql, linked)
        assert ok is False
        assert any("not_a_col" in x for x in invalid)

    def test_validate_columns_ignores_datasource_and_schema_qualifiers(self):
        gen = CandidateGenerator(mind=Mock(), mindsdb_client=None)
        linked = LinkedSchema(
            tables=["ds.sch.orders"],
            columns={"ds.sch.orders": ["id", "user_id"]},
            joins=[],
        )
        # `ds.user_id` is not a real column ref; it should be ignored as a qualifier and not flagged.
        sql = "SELECT ds.user_id FROM ds.sch.orders"
        ok, invalid = gen.validate_columns(sql, linked)
        assert ok is True
        assert invalid == []


class TestCandidateGeneratorPreflightScore:
    def test_preflight_score_without_client_returns_zero(self):
        gen = CandidateGenerator(mind=Mock(), mindsdb_client=None)
        score, sanitized, error = gen.preflight_score("SELECT 1")
        assert score == 0
        assert sanitized == "SELECT 1"
        assert "No MindsDB client" in error

    def test_preflight_score_sanitize_failure_returns_zero(self):
        gen = CandidateGenerator(mind=Mock(), mindsdb_client=Mock())

        def sanitize(_sql: str) -> str:
            raise ValueError("bad sql")

        score, sanitized, error = gen.preflight_score("SELECT 1", sanitize_fn=sanitize)
        assert score == 0
        assert sanitized == "SELECT 1"
        assert error == "bad sql"

    def test_preflight_score_explain_and_exec_success_returns_two(self):
        client = Mock()
        explain_result = Mock()
        explain_result.fetch.return_value = Mock()
        exec_result = Mock()
        exec_result.fetch.return_value = Mock()
        client.query.side_effect = [explain_result, exec_result]

        gen = CandidateGenerator(mind=Mock(), mindsdb_client=client)
        score, sanitized, error = gen.preflight_score("SELECT 1")
        assert score == 2
        assert sanitized == "SELECT 1"
        assert error == ""

    def test_preflight_score_exec_failure_returns_one_and_error(self):
        client = Mock()
        explain_result = Mock()
        explain_result.fetch.return_value = Mock()
        exec_result = Mock()
        exec_result.fetch.side_effect = RuntimeError("exec failed")
        client.query.side_effect = [explain_result, exec_result]

        gen = CandidateGenerator(mind=Mock(), mindsdb_client=client)
        score, sanitized, error = gen.preflight_score("SELECT 1")
        assert score == 1
        assert sanitized == "SELECT 1"
        assert "exec failed" in error

    def test_preflight_score_native_dialect_skips_explain(self):
        client = Mock()
        exec_result = Mock()
        exec_result.fetch.return_value = Mock()
        client.query.return_value = exec_result

        gen = CandidateGenerator(mind=Mock(), mindsdb_client=client)
        sql = "SELECT * FROM snowflake_ds (SELECT 1)"
        score, sanitized, error = gen.preflight_score(sql)
        assert score == 2
        assert sanitized == sql
        assert error == ""
        # Only the execution query should be called (no EXPLAIN).
        client.query.assert_called_once_with(sql)


class TestCandidateGeneratorAsyncPaths:
    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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
