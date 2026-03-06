from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from minds.agents.candidate_sql_agent.linker_agent.agent import LinkedSchema
from minds.agents.candidate_sql_agent.text_to_sql_agents.agents import (
    PlanningAgentDeps,
    PlanningAgentRetryDeps,
    SQLGenAgentDeps,
    SQLGenRetryAgentDeps,
    SummarizeAgentDeps,
    TextToSQLPipeline,
    planning_instructions,
    planning_retry_instructions,
    sql_gen_instructions,
    sql_retry_instructions,
    summarize_instructions,
)
from minds.agents.candidate_sql_agent.text_to_sql_agents.models import (
    AcquiredKnowledge,
    DataCatalogSubset,
    QueryPlan,
    QueryPlanStep,
    QueryPlanStepType,
)
from minds.agents.exceptions import DataCatalogValidationError, QueryGenerationError, QueryPlanningError


def _mk_catalog(datasource_name: str, table_names: list[str]):
    """Build a minimal DataCatalog-shaped object used by TextToSQLPipeline validation/pruning."""
    tables = [SimpleNamespace(table=SimpleNamespace(name=t)) for t in table_names]
    return SimpleNamespace(
        mind_datasource=SimpleNamespace(
            datasource=SimpleNamespace(name=datasource_name),
            mind_datasource_tables=tables,
        )
    )


def _mk_mind(name: str = "test-mind", ds_names: list[str] | None = None):
    mind = Mock()
    mind.name = name
    if ds_names:
        mind.mind_datasources = [SimpleNamespace(datasource=SimpleNamespace(name=ds_names[0]))]
    else:
        mind.mind_datasources = []
    return mind


class TestTextToSQLPipelineHelpers:
    def test_set_native_datasource_from_linked_schema_selects_most_frequent(self):
        mind = _mk_mind(ds_names=["fallback_ds"])
        p = TextToSQLPipeline(mind=mind, mindsdb_client=Mock(), is_native_query_mode_enabled=True)

        linked = LinkedSchema(
            tables=["ds1.t1", "ds1.t2", "ds2.t3"],
            columns={},
            joins=[],
        )
        p._set_native_datasource_from_linked_schema(linked)
        assert p._get_native_datasource_name() == "ds1"

    def test_strip_datasource_prefix_for_native_is_case_insensitive(self):
        mind = _mk_mind()
        p = TextToSQLPipeline(mind=mind, mindsdb_client=Mock(), is_native_query_mode_enabled=False)

        q = 'SELECT * FROM "Ds1".t JOIN ds1 . u ON t.id = u.id'
        assert p._strip_datasource_prefix_for_native(q, "ds1") == 'SELECT * FROM "Ds1".t JOIN  u ON t.id = u.id'

    def test_validate_query_plan_errors(self):
        mind = _mk_mind()
        p = TextToSQLPipeline(mind=mind, mindsdb_client=Mock())

        with pytest.raises(QueryPlanningError, match="at least one step"):
            p._validate_query_plan(QueryPlan(steps=[]))

        exploratory = QueryPlanStep(
            description="explore",
            type=QueryPlanStepType.EXPLORATORY,
            data_catalog_subset=DataCatalogSubset(datasources=["ds"], tables=["ds.t"]),
        )
        # last step must be FINAL
        with pytest.raises(QueryPlanningError, match="Last step must be a final step"):
            p._validate_query_plan(QueryPlan(steps=[exploratory]))

        final1 = QueryPlanStep(
            description="final",
            type=QueryPlanStepType.FINAL,
            data_catalog_subset=DataCatalogSubset(datasources=["ds"], tables=["ds.t"]),
        )
        final2 = QueryPlanStep(
            description="final2",
            type=QueryPlanStepType.FINAL,
            data_catalog_subset=DataCatalogSubset(datasources=["ds"], tables=["ds.t"]),
        )
        with pytest.raises(QueryPlanningError, match="exactly one final step"):
            p._validate_query_plan(QueryPlan(steps=[exploratory, final1, final2]))

        # ok: exploratory then final
        p._validate_query_plan(QueryPlan(steps=[exploratory, final1]))

    def test_extract_plan_table_names_dedup_and_sorted(self):
        mind = _mk_mind()
        p = TextToSQLPipeline(mind=mind, mindsdb_client=Mock())

        steps = [
            QueryPlanStep(
                description="s1",
                type=QueryPlanStepType.EXPLORATORY,
                data_catalog_subset=DataCatalogSubset(tables=["b.t2", "a.t1"]),
            ),
            QueryPlanStep(
                description="s2",
                type=QueryPlanStepType.FINAL,
                data_catalog_subset=DataCatalogSubset(tables=["a.t1", "a.t3"]),
            ),
        ]
        plan = QueryPlan(steps=steps)
        assert p._extract_plan_table_names(plan) == ["a.t1", "a.t3", "b.t2"]

    def test_validate_tables_exist_in_catalog_ok_and_errors(self):
        mind = _mk_mind()
        p = TextToSQLPipeline(mind=mind, mindsdb_client=Mock())

        catalogs = [_mk_catalog("ds1", ["t1", "t2"]), _mk_catalog("ds2", ["t3"])]

        # ok, including ignored system table
        p._validate_tables_exist_in_catalog(["ds1.t1", "information_schema.meta_tables"], catalogs)

        # missing datasource prefix
        with pytest.raises(DataCatalogValidationError, match="missing datasource prefix"):
            p._validate_tables_exist_in_catalog(["t1"], catalogs)

        # missing datasource
        with pytest.raises(DataCatalogValidationError, match="Datasource 'dsx' not found"):
            p._validate_tables_exist_in_catalog(["dsx.t1"], catalogs)

        # missing table in existing datasource, includes available tables in message
        with pytest.raises(DataCatalogValidationError, match=r"Available tables in 'ds1': t1, t2"):
            p._validate_tables_exist_in_catalog(["ds1.nope"], catalogs)

    def test_normalize_selected_tables_and_resolve_selected_datasources(self):
        mind = _mk_mind()
        p = TextToSQLPipeline(mind=mind, mindsdb_client=Mock())
        catalogs = [_mk_catalog("ds1", ["t1"]), _mk_catalog("ds2", ["t2"])]

        selected_ds_lower, selected_fq, selected_tables_only, selected_by_ds = p._normalize_selected_tables(
            selected_ds={"DOES_NOT_EXIST"},
            selected_tables={"ds2.t2", "t1"},
        )
        assert selected_fq == {"ds2.t2"}
        assert selected_tables_only == {"t1"}
        assert selected_by_ds == {"ds2": {"t2"}}

        resolved = p._resolve_selected_datasources(selected_ds_lower, selected_by_ds, catalogs)
        # explicit DS doesn't match catalogs, so it falls back to inferred ds2
        assert resolved == {"ds2"}

    def test_sanitize_and_validate_sql_mindsdb_rejects_bad_inputs(self):
        mind = _mk_mind()
        p = TextToSQLPipeline(mind=mind, mindsdb_client=Mock(), is_native_query_mode_enabled=False)

        with pytest.raises(QueryGenerationError, match="Empty SQL"):
            p._sanitize_and_validate_sql_mindsdb("")

        with pytest.raises(QueryGenerationError, match="error string"):
            p._sanitize_and_validate_sql_mindsdb("Error: something")

        with pytest.raises(QueryGenerationError, match="Double quotes are not allowed"):
            p._sanitize_and_validate_sql_mindsdb('SELECT "A" FROM t')

        with pytest.raises(QueryGenerationError, match="Only SELECT queries are allowed"):
            p._sanitize_and_validate_sql_mindsdb("DELETE FROM t")

        with pytest.raises(QueryGenerationError, match="LOG10 function is not supported"):
            p._sanitize_and_validate_sql_mindsdb("SELECT LOG10(x) FROM t")

        with pytest.raises(QueryGenerationError, match="Multiple SQL statements"):
            p._sanitize_and_validate_sql_mindsdb("SELECT 1; SELECT 2;")

    def test_sanitize_and_validate_sql_mindsdb_allows_native_double_quotes_and_strips_semicolon(self):
        mind = _mk_mind(ds_names=["ds1"])
        p = TextToSQLPipeline(mind=mind, mindsdb_client=Mock(), is_native_query_mode_enabled=True)

        assert p._sanitize_and_validate_sql_mindsdb('SELECT "A" FROM t;') == 'SELECT "A" FROM t'

    def test_generate_markdown_table_empty_and_truncation_and_numeric_formatting(self):
        mind = _mk_mind()
        p = TextToSQLPipeline(mind=mind, mindsdb_client=Mock())

        empty = pd.DataFrame()
        assert "(empty)" in p._generate_markdown_table(empty)

        df = pd.DataFrame({"i": [1000, 2000, 3000], "f": [1.2, 2000.25, 3000000.0]})
        md = p._generate_markdown_table(df, override_max_rows=2)
        assert "[Showing 2 of 3 rows]" in md
        # numeric formatting: ints have comma separators; floats have 1 decimal
        assert "1,000" in md
        assert "2,000.2" in md


@pytest.mark.asyncio
async def test_plan_with_retry_uses_retry_agent_after_failure():
    """Exercise the retry path without invoking any real LLM/model."""
    mind = _mk_mind()
    p = TextToSQLPipeline(mind=mind, mindsdb_client=Mock())

    good_plan = QueryPlan(
        steps=[
            QueryPlanStep(
                description="explore",
                type=QueryPlanStepType.EXPLORATORY,
                data_catalog_subset=DataCatalogSubset(tables=["ds.t"]),
            ),
            QueryPlanStep(
                description="final",
                type=QueryPlanStepType.FINAL,
                data_catalog_subset=DataCatalogSubset(tables=["ds.t"]),
            ),
        ]
    )

    mock_streamer = Mock()
    mock_streamer.push = AsyncMock()

    class _Res:
        def __init__(self, output):
            self.output = output

    with (
        patch("minds.agents.candidate_sql_agent.text_to_sql_agents.agents.planning_agent") as planning_agent,
        patch(
            "minds.agents.candidate_sql_agent.text_to_sql_agents.agents.planning_retry_agent"
        ) as planning_retry_agent,
        patch("minds.agents.candidate_sql_agent.text_to_sql_agents.agents.model_for", return_value=Mock()),
    ):
        planning_agent.run = AsyncMock(side_effect=RuntimeError("boom"))
        planning_retry_agent.run = AsyncMock(return_value=_Res(good_plan))

        # keep test fast: one retry is enough to validate path
        with patch(
            "minds.agents.candidate_sql_agent.text_to_sql_agents.agents.agent_settings.max_planning_retries",  # type: ignore[attr-defined]
            2,
        ):
            res = await p._plan_with_retry(
                prompt="q",
                message_history=[],
                data_catalog_context_str="ctx",
                streamer=mock_streamer,
                usage=Mock(),
                usage_limits=Mock(),
            )

    assert res == good_plan


@pytest.mark.asyncio
async def test_prune_data_catalogs_for_step_filters_tables_and_restores_original_list():
    mind = _mk_mind()
    p = TextToSQLPipeline(mind=mind, mindsdb_client=Mock())

    class Catalog:
        def __init__(self, ds: str, tables: list[str]):
            self.mind_datasource = SimpleNamespace(
                datasource=SimpleNamespace(name=ds),
                mind_datasource_tables=[SimpleNamespace(table=SimpleNamespace(name=t)) for t in tables],
            )
            self.modified_at = "now"

        def to_context_str(self, **_):
            return ",".join([t.table.name for t in self.mind_datasource.mind_datasource_tables])

    dc1 = Catalog("ds1", ["t1", "t2"])
    dc2 = Catalog("ds2", ["t3"])
    subset = DataCatalogSubset(datasources=None, tables=["ds1.t2", "t3"])

    pruned = await p._prune_data_catalogs_for_step([dc1, dc2], subset)
    assert len(pruned) == 2

    # ds1 should be wrapped and filtered to only t2
    ctx = pruned[0].to_context_str()
    assert ctx == "t2"

    # wrapper must restore underlying list after generating context
    assert dc1.to_context_str() == "t1,t2"


@pytest.mark.asyncio
async def test_execute_exploratory_step_success_records_acquired_knowledge():
    mind = _mk_mind()
    p = TextToSQLPipeline(mind=mind, mindsdb_client=Mock(), is_native_query_mode_enabled=False)
    ak = AcquiredKnowledge()

    streamer = Mock()
    streamer.push = AsyncMock()

    class _Res:
        def __init__(self, query: str):
            self.output = SimpleNamespace(query=query)

    with (
        patch("minds.agents.candidate_sql_agent.text_to_sql_agents.agents.sql_gen_agent") as sql_gen_agent,
        patch("minds.agents.candidate_sql_agent.text_to_sql_agents.agents.agent_settings.max_sql_retries", 1),
    ):
        sql_gen_agent.run = AsyncMock(return_value=_Res("SELECT 1"))
        p._execute_sql = Mock(return_value=pd.DataFrame({"x": [1]}))

        await p._execute_exploratory_step(
            step_description="explore",
            message_history=[],
            data_catalog_subset_context="ctx",
            acquired_knowledge=ak,
            streamer=streamer,
            usage=Mock(),
            usage_limits=Mock(),
        )

    assert len(ak.items) == 1
    assert "SELECT 1" in ak.items[0].attempts[0].query
    assert "x" in (ak.items[0].attempts[0].result or "")


@pytest.mark.asyncio
async def test_execute_multi_path_step_invalid_columns_is_recorded():
    mind = _mk_mind()
    p = TextToSQLPipeline(mind=mind, mindsdb_client=Mock())
    ak = AcquiredKnowledge()

    streamer = Mock()
    streamer.push = AsyncMock()

    candidate = SimpleNamespace(query="SELECT t.bad FROM t", strategy="direct", executed=False, execution_error=None)

    class FakeCG:
        def __init__(self, *_a, **_k): ...

        async def generate(self, **_):
            return [candidate]

        def validate_columns(self, *_a, **_k):
            return False, ["t.bad"]

        def preflight_score(self, *_a, **_k):
            raise AssertionError("should not be called when columns invalid")

    linked_schema = LinkedSchema(tables=["ds.t"], columns={"ds.t": ["good"]}, joins=[])

    with patch("minds.agents.candidate_sql_agent.text_to_sql_agents.agents.CandidateGenerator", FakeCG):
        await p._execute_multi_path_step(
            step_description="step",
            data_catalog_subset_context="ctx",
            acquired_knowledge=ak,
            streamer=streamer,
            linked_schema=linked_schema,
        )

    assert len(ak.items) == 1
    assert ak.items[0].attempts
    assert "Hallucinated columns" in (ak.items[0].attempts[0].error or "")


@pytest.mark.asyncio
async def test_execute_multi_path_step_early_exits_on_first_successful_candidate():
    mind = _mk_mind()
    p = TextToSQLPipeline(mind=mind, mindsdb_client=Mock())
    ak = AcquiredKnowledge()

    streamer = Mock()
    streamer.push = AsyncMock()

    c1 = SimpleNamespace(
        query="SELECT 1",
        strategy="a",
        executed=False,
        execution_error=None,
        execution_result=None,
        preflight_score=0,
    )
    c2 = SimpleNamespace(
        query="SELECT 2",
        strategy="b",
        executed=False,
        execution_error=None,
        execution_result=None,
        preflight_score=0,
    )

    class FakeCG:
        calls = 0

        def __init__(self, *_a, **_k): ...

        async def generate(self, **_):
            return [c1, c2]

        def validate_columns(self, *_a, **_k):
            return True, []

        def preflight_score(self, query, sanitize_fn=None):
            FakeCG.calls += 1
            sanitized = sanitize_fn(query) if sanitize_fn else query
            return 2, sanitized, ""

    p._execute_sql = Mock(return_value=pd.DataFrame({"x": [1]}))

    with patch("minds.agents.candidate_sql_agent.text_to_sql_agents.agents.CandidateGenerator", FakeCG):
        await p._execute_multi_path_step(
            step_description="step",
            data_catalog_subset_context="ctx",
            acquired_knowledge=ak,
            streamer=streamer,
            linked_schema=LinkedSchema(tables=[], columns={}, joins=[]),
        )

    assert FakeCG.calls == 1
    assert c1.executed is True
    assert c2.executed is False


@pytest.mark.asyncio
async def test_execute_final_step_returns_chart_intent():
    mind = _mk_mind(ds_names=["ds1"])
    p = TextToSQLPipeline(mind=mind, mindsdb_client=Mock(), is_native_query_mode_enabled=True)

    streamer = Mock()
    streamer.push = AsyncMock()

    class _Res:
        def __init__(self):
            self.output = SimpleNamespace(query='SELECT "A" FROM t', chart_intent={"type": "bar", "x": "A", "y": "A"})

    with (
        patch("minds.agents.candidate_sql_agent.text_to_sql_agents.agents.sql_gen_agent") as sql_gen_agent,
        patch("minds.agents.candidate_sql_agent.text_to_sql_agents.agents.agent_settings.max_sql_retries", 1),
    ):
        sql_gen_agent.run = AsyncMock(return_value=_Res())
        p._execute_sql = Mock(return_value=pd.DataFrame({"A": [1]}))

        query, df, intent = await p._execute_final_step(
            step_description="final",
            message_history=[],
            data_catalog_subset_context="ctx",
            acquired_knowledge=AcquiredKnowledge(),
            streamer=streamer,
            usage=Mock(),
            usage_limits=Mock(),
        )

    assert "SELECT * FROM ds1 (" in query  # wrapped for native mode
    assert not df.empty
    assert intent and intent["type"] == "bar"


@pytest.mark.asyncio
async def test_pipeline_run_uses_schema_linking_and_summary_mode():
    mind = _mk_mind(name="m", ds_names=["ds1"])
    p = TextToSQLPipeline(mind=mind, mindsdb_client=Mock(), is_native_query_mode_enabled=True)

    streamer = Mock()
    streamer.push = AsyncMock()

    class Catalog:
        def __init__(self):
            self.modified_at = "now"
            self.mind_datasource = SimpleNamespace(
                datasource=SimpleNamespace(name="ds1"),
                mind_datasource_tables=[SimpleNamespace(table=SimpleNamespace(name="t1"))],
            )

        def get_table_list_summary(self, **_):
            return "Table: ds1.t1"

        def to_context_str(self, **_):
            return "FULL"

    catalogs = [Catalog()]
    plan = QueryPlan(
        steps=[
            QueryPlanStep(
                description="explore",
                type=QueryPlanStepType.EXPLORATORY,
                data_catalog_subset=DataCatalogSubset(tables=["ds1.t1"]),
            ),
            QueryPlanStep(
                description="final",
                type=QueryPlanStepType.FINAL,
                data_catalog_subset=DataCatalogSubset(tables=["ds1.t1"]),
            ),
        ]
    )

    with (
        patch(
            "minds.agents.candidate_sql_agent.text_to_sql_agents.agents.data_catalog_cache.load",
            AsyncMock(return_value=catalogs),
        ),
        patch("minds.agents.candidate_sql_agent.text_to_sql_agents.agents.SchemaLinker") as SchemaLinker,
        patch(
            "minds.agents.candidate_sql_agent.text_to_sql_agents.agents.agent_settings.large_catalog_table_threshold", 0
        ),
        patch.object(p, "_plan_with_retry", AsyncMock(return_value=plan)),
        patch.object(p, "_execute", AsyncMock(return_value=("Q", "R", 2, AcquiredKnowledge()))),
    ):
        SchemaLinker.return_value.link = AsyncMock(
            return_value=LinkedSchema(tables=["ds1.t1"], columns={}, joins=[], reasoning="r")
        )
        SchemaLinker.return_value.filter_catalogs_by_linked_schema = Mock(return_value="FILTERED")

        res = await p.run(prompt="p", message_history=[], streamer=streamer)

    assert res.final_query == "Q"
    assert res.execution_result == "R"
    assert res.steps_executed == 2


@pytest.mark.asyncio
async def test_instruction_builders_include_expected_context_and_mode_specific_rules():
    mind = _mk_mind(ds_names=["ds1"])

    p = await planning_instructions(SimpleNamespace(deps=PlanningAgentDeps(mind=mind, data_catalog_context="CAT")))
    assert "CAT" in p

    pr = await planning_retry_instructions(
        SimpleNamespace(
            deps=PlanningAgentRetryDeps(
                mind=mind,
                data_catalog_context="CAT",
                failed_query_plan="BAD PLAN",
                error_message="oops",
            )
        )
    )
    assert "BAD PLAN" in pr
    assert "oops" in pr

    g_native = await sql_gen_instructions(
        SimpleNamespace(
            deps=SQLGenAgentDeps(
                mind=mind,
                data_catalog_subset_context="SUB",
                acquired_knowledge="K",
                is_native_query_mode_enabled=True,
            )
        )
    )
    assert "SNOWFLAKE CASE SENSITIVITY" in g_native

    g_mindsdb = await sql_gen_instructions(
        SimpleNamespace(
            deps=SQLGenAgentDeps(
                mind=mind,
                data_catalog_subset_context="SUB",
                acquired_knowledge="K",
                is_native_query_mode_enabled=False,
            )
        )
    )
    assert "EXACT datasource name" in g_mindsdb

    retry = await sql_retry_instructions(
        SimpleNamespace(
            deps=SQLGenRetryAgentDeps(
                mind=mind,
                data_catalog_subset_context="SUB",
                acquired_knowledge="K",
                is_native_query_mode_enabled=False,
                failed_query="SELECT 1",
                error_message="bad",
                previous_attempts=[],
            )
        )
    )
    assert "SELECT 1" in retry
    assert "bad" in retry

    summ = await summarize_instructions(
        SimpleNamespace(
            deps=SummarizeAgentDeps(
                mind=mind,
                prompt="P",
                step_description="final step",
                acquired_knowledge="K",
            )
        )
    )
    assert "final step" in summ


@pytest.mark.asyncio
async def test_execute_stops_when_max_steps_exceeded():
    mind = _mk_mind()
    p = TextToSQLPipeline(mind=mind, mindsdb_client=Mock())

    streamer = Mock()
    streamer.push = AsyncMock()

    plan = QueryPlan(
        steps=[
            QueryPlanStep(
                description="explore",
                type=QueryPlanStepType.EXPLORATORY,
                data_catalog_subset=DataCatalogSubset(tables=["ds.t"]),
            )
        ]
    )

    with patch("minds.agents.candidate_sql_agent.text_to_sql_agents.agents.agent_settings.max_query_plan_steps", 0):
        q, msg, steps, ak = await p._execute(
            query_plan=plan,
            prompt="p",
            message_history=[],
            data_catalogs=[],
            data_catalog_context_str="ctx",
            streamer=streamer,
            usage=Mock(),
            usage_limits=Mock(),
            linked_schema=None,
        )

    assert q == ""
    assert "exceeded maximum step limit" in msg.lower()
    assert steps == 1
    assert ak is not None
