"""
Unit tests for SchemaLinker helpers and controller routing utilities.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from minds.agents.candidate_sql_agent.controller_agents.agents import _compact_table_list_summary
from minds.agents.candidate_sql_agent.linker_agent.agent import LinkedSchema, SchemaLinker


def _make_catalog(datasource_name: str, table_names: list[str]):
    catalog = Mock()
    md = Mock()
    md.datasource = Mock()
    md.datasource.name = datasource_name

    md_tables = []
    for t in table_names:
        tbl = Mock()
        tbl.name = t
        tbl.description = None
        tbl.primary_key_constraints = []
        tbl.foreign_key_constraints = []
        tbl.columns = []

        mdt = Mock()
        mdt.table = tbl
        md_tables.append(mdt)

    md.mind_datasource_tables = md_tables
    catalog.mind_datasource = md
    return catalog


class TestLinkedSchemaModel:
    def test_get_table_list_strips_datasource_prefix(self):
        linked = LinkedSchema(tables=["ds.users", "orders"], columns={}, joins=[])
        assert linked.get_table_list() == ["users", "orders"]

    def test_to_context_str_includes_tables_and_columns(self):
        linked = LinkedSchema(
            tables=["ds.users"],
            columns={"ds.users": ["id", "name"]},
            joins=[],
        )
        ctx = linked.to_context_str()
        assert "Table: ds.users" in ctx
        assert "Columns: id, name" in ctx


class TestSchemaLinker:
    def test_build_schema_summary_includes_qualified_table_names(self):
        linker = SchemaLinker(mind=Mock())
        catalogs = [_make_catalog("ds", ["users", "orders"])]
        summary = linker._build_schema_summary(catalogs)
        assert "## Datasource: ds" in summary
        assert "Table: ds.users" in summary
        assert "Table: ds.orders" in summary

    def test_filter_catalogs_by_linked_schema_includes_only_linked_tables(self):
        linker = SchemaLinker(mind=Mock())
        catalogs = [_make_catalog("ds", ["users", "orders"])]
        linked = LinkedSchema(
            tables=["ds.users"],
            columns={"ds.users": ["id"]},
            joins=[],
        )
        filtered = linker.filter_catalogs_by_linked_schema(catalogs, linked)
        assert "Table: ds.users" in filtered
        assert "Table: ds.orders" not in filtered

    def test_filter_catalogs_by_linked_schema_matches_by_table_name(self):
        linker = SchemaLinker(mind=Mock())
        catalogs = [_make_catalog("ds", ["users"])]
        # Linked schema table without datasource prefix should still match by table name.
        linked = LinkedSchema(
            tables=["users"],
            columns={"users": ["id"]},
            joins=[],
        )
        filtered = linker.filter_catalogs_by_linked_schema(catalogs, linked)
        assert "Table: ds.users" in filtered

    @pytest.mark.asyncio
    async def test_link_success_returns_linked_schema(self):
        mind = Mock()
        linker = SchemaLinker(mind=mind)
        catalogs = [_make_catalog("ds", ["users"])]

        out = LinkedSchema(tables=["ds.users"], columns={"ds.users": ["id"]}, joins=[], reasoning="r")

        class _Res:
            def __init__(self, output):
                self.output = output

        with patch("minds.agents.candidate_sql_agent.linker_agent.agent.schema_linker_agent") as agent:
            agent.run = AsyncMock(return_value=_Res(out))
            res = await linker.link(question="q", data_catalogs=catalogs)

        assert res.tables == ["ds.users"]
        assert "ds.users" in res.columns

    @pytest.mark.asyncio
    async def test_link_failure_returns_empty_schema(self):
        mind = Mock()
        linker = SchemaLinker(mind=mind)
        catalogs = [_make_catalog("ds", ["users"])]

        with patch("minds.agents.candidate_sql_agent.linker_agent.agent.schema_linker_agent") as agent:
            agent.run = AsyncMock(side_effect=RuntimeError("boom"))
            res = await linker.link(question="q", data_catalogs=catalogs)

        assert res.tables == []
        assert res.columns == {}
        assert "failed" in res.reasoning.lower()


class TestCompactTableListSummary:
    def test_compact_table_list_summary_truncates_with_suffix(self):
        catalog = _make_catalog("ds", [f"t{i}" for i in range(60)])
        s = _compact_table_list_summary(catalog, max_tables=50)
        assert "Total Tables: 60" in s
        assert "(+10 more)" in s
