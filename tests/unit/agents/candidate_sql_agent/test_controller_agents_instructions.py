from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from minds.agents.candidate_sql_agent.controller_agents.agents import (
    FeedbackAgentDeps,
    LightweightRouterAgentDeps,
    feedback_instructions,
    lightweight_router_instructions,
)


@pytest.mark.asyncio
async def test_lightweight_router_instructions_includes_compact_table_list():
    mind = Mock()
    deps = LightweightRouterAgentDeps(mind=mind)
    ctx = SimpleNamespace(deps=deps)

    catalog = SimpleNamespace(
        mind_datasource=SimpleNamespace(
            datasource=SimpleNamespace(name="ds1"),
            mind_datasource_tables=[SimpleNamespace(table=SimpleNamespace(name="t1"))],
        )
    )

    with patch(
        "minds.agents.candidate_sql_agent.controller_agents.agents.data_catalog_cache.load",
        AsyncMock(return_value=[catalog]),
    ):
        prompt = await lightweight_router_instructions(ctx)

    assert "Data Source: ds1" in prompt
    assert "Tables: t1" in prompt


@pytest.mark.asyncio
async def test_feedback_instructions_renders_catalog_context_and_respects_token_budget():
    mind = Mock()
    deps = FeedbackAgentDeps(mind=mind, is_native_query_mode_enabled=False)
    ctx = SimpleNamespace(deps=deps)

    class Catalog:
        def __init__(self, name: str):
            self.mind_datasource = SimpleNamespace(datasource=SimpleNamespace(name=name), mind_datasource_tables=[])

        def to_context_str(self, **kwargs):
            # expose the token budgeting behavior for coverage
            assert "max_tokens" in kwargs
            return f"CTX:{self.mind_datasource.datasource.name}"

    with patch(
        "minds.agents.candidate_sql_agent.controller_agents.agents.data_catalog_cache.load",
        AsyncMock(return_value=[Catalog("ds1"), Catalog("ds2")]),
    ):
        prompt = await feedback_instructions(ctx)

    assert "CTX:ds1" in prompt
    assert "CTX:ds2" in prompt
