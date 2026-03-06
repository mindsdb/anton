"""
Unit tests for SelectionAgent logic (pairwise selection orchestration).
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from minds.agents.candidate_sql_agent.candidate_generator_agent.agent import SQLCandidate
from minds.agents.candidate_sql_agent.selection_agent.agent import QueryComplexityClassifier, SelectionAgent


class TestSelectionAgent:
    @pytest.mark.asyncio
    async def test_select_raises_on_empty(self):
        agent = SelectionAgent(mind=Mock())
        with pytest.raises(ValueError, match="No candidates"):
            await agent.select([], question="q", schema_context="s")

    @pytest.mark.asyncio
    async def test_select_returns_single_candidate(self):
        agent = SelectionAgent(mind=Mock())
        c = SQLCandidate(query="SELECT 1", strategy="direct")
        assert await agent.select([c], question="q", schema_context="s") is c

    @pytest.mark.asyncio
    async def test_select_returns_only_successfully_executed_candidate(self):
        agent = SelectionAgent(mind=Mock())
        good = SQLCandidate(query="SELECT 1", strategy="a", executed=True, execution_error=None)
        bad = SQLCandidate(query="SELECT 2", strategy="b", executed=True, execution_error="boom")
        not_run = SQLCandidate(query="SELECT 3", strategy="c", executed=False)
        assert await agent.select([bad, not_run, good], question="q", schema_context="s") is good

    @pytest.mark.asyncio
    async def test_select_uses_pairwise_select_for_multiple_executed(self):
        agent = SelectionAgent(mind=Mock())
        c1 = SQLCandidate(query="SELECT 1", strategy="a", executed=True)
        c2 = SQLCandidate(query="SELECT 2", strategy="b", executed=True)

        with patch.object(agent, "_pairwise_select", new=AsyncMock(return_value=c2)) as mock_pairwise:
            chosen = await agent.select([c1, c2], question="q", schema_context="s")
            assert chosen is c2
            mock_pairwise.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pairwise_select_counts_wins_and_returns_max(self):
        mind = Mock()
        agent = SelectionAgent(mind=mind)
        candidates = [
            SQLCandidate(query="A", strategy="a"),
            SQLCandidate(query="B", strategy="b"),
            SQLCandidate(query="C", strategy="c"),
        ]

        # Force candidate 1 to win all comparisons.
        async def _compare_pair(_a, _b, idx_a, idx_b, *_args, **_kwargs):
            return idx_b  # always give point to second in the pair

        agent._compare_pair = AsyncMock(side_effect=_compare_pair)
        winner = await agent._pairwise_select(candidates, question="q", schema_context="s")
        assert winner is candidates[2]

    @pytest.mark.asyncio
    async def test_pairwise_select_on_compare_exception_gives_point_to_first(self):
        agent = SelectionAgent(mind=Mock())
        c1 = SQLCandidate(query="A", strategy="a")
        c2 = SQLCandidate(query="B", strategy="b")

        agent._compare_pair = AsyncMock(side_effect=RuntimeError("boom"))
        winner = await agent._pairwise_select([c1, c2], question="q", schema_context="s")
        assert winner is c1

    @pytest.mark.asyncio
    async def test_compare_pair_handles_winner_a_b_and_unclear(self):
        agent = SelectionAgent(mind=Mock())
        a = SQLCandidate(query="SELECT 1", strategy="a")
        b = SQLCandidate(query="SELECT 2", strategy="b")

        class _Res:
            def __init__(self, winner: str):
                self.output = Mock(winner=winner, reasoning="because")

        with patch("minds.agents.candidate_sql_agent.selection_agent.agent.selection_agent") as sel:
            sel.run = AsyncMock(return_value=_Res("B"))
            assert await agent._compare_pair(a, b, 0, 1, "q", "s") == 1

            sel.run = AsyncMock(return_value=_Res("A"))
            assert await agent._compare_pair(a, b, 0, 1, "q", "s") == 0

            sel.run = AsyncMock(return_value=_Res("???"))
            assert await agent._compare_pair(a, b, 0, 1, "q", "s") == 0

    def test_format_execution_info_branches(self):
        agent = SelectionAgent(mind=Mock())
        ok = SQLCandidate(query="q", strategy="s", executed=True, execution_error=None)
        err = SQLCandidate(query="q", strategy="s", executed=True, execution_error="boom")
        not_run = SQLCandidate(query="q", strategy="s", executed=False, execution_error=None)

        assert "executed successfully" in agent._format_execution_info(ok, "A")
        assert "failed with error" in agent._format_execution_info(err, "A")
        assert "has not been executed" in agent._format_execution_info(not_run, "A")


class TestQueryComplexityClassifier:
    def test_is_simple_false_when_multiple_tables(self):
        c = QueryComplexityClassifier()
        assert c.is_simple("show users", num_tables=2) is False

    def test_is_simple_false_when_complex_indicator_present(self):
        c = QueryComplexityClassifier()
        assert c.is_simple("show users with join", num_tables=1) is False

    def test_is_simple_true_for_single_table_simple_question(self):
        c = QueryComplexityClassifier()
        assert c.is_simple("list all users", num_tables=1) is True
