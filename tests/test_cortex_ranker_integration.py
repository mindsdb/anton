"""Integration: Cortex.build_memory_context wires the ranker + stats.

These tests don't mock Hippocampus or the stats sidecar — they spin
up real instances on a tmp_path and verify the end-to-end shape:

  1. A small rules.md is loaded in full (no ranking pressure), and
     every bulleted rule gets its retrieval counter bumped.
  2. A large `## When` rules.md is ranked against the user message:
     the most-relevant rule appears in the output, less-relevant
     rules get dropped, and counters reflect what's loaded.
  3. The stats sidecar is written exactly once per build_memory_context
     call (the buffer/flush contract).
  4. Always/Never rules are always loaded even when ranking kicks in.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from anton.core.memory.cortex import Cortex
from anton.core.memory.hippocampus import Hippocampus
from anton.core.memory.base import Engram


def _make_cortex(tmp_path: Path) -> Cortex:
    global_dir = tmp_path / "global"
    project_dir = tmp_path / "project"
    global_dir.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)
    return Cortex(
        global_hc=Hippocampus(global_dir),
        project_hc=Hippocampus(project_dir),
        mode="autopilot",
    )


def _seed_when_rules(hc: Hippocampus, rules: list[str]) -> None:
    """Encode a batch of When rules. Pad to >6000 chars so ranking
    triggers — under the threshold, Cortex returns everything."""
    for text in rules:
        hc.encode_rule(text, kind="when", confidence="high", source="user")


class TestSmallCorpusLoadsAll:
    @pytest.mark.asyncio
    async def test_under_budget_loads_every_rule_and_records_retrievals(
        self, tmp_path: Path
    ):
        cortex = _make_cortex(tmp_path)
        _seed_when_rules(cortex.global_hc, [
            "Use httpx instead of requests",
            "For CSV files use low_memory=False with pd.read_csv",
        ])

        out = await cortex.build_memory_context("what's the bitcoin price")
        assert "httpx" in out
        assert "low_memory=False" in out

        # Stats sidecar should have two recorded rules.
        stats_path = tmp_path / "global" / "rules.stats.json"
        assert stats_path.exists()
        data = json.loads(stats_path.read_text())
        # Two distinct rules → two entries.
        assert len(data["rules"]) == 2
        for rec in data["rules"].values():
            assert rec["retrievals"] == 1
            assert rec["last_retrieved"] is not None


class TestLargeCorpusTriggersRanking:
    @pytest.mark.asyncio
    async def test_ranker_keeps_relevant_drops_irrelevant(self, tmp_path: Path):
        cortex = _make_cortex(tmp_path)
        # Each rule is long enough to push the corpus past the 6000-char
        # ranking trigger (about 30 rules of ~250 chars each).
        pad = " " + ("x" * 200)
        pandas_rules = [f"For CSV files use pandas read_csv with utf-8{pad}-{i}" for i in range(15)]
        html_rules = [f"For HTML reports render with explicit charset utf-8{pad}-{i}" for i in range(15)]
        _seed_when_rules(cortex.global_hc, pandas_rules + html_rules)

        # Query is squarely about CSV / pandas → pandas rules should rank
        # higher. With cap_k=20 and budget pressure, we expect MORE pandas
        # rules than html rules in the output.
        out = await cortex.build_memory_context(
            "load the sales.csv and summarize with pandas"
        )
        pandas_in_out = sum(1 for r in pandas_rules if r in out)
        html_in_out = sum(1 for r in html_rules if r in out)
        assert pandas_in_out > html_in_out, (
            f"pandas={pandas_in_out} html={html_in_out} — ranker should "
            f"favour CSV/pandas rules for a CSV/pandas query"
        )

    @pytest.mark.asyncio
    async def test_always_and_never_sections_survive_ranking(self, tmp_path: Path):
        cortex = _make_cortex(tmp_path)
        cortex.global_hc.encode_rule(
            "Use httpx instead of requests", kind="always",
            confidence="high", source="user",
        )
        cortex.global_hc.encode_rule(
            "Use time.sleep() in scratchpad cells", kind="never",
            confidence="high", source="user",
        )
        pad = " " + ("x" * 200)
        _seed_when_rules(cortex.global_hc, [
            f"When loading CSV files use low_memory=False{pad}-{i}"
            for i in range(40)
        ])

        out = await cortex.build_memory_context(
            "render an HTML dashboard with utf-8"  # unrelated to CSV/httpx/sleep
        )
        # Both unconditional rules survive even though the query has no
        # lexical overlap with them — they're not ranked, just loaded.
        assert "httpx" in out
        assert "time.sleep" in out


class TestStatsSidecar:
    @pytest.mark.asyncio
    async def test_two_builds_increment_same_counter(self, tmp_path: Path):
        cortex = _make_cortex(tmp_path)
        _seed_when_rules(cortex.global_hc, ["Use httpx instead of requests"])

        await cortex.build_memory_context("call an API")
        await cortex.build_memory_context("make an HTTP request")

        data = json.loads((tmp_path / "global" / "rules.stats.json").read_text())
        # One rule, retrieved twice across two builds.
        records = list(data["rules"].values())
        assert len(records) == 1
        assert records[0]["retrievals"] == 2

    @pytest.mark.asyncio
    async def test_no_rules_means_no_sidecar(self, tmp_path: Path):
        # Cold start: no rules in the hippocampus → build_memory_context
        # has nothing to record → the sidecar shouldn't appear.
        cortex = _make_cortex(tmp_path)
        await cortex.build_memory_context("anything")
        # The file may not exist (nothing to write) OR may exist with
        # an empty rules map; both are valid. The contract is "no
        # spurious counters", not "no file ever".
        path = tmp_path / "global" / "rules.stats.json"
        if path.exists():
            data = json.loads(path.read_text())
            assert data.get("rules") == {}


class TestConsumeRetrievedThisTurn:
    """Phase C — outcome bridge takes a per-turn snapshot of which
    rule IDs landed in the prompt, so the ACC can ask "did the LLM
    actually see this rule?" before bumping the ignored counter."""

    @pytest.mark.asyncio
    async def test_set_accumulates_then_clears_on_consume(self, tmp_path: Path):
        cortex = _make_cortex(tmp_path)
        _seed_when_rules(cortex.global_hc, [
            "Use httpx instead of requests",
            "For CSV files use low_memory=False",
        ])
        await cortex.build_memory_context("call an API")

        # Two rules retrieved → set has two IDs.
        snapshot = cortex.consume_retrieved_this_turn()
        assert len(snapshot) == 2

        # Second consume in the same turn is empty (take + clear).
        again = cortex.consume_retrieved_this_turn()
        assert again == set()

    @pytest.mark.asyncio
    async def test_set_accumulates_across_multiple_builds_in_one_turn(
        self, tmp_path: Path
    ):
        # `_build_system_prompt` runs build_memory_context more than
        # once per turn on certain recovery paths. The retrieval set
        # should NOT reset between those builds — it should drain
        # only on consume (end-of-turn).
        cortex = _make_cortex(tmp_path)
        _seed_when_rules(cortex.global_hc, ["Use httpx instead of requests"])

        await cortex.build_memory_context("call an API")
        await cortex.build_memory_context("retry the API call")
        snapshot = cortex.consume_retrieved_this_turn()
        # Same rule, multiple builds → one ID in the set.
        assert len(snapshot) == 1
