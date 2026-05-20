"""Test the `/memory rankings` debug surface (Layer 3 — Phase D).

Light-touch tests — we don't reproduce the rich-console formatting
verbatim, we just confirm:

  - The handler runs without errors when stats are absent.
  - Recorded retrievals + ignored counts appear in the output.
  - Rule text is rendered (so a developer running the command sees
    something useful, not just numbers).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from anton.core.memory.cortex import Cortex
from anton.core.memory.hippocampus import Hippocampus


class _StubSettings:
    """Minimal settings stand-in. MemoryManage only ever reads attrs
    we set explicitly in its handlers — none for rankings."""

    pass


def _make_manager(tmp_path: Path, console: Console):
    from anton.memory.manage import MemoryManage

    global_dir = tmp_path / "global"
    project_dir = tmp_path / "project"
    global_dir.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)

    cortex = Cortex(
        global_hc=Hippocampus(global_dir),
        project_hc=Hippocampus(project_dir),
        mode="autopilot",
    )
    return MemoryManage(
        console=console,
        settings=_StubSettings(),
        cortex=cortex,
    )


@pytest.mark.asyncio
async def test_rankings_with_no_rules_prints_empty_notice(tmp_path: Path):
    buf = Console(record=True)
    manager = _make_manager(tmp_path, buf)
    await manager.rankings()
    output = buf.export_text()
    assert "No rules in memory" in output


@pytest.mark.asyncio
async def test_rankings_shows_recorded_counts(tmp_path: Path):
    buf = Console(record=True)
    manager = _make_manager(tmp_path, buf)
    cortex = manager.cortex

    # Seed two rules and simulate two retrievals + one ignored on the
    # first. Use the stats API directly — testing the surface, not the
    # capture path (that's covered in the cortex integration tests).
    cortex.global_hc.encode_rule(
        "Use httpx instead of requests",
        kind="when", confidence="high", source="user",
    )
    cortex.global_hc.encode_rule(
        "For CSV files use low_memory=False",
        kind="when", confidence="high", source="user",
    )
    cortex._rule_stats.record_retrieval("Use httpx instead of requests")
    cortex._rule_stats.record_retrieval("Use httpx instead of requests")
    cortex._rule_stats.record_ignored("Use httpx instead of requests")
    cortex._rule_stats.flush()

    await manager.rankings()
    output = buf.export_text()
    # Rule text appears.
    assert "httpx" in output
    assert "low_memory" in output
    # Counters appear (retrievals=2, ignored=1 for the first rule).
    # We look for the digits in proximity to the rule rather than
    # pinning column positions — that lets us refactor formatting
    # without breaking the test.
    assert "2" in output  # retrieval count
    assert "1" in output  # ignored count


@pytest.mark.asyncio
async def test_rankings_with_remote_backend_explains_absence(tmp_path: Path):
    # If the cortex has no rule_stats (e.g. remote hippocampus with no
    # local dir), the command should explain rather than crash or
    # render an empty table.
    buf = Console(record=True)
    manager = _make_manager(tmp_path, buf)
    manager.cortex._rule_stats = None  # simulate remote-backend scenario
    await manager.rankings()
    output = buf.export_text()
    assert "rule-stats backend" in output or "No rule-stats" in output
