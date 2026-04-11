"""End-to-end test of the skills loop.

Simulates the full happy path with mocked LLM at each call site:

1. Save a skill from recent work via `/skill save` (mock LLM drafts JSON)
2. "Restart" by building a fresh prompt — the saved skill appears in the
   procedural memory section
3. Simulate the LLM calling `recall_skill(label)` via the tool registry
4. Verify the tool result + counter increment
5. Recall again with a typo — closest_match recovers, counter increments

This is the test that proves the loop is wired correctly across all
prior steps. If it passes, the v1 build is functional end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.console import Console

from anton.commands.skills import _SkillDraft, handle_skill_save
from anton.core.llm.prompt_builder import ChatSystemPromptBuilder
from anton.core.memory.skills import SkillStore
from anton.core.tools.recall_skill import RECALL_SKILL_TOOL
from anton.core.tools.registry import ToolRegistry


@pytest.fixture()
def store_root(tmp_path: Path) -> Path:
    return tmp_path / "skills"


@pytest.fixture()
def console() -> Console:
    return Console(record=True, width=120)


def _make_session_for_save(
    store: SkillStore, draft: _SkillDraft, cells: list, history: list
) -> MagicMock:
    """A session-like mock with the LLM ready to return `draft`."""
    session = MagicMock()
    session._history = history
    session._skill_store = store
    pad = SimpleNamespace(cells=cells)
    session._scratchpads = SimpleNamespace(_pads={"work": pad})
    session._llm = MagicMock()
    session._llm.generate_object = AsyncMock(return_value=draft)
    return session


def _fake_cell(code: str, stdout: str = "ok"):
    return SimpleNamespace(
        code=code, stdout=stdout, stderr="", error=None, description=""
    )


@pytest.mark.asyncio
async def test_full_skills_loop(console, store_root):
    # ── Setup: empty store, simulated session with some scratchpad work ─
    store = SkillStore(root=store_root)
    assert store.list_all() == []

    cells = [
        _fake_cell(
            "import pandas as pd\n"
            "df = pd.read_csv('sales_q3.csv')\n"
            "print(df.shape)",
            stdout="(12000, 8)",
        ),
        _fake_cell(
            "print(df.describe())\n"
            "print(df.dtypes)",
            stdout="<summary statistics>",
        ),
        _fake_cell(
            "df['amount'].plot.hist(bins=30)",
            stdout="<plot>",
        ),
    ]
    history = [
        {"role": "user", "content": "take a quick look at sales_q3.csv"},
        {
            "role": "assistant",
            "content": "Loaded the CSV and summarized it. 12k rows, 8 cols.",
        },
    ]

    draft = _SkillDraft(
        label="csv_summary",
        name="CSV Summary",
        description="Load a CSV with pandas, print shape/describe/dtypes, plot a histogram.",
        when_to_use="User asks to explore, summarize, or describe a CSV file.",
        declarative_md=(
            "1. Use `pandas.read_csv()` to load the file.\n"
            "2. Print `df.shape` and `df.dtypes`.\n"
            "3. Run `df.describe()` for summary stats.\n"
            "4. For numeric columns of interest, plot a histogram with "
            "`df[col].plot.hist(bins=30)`."
        ),
    )
    session = _make_session_for_save(store, draft, cells, history)

    # ── Step 1: /skill save ─────────────────────────────────────────────
    await handle_skill_save(console, session, store=store)

    skills = store.list_all()
    assert len(skills) == 1
    saved = skills[0]
    assert saved.label == "csv_summary"
    assert saved.name == "CSV Summary"
    assert "pandas.read_csv" in saved.declarative_md
    assert saved.stats.stage_1.recommended == 0  # no recalls yet

    # ── Step 2: "restart" — build a fresh system prompt ─────────────────
    # This simulates a new session reading the same disk store
    fresh_store = SkillStore(root=store_root)
    builder = ChatSystemPromptBuilder()
    prompt = builder.build(
        current_datetime="2026-04-10T13:00:00+00:00",
        runtime_context="test",
        proactive_dashboards=False,
        output_dir="/tmp/x",
        skill_store=fresh_store,
    )
    assert "## Procedural memory" in prompt
    assert "`csv_summary`" in prompt
    assert "explore, summarize, or describe a CSV" in prompt

    # ── Step 3: LLM "decides" to recall the skill via the tool registry ─
    registry = ToolRegistry()
    registry.register_tool(RECALL_SKILL_TOOL)
    fresh_session = SimpleNamespace(_skill_store=fresh_store)

    result = await registry.dispatch_tool(
        fresh_session, "recall_skill", {"label": "csv_summary"}
    )

    assert "CSV Summary" in result
    assert "pandas.read_csv" in result
    assert "describe" in result

    # Counter incremented to 1
    after_recall_1 = fresh_store.load("csv_summary")
    assert after_recall_1 is not None
    assert after_recall_1.stats.stage_1.recommended == 1
    assert after_recall_1.stats.total_recalls == 1
    assert after_recall_1.stats.stage_1.last_used  # ISO timestamp present

    # ── Step 4: Recall again with a typo — closest_match recovers ───────
    typo_result = await registry.dispatch_tool(
        fresh_session, "recall_skill", {"label": "csv_sumary"}  # missing 'm'
    )
    assert "⚠" in typo_result
    assert "csv_summary" in typo_result
    assert "pandas.read_csv" in typo_result  # full procedure still returned

    # Counter is now 2 (typo credited to the resolved label)
    after_recall_2 = fresh_store.load("csv_summary")
    assert after_recall_2 is not None
    assert after_recall_2.stats.stage_1.recommended == 2
    assert after_recall_2.stats.total_recalls == 2

    # ── Step 5: Disk verification ───────────────────────────────────────
    skill_dir = store_root / "csv_summary"
    assert skill_dir.is_dir()
    assert (skill_dir / "meta.json").is_file()
    assert (skill_dir / "declarative.md").is_file()
    assert (skill_dir / "stats.json").is_file()

    stats_on_disk = json.loads((skill_dir / "stats.json").read_text())
    assert stats_on_disk["total_recalls"] == 2
    assert stats_on_disk["stage_1"]["recommended"] == 2

    meta_on_disk = json.loads((skill_dir / "meta.json").read_text())
    assert meta_on_disk["label"] == "csv_summary"
    assert meta_on_disk["stage_1_present"] is True
    assert meta_on_disk["stage_2_present"] is False
    assert meta_on_disk["stage_3_present"] is False

    declarative_on_disk = (skill_dir / "declarative.md").read_text()
    assert "pandas.read_csv" in declarative_on_disk
