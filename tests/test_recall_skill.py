"""Tests for the `recall_skill` tool handler."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from anton.core.memory.skills import Skill, SkillStore
from anton.core.tools.recall_skill import (
    RECALL_SKILL_TOOL,
    handle_recall_skill,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def store(tmp_path: Path) -> SkillStore:
    s = SkillStore(root=tmp_path / "skills")
    s.save(
        Skill(
            label="csv_summary",
            name="CSV Summary",
            description="Load a CSV, infer schema, compute summary stats.",
            declarative_md="1. Load the CSV.\n2. Infer types.\n3. Print summary.",
            created_at="2026-04-10T12:00:00+00:00",
            provenance="manual",
        )
    )
    s.save(
        Skill(
            label="web_scraping",
            name="Web Scraping",
            description="Fetch and parse HTML to extract structured data.",
            declarative_md="1. Fetch the URL.\n2. Parse with BeautifulSoup.\n3. Select elements.",
            created_at="2026-04-10T12:00:00+00:00",
            provenance="manual",
        )
    )
    s.save(
        Skill(
            label="api_fetcher",
            name="API Fetcher",
            description="Call a REST API and normalize the response.",
            declarative_md="1. Identify auth.\n2. Call endpoint.\n3. Normalize.",
            created_at="2026-04-10T12:00:00+00:00",
            provenance="manual",
        )
    )
    return s


def _session_with(store: SkillStore) -> SimpleNamespace:
    """Build a minimal session-like object exposing only `_skill_store`."""
    return SimpleNamespace(_skill_store=store)


# ─────────────────────────────────────────────────────────────────────────────
# Tool def basics
# ─────────────────────────────────────────────────────────────────────────────


class TestToolDef:
    def test_tool_name(self):
        assert RECALL_SKILL_TOOL.name == "recall_skill"

    def test_required_label_param(self):
        schema = RECALL_SKILL_TOOL.input_schema
        assert "label" in schema["properties"]
        assert schema["required"] == ["label"]

    def test_handler_is_wired(self):
        assert RECALL_SKILL_TOOL.handler is handle_recall_skill


# ─────────────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────────────


class TestExactMatch:
    @pytest.mark.asyncio
    async def test_returns_procedure(self, store: SkillStore):
        session = _session_with(store)
        result = await handle_recall_skill(session, {"label": "csv_summary"})
        # ENG-2248 tier 1: a served procedure is an explicit success.
        assert result.ok is True
        assert "CSV Summary" in result.content
        assert "Load the CSV" in result.content
        assert "Infer types" in result.content
        assert "Print summary" in result.content

    @pytest.mark.asyncio
    async def test_increments_recommended_counter(self, store: SkillStore):
        session = _session_with(store)
        await handle_recall_skill(session, {"label": "csv_summary"})
        loaded = store.load("csv_summary")
        assert loaded is not None
        assert loaded.stats.stage_1.recommended == 1

    @pytest.mark.asyncio
    async def test_repeated_calls_accumulate(self, store: SkillStore):
        session = _session_with(store)
        for _ in range(3):
            await handle_recall_skill(session, {"label": "web_scraping"})
        loaded = store.load("web_scraping")
        assert loaded is not None
        assert loaded.stats.stage_1.recommended == 3
        assert loaded.stats.total_recalls == 3

    @pytest.mark.asyncio
    async def test_does_not_cross_contaminate(self, store: SkillStore):
        session = _session_with(store)
        await handle_recall_skill(session, {"label": "csv_summary"})
        await handle_recall_skill(session, {"label": "csv_summary"})
        await handle_recall_skill(session, {"label": "web_scraping"})
        csv = store.load("csv_summary")
        web = store.load("web_scraping")
        api = store.load("api_fetcher")
        assert csv is not None and csv.stats.stage_1.recommended == 2
        assert web is not None and web.stats.stage_1.recommended == 1
        assert api is not None and api.stats.stage_1.recommended == 0


# ─────────────────────────────────────────────────────────────────────────────
# Typo fallback
# ─────────────────────────────────────────────────────────────────────────────


class TestTypoFallback:
    @pytest.mark.asyncio
    async def test_typo_returns_closest_match(self, store: SkillStore):
        session = _session_with(store)
        result = await handle_recall_skill(session, {"label": "csv_sumary"})
        # ENG-2248 tier 1: a SUBSTITUTION is still a served request, so the
        # closest-match path is ok=True, warning and all.
        assert result.ok is True
        assert "⚠" in result.content
        assert "csv-summary" in result.content
        # The full procedure is still included after the warning
        assert "Load the CSV" in result.content

    @pytest.mark.asyncio
    async def test_typo_credits_resolved_label_not_input(self, store: SkillStore):
        session = _session_with(store)
        await handle_recall_skill(session, {"label": "csv_sumary"})
        loaded = store.load("csv_summary")
        assert loaded is not None
        assert loaded.stats.stage_1.recommended == 1

    @pytest.mark.asyncio
    async def test_dash_to_underscore_recovered(self, store: SkillStore):
        session = _session_with(store)
        result = await handle_recall_skill(session, {"label": "web__scraping"})
        assert result.ok is True
        assert "web-scraping" in result.content
        # Could match exactly via slugify, in which case there's no warning,
        # or via fuzzy match. Either way the procedure should be returned.
        assert "BeautifulSoup" in result.content


# ─────────────────────────────────────────────────────────────────────────────
# Unknown / error paths
# ─────────────────────────────────────────────────────────────────────────────


class TestUnknownSlug:
    @pytest.mark.asyncio
    async def test_unrelated_returns_no_match_with_listing(
        self, store: SkillStore
    ):
        session = _session_with(store)
        result = await handle_recall_skill(session, {"label": "xyzzy_quark"})
        assert "NO MATCH" in result
        # Should mention all available labels
        assert "csv-summary" in result
        assert "web-scraping" in result
        assert "api-fetcher" in result

    @pytest.mark.asyncio
    async def test_unrelated_does_not_increment_counters(
        self, store: SkillStore
    ):
        session = _session_with(store)
        await handle_recall_skill(session, {"label": "xyzzy_quark"})
        for label in ("csv-summary", "web-scraping", "api-fetcher"):
            loaded = store.load(label)
            assert loaded is not None
            assert loaded.stats.stage_1.recommended == 0
            assert loaded.stats.total_recalls == 0

    @pytest.mark.asyncio
    async def test_empty_label_returns_error(self, store: SkillStore):
        session = _session_with(store)
        result = await handle_recall_skill(session, {"label": ""})
        # ENG-2248 tier 2, and this is the accepted BEHAVIOUR CHANGE: ok=False
        # feeds the per-tool error streak, so two of these in a row now nudge
        # the model and five trip the breaker. Deliberate — a call with no
        # label cannot succeed on retry, so repeating it is thrash.
        assert result.ok is False
        assert result.reason == "missing_name"
        assert "ERROR" in result.content

    @pytest.mark.asyncio
    async def test_missing_label_returns_error(self, store: SkillStore):
        session = _session_with(store)
        result = await handle_recall_skill(session, {})
        assert result.ok is False
        assert result.reason == "missing_name"
        assert "ERROR" in result.content

    @pytest.mark.asyncio
    async def test_no_store_on_session_returns_error(self):
        session = SimpleNamespace()  # no _skill_store
        result = await handle_recall_skill(session, {"label": "csv_summary"})
        # `store_unavailable` is an existing sentinel key (external_wall /
        # service_unavailable), reused rather than invented.
        assert result.ok is False
        assert result.reason == "store_unavailable"
        assert "ERROR" in result.content

    @pytest.mark.asyncio
    async def test_empty_store_unrelated_label(self, tmp_path: Path):
        store = SkillStore(root=tmp_path / "empty_skills")
        session = _session_with(store)
        result = await handle_recall_skill(session, {"label": "anything"})
        assert "NO MATCH" in result
        assert "empty" in result.lower()
