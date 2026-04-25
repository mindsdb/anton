from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from anton.core.memory.cortex import Cortex
from anton.core.memory.hippocampus import Engram, Hippocampus


@pytest.fixture()
def dirs(tmp_path):
    g = tmp_path / "global"
    p = tmp_path / "project"
    g.mkdir()
    p.mkdir()
    return g, p


@pytest.fixture()
def cortex(dirs):
    g, p = dirs
    return Cortex(global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="copilot")


class TestBuildMemoryContext:
    async def test_empty_returns_empty(self, cortex):
        assert await cortex.build_memory_context() == ""

    async def test_includes_identity(self, cortex, dirs):
        g, _ = dirs
        Hippocampus(g).rewrite_identity(["Name: Jorge", "TZ: PST"])
        result = await cortex.build_memory_context()
        assert "Identity" in result
        assert "Name: Jorge" in result

    async def test_includes_global_rules(self, cortex, dirs):
        g, _ = dirs
        Hippocampus(g).encode_rule("Use httpx", kind="always", confidence="high", source="user")
        result = await cortex.build_memory_context()
        assert "Global Rules" in result
        assert "Use httpx" in result

    async def test_includes_project_rules(self, cortex, dirs):
        _, p = dirs
        Hippocampus(p).encode_rule("Use Django ORM", kind="always", confidence="high", source="user")
        result = await cortex.build_memory_context()
        assert "Project Rules" in result
        assert "Use Django ORM" in result

    async def test_includes_lessons(self, cortex, dirs):
        g, p = dirs
        Hippocampus(g).encode_lesson("Global fact")
        Hippocampus(p).encode_lesson("Project fact")
        result = await cortex.build_memory_context()
        assert "Global Lessons" in result
        assert "Project Lessons" in result
        assert "Global fact" in result
        assert "Project fact" in result


class TestGetScratchpadContext:
    def test_empty_returns_empty(self, cortex):
        assert cortex.get_scratchpad_context() == ""

    def test_combines_scopes(self, cortex, dirs):
        g, p = dirs
        (g / "rules.md").write_text("# Rules\n\n## Always\n\n## Never\n\n## When\n- If slow → batch\n")
        (p / "lessons.md").write_text("# Lessons\n- Scratchpad times out at 30s\n")
        result = cortex.get_scratchpad_context()
        assert "slow" in result
        assert "Scratchpad times out" in result


class TestEncode:
    async def test_encode_rule_to_project(self, cortex, dirs):
        _, p = dirs
        engram = Engram(text="Use httpx", kind="always", scope="project", confidence="high")
        actions = await cortex.encode([engram])
        assert any("always rule" in a.lower() for a in actions)
        assert (p / "rules.md").exists()

    async def test_encode_lesson_to_global(self, cortex, dirs):
        g, _ = dirs
        engram = Engram(text="CoinGecko rate limit", kind="lesson", scope="global", topic="api")
        actions = await cortex.encode([engram])
        assert any("lesson" in a.lower() for a in actions)
        assert (g / "lessons.md").exists()

    async def test_encode_profile(self, cortex, dirs):
        g, _ = dirs
        engram = Engram(text="Name: Jorge", kind="profile", scope="global", confidence="high")
        actions = await cortex.encode([engram])
        assert any("identity" in a.lower() for a in actions)
        assert (g / "profile.md").exists()
        assert "Name: Jorge" in (g / "profile.md").read_text()

    async def test_encode_profile_with_project_scope_routes_to_global(self, cortex, dirs):
        g, p = dirs
        engram = Engram(text="Name: Jorge", kind="profile", scope="project")
        await cortex.encode([engram])
        assert (g / "profile.md").exists()
        assert "Name: Jorge" in (g / "profile.md").read_text()
        assert not (p / "profile.md").exists()

    async def test_off_mode_returns_disabled(self, dirs):
        g, p = dirs
        cortex = Cortex(global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="off")
        engram = Engram(text="test", kind="lesson", scope="global")
        actions = await cortex.encode([engram])
        assert any("disabled" in a.lower() for a in actions)


class TestOrphanedIdentityMigration:
    def test_migrates_project_identity_to_global_on_init(self, dirs):
        g, p = dirs
        # Simulate orphaned state from the old bug: identity entries in project scope.
        Hippocampus(p).rewrite_identity(["Name: Jorge", "TZ: PST"])
        assert (p / "profile.md").exists()

        Cortex(global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="copilot")

        assert (g / "profile.md").exists()
        merged = (g / "profile.md").read_text()
        assert "Name: Jorge" in merged
        assert "TZ: PST" in merged
        assert not (p / "profile.md").exists()

    def test_migration_does_not_overwrite_fresh_global_entries(self, dirs):
        # Orphaned project data is likely stale (old bug wrote it, user may
        # have since corrected to global). Global must win on key conflicts.
        g, p = dirs
        Hippocampus(g).rewrite_identity(["Name: Alejandro"])
        Hippocampus(p).rewrite_identity(["Name: Alec", "TZ: PST"])

        Cortex(global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="copilot")

        merged = (g / "profile.md").read_text()
        assert "Name: Alejandro" in merged
        assert "Name: Alec" not in merged
        assert "TZ: PST" in merged  # non-conflicting keys still migrate
        assert not (p / "profile.md").exists()

    def test_migration_noop_when_project_identity_empty(self, dirs):
        g, p = dirs
        Cortex(global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="copilot")
        assert not (g / "profile.md").exists()
        assert not (p / "profile.md").exists()


class TestEncodingGate:
    def test_autopilot_never_confirms(self, dirs):
        g, p = dirs
        cortex = Cortex(global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="autopilot")
        engram = Engram(text="test", kind="lesson", scope="global", confidence="low")
        assert cortex.encoding_gate(engram) is False

    def test_off_never_confirms(self, dirs):
        g, p = dirs
        cortex = Cortex(global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="off")
        engram = Engram(text="test", kind="lesson", scope="global", confidence="high")
        assert cortex.encoding_gate(engram) is False

    def test_copilot_confirms_low_confidence(self, dirs):
        g, p = dirs
        cortex = Cortex(global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="copilot")
        low = Engram(text="test", kind="lesson", scope="global", confidence="medium")
        high = Engram(text="test", kind="lesson", scope="global", confidence="high")
        assert cortex.encoding_gate(low) is True
        assert cortex.encoding_gate(high) is False


class TestNeedsCompaction:
    def test_below_threshold(self, cortex):
        assert cortex.needs_compaction() is False

    def test_above_threshold(self, cortex, dirs):
        g, _ = dirs
        hc = Hippocampus(g)
        for i in range(55):
            hc.encode_lesson(f"Fact number {i}")
        assert cortex.needs_compaction() is True


class TestMaybeUpdateIdentity:
    async def test_no_llm_does_nothing(self, cortex, dirs):
        # cortex has no LLM by default in fixture
        await cortex.maybe_update_identity("I'm Jorge")
        g, _ = dirs
        assert not (g / "profile.md").exists()

    async def test_off_mode_does_nothing(self, dirs):
        g, p = dirs
        mock_llm = AsyncMock()
        cortex = Cortex(global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="off", llm_client=mock_llm)
        await cortex.maybe_update_identity("I'm Jorge")
        mock_llm.generate_object_code.assert_not_called()

    async def test_extracts_identity(self, dirs):
        from anton.core.memory.cortex import _IdentityFacts

        g, p = dirs
        mock_llm = AsyncMock()
        mock_llm.generate_object_code = AsyncMock(
            return_value=_IdentityFacts(facts=["Name: Jorge"])
        )
        cortex = Cortex(global_hc=Hippocampus(g), project_hc=Hippocampus(p), mode="copilot", llm_client=mock_llm)
        await cortex.maybe_update_identity("Hi, I'm Jorge")
        assert (g / "profile.md").exists()
        assert "Name: Jorge" in (g / "profile.md").read_text()
