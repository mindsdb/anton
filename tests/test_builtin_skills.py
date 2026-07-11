"""Built-in packaged skills: shipped content, store integration, hint-label drift."""

from pathlib import Path

import pytest

from anton.core.memory import skills as skills_mod
from anton.core.memory.skills import Skill, SkillStore

REAL_BUILTIN_ROOT = (
    Path(skills_mod.__file__).parent / "builtin_skills"
)
SHIPPED_LABELS = {"build-fullstack-backend", "build-html-dashboard"}


@pytest.fixture()
def store(tmp_path) -> SkillStore:
    """Store with an empty user root and the REAL packaged builtin root."""
    return SkillStore(root=tmp_path / "user-skills", builtin_root=REAL_BUILTIN_ROOT)


class TestShippedSkills:
    def test_shipped_skills_present(self, store):
        labels = {s["label"] for s in store.list_summaries()}
        assert SHIPPED_LABELS <= labels

    def test_load_returns_full_body(self, store):
        backend = store.load("build-fullstack-backend")
        assert backend is not None
        assert backend.provenance == "builtin"
        # spot-check load-bearing contract pieces survived extraction
        assert "Mangum(app, lifespan=\"off\")" in backend.declarative_md
        assert "/api/" in backend.declarative_md
        assert "requirements.txt" in backend.declarative_md

        dashboard = store.load("build-html-dashboard")
        assert dashboard is not None
        assert dashboard.provenance == "builtin"
        assert len(dashboard.declarative_md) > 5_000

    def test_descriptions_fit_prompt_budget(self, store):
        for summary in store.list_summaries():
            assert len(summary["description"]) <= 1024


class TestStoreIntegration:
    def test_user_skill_shadows_builtin(self, store):
        store.save(
            Skill(
                label="build-fullstack-backend",
                name="My Override",
                description="user override",
                declarative_md="user body",
                created_at="",
                provenance="manual",
            )
        )
        loaded = store.load("build-fullstack-backend")
        assert loaded.provenance == "manual"
        assert loaded.declarative_md == "user body"
        # no duplicate rows in listings
        labels = [s["label"] for s in store.list_summaries()]
        assert labels.count("build-fullstack-backend") == 1

    def test_delete_builtin_is_noop(self, store):
        assert store.delete("build-fullstack-backend") is False
        assert store.load("build-fullstack-backend") is not None

    def test_increment_recommended_is_safe(self, store):
        # stats live in the (read-only) builtin dir only for user skills;
        # builtins must no-op without raising or writing.
        store.increment_recommended("build-fullstack-backend")
        assert not (REAL_BUILTIN_ROOT / "build-fullstack-backend" / "stats.json").exists()

    def test_list_all_includes_builtins_sorted(self, store):
        labels = [s.label for s in store.list_all()]
        assert labels == sorted(labels)
        assert SHIPPED_LABELS <= set(labels)

    def test_missing_builtin_root_is_harmless(self, tmp_path):
        s = SkillStore(root=tmp_path / "u", builtin_root=tmp_path / "nope")
        assert s.list_summaries() == []
        assert s.load("build-fullstack-backend") is None


class TestHintLabelDrift:
    """The prompts/tool descriptions reference skills by label; those labels
    must exist as shipped built-ins or the recall hint dead-ends."""

    def _referenced_labels(self, *texts: str) -> set[str]:
        import re

        out: set[str] = set()
        for t in texts:
            out.update(re.findall(r'recall_skill\(\\?"([a-z0-9-]+)\\?"\)', t))
        return out

    def test_prompt_hints_reference_shipped_skills(self, store):
        from anton.core.llm.prompts import (
            BACKEND_GENERATION_PROMPT,
            VISUALIZATIONS_HTML_OUTPUT_FORMAT_PROMPT,
            VISUALIZATIONS_MARKDOWN_OUTPUT_FORMAT_PROMPT,
        )

        referenced = self._referenced_labels(
            BACKEND_GENERATION_PROMPT,
            VISUALIZATIONS_HTML_OUTPUT_FORMAT_PROMPT,
            VISUALIZATIONS_MARKDOWN_OUTPUT_FORMAT_PROMPT,
        )
        assert referenced  # hints actually exist
        for label in referenced:
            assert store.load(label) is not None, f"hinted skill missing: {label}"

    def test_tool_descriptions_reference_shipped_skills(self, store):
        from anton.core.tools.tool_defs import CREATE_ARTIFACT_TOOL, LAUNCH_BACKEND_TOOL

        referenced = self._referenced_labels(
            CREATE_ARTIFACT_TOOL.description, LAUNCH_BACKEND_TOOL.description
        )
        assert referenced
        for label in referenced:
            assert store.load(label) is not None, f"hinted skill missing: {label}"


class TestBrokenShadowFallback:
    def test_unreadable_user_dir_falls_back_to_builtin(self, store):
        bad = store.root / "build-fullstack-backend"
        bad.mkdir(parents=True)
        (bad / "SKILL.md").write_text("not: [valid: yaml\nno frontmatter fence")
        loaded = store.load("build-fullstack-backend")
        assert loaded is not None
        assert loaded.provenance == "builtin"
        assert "Mangum" in loaded.declarative_md

    def test_unreadable_dir_without_builtin_still_none(self, tmp_path):
        s = SkillStore(root=tmp_path / "u", builtin_root=tmp_path / "nope")
        bad = s.root / "some-skill"
        bad.mkdir(parents=True)
        (bad / "SKILL.md").write_text("garbage")
        assert s.load("some-skill") is None


class TestRecallIdempotence:
    def _session(self, store, history):
        from types import SimpleNamespace

        return SimpleNamespace(_skill_store=store, history=history)

    @pytest.mark.asyncio
    async def test_second_recall_returns_stub(self, store):
        from anton.core.tools.recall_skill import handle_recall_skill

        history: list = []
        session = self._session(store, history)
        first = await handle_recall_skill(session, {"label": "build-html-dashboard"})
        assert "## Procedure" in first
        # simulate the tool result landing in history
        history.append({"role": "user", "content": [{"type": "tool_result", "content": first}]})
        second = await handle_recall_skill(session, {"label": "build-html-dashboard"})
        assert "already recalled" in second
        assert "## Procedure" not in second

    @pytest.mark.asyncio
    async def test_compacted_history_resends_full_body(self, store):
        from anton.core.tools.recall_skill import handle_recall_skill

        # a summary that mentions the skill but lacks the recall marker
        history = [{"role": "user", "content": "[compacted] recalled build-html-dashboard earlier"}]
        session = self._session(store, history)
        result = await handle_recall_skill(session, {"label": "build-html-dashboard"})
        assert "## Procedure" in result
