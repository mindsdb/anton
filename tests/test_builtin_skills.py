"""Built-in packaged skills: shipped content, store integration, hint-label drift."""

from pathlib import Path

import pytest

from anton.core.memory import skills as skills_mod
from anton.core.memory.skills import Skill, SkillStore

REAL_BUILTIN_ROOT = (
    Path(skills_mod.__file__).parent / "builtin_skills"
)
SHIPPED_LABELS = {
    "build-fullstack-backend",
    "build-html-dashboard",
    "public-data-sources",
    "cowork-product",
}


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

        data_sources = store.load("public-data-sources")
        assert data_sources is not None
        assert data_sources.provenance == "builtin"
        # spot-check endpoints + single-brace URL params (SKILL.md bodies are
        # not str.format()'d, unlike the CHAT_SYSTEM_PROMPT they moved out of)
        assert "feedparser" in data_sources.declarative_md
        assert "api.worldbank.org/v2/country/{code}/indicator/{indicator}" in data_sources.declarative_md

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
            CHAT_SYSTEM_PROMPT,
            VISUALIZATIONS_HTML_OUTPUT_FORMAT_PROMPT,
            VISUALIZATIONS_MARKDOWN_OUTPUT_FORMAT_PROMPT,
        )

        referenced = self._referenced_labels(
            BACKEND_GENERATION_PROMPT,
            CHAT_SYSTEM_PROMPT,
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


class TestDataCatalogMovedToSkill:
    """The public-data endpoint catalog must live in the recalled skill only,
    not inline in the always-on CHAT_SYSTEM_PROMPT (that duplication is the
    token cost this move removes)."""

    CATALOG_MARKERS = ("api.worldbank.org", "hacker-news.firebaseio.com", "feedparser")

    def test_catalog_not_inline_in_base_prompt(self):
        from anton.core.llm.prompts import CHAT_SYSTEM_PROMPT

        for marker in self.CATALOG_MARKERS:
            assert marker not in CHAT_SYSTEM_PROMPT, f"catalog leaked into base prompt: {marker}"
        assert 'recall_skill("public-data-sources")' in CHAT_SYSTEM_PROMPT

    def test_catalog_lives_in_skill(self, store):
        skill = store.load("public-data-sources")
        assert skill is not None and skill.provenance == "builtin"
        for marker in self.CATALOG_MARKERS:
            assert marker in skill.declarative_md, f"catalog marker missing from skill: {marker}"


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
        # ENG-2248: both paths are successes — the stub is a success with a
        # deliberately short body, not a failure.
        assert first.ok is True
        assert "## Procedure" in first.content
        # Simulate the tool result landing in history. `.content` is what the
        # dispatch loop puts there (`result = outcome.content`), so passing the
        # ToolOutcome itself would make this test diverge from production.
        history.append(
            {"role": "user",
             "content": [{"type": "tool_result", "content": first.content}]}
        )
        second = await handle_recall_skill(session, {"label": "build-html-dashboard"})
        assert second.ok is True
        assert "already recalled" in second.content
        assert "## Procedure" not in second.content

    @pytest.mark.asyncio
    async def test_compacted_history_resends_full_body(self, store):
        from anton.core.tools.recall_skill import handle_recall_skill

        # a summary that mentions the skill but lacks the recall marker
        history = [{"role": "user", "content": "[compacted] recalled build-html-dashboard earlier"}]
        session = self._session(store, history)
        result = await handle_recall_skill(session, {"label": "build-html-dashboard"})
        assert result.ok is True
        assert "## Procedure" in result.content

    @pytest.mark.asyncio
    async def test_surviving_stub_does_not_suppress_resend(self, store):
        """Compaction can evict the full body while keeping a newer stub.
        The stub must not satisfy the already-recalled check, or recall
        returns stubs forever and the contract is never re-sent."""
        from anton.core.tools.recall_skill import handle_recall_skill

        history: list = []
        session = self._session(store, history)
        full = await handle_recall_skill(session, {"label": "build-html-dashboard"})
        history.append(
            {"role": "user",
             "content": [{"type": "tool_result", "content": full.content}]}
        )
        stub = await handle_recall_skill(session, {"label": "build-html-dashboard"})
        assert "already recalled" in stub.content
        # simulate compaction: full body evicted, stub survives
        history.clear()
        history.append(
            {"role": "user",
             "content": [{"type": "tool_result", "content": stub.content}]}
        )
        result = await handle_recall_skill(session, {"label": "build-html-dashboard"})
        assert result.ok is True
        assert "## Procedure" in result.content

    @pytest.mark.asyncio
    async def test_summary_quoting_marker_does_not_suppress_resend(self, store):
        """A compaction summary that quotes the marker text (but not the
        procedure) must not count as the contract being present."""
        from anton.core.tools.recall_skill import _recall_marker, handle_recall_skill

        history = [{"role": "user", "content": f"[compacted] earlier: {_recall_marker('build-html-dashboard')}"}]
        session = self._session(store, history)
        result = await handle_recall_skill(session, {"label": "build-html-dashboard"})
        assert result.ok is True
        assert "## Procedure" in result.content


class TestCoworkProductSkill:
    """ENG-2423 — the facts the agent must not improvise.

    Asserted on content, not just presence: a skill that ships but says the
    wrong thing is the bug, not the fix. Each fact here was read out of the
    code that implements it, and each is one a prod answer got wrong.
    """

    def test_recallable_by_the_label_the_system_prompt_names(self, store):
        """The prompt tells the model to call `recall_skill("cowork-product")`.
        If that label does not resolve, the pointer is dead and the model falls
        back to exactly the priors this ticket exists to stop."""
        skill = store.load("cowork-product")
        assert skill is not None
        assert skill.provenance == "builtin"

    def test_names_the_installed_app_and_the_download_page(self, store):
        """`productName` in cowork's package.json is "MindsHub Cowork" — the
        string a user looks for in Applications. Getting this wrong is what
        "drag ChatGPT to the Trash" was."""
        body = store.load("cowork-product").declarative_md
        assert "MindsHub Cowork" in body
        assert "https://mindshub.ai/download" in body

    def test_carries_every_platform_that_ships_an_installer(self, store):
        """build-installers.yml uploads macOS .pkg, Windows .exe and Linux .deb
        (amd64 + arm64). Omitting Linux would make a correct answer wrong for
        the users who have it."""
        body = store.load("cowork-product").declarative_md
        for platform in ("macOS", "Windows", "Linux"):
            assert platform in body

    def test_states_the_harness_split_in_the_direction_the_code_implements(self, store):
        """SettingsView gates the account-wide Agent Harness group on
        `host.isWeb`, and Coding Mode (where desktop picks a harness) is
        Electron-only. A fact sheet with this backwards would send the desktop
        user who filed this ticket hunting for a setting that is not there."""
        body = store.load("cowork-product").declarative_md
        assert "Settings → Agent Harness" in body
        assert "browser only" in body
        assert "Coding Mode: **desktop only**" in body

    def test_forbids_answering_from_general_knowledge(self, store):
        body = store.load("cowork-product").declarative_md
        assert "Never answer a question about Cowork from general" in body
        # And says what to do instead of inventing a missing fact.
        assert "say you are not certain" in body


class TestPromptAndSkillAgree:
    """The always-on block and the on-demand skill must state the same facts.

    They are two sources for one truth, and they have already drifted once: the
    skill said macOS/Windows/Linux while the prompt line said macOS/Windows,
    with Linux `.deb`s shipping on every release. That drift matters more in
    one direction than the other — the prompt block ships unconditionally, the
    skill only if the model recalls it — so the wrong fact was the one with the
    higher authority, which is ENG-2423's own failure mode reproduced with our
    fact instead of the base model's prior.
    """

    PLATFORMS = ("macOS", "Windows", "Linux")

    def _named(self, text: str) -> set[str]:
        low = text.lower()
        return {p for p in self.PLATFORMS if p.lower() in low}

    def test_platform_lists_match(self, store):
        from anton.core.llm.identity import product_lines

        in_prompt = self._named("\n".join(product_lines("desktop")))
        in_skill = self._named(store.load("cowork-product").declarative_md)

        assert in_prompt == in_skill, (
            f"prompt names {sorted(in_prompt)}, skill names {sorted(in_skill)} — "
            "the always-on block is the one users meet first, so it must not be "
            "the narrower of the two"
        )

    def test_both_name_every_platform_that_ships_an_installer(self, store):
        """Pinned to the build, not just to each other: two sources agreeing on
        a wrong list would pass the test above."""
        from anton.core.llm.identity import product_lines

        for source in (
            "\n".join(product_lines("desktop")),
            store.load("cowork-product").declarative_md,
        ):
            assert self._named(source) == set(self.PLATFORMS)

    def test_both_name_the_product_and_its_maker(self, store):
        from anton.core.llm.identity import product_lines

        for source in (
            "\n".join(product_lines("web")),
            store.load("cowork-product").declarative_md,
        ):
            assert "Cowork" in source
            assert "MindsHub" in source
