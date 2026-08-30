"""Built-in packaged skills: shipped content, store integration, hint-label drift."""

from pathlib import Path

import pytest

from anton.core.memory import skills as skills_mod
from anton.core.memory.skills import Skill, SkillStore

REAL_BUILTIN_ROOT = (
    Path(skills_mod.__file__).parent / "builtin_skills"
)
SHIPPED_LABELS = {"build-fullstack-backend", "build-html-dashboard", "public-data-sources"}


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

    @pytest.mark.asyncio
    async def test_surviving_stub_does_not_suppress_resend(self, store):
        """Compaction can evict the full body while keeping a newer stub.
        The stub must not satisfy the already-recalled check, or recall
        returns stubs forever and the contract is never re-sent."""
        from anton.core.tools.recall_skill import handle_recall_skill

        history: list = []
        session = self._session(store, history)
        full = await handle_recall_skill(session, {"label": "build-html-dashboard"})
        history.append({"role": "user", "content": [{"type": "tool_result", "content": full}]})
        stub = await handle_recall_skill(session, {"label": "build-html-dashboard"})
        assert "already recalled" in stub
        # simulate compaction: full body evicted, stub survives
        history.clear()
        history.append({"role": "user", "content": [{"type": "tool_result", "content": stub}]})
        result = await handle_recall_skill(session, {"label": "build-html-dashboard"})
        assert "## Procedure" in result

    @pytest.mark.asyncio
    async def test_summary_quoting_marker_does_not_suppress_resend(self, store):
        """A compaction summary that quotes the marker text (but not the
        procedure) must not count as the contract being present."""
        from anton.core.tools.recall_skill import _recall_marker, handle_recall_skill

        history = [{"role": "user", "content": f"[compacted] earlier: {_recall_marker('build-html-dashboard')}"}]
        session = self._session(store, history)
        result = await handle_recall_skill(session, {"label": "build-html-dashboard"})
        assert "## Procedure" in result


class TestArtifactWorkflowPointsAtTheTool:
    """The always-on section must name the normal path.

    Before this change it described only manual assembly, even though
    `generate_artifact` is registered alongside the other artifact tools whenever a
    workspace is bound to the session (`session.py:940-947`), and its own
    `ToolDef.prompt` demands using it INSTEAD of writing files by hand. The agent
    received two mutually exclusive instructions.
    """

    def test_workflow_names_the_generator_for_supported_types(self):
        from anton.core.llm.prompts import ARTIFACTS_PROMPT

        assert "generate_artifact" in ARTIFACTS_PROMPT
        for t in ("html-app", "fullstack-stateless-app", "fullstack-stateful-app"):
            assert t in ARTIFACTS_PROMPT

    def test_workflow_keeps_the_manual_path_for_the_other_types(self):
        """The generator does not support document/dataset/image — their path stays manual."""
        from anton.core.llm.prompts import ARTIFACTS_PROMPT

        low = ARTIFACTS_PROMPT.lower()
        assert "document" in low and "dataset" in low
        assert "yourself" in low or "by hand" in low

    def test_workflow_still_requires_registration_first(self):
        """create_artifact is still the first step — the generator takes a ready slug."""
        from anton.core.llm.prompts import ARTIFACTS_PROMPT

        assert "create_artifact" in ARTIFACTS_PROMPT
        assert "BEFORE" in ARTIFACTS_PROMPT


class TestNoPathStillNamesASeparatePrdStep:
    """The two-step pipeline is gone, and so is the lock that guarded its
    order (invariant 13).

    `generate_prd` was a separate tool with no `ToolDef.prompt` of its own, so
    these always-on blocks were the only place the model could learn the step
    existed — which is why an ordering lock was worth having. There is one
    tool now and it agrees the requirements itself, so a block still naming a
    PRD step would be sending the agent to call something that does not exist.
    """

    def _blocks(self) -> dict[str, str]:
        from anton.core.llm.prompts import (
            ARTIFACTS_PROMPT,
            BACKEND_GENERATION_PROMPT,
            VISUALIZATIONS_HTML_OUTPUT_FORMAT_PROMPT,
            VISUALIZATIONS_MARKDOWN_OUTPUT_FORMAT_PROMPT,
        )

        return {
            "artifacts": ARTIFACTS_PROMPT,
            "backend": BACKEND_GENERATION_PROMPT,
            "viz_html": VISUALIZATIONS_HTML_OUTPUT_FORMAT_PROMPT,
            "viz_markdown": VISUALIZATIONS_MARKDOWN_OUTPUT_FORMAT_PROMPT,
        }

    def test_no_block_sends_the_agent_to_a_tool_that_does_not_exist(self):
        for name, text in self._blocks().items():
            assert "generate_prd" not in text, name

    def test_every_block_still_names_the_generator(self):
        for name, text in self._blocks().items():
            assert "generate_artifact" in text, name

    def test_the_artifacts_block_says_the_tool_agrees_the_requirements_itself(self):
        """The agent must not pre-interview the user for something the tool
        will ask about anyway, nor write a PRD by hand."""
        from anton.core.llm.prompts import ARTIFACTS_PROMPT

        assert "agrees a short brief" in ARTIFACTS_PROMPT
        assert "Do NOT write a PRD yourself" in ARTIFACTS_PROMPT


class TestRecallIsConditionalNow:
    """`recall_skill` stays, but stops being an unconditional instruction."""

    def _texts(self):
        from anton.core.llm.prompts import (
            BACKEND_GENERATION_PROMPT,
            VISUALIZATIONS_HTML_OUTPUT_FORMAT_PROMPT,
            VISUALIZATIONS_MARKDOWN_OUTPUT_FORMAT_PROMPT,
        )

        return {
            "backend": BACKEND_GENERATION_PROMPT,
            "viz_html": VISUALIZATIONS_HTML_OUTPUT_FORMAT_PROMPT,
            "viz_markdown": VISUALIZATIONS_MARKDOWN_OUTPUT_FORMAT_PROMPT,
        }

    def test_no_unconditional_mandatory_recall(self):
        """"MANDATORY: call recall_skill …" is the very instruction that
        conflicted with generate_artifact's ToolDef.prompt."""
        for name, text in self._texts().items():
            assert "MANDATORY: call `recall_skill" not in text, name
            assert "MANDATORY: BEFORE writing" not in text, name

    def test_the_generator_is_named_as_the_normal_path(self):
        for name, text in self._texts().items():
            assert "generate_artifact" in text, name

    def test_recall_is_tied_to_the_manual_path(self):
        """The skill is not discarded — manual edits and unsupported types need it."""
        for name, text in self._texts().items():
            assert "recall_skill" in text, name
            low = text.lower()
            assert "by hand" in low or "yourself" in low, name

    def test_hints_still_point_at_shipped_skills(self, store):
        """The rewording must not leave a pointer to a non-existent skill."""
        import re

        for text in self._texts().values():
            for label in re.findall(r'recall_skill\(\\?"([a-z0-9-]+)\\?"\)', text):
                assert store.load(label) is not None, label


class TestToolDescriptionsAfterTheSwitch:
    def test_create_artifact_no_longer_demands_recall_before_any_write(self):
        from anton.core.tools.tool_defs import CREATE_ARTIFACT_TOOL

        d = CREATE_ARTIFACT_TOOL.description
        assert "SKILL PREREQUISITE" not in d
        assert "BEFORE writing any files" not in d
        assert "generate_artifact" in d

    def test_create_artifact_keeps_a_conditional_recall_hint(self):
        """The manual path does use the skill, and
        test_tool_descriptions_reference_shipped_skills depends on this mention."""
        from anton.core.tools.tool_defs import CREATE_ARTIFACT_TOOL

        d = CREATE_ARTIFACT_TOOL.description
        assert "recall_skill" in d
        assert "by hand" in d.lower() or "yourself" in d.lower()

    def test_launch_backend_says_the_generator_launches_on_its_own(self):
        from anton.core.tools.tool_defs import LAUNCH_BACKEND_TOOL

        d = LAUNCH_BACKEND_TOOL.description
        assert "generate_artifact" in d
        assert "If you haven't recalled it this conversation" not in d


class TestSkillDescriptionsSteerAway:
    """The description is all the thalamus sees when deciding to preload a skill.

    `_inject_recalled_skills` puts the FULL body of up to three skills into the
    context per turn — i.e. ~270 lines of "do it by hand" land in front of the
    agent before it picks a path. So the description must open with the
    applicability condition, not with "MANDATORY reading before ANY…".
    """

    LABELS = ("build-html-dashboard", "build-fullstack-backend")

    def test_no_mandatory_trigger_words(self, store):
        for label in self.LABELS:
            d = store.load(label).description
            assert "MANDATORY" not in d, label
            assert "before writing" not in d.lower(), label
            assert "when in doubt" not in d.lower(), label

    def test_description_opens_with_the_applicability_condition(self, store):
        """The condition must come FIRST, not somewhere inside the first sentence.

        A weak assert ("by hand" anywhere in the first sentence) also passes for a
        topic-led wording: "For building a dashboard, chart or browser
        visualization BY HAND …" — there the first ~56 characters are pure bait for
        matching, while the catalog's framing text reads "When the user's request
        matches one of them, call recall_skill(label)". So we pin the prefix.
        """
        for label in self.LABELS:
            d = store.load(label).description
            assert d.startswith("ONLY "), f"{label}: {d[:50]!r}"
            assert "by hand" in d[:55].lower(), f"{label}: {d[:55]!r}"

    def test_normal_path_disclaimer_precedes_the_contents(self, store):
        """"NOT needed on the normal path" must come ABOVE the contents listing.

        Otherwise the description advertises the contract first and hides the
        disclaimer in the tail.
        """
        markers = {
            "build-html-dashboard": "ECharts",
            "build-fullstack-backend": "FastAPI",
        }
        for label, content_marker in markers.items():
            d = store.load(label).description
            assert d.index("NOT needed") < d.index(content_marker), label

    def test_description_points_at_the_generator(self, store):
        for label in self.LABELS:
            assert "generate_artifact" in store.load(label).description, label

    def test_description_stays_findable_for_the_fallback(self, store):
        """Steering away is not hiding: the contents must stay recognisable."""
        html = store.load("build-html-dashboard").description
        assert "dashboard" in html.lower() and "chart" in html.lower()
        backend = store.load("build-fullstack-backend").description
        assert "backend" in backend.lower() and "fastapi" in backend.lower()

    def test_descriptions_still_fit_the_budget(self, store):
        for label in self.LABELS:
            assert len(store.load(label).description) <= 1024, label

    def test_skill_bodies_are_untouched(self, store):
        """Design decision 1: bodies are not rewritten, the manual path still needs them."""
        html = store.load("build-html-dashboard").declarative_md
        assert "REROUND DISCIPLINE" in html
        assert "cdn.jsdelivr.net/npm/echarts" in html
        backend = store.load("build-fullstack-backend").declarative_md
        assert "handler = Mangum(app, lifespan=\"off\")" in backend
        assert "DS_<ENGINE>_<NAME>__<FIELD>" in backend
