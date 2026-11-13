"""edit_artifact: the tool, its place in the session, and the routing prompts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from anton.core.artifact_editor.records import LOG_ENV_VAR
from anton.core.artifacts import ArtifactStore
from anton.core.settings import CoreSettings
from anton.core.tools.edit_artifact import (
    SECTION_ROUTES,
    editor_enabled,
    handle_edit_artifact,
    route_section_to_editor,
)
from anton.core.tools.progress import ToolProgress
from anton.core.tools.registry import ToolOutcome, ToolRegistry
from anton.core.tools.tool_handlers import PageKey, editor_offered

from tests.test_artifact_editor_loop import COLOR_PATCH, PAGE, BAD_PATCH, ScriptedLLM, diff, reply


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv(LOG_ENV_VAR, str(tmp_path / "record.jsonl"))
    monkeypatch.delenv("ANTON_RENDER_CHECK_URL", raising=False)
    monkeypatch.delenv("ANTON_ARTIFACT_EDITOR", raising=False)


def _session(tmp_path, llm, **extra):
    from anton.core.tools.edit_artifact import EDIT_ARTIFACT_TOOL

    registry = ToolRegistry()
    registry.register_tool(EDIT_ARTIFACT_TOOL)
    return SimpleNamespace(
        _workspace=SimpleNamespace(artifacts_dir=tmp_path / "artifacts"), _llm=llm, _settings=None,
        _page_checks={}, _repair_cells={}, _out_of_repairs=set(), _artifacts_touched=set(), _session_id="conv-1",
        _turn_count=0, _failed_views={}, _unverified_findings_listed=set(), tool_registry=registry, **extra,
    )


def _artifact(tmp_path, page=PAGE):
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = store.create(name="Sales", description="sales page", type="html-app", primary="index.html")
    (store.folder_for(artifact.slug) / "index.html").write_text(page)
    return store, artifact.slug


async def _drive(session, tc_input):
    progress, final = [], None
    async for item in handle_edit_artifact(session, tc_input):
        if isinstance(item, ToolProgress):
            progress.append(item.text)
        else:
            final = item
    return progress, final


async def test_edit_lands_in_one_call_and_says_it_is_unverified_without_a_browser(tmp_path):
    store, slug = _artifact(tmp_path)
    session = _session(tmp_path, ScriptedLLM([reply(diff(COLOR_PATCH, note="Heading is navy."))]))
    progress, outcome = await _drive(session, {"slug": slug, "task": "make the heading navy"})

    assert isinstance(outcome, ToolOutcome) and outcome.ok is True
    assert "in 1 model call" in outcome.content and "Heading is navy." in outcome.content
    assert "not verified in a browser" in outcome.content
    assert "color: navy" in (store.folder_for(slug) / "index.html").read_text()
    assert slug in session._artifacts_touched
    assert session._page_checks[PageKey(slug, "index.html")] == "unverified"
    assert any("thought" in line for line in progress)


async def test_failed_edit_puts_the_files_back_and_says_so(tmp_path):
    store, slug = _artifact(tmp_path)
    session = _session(tmp_path, ScriptedLLM([reply(diff(COLOR_PATCH, complete=False)), RuntimeError("boom")]))
    _, outcome = await _drive(session, {"slug": slug, "task": "navy"})
    assert outcome.ok is False and outcome.reason == "edit_not_applied"
    assert "back as it was" in outcome.content
    assert (store.folder_for(slug) / "index.html").read_text() == PAGE
    assert slug not in session._artifacts_touched


async def test_an_answer_changes_nothing(tmp_path):
    store, slug = _artifact(tmp_path)
    session = _session(tmp_path, ScriptedLLM([reply(text="There is no region column.", stop="stop")]))
    _, outcome = await _drive(session, {"slug": slug, "task": "add a region filter"})
    assert outcome.ok is True and "changed nothing" in outcome.content
    assert (store.folder_for(slug) / "index.html").read_text() == PAGE


async def test_repair_of_a_failing_page_spends_a_repair_cell(tmp_path):
    store, slug = _artifact(tmp_path)
    session = _session(tmp_path, ScriptedLLM([reply(diff(COLOR_PATCH))]))
    key = PageKey(slug, "index.html")
    session._page_checks[key] = "failed"
    await _drive(session, {"slug": slug, "task": "fix it"})
    assert session._repair_cells[key] == 1


@pytest.mark.parametrize("tc_input, reason", [
    ({"slug": "", "task": "x"}, "missing_slug"),
    ({"slug": "nope", "task": ""}, "missing_task"),
    ({"slug": "nope", "task": "x"}, "artifact_not_found"),
])
async def test_bad_arguments(tmp_path, tc_input, reason):
    _artifact(tmp_path)
    _, outcome = await _drive(_session(tmp_path, ScriptedLLM([])), tc_input)
    assert outcome.ok is False and outcome.reason == reason


async def test_a_page_out_of_repairs_is_refused(tmp_path):
    _, slug = _artifact(tmp_path)
    session = _session(tmp_path, ScriptedLLM([]))
    session._out_of_repairs.add(PageKey(slug, "index.html"))
    _, outcome = await _drive(session, {"slug": slug, "task": "x"})
    assert outcome.ok is False and outcome.reason == "out_of_repairs"


async def test_rejected_patch_then_fix(tmp_path):
    store, slug = _artifact(tmp_path)
    session = _session(tmp_path, ScriptedLLM([reply(diff(BAD_PATCH)), reply(diff(COLOR_PATCH))]))
    _, outcome = await _drive(session, {"slug": slug, "task": "navy"})
    assert outcome.ok is True and "in 2 model calls" in outcome.content


# ── flag and routing ─────────────────────────────────────────────────


def test_flag(monkeypatch):
    assert editor_enabled(CoreSettings(artifact_editor="on"))
    assert not editor_enabled(CoreSettings(artifact_editor="off"))
    monkeypatch.setenv("ANTON_ARTIFACT_EDITOR", "off")
    assert not editor_enabled(None)
    assert not editor_enabled(CoreSettings())


def test_every_artifacts_section_routes_edits_to_the_editor():
    from anton.core.llm import prompts
    from anton.core.llm.static_artifact_prompt import ARTIFACTS, STATIC_ARTIFACTS

    sections = [ARTIFACTS, STATIC_ARTIFACTS, prompts.ARTIFACTS_PROMPT, prompts.NO_DIRECT_REPORTS_ARTIFACTS_PROMPT,
                prompts.STATIC_ARTIFACTS_PROMPT, prompts.NO_DIRECT_REPORTS_STATIC_ARTIFACTS_PROMPT]
    for section in sections:
        routed = route_section_to_editor(section)
        assert routed != section
        assert "edit_artifact" in routed
        for old, _new in SECTION_ROUTES:
            assert old not in routed


def _tool_names(session) -> list[str]:
    session._build_tools()
    return [tool.name for tool in session.tool_registry.get_tool_defs()]


async def test_session_offers_the_editor_and_routes_generate(make_session, tmp_path):
    from anton.workspace import Workspace

    workspace = Workspace(tmp_path)
    on = make_session(workspace=workspace, settings=CoreSettings(artifact_editor="on"))
    names = _tool_names(on)
    assert "edit_artifact" in names and "generate_artifact" in names
    generate = next(t for t in on.tool_registry.get_tool_defs() if t.name == "generate_artifact")
    assert "call edit_artifact instead" in generate.description
    assert editor_offered(on)
    prompt = await on._build_system_prompt("make the chart blue")
    assert "edit_artifact(slug, task)" in prompt
    assert "EDITING AN EXISTING ARTIFACT" in prompt

    off = make_session(workspace=workspace, settings=CoreSettings(artifact_editor="off"))
    assert "edit_artifact" not in _tool_names(off)
    assert not editor_offered(off)


async def test_browser_check_failed_routes_the_repair_to_the_editor(tmp_path):
    import json

    from anton.core.tools.generate_artifact.state import BrowserCheckFailed
    from anton.core.tools.tool_handlers import _browser_check_failed_outcome

    store, slug = _artifact(tmp_path)
    session = _session(tmp_path, ScriptedLLM([]))
    result = BrowserCheckFailed(page="index.html", findings=("uncaught_exception: boom",), views=(),
                                screenshot=None, error="browser check failed")
    outcome = await _browser_check_failed_outcome(session, store=store, slug=slug,
                                                  folder=store.folder_for(slug), result=result)
    body = json.loads(outcome.content)
    assert "edit_artifact(slug, task)" in body["instruction"]


async def test_the_artifact_may_be_named_instead_of_its_slug(tmp_path):
    store, slug = _artifact(tmp_path)
    session = _session(tmp_path, ScriptedLLM([reply(diff(COLOR_PATCH))]))
    _, outcome = await _drive(session, {"slug": "sales", "task": "navy"})
    assert outcome.ok is True and slug in outcome.content
    assert "color: navy" in (store.folder_for(slug) / "index.html").read_text()


async def test_an_ambiguous_name_lists_the_candidates(tmp_path):
    _artifact(tmp_path)
    _artifact(tmp_path)
    _, outcome = await _drive(_session(tmp_path, ScriptedLLM([])), {"slug": "Sales", "task": "navy"})
    assert outcome.ok is False and "matches several artifacts" in outcome.content


async def test_a_loose_name_finds_the_artifact_and_a_wrong_one_lists_them(tmp_path):
    store, slug = _artifact(tmp_path)
    session = _session(tmp_path, ScriptedLLM([reply(diff(COLOR_PATCH))]))
    _, outcome = await _drive(session, {"slug": "the Sales report", "task": "navy"})
    assert outcome.ok is True
    _, outcome = await _drive(_session(tmp_path, ScriptedLLM([])), {"slug": "Inventory", "task": "navy"})
    assert outcome.ok is False and f"`{slug}` (Sales)" in outcome.content
