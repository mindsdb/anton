from types import SimpleNamespace

import pytest

from anton.core.artifacts.store import ArtifactStore
from anton.core.tools import tool_handlers
from anton.core.tools.generate_artifact import fast_static, orchestrator
from anton.core.tools.generate_artifact.state import BrowserCheck, GenState
from anton.core.llm.provider import LLMResponse, StreamComplete


SKETCH = '<html><head><meta name="viewport" content="width=device-width"></head><body><svg id="world-map"><path d="M0,0L20,0L10,10Z"/></svg></body></html>'
TABLE = '<html><head><meta name="viewport" content="width=device-width"></head><body><table><tr><td>42</td></tr></table></body></html>'


@pytest.fixture
def pages(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTON_RENDER_CHECK_URL", raising=False)
    monkeypatch.setattr("anton.core.artifacts.html_lint.lint_html", lambda *a, **kw: [])
    store = ArtifactStore(tmp_path)
    artifact = store.create(name="World map", description="Show sites on a world map", type="html-app", primary="index.html")
    folder = store.root / artifact.slug
    return store, artifact.slug, folder


async def test_direct_checks_retain_requirement_when_repair_replaces_map_with_table(pages):
    store, slug, folder = pages
    session = SimpleNamespace(_artifact_geography_request="Show a world map of sites", _geography_requirements={}, _page_checks={}, _repair_cells={})
    before = tool_handlers.snapshot_existing_artifacts(store)
    (folder / "index.html").write_text(SKETCH)
    first = await tool_handlers.check_cell_artifacts(session, store=store, before=before)
    assert first.page_checks[(slug, "index.html")] == "failed"
    assert any("geograph" in line.lower() for line in first.messages)
    before = tool_handlers.snapshot_existing_artifacts(store)
    session._artifact_geography_request = "Fix the failure"
    (folder / "index.html").write_text(TABLE)
    repaired = await tool_handlers.check_cell_artifacts(session, store=store, before=before)
    assert repaired.page_checks[(slug, "index.html")] == "failed"
    assert session._repair_cells[(slug, "index.html")] == 1


async def test_direct_requirement_does_not_spill_into_secondary_page(pages):
    store, slug, folder = pages
    session = SimpleNamespace(_artifact_geography_request="Show a world map and a notes page", _geography_requirements={})
    before = tool_handlers.snapshot_existing_artifacts(store)
    (folder / "index.html").write_text(SKETCH)
    (folder / "notes.html").write_text(TABLE)
    result = await tool_handlers.check_cell_artifacts(session, store=store, before=before)
    assert result.page_checks[(slug, "index.html")] == "failed"
    assert result.page_checks[(slug, "notes.html")] == "passed"
    assert (slug, "notes.html") not in session._geography_requirements


async def test_ambiguous_multiple_artifacts_get_advice_without_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTON_RENDER_CHECK_URL", raising=False)
    monkeypatch.setattr("anton.core.artifacts.html_lint.lint_html", lambda *a, **kw: [])
    store = ArtifactStore(tmp_path)
    artifacts = [store.create(name=name, description="", type="html-app", primary="index.html") for name in ("First", "Second")]
    before = tool_handlers.snapshot_existing_artifacts(store)
    for artifact in artifacts:
        (store.root / artifact.slug / "index.html").write_text(TABLE)
    session = SimpleNamespace(_artifact_geography_request="Build a world map and supporting notes", _geography_requirements={})
    result = await tool_handlers.check_cell_artifacts(session, store=store, before=before)
    assert set(result.page_checks.values()) == {"passed"}
    assert any("geograph" in line.lower() for line in result.messages)


async def test_fast_writer_rejects_sketch_before_browser_and_keeps_requirement(tmp_path, monkeypatch):
    calls = []
    async def stream(**kwargs):
        calls.append(kwargs)
        yield StreamComplete(LLMResponse(content=SKETCH))
    browser = []
    monkeypatch.setattr(fast_static.verifiers, "verify_frontend_live", lambda *a, **kw: browser.append(a) or BrowserCheck(status="passed"))
    session = SimpleNamespace(_llm=SimpleNamespace(code_stream=stream), history=[])
    state = GenState(session=session, artifact_type="html-app", artifact_path=tmp_path, slug="map", user_request="Build a world map")
    result = await fast_static.run(state)
    assert not isinstance(result, dict) or result.get("status") != "generated"
    assert not browser
    assert len(calls) == 2
    assert "geograph" in str(calls[-1]["messages"]).lower()
    assert session._geography_requirements[("map", "index.html")].required


@pytest.mark.parametrize("tool_request,active_request", [
    ("Build an Earth globe", ""),
    ("Build the dashboard", "Build an Earth globe"),
])
async def test_standard_writer_passes_original_request_into_static_check(tmp_path, monkeypatch, tool_request, active_request):
    async def write(**kwargs):
        (tmp_path / "index.html").write_text(SKETCH)
        return {"files_written": ["index.html"], "rounds_used": 1, "summary": "done"}
    monkeypatch.setattr(orchestrator.engine, "_run_loop", write)
    browser = []
    monkeypatch.setattr(orchestrator.verifiers, "verify_frontend_live", lambda *a, **kw: browser.append(a) or BrowserCheck(status="passed"))
    session = SimpleNamespace(_scratchpads=SimpleNamespace(cell_sandbox=None), _artifact_geography_request=active_request)
    state = GenState(session=session, artifact_type="html-app", artifact_path=tmp_path, slug="map", user_request=tool_request, brief="Show sites", prd="Show sites")
    error = await orchestrator._gen_verify_frontend(state)
    assert error is not None
    assert "geograph" in str(error).lower()
    assert not browser
    assert session._geography_requirements[("map", "index.html")].kind == "globe"


async def test_streaming_exec_uses_current_user_request_and_next_turn_resets_it(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from anton.core.backends.base import Cell
    from anton.core.session import ChatSession, ChatSessionConfig, _VerifierVerdict
    from tests.conftest import make_mock_llm
    from tests.test_streaming_exec_artifact_lint import _FakeAsyncIter, _scratchpad_exec_response, _text_response

    monkeypatch.delenv("ANTON_RENDER_CHECK_URL", raising=False)
    monkeypatch.setattr("anton.core.artifacts.html_lint.lint_html", lambda *a, **kw: [])
    workspace = MagicMock(base=tmp_path)
    workspace.artifacts_dir = tmp_path / "artifacts"
    store = ArtifactStore(workspace.artifacts_dir)
    artifact = store.create(name="Sites", description="", type="html-app", primary="index.html")
    page = store.root / artifact.slug / "index.html"

    class Pad:
        async def execute_streaming(self, code, **kwargs):
            page.write_text(SKETCH)
            yield Cell(code=code, stdout="saved", stderr="", error=None, description="build map")

    llm = make_mock_llm()
    llm.generate_object_code = AsyncMock(return_value=_VerifierVerdict(status="COMPLETE", reason="done"))
    responses = iter([_scratchpad_exec_response("Build map"), _text_response("Done")])
    llm.plan_stream = lambda **kw: _FakeAsyncIter([StreamComplete(response=next(responses, _text_response("Done")))])
    session = ChatSession(ChatSessionConfig(llm_client=llm, workspace=workspace))
    session._scratchpads.get_or_create = AsyncMock(return_value=Pad())
    try:
        async for _ in session.turn_stream("Build a world map with these sites"):
            pass
        key = (artifact.slug, "index.html")
        assert session._page_checks[key] == "failed"
        assert session._geography_requirements[key].required
        assert any("Geography check:" in str(message) for message in session.history if message["role"] == "user")
        async for _ in session.turn_stream("Explain the totals in a paragraph"):
            pass
        assert not session._geography_requirements
    finally:
        await session.close()


async def test_later_unrelated_artifact_does_not_inherit_required_map(pages):
    store, slug, folder = pages
    session = SimpleNamespace(_artifact_geography_request="Build a world map and supporting notes", _geography_requirements={})
    before = tool_handlers.snapshot_existing_artifacts(store)
    (folder / "index.html").write_text(SKETCH)
    await tool_handlers.check_cell_artifacts(session, store=store, before=before)
    notes = store.create(name="Notes", description="Sources and caveats", type="html-app", primary="index.html")
    before = tool_handlers.snapshot_existing_artifacts(store)
    (store.root / notes.slug / "index.html").write_text(TABLE)
    result = await tool_handlers.check_cell_artifacts(session, store=store, before=before)
    assert result.page_checks[(notes.slug, "index.html")] == "passed"
    assert (notes.slug, "index.html") not in session._geography_requirements


async def test_notes_written_before_map_do_not_acquire_map_requirement(pages):
    store, slug, folder = pages
    session = SimpleNamespace(_artifact_geography_request="Build a world map and supporting notes", _geography_requirements={})
    notes = store.create(name="Notes", description="Sources and caveats", type="html-app", primary="index.html")
    before = tool_handlers.snapshot_existing_artifacts(store)
    (store.root / notes.slug / "index.html").write_text(TABLE)
    result = await tool_handlers.check_cell_artifacts(session, store=store, before=before)
    assert result.page_checks[(notes.slug, "index.html")] == "passed"
    assert (notes.slug, "index.html") not in session._geography_requirements
    before = tool_handlers.snapshot_existing_artifacts(store)
    (folder / "index.html").write_text(SKETCH)
    result = await tool_handlers.check_cell_artifacts(session, store=store, before=before)
    assert result.page_checks[(slug, "index.html")] == "failed"


async def test_fast_writer_preserves_ambiguous_geography_advisory(tmp_path, monkeypatch):
    async def stream(**kwargs):
        yield StreamComplete(LLMResponse(content=SKETCH))
    monkeypatch.setattr(fast_static.verifiers, "verify_frontend_live", lambda *a, **kw: BrowserCheck(status="passed"))
    session = SimpleNamespace(_llm=SimpleNamespace(code_stream=stream), history=[])
    state = GenState(session=session, artifact_type="html-app", artifact_path=tmp_path, slug="map", user_request="Build a map")
    state.verify_warnings = ["The supplied structured source could not be read."]
    result = await fast_static.run(state)
    assert result["status"] == "generated"
    assert any("Geography check:" in warning for warning in result["warnings"])
    assert "The supplied structured source could not be read." in result["warnings"]


def test_generation_can_add_globe_requirement_without_dropping_existing_map(tmp_path):
    from anton.core.artifacts.geography_pages import generated_requirement

    state = GenState(session=SimpleNamespace(), artifact_type="html-app", artifact_path=tmp_path, slug="map", user_request="Build a world map")
    generated_requirement(state, tmp_path / "index.html")
    state.user_request = "Add an Earth globe"
    assert generated_requirement(state, tmp_path / "index.html").kind == "map_and_globe"
    state.user_request = "Fix the loading failure"
    assert generated_requirement(state, tmp_path / "index.html").kind == "map_and_globe"
