"""anton compares each page with how it was when the turn began.

The fake render-check service from `test_render_check` answers each
request. These tests check what anton sends (`previous_html`), what the
agent hears, what anton puts back, and what closes the turn.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import anton.core.tools.generate_artifact as gen_pkg
from anton.core.artifacts import ArtifactStore
from anton.core.artifacts.page_copies import TurnPages
from anton.core.artifacts.render_check import Comparison
from anton.core.session import ChatSession
from anton.core.tools.generate_artifact import engine, orchestrator, verifiers
from anton.core.tools.generate_artifact.state import BrowserCheckFailed
from anton.core.tools import tool_handlers
from anton.core.tools.progress import ToolProgress
from anton.core.tools.registry import ToolOutcome
from anton.core.tools.tool_handlers import PageKey, handle_generate_artifact
from tests.test_render_check import (  # noqa: F401  (policy, project and service are fixtures)
    _PAGE,
    _TIMEOUT,
    _VIEWS,
    _answer,
    _comparison,
    _gen_state,
    _writes_page,
    policy,
    project,
    service,
)

_BEFORE = _PAGE.replace("hi", "before")
_BLANK_MAP = {"kind": "blank_view", "viewport": "desktop", "detail": "behind “Map”: canvas#map has no drawn content"}
_BROKE_MAP = {"pressed": "Map", "reason": "new_problem", "viewports": ["desktop"], "findings": [_BLANK_MAP]}
_BLANK_DETAILS = {"pressed": "Details", "reason": "no_view", "viewports": ["desktop"], "findings": []}
_GLOBE_GONE = {"removed_views": [{"pressed": "Globe"}], "removed_controls": ["Globe"]}


def _session(**fields) -> SimpleNamespace:
    return SimpleNamespace(**{
        "_page_checks": {}, "_unverified_findings_listed": set(), "_repair_cells": {}, "_failed_views": {},
        "_scratchpads": None, "_turn_pages": TurnPages(), **fields,
    })


def _artifact(root: Path) -> tuple[ArtifactStore, str]:
    store = ArtifactStore(root / "artifacts")
    slug = store.create(name="Control tower", description="d", type="html-app", primary="index.html").slug
    (store.root / slug / "index.html").write_text(_BEFORE)
    return store, slug


async def _cell(session, store, slug: str, text: str):
    """One cell that rewrites the page, run as `handle_scratchpad` runs one:
    the turn's copies, the snapshot, the write, the check. The page's mtime
    is set back first: a same-size write within one mtime tick would look
    unchanged to the snapshot."""
    page = store.root / slug / "index.html"
    await tool_handlers.capture_turn_pages(session, store)
    if page.exists():
        os.utime(page, ns=(1_000_000_000, 1_000_000_000))
    before = tool_handlers.snapshot_existing_artifacts(store)
    page.write_text(_PAGE.replace("hi", text))
    return await tool_handlers.check_cell_artifacts(session, store=store, before=before)


def _passed(**comparison) -> tuple:
    return (200, {**_answer("passed", views=_VIEWS, unopened=[]), "comparison": _comparison(**comparison)}, 0.0)


# ── What a cell's check sends ────────────────────────────────────────────────


async def test_every_cell_sends_the_page_as_the_turn_found_it(service, tmp_path):
    store, slug = _artifact(tmp_path)
    session = _session()

    await _cell(session, store, slug, "first edit")
    await _cell(session, store, slug, "second edit")

    assert [request["body"].get("previous_html") for request in service.requests] == [_BEFORE, _BEFORE]


async def test_a_page_the_turn_made_is_sent_without_a_previous_page(service, tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    session = _session()
    await tool_handlers.capture_turn_pages(session, store)
    slug = store.create(name="New", description="d", type="html-app", primary="index.html").slug

    await _cell(session, store, slug, "built")

    assert "previous_html" not in service.requests[0]["body"]


async def test_with_less_than_a_whole_check_left_the_page_goes_without_its_copy(service, tmp_path, monkeypatch):
    monkeypatch.setattr(tool_handlers, "_RENDER_CHECK_CELL_BUDGET_SECONDS", 30.0)
    store, slug = _artifact(tmp_path)

    await _cell(_session(), store, slug, "edit")

    assert "previous_html" not in service.requests[0]["body"]


def test_both_exec_paths_keep_the_turns_copies_before_their_snapshot():
    """`handle_scratchpad` and the session's inline streaming exec each keep
    the copies before the snapshot a cell's check diffs against."""
    for source in (inspect.getsource(tool_handlers.handle_scratchpad), inspect.getsource(ChatSession)):
        capture = source.index("capture_turn_pages(")
        assert capture < source.index("snapshot_existing_artifacts(", capture)


# ── What the agent hears ─────────────────────────────────────────────────────


async def test_a_passing_page_hears_which_views_the_turn_blanked_and_removed(service, tmp_path):
    store, slug = _artifact(tmp_path)
    service.answer(_passed(broken_views=[_BLANK_DETAILS], **_GLOBE_GONE))

    lint = await _cell(_session(), store, slug, "edited")

    label = f"{slug}/index.html"
    assert lint.messages[1:] == [
        f"{label} — “Details” showed a graphic before this turn's edits and shows none now. If the user did not "
        "ask to remove it, the edits broke it: fix it",
        f"{label} — this turn's edits removed “Globe”. If the user did not ask for that, put it back; if they "
        "did, tell them it is gone",
    ]


async def test_a_view_the_turn_broke_is_named_before_the_repair_cells_left(service, tmp_path):
    store, slug = _artifact(tmp_path)
    service.answer((200, {
        **_answer("failed", findings=[_BLANK_MAP], views=_VIEWS, unopened=[]),
        "comparison": _comparison(broken_views=[_BROKE_MAP]),
    }, 0.0))

    lint = await _cell(_session(), store, slug, "breaks the map")

    label = f"{slug}/index.html"
    assert lint.messages == [
        f"{label} — blank_view (desktop): {_BLANK_MAP['detail']}",
        f"{label} — “Map” drew before this turn's edits and fails now: this turn's edits broke it. Fix it and "
        "keep the view",
        f"{label} — {tool_handlers._repairs_left_line(2)}",
    ]


async def test_the_same_blank_view_is_named_once_a_turn(service, tmp_path):
    store, slug = _artifact(tmp_path)
    session = _session()
    service.answer(_passed(broken_views=[_BLANK_DETAILS]))

    first = await _cell(session, store, slug, "one")
    second = await _cell(session, store, slug, "two")

    assert any("showed a graphic" in message for message in first.messages)
    assert not any("showed a graphic" in message for message in second.messages)


async def test_an_unverified_check_names_no_change(service, tmp_path):
    store, slug = _artifact(tmp_path)
    service.answer((200, {**_answer("unverified", findings=[_TIMEOUT]), "comparison": _comparison(**_GLOBE_GONE)}, 0.0))

    lint = await _cell(_session(), store, slug, "edited")

    assert not any("removed" in message for message in lint.messages)


# ── What ends the turn ───────────────────────────────────────────────────────


async def test_the_turn_ends_naming_a_view_an_edit_removed(service, project, policy):
    store, slug = _artifact(project)
    session = _session(
        _scratchpads=SimpleNamespace(cell_sandbox=policy), _workspace=SimpleNamespace(artifacts_dir=store.root),
    )
    service.answer(_passed(**_GLOBE_GONE))

    await _cell(session, store, slug, "no globe")

    assert ChatSession._unverified_page_lines(session) == [
        "“Control tower” no longer has “Globe”: this turn removed it."
    ]
    session._scratchpads = SimpleNamespace(cell_sandbox=None)
    assert ChatSession._unverified_page_lines(session) == []


# ── Putting the page back ────────────────────────────────────────────────────

_FAILED_MAP = (200, {
    **_answer("failed", findings=[_BLANK_MAP], views=_VIEWS, unopened=[]),
    "comparison": _comparison(broken_views=[_BROKE_MAP], **_GLOBE_GONE),
}, 0.0)


def _sandboxed(store: ArtifactStore, policy) -> SimpleNamespace:
    return _session(
        _scratchpads=SimpleNamespace(cell_sandbox=policy), _workspace=SimpleNamespace(artifacts_dir=store.root),
    )


async def test_a_page_the_turn_broke_past_its_repair_cells_goes_back_as_it_was(service, project, policy):
    store, slug = _artifact(project)
    session = _sandboxed(store, policy)
    key = tool_handlers.PageKey(slug, "index.html")
    service.answer(_FAILED_MAP)

    await _cell(session, store, slug, "breaks the map")
    await _cell(session, store, slug, "first repair")
    last = await _cell(session, store, slug, "second repair")

    assert last.messages == [f"{slug}/index.html — {tool_handlers._RESTORED}"]
    assert (store.root / slug / "index.html").read_text() == _BEFORE
    assert session._page_checks == {key: "passed"}
    assert session._failed_views == {}
    # Only the line for the page put back: what the broken version lost is not the user's page any more.
    assert ChatSession._unverified_page_lines(session) == [
        "“Control tower”: the edit could not be made safely, so the page is back as it was before this turn."
    ]


async def test_a_page_that_failed_the_same_way_before_the_turn_stays(service, tmp_path):
    store, slug = _artifact(tmp_path)
    session = _session()
    service.answer((200, {
        **_answer("failed", findings=[_BLANK_MAP], views=_VIEWS, unopened=[]),
        "comparison": _comparison(previous_status="failed"),
    }, 0.0))

    for text in ("edit", "first repair"):
        await _cell(session, store, slug, text)
    last = await _cell(session, store, slug, "second repair")

    assert last.messages == [f"{slug}/index.html — {tool_handlers._NO_REPAIRS_LEFT}"]
    assert (store.root / slug / "index.html").read_text() == _PAGE.replace("hi", "second repair")


async def test_a_page_that_could_not_be_put_back_says_no_repair_cells_are_left(service, tmp_path, monkeypatch):
    store, slug = _artifact(tmp_path)
    session = _session()
    service.answer(_FAILED_MAP)
    monkeypatch.setattr(TurnPages, "restore", lambda self, key, store: False)

    for text in ("edit", "first repair"):
        await _cell(session, store, slug, text)
    last = await _cell(session, store, slug, "second repair")

    assert last.messages == [f"{slug}/index.html — {tool_handlers._NO_REPAIRS_LEFT}"]
    assert session._page_checks == {tool_handlers.PageKey(slug, "index.html"): "failed"}


async def test_a_page_put_back_without_a_comparison_is_neither_failed_nor_unverified(service, project, policy):
    """No check this turn vouched for the page the turn found, but it is that
    page, byte for byte: the turn ends with the line that it went back, and
    no line that it failed or was not verified."""
    store, slug = _artifact(project)
    session = _sandboxed(store, policy)
    service.answer((200, _answer("failed", findings=[_BLANK_MAP], views=_VIEWS, unopened=[]), 0.0))

    for text in ("edit", "first repair", "second repair"):
        await _cell(session, store, slug, text)

    assert (store.root / slug / "index.html").read_text() == _BEFORE
    assert session._page_checks == {}
    assert session._failed_views == {}
    assert ChatSession._unverified_page_lines(session) == [
        "“Control tower”: the edit could not be made safely, so the page is back as it was before this turn."
    ]


async def test_a_page_put_back_forgets_the_views_its_broken_version_failed_on(service, tmp_path):
    """The page failed before the turn too, so it goes back as failed, but
    the next edit is held to the views of the page the turn found, not to
    those of the version anton replaced."""
    store, slug = _artifact(tmp_path)
    session = _session()
    key = tool_handlers.PageKey(slug, "index.html")
    service.answer((200, {
        **_answer("failed", findings=[_BLANK_MAP], views=_VIEWS, unopened=[]),
        "comparison": _comparison(previous_status="failed", broken_views=[_BROKE_MAP]),
    }, 0.0))

    for text in ("edit", "first repair", "second repair"):
        await _cell(session, store, slug, text)

    assert (store.root / slug / "index.html").read_text() == _BEFORE
    assert session._page_checks == {key: "failed"}
    assert session._failed_views == {}


async def test_after_the_page_went_back_a_new_edit_hears_its_blank_view_again(service, tmp_path):
    store, slug = _artifact(tmp_path)
    session = _session()
    blank = (200, {
        **_answer("failed", findings=[_BLANK_MAP], views=_VIEWS, unopened=[]),
        "comparison": _comparison(broken_views=[_BROKE_MAP, _BLANK_DETAILS]),
    }, 0.0)
    service.answer(blank, blank, blank, _passed(broken_views=[_BLANK_DETAILS]))

    for text in ("edit", "first repair", "second repair"):
        await _cell(session, store, slug, text)
    after = await _cell(session, store, slug, "a new edit")

    assert any("“Details” showed a graphic" in message for message in after.messages)


async def test_a_page_edited_again_after_it_went_back_no_longer_reads_as_put_back(service, project, policy):
    """The cell after the restore rewrote the page, and its check could not
    vouch for it: the turn must not end saying the page is as it was."""
    store, slug = _artifact(project)
    session = _sandboxed(store, policy)
    service.answer(_FAILED_MAP, _FAILED_MAP, _FAILED_MAP, (503, {"detail": "the render check isn't ready: starting"}, 0.0))

    for text in ("breaks the map", "first repair", "second repair"):
        await _cell(session, store, slug, text)
    assert any("back as it was" in line for line in ChatSession._unverified_page_lines(session))
    await _cell(session, store, slug, "a new edit")

    assert (store.root / slug / "index.html").read_text() == _PAGE.replace("hi", "a new edit")
    assert ChatSession._unverified_page_lines(session) == [
        "“Control tower” was not verified in a browser, so it may not display correctly in the preview."
    ]


# ── generate_artifact's builds ───────────────────────────────────────────────


def test_the_generators_check_sends_the_page_as_it_was_and_keeps_the_comparison(service, tmp_path):
    page = tmp_path / "index.html"
    page.write_text(_PAGE)
    service.answer(_passed(**_GLOBE_GONE))

    live = verifiers.verify_frontend_live(page, cell_sandbox=None, previous_html=_BEFORE)
    verifiers.verify_frontend_live(page, cell_sandbox=None)

    assert service.requests[0]["body"]["previous_html"] == _BEFORE
    assert "previous_html" not in service.requests[1]["body"]
    assert [view.pressed for view in live.comparison.removed_views] == ["Globe"]


async def test_each_build_is_compared_with_the_page_as_the_turn_found_it(service, project, policy, monkeypatch):
    st = _gen_state(project, policy)
    st.previous_page = (tool_handlers.PageKey("dash", "index.html"), _BEFORE)
    monkeypatch.setattr(orchestrator.engine, "_run_loop", _writes_page(project, []))
    service.answer(_passed(**_GLOBE_GONE))

    assert await orchestrator._gen_verify_frontend(st) is None

    assert service.requests[0]["body"]["previous_html"] == _BEFORE
    assert orchestrator._result_shell(st)["changes_from_previous"] == {
        "broke": [], "blank": [], "removed": ["“Globe”"],
    }


async def test_a_new_artifacts_build_is_compared_with_nothing(service, project, policy, monkeypatch):
    st = _gen_state(project, policy)
    monkeypatch.setattr(orchestrator.engine, "_run_loop", _writes_page(project, []))

    assert await orchestrator._gen_verify_frontend(st) is None

    assert "previous_html" not in service.requests[0]["body"]
    assert "changes_from_previous" not in orchestrator._result_shell(st)


async def test_a_build_left_to_repair_carries_its_comparison(service, project, policy, monkeypatch):
    st = _gen_state(project, policy)
    st.previous_page = (tool_handlers.PageKey("dash", "index.html"), _BEFORE)
    monkeypatch.setattr(orchestrator.engine, "_run_loop", _writes_page(project, []))
    service.answer(_FAILED_MAP)

    result = await orchestrator._gen_verify_frontend(st)

    assert isinstance(result, BrowserCheckFailed)
    assert result.comparison is not None and result.comparison.has_new_problem


def test_embedding_writes_nothing_once_the_run_is_abandoned(tmp_path, monkeypatch):
    st = _gen_state(tmp_path)
    st.attachments = [object()]
    page = tmp_path / "index.html"
    page.write_text(_PAGE)
    monkeypatch.setattr(orchestrator.attachments, "embed_assets", lambda html, *args, **kwargs: html + "<!-- embedded -->")

    st.abandoned.set()
    orchestrator._embed_attachments(st, page)
    assert page.read_text() == _PAGE

    st.abandoned.clear()
    orchestrator._embed_attachments(st, page)
    assert page.read_text().endswith("<!-- embedded -->")


def test_the_runs_comparison_reaches_the_turns_record(tmp_path):
    session = _session()
    comparison = Comparison.model_validate(_comparison(**_GLOBE_GONE))

    engine._record_page_check(session, state=SimpleNamespace(
        slug="dash", browser_status="passed", browser_findings_listed=False, files_written=["index.html"],
        is_fullstack=False, primary="index.html", comparison=comparison,
    ), result={"status": "generated"})

    store = ArtifactStore(tmp_path / "artifacts")
    assert session._turn_pages.end_lines(store, checks=session._page_checks) == [
        "“dash” no longer has “Globe”: this turn removed it."
    ]


def test_a_builds_comparison_is_kept_under_the_page_its_check_loaded():
    session = _session()

    engine._record_page_check(session, state=SimpleNamespace(
        slug="dash", browser_status="passed", browser_findings_listed=False, files_written=["app/main.html"],
        is_fullstack=False, primary="index.html", browser_page="app/main.html",
        comparison=Comparison.model_validate(_comparison()),
    ), result={"status": "generated"})

    assert session._turn_pages.status_before(PageKey("dash", "app/main.html"), None) == "passed"
    assert session._turn_pages.status_before(PageKey("dash", "index.html"), None) == "unverified"


@pytest.mark.parametrize("result", [
    pytest.param("the run stopped at the brief", id="str"),
    pytest.param({"status": "stopped_over_budget"}, id="stopped-over-budget"),
])
def test_a_run_that_built_no_page_keeps_no_comparison(tmp_path, result):
    """The state's comparison may be an earlier attempt's, or a page the run
    renamed, and the page is about to go back: it describes no page."""
    session = _session()

    engine._record_page_check(session, state=SimpleNamespace(
        slug="dash", browser_status="passed", browser_findings_listed=False, files_written=["index.html"],
        is_fullstack=False, primary="index.html", comparison=Comparison.model_validate(_comparison(**_GLOBE_GONE)),
    ), result=result)

    store = ArtifactStore(tmp_path / "artifacts")
    assert session._turn_pages.end_lines(store, checks=session._page_checks) == []


def test_a_build_left_to_repairs_comparison_reaches_the_turns_record(tmp_path):
    session = _session()
    comparison = Comparison.model_validate(_comparison(broken_views=[_BROKE_MAP]))

    engine._record_page_check(
        session,
        state=SimpleNamespace(slug="dash", browser_status="failed", files_written=["index.html"],
                              is_fullstack=False, primary="index.html"),
        result=BrowserCheckFailed(page="index.html", findings=("x",), views=(), screenshot=None, error="e",
                                  comparison=comparison),
    )

    store = ArtifactStore(tmp_path / "artifacts")
    assert session._turn_pages.end_lines(store, checks=session._page_checks) == [
        "“dash”: “Map” drew before this turn and fails now."
    ]


# ── generate_artifact puts the page back ─────────────────────────────────────

_HALF = "<p>half written</p>"
_GENERATED = {"status": "generated", "files_written": ["index.html"], "summary": "s", "trace": []}
_OVER_BUDGET = {"status": "stopped_over_budget", "files_written": [], "summary": "s", "trace": []}
_LEFT_TO_REPAIR = BrowserCheckFailed(page="index.html", findings=("x",), views=(), screenshot=None, error="e")


@pytest.fixture
def configured(monkeypatch):
    """A render-check service to compare with: copies are kept only then."""
    monkeypatch.setenv("ANTON_RENDER_CHECK_URL", "http://127.0.0.1:9")


def _tc(slug: str) -> dict:
    return {"slug": slug, "user_request": "make it premium", "agent_understanding": "restyle", "user_language": "English"}


def _gen_session(store: ArtifactStore) -> SimpleNamespace:
    return _session(_workspace=SimpleNamespace(artifacts_dir=store.root))


def _run(outcome):
    """A run that half-writes the page, then ends with OUTCOME (raised when it is an exception)."""
    async def fake_generate(**kw):
        (kw["artifact_path"] / "index.html").write_text(_HALF)
        if isinstance(outcome, BaseException):
            raise outcome
        # A copy: the handler adds keys to a dict result.
        return dict(outcome) if isinstance(outcome, dict) else outcome

    return fake_generate


async def _result(session, slug: str) -> ToolOutcome:
    result = None
    async for item in handle_generate_artifact(session, _tc(slug)):
        if not isinstance(item, ToolProgress):
            result = item
    return result


async def _until(condition, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        assert loop.time() < deadline, "timed out"
        await asyncio.sleep(0.01)


@pytest.mark.parametrize("outcome", ["Frontend verification failed after 2 attempt(s): x", _OVER_BUDGET])
async def test_a_run_that_did_not_build_the_page_puts_it_back(configured, tmp_path, monkeypatch, outcome):
    store, slug = _artifact(tmp_path)
    session = _gen_session(store)
    monkeypatch.setattr(gen_pkg, "generate", _run(outcome))

    result = await _result(session, slug)

    assert (store.root / slug / "index.html").read_text() == _BEFORE
    assert "The page is back as it was before this run." in result.content
    # No status is known for the page the turn found, so it has no record: no line that it failed.
    assert session._page_checks == {}


async def test_a_crashed_run_puts_the_page_back_before_it_says_so(configured, tmp_path, monkeypatch):
    store, slug = _artifact(tmp_path)
    monkeypatch.setattr(gen_pkg, "generate", _run(RuntimeError("kaput")))
    outcomes = 0

    async for item in handle_generate_artifact(_gen_session(store), _tc(slug)):
        if isinstance(item, ToolOutcome):
            outcomes += 1
            assert (store.root / slug / "index.html").read_text() == _BEFORE
            assert "kaput" in item.content
            assert item.content.endswith("The page is back as it was before this run.")

    assert outcomes == 1


@pytest.mark.parametrize("outcome", [_GENERATED, _LEFT_TO_REPAIR])
async def test_a_built_page_or_one_left_to_repair_stays(configured, tmp_path, monkeypatch, outcome):
    store, slug = _artifact(tmp_path)
    monkeypatch.setattr(gen_pkg, "generate", _run(outcome))

    await _result(_gen_session(store), slug)

    assert (store.root / slug / "index.html").read_text() == _HALF


@pytest.mark.parametrize("status", ["needs_confirmation", "cancelled"])
async def test_a_run_that_never_touched_the_page_writes_nothing(configured, tmp_path, monkeypatch, status):
    store, slug = _artifact(tmp_path)
    page = store.root / slug / "index.html"
    inode = page.stat().st_ino

    async def fake_generate(**kw):
        return {"status": status, "files_written": [], "summary": "s", "trace": []}

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    result = await _result(_gen_session(store), slug)

    assert page.stat().st_ino == inode
    assert "restored" not in json.loads(result.content)


async def test_a_new_artifact_has_nothing_to_put_back(configured, tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "artifacts")
    slug = store.create(name="New", description="d", type="html-app", primary="index.html").slug
    monkeypatch.setattr(gen_pkg, "generate", _run("Frontend verification failed after 2 attempt(s): x"))

    await _result(_gen_session(store), slug)

    assert (store.root / slug / "index.html").read_text() == _HALF


async def test_a_run_the_turn_abandoned_puts_the_page_back_once_it_stops(configured, tmp_path, monkeypatch):
    store, slug = _artifact(tmp_path)

    async def fake_generate(**kw):
        (kw["artifact_path"] / "index.html").write_text(_HALF)
        kw["progress"].put_nowait("Writing the page")
        await asyncio.Event().wait()

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    agen = handle_generate_artifact(_gen_session(store), _tc(slug))
    assert await agen.__anext__() == ToolProgress("Writing the page")

    await agen.aclose()

    await _until(lambda: (store.root / slug / "index.html").read_text() == _BEFORE)


@pytest.mark.parametrize(("outcome", "put_back"), [(_GENERATED, False), (_LEFT_TO_REPAIR, False), (_OVER_BUDGET, True)])
async def test_closing_the_handler_after_the_run_ended_follows_the_runs_outcome(
    configured, tmp_path, monkeypatch, outcome, put_back
):
    store, slug = _artifact(tmp_path)
    page = store.root / slug / "index.html"

    async def fake_generate(**kw):
        page.write_text(_HALF)
        kw["progress"].put_nowait("Wrote the page")
        return dict(outcome) if isinstance(outcome, dict) else outcome

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    agen = handle_generate_artifact(_gen_session(store), _tc(slug))
    assert await agen.__anext__() == ToolProgress("Wrote the page")

    await agen.aclose()

    if put_back:
        await _until(lambda: page.read_text() == _BEFORE)
    else:
        await asyncio.sleep(0.1)
        assert page.read_text() == _HALF


async def test_an_embed_still_running_when_the_turn_ends_writes_nothing_over_the_restored_page(
    configured, tmp_path, monkeypatch
):
    store, slug = _artifact(tmp_path)
    page = store.root / slug / "index.html"
    started, release, done = threading.Event(), threading.Event(), threading.Event()

    async def fake_generate(**kw):
        page.write_text(_HALF)

        def embed():
            # Stands in for `_embed_attachments`, whose thread outlives the cancelled run.
            started.set()
            release.wait(5)
            with kw["page_lock"]:
                if not kw["abandoned"].is_set():
                    page.write_text("<p>embedded</p>")
            done.set()

        kw["progress"].put_nowait("Embedding")
        await asyncio.to_thread(embed)

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    agen = handle_generate_artifact(_gen_session(store), _tc(slug))
    assert await agen.__anext__() == ToolProgress("Embedding")
    # Cancelling the run before a thread picks up `embed` would drop it unrun.
    await _until(started.is_set)

    await agen.aclose()
    await _until(lambda: page.read_text() == _BEFORE)
    release.set()
    await _until(done.is_set)

    assert page.read_text() == _BEFORE


async def test_a_put_back_that_raises_is_logged_not_raised(configured, tmp_path, monkeypatch, caplog):
    store, slug = _artifact(tmp_path)
    monkeypatch.setattr(gen_pkg, "generate", _run("Frontend verification failed after 2 attempt(s): x"))

    def boom(self, key, store):
        raise RuntimeError("disk")

    monkeypatch.setattr(TurnPages, "restore_if_changed", boom)

    result = await _result(_gen_session(store), slug)

    assert "The page is back" not in result.content
    assert "could not put" in caplog.text


async def test_a_rebuild_that_removed_views_tells_the_agent_to_name_them(configured, tmp_path, monkeypatch):
    store, slug = _artifact(tmp_path)
    outcome = {**_GENERATED, "verified_in_browser": True,
               "changes_from_previous": {"broke": [], "blank": [], "removed": ["“Globe”"]}}
    monkeypatch.setattr(gen_pkg, "generate", _run(outcome))

    result = await _result(_gen_session(store), slug)

    data = json.loads(result.content)
    assert data["changes_from_previous"]["removed"] == ["“Globe”"]
    assert tool_handlers._CHANGES_INSTRUCTION in data["instruction"]


async def test_a_page_put_back_drops_what_the_replaced_build_changed(configured, tmp_path, monkeypatch):
    store, slug = _artifact(tmp_path)
    outcome = {**_OVER_BUDGET, "changes_from_previous": {"broke": ["“Map”"], "blank": [], "removed": []}}
    monkeypatch.setattr(gen_pkg, "generate", _run(outcome))

    result = await _result(_gen_session(store), slug)

    data = json.loads(result.content)
    assert data["restored"] == "The page is back as it was before this run."
    assert "changes_from_previous" not in data
    assert tool_handlers._CHANGES_INSTRUCTION not in data["instruction"]


async def _edited_this_turn(session, store, slug: str, *, status=None) -> Path:
    """A cell earlier in the turn: the turn's copies, then its edit, and
    STATUS, how that cell's check found the edit."""
    page = store.root / slug / "index.html"
    await tool_handlers.capture_turn_pages(session, store)
    page.write_text("<p>the edit the user asked for</p>")
    if status is not None:
        session._page_checks[PageKey(slug, "index.html")] = status
    return page


@pytest.mark.parametrize("outcome", ["Frontend verification failed after 2 attempt(s): x", _OVER_BUDGET])
async def test_a_run_after_an_edit_this_turn_goes_back_to_that_edit(configured, tmp_path, monkeypatch, outcome):
    """The page goes back as the run found it, which is not how the turn
    found it: the run says so, the turn's last lines do not."""
    store, slug = _artifact(tmp_path)
    session = _gen_session(store)
    page = await _edited_this_turn(session, store, slug)
    monkeypatch.setattr(gen_pkg, "generate", _run(outcome))

    result = await _result(session, slug)

    assert page.read_text() == "<p>the edit the user asked for</p>"
    assert tool_handlers._PAGE_PUT_BACK in result.content
    if isinstance(outcome, dict):
        assert json.loads(result.content)["restored"] == tool_handlers._PAGE_PUT_BACK
    assert session._turn_pages.has_restored() is False
    assert session._turn_pages.end_lines(store, checks=session._page_checks) == []


async def test_a_run_after_an_edit_this_turn_that_wrote_nothing_leaves_it(configured, tmp_path, monkeypatch):
    store, slug = _artifact(tmp_path)
    session = _gen_session(store)
    page = await _edited_this_turn(session, store, slug)
    inode = page.stat().st_ino

    async def fake_generate(**kw):
        return {"status": "needs_confirmation", "files_written": [], "summary": "s", "trace": []}

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    result = await _result(session, slug)

    assert page.stat().st_ino == inode
    assert "restored" not in json.loads(result.content)


async def test_a_page_put_back_to_an_edit_this_turn_keeps_how_that_edit_fared(configured, tmp_path, monkeypatch):
    store, slug = _artifact(tmp_path)
    session = _gen_session(store)
    key = PageKey(slug, "index.html")
    page = await _edited_this_turn(session, store, slug, status="passed")

    async def fake_generate(**kw):
        page.write_text(_HALF)
        # As `engine._record_page_check` records a run that wrote a page no browser loaded.
        tool_handlers.record_page_checks(kw["session"], checks={key: "unverified"})
        return "Frontend verification failed after 2 attempt(s): x"

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    await _result(session, slug)

    assert page.read_text() == "<p>the edit the user asked for</p>"
    assert session._page_checks[key] == "passed"


async def test_a_handler_that_crashed_mid_run_stops_the_run_before_it_puts_the_page_back(
    configured, tmp_path, monkeypatch
):
    store, slug = _artifact(tmp_path)
    page = store.root / slug / "index.html"
    tasks: list[asyncio.Task] = []

    async def fake_generate(**kw):
        tasks.append(asyncio.current_task())
        page.write_text(_HALF)
        kw["progress"].put_nowait("Writing the page")
        await asyncio.sleep(0.2)
        page.write_text("<p>late</p>")

    async def broken_drain(queue):
        yield ToolProgress(await queue.get())
        raise RuntimeError("progress broke")

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    monkeypatch.setattr(tool_handlers, "_drain_progress", broken_drain)
    outcomes = 0

    async for item in handle_generate_artifact(_gen_session(store), _tc(slug)):
        if isinstance(item, ToolOutcome):
            outcomes += 1
            assert tasks[0].done()
            assert page.read_text() == _BEFORE
            await asyncio.sleep(0.3)
            assert page.read_text() == _BEFORE

    assert outcomes == 1


async def test_a_page_put_back_keeps_how_it_last_fared(configured, tmp_path, monkeypatch):
    store, slug = _artifact(tmp_path)
    session = _gen_session(store)
    key = PageKey(slug, "index.html")
    # An earlier check this turn found the page as the turn found it passing.
    session._turn_pages.record({key: Comparison.model_validate(_comparison())})
    monkeypatch.setattr(gen_pkg, "generate", _run(_OVER_BUDGET))

    await _result(session, slug)

    assert (store.root / slug / "index.html").read_text() == _BEFORE
    assert session._page_checks[key] == "passed"


async def test_a_page_put_back_drops_the_replaced_builds_screenshot_and_browser_fields(
    configured, tmp_path, monkeypatch
):
    store, slug = _artifact(tmp_path)
    outcome = {**_OVER_BUDGET, "screenshot": object(), "screenshot_page": "index.html", "verified_in_browser": True,
               "browser_checked": ["the page as it opens"], "browser_unchecked": []}
    monkeypatch.setattr(gen_pkg, "generate", _run(outcome))

    result = await _result(_gen_session(store), slug)

    assert isinstance(result.content, str)
    data = json.loads(result.content)
    for name in ("screenshot", "screenshot_page", "verified_in_browser", "browser_checked", "browser_unchecked"):
        assert name not in data
    assert data["instruction"].endswith(tool_handlers._RESTORED_INSTRUCTION)
