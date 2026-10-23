"""anton compares each page with how it was when the turn began.

The fake render-check service from `test_render_check` answers each
request. These tests check what anton sends (`previous_html`), what the
agent hears, what anton puts back, and what closes the turn.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from anton.core.artifacts import ArtifactStore
from anton.core.artifacts.page_copies import TurnPages
from anton.core.artifacts.render_check import Comparison
from anton.core.session import ChatSession
from anton.core.tools.generate_artifact import engine, orchestrator, verifiers
from anton.core.tools.generate_artifact.state import BrowserCheckFailed
from anton.core.tools import tool_handlers
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
    the turn's copies, the snapshot, the write, the check."""
    await tool_handlers.capture_turn_pages(session, store)
    before = tool_handlers.snapshot_existing_artifacts(store)
    (store.root / slug / "index.html").write_text(_PAGE.replace("hi", text))
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
