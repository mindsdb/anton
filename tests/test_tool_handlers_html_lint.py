"""Coverage for the exec-time artifact lint hook's html path.

Mirrors `test_tool_handlers_xlsx_lint.py` — same `_FakeStore`/`_make_artifact`
shape, exercising `lint_changed_artifact_files` directly rather than the
standalone `lint_html` (see `test_html_lint.py` for that).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from anton.core.tools.tool_handlers import ArtifactFiles, lint_changed_artifact_files, snapshot_existing_artifacts

requires_browser = pytest.mark.skipif(
    not os.environ.get("ANTON_HTML_LINT_BROWSER"),
    reason="ANTON_HTML_LINT_BROWSER not set to a real browser binary",
)

_BROKEN_HTML = """<!DOCTYPE html>
<html>
<head><script src="missing.js"></script></head>
<body><script>undefinedFunctionCallXYZ();</script></body>
</html>
"""


class _FakeStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def read_metadata(self, slug: str) -> None:
        # The real store's answer for this file's `{}` metadata.json.
        return None


@pytest.fixture
def store(tmp_path: Path) -> _FakeStore:
    return _FakeStore(tmp_path)


def _make_artifact(store: _FakeStore, slug: str) -> Path:
    folder = store.root / slug
    folder.mkdir()
    (folder / "metadata.json").write_text("{}")
    return folder


def test_no_browser_configured_says_the_page_was_not_checked(monkeypatch, store: _FakeStore):
    """No browser configured (the cloud, a bare CLI) means the checker can't
    run at all. The cell says so: silence used to read as a pass to an agent
    told that a page that could not be checked gets a line."""
    monkeypatch.delenv("ANTON_HTML_LINT_BROWSER", raising=False)
    monkeypatch.delenv("ANTON_RENDER_CHECK_URL", raising=False)
    folder = _make_artifact(store, "dash-abc12345")
    (folder / "dash.html").write_text(_BROKEN_HTML)
    before = {"dash-abc12345": ArtifactFiles()}

    lint = lint_changed_artifact_files(store, before)

    assert lint.messages == ["dash-abc12345/dash.html — not checked in a browser: no browser on this deployment"]
    assert lint.page_checks == {("dash-abc12345", "dash.html"): "unverified"}


@pytest.mark.skipif(shutil.which("true") is None, reason="no 'true' binary on PATH")
def test_a_configured_browser_that_fails_to_produce_output_says_so(
    monkeypatch, store: _FakeStore
):
    """A browser IS configured here — the run itself just didn't produce a
    usable result (stands in for a crash/timeout/malformed output). The
    cell says the page was not checked, never nothing.

    `shutil.which`, not a hardcoded `/bin/true` — that path doesn't exist on
    macOS (`/usr/bin/true` there), which silently degenerated this into the
    no-browser-at-all case above instead of what it claims to test.
    """
    monkeypatch.setenv("ANTON_HTML_LINT_BROWSER", shutil.which("true"))  # exits 0, no output
    monkeypatch.delenv("ANTON_RENDER_CHECK_URL", raising=False)
    folder = _make_artifact(store, "dash-abc12345")
    (folder / "dash.html").write_text(_BROKEN_HTML)
    before = {"dash-abc12345": ArtifactFiles()}

    assert lint_changed_artifact_files(store, before).messages == [
        "dash-abc12345/dash.html — not checked in a browser: the local browser check gave no answer"
    ]


def test_a_clean_load_in_the_local_browser_adds_no_line(monkeypatch, store: _FakeStore):
    """A pass adds no line: a page no browser loaded still gets one, so
    silence means a pass. The record still says it passed."""
    from anton.core.tools import tool_handlers

    monkeypatch.delenv("ANTON_RENDER_CHECK_URL", raising=False)
    monkeypatch.setattr("anton.core.artifacts.html_lint.lint_html", lambda path, *, cell_sandbox: [])
    folder = _make_artifact(store, "dash-abc12345")
    (folder / "dash.html").write_text("<html><body>ok</body></html>")

    lint = tool_handlers.lint_changed_artifact_files(store, {"dash-abc12345": ArtifactFiles()})

    assert lint.messages == []
    assert lint.page_checks == {("dash-abc12345", "dash.html"): "passed"}


def test_a_page_over_the_size_ceiling_says_it_was_not_checked(monkeypatch, store: _FakeStore):
    """The size ceiling used to drop the page before it reached the browser
    pass, with no line and no record."""
    from anton.core.tools import tool_handlers

    monkeypatch.delenv("ANTON_RENDER_CHECK_URL", raising=False)
    monkeypatch.setattr(tool_handlers, "_ARTIFACT_LINT_SIZE_CEILING", 1024 * 1024)
    loaded: list[Path] = []
    monkeypatch.setattr(
        "anton.core.artifacts.html_lint.lint_html", lambda path, *, cell_sandbox: loaded.append(path) or []
    )
    folder = _make_artifact(store, "dash-abc12345")
    (folder / "dash.html").write_text("<html><body>" + "x" * (1024 * 1024) + "</body></html>")

    lint = tool_handlers.lint_changed_artifact_files(store, {"dash-abc12345": ArtifactFiles()})

    assert loaded == []
    assert lint.messages == ["dash-abc12345/dash.html — not checked in a browser: the page is over 1 MB"]
    assert lint.page_checks == {("dash-abc12345", "dash.html"): "unverified"}


@requires_browser
def test_flags_a_changed_html_artifact(store: _FakeStore):
    folder = _make_artifact(store, "dash-abc12345")
    (folder / "dash.html").write_text(_BROKEN_HTML)
    before = {"dash-abc12345": ArtifactFiles()}

    messages = lint_changed_artifact_files(store, before).messages

    assert any("dash-abc12345/dash.html" in m for m in messages)


@requires_browser
def test_skips_artifacts_the_cell_did_not_touch(store: _FakeStore):
    """A broken page already on disk before this cell ran must not be
    re-flagged every unrelated turn: only a change to its files counts."""
    folder = _make_artifact(store, "dash-abc12345")
    page = folder / "dash.html"
    page.write_text(_BROKEN_HTML)
    before = snapshot_existing_artifacts(store)

    assert lint_changed_artifact_files(store, before).messages == []


def _dashboard(
    tmp_path: Path, *, primary: str | None = "index.html", pages: tuple[str, ...] = ("index.html", "details.html")
):
    """A dashboard whose `pages` read `data.csv`. By default its primary
    `index.html` has a second page beside it."""
    from anton.core.artifacts.store import ArtifactStore

    store = ArtifactStore(tmp_path / "artifacts")
    slug = store.create(name="Dash", description="d", type="html-app", primary=primary).slug
    folder = store.root / slug
    for name in pages:
        (folder / name).write_text(f"<html><body>{name}</body></html>")
    (folder / "data.csv").write_text("a,b\n1,2\n")
    return store, slug


@pytest.mark.parametrize(
    ("primary", "pages", "cell", "loaded_pages"),
    [
        ("index.html", ("index.html", "details.html"), "data.csv", ["index.html"]),
        ("index.html", ("index.html", "details.html"), "details.html", ["details.html"]),
        (None, ("index.html", "details.html"), "data.csv", ["index.html"]),
        (None, ("dashboard.html",), "data.csv", ["dashboard.html"]),
        (None, ("summary.html", "details.html"), "data.csv", ["details.html"]),
        ("summary.html", ("summary.html", "details.html"), "data.csv", ["summary.html"]),
    ],
)
def test_the_local_browser_loads_only_the_edited_pages_and_says_nothing_on_a_pass(
    monkeypatch, tmp_path: Path, primary: str | None, pages: tuple[str, ...], cell: str, loaded_pages: list[str]
):
    """Without the render-check service, the local browser loads the page
    the cell edited, or the artifact's main page when the cell changed only
    a file the page reads, the same pages the service would get. The main
    page is the primary, else `index.html`, else the first page by name,
    so an artifact created without a primary still gets one checked. A
    clean page adds no line: the result stays in history and is resent on
    every model call, and silence already means a pass, since a page the
    browser could not load still gets a line."""
    from anton.core.tools import tool_handlers

    monkeypatch.delenv("ANTON_RENDER_CHECK_URL", raising=False)
    loaded: list[str] = []
    monkeypatch.setattr(
        "anton.core.artifacts.html_lint.lint_html",
        lambda path, *, cell_sandbox: loaded.append(path.name) or [],
    )
    store, slug = _dashboard(tmp_path, primary=primary, pages=pages)
    before = tool_handlers.snapshot_existing_artifacts(store)
    (store.root / slug / cell).write_text("changed by this cell")

    lint = tool_handlers.lint_changed_artifact_files(store, before)

    assert loaded == loaded_pages
    assert lint.messages == []
    assert lint.page_checks == {(slug, page): "passed" for page in loaded_pages}


def test_a_host_with_no_browser_names_only_the_page_the_cell_edited(monkeypatch, tmp_path: Path):
    """A cell that edits one page of a two-page artifact, on a host with no
    browser, gets one line, for that page, saying why it was not checked."""
    from anton.core.tools import tool_handlers

    monkeypatch.delenv("ANTON_RENDER_CHECK_URL", raising=False)
    monkeypatch.delenv("ANTON_HTML_LINT_BROWSER", raising=False)
    store, slug = _dashboard(tmp_path)
    before = tool_handlers.snapshot_existing_artifacts(store)
    (store.root / slug / "details.html").write_text("<html><body>new details</body></html>")

    lint = tool_handlers.lint_changed_artifact_files(store, before)

    assert lint.messages == [f"{slug}/details.html — not checked in a browser: no browser on this deployment"]
    assert lint.page_checks == {(slug, "details.html"): "unverified"}


def test_a_finding_on_an_edited_page_still_reaches_the_agent(monkeypatch, tmp_path: Path):
    from anton.core.artifacts.html_lint import HtmlFinding
    from anton.core.tools import tool_handlers

    monkeypatch.delenv("ANTON_RENDER_CHECK_URL", raising=False)
    monkeypatch.setattr(
        "anton.core.artifacts.html_lint.lint_html",
        lambda path, *, cell_sandbox: [HtmlFinding(kind="console_error", detail="boom")],
    )
    store, slug = _dashboard(tmp_path)
    before = tool_handlers.snapshot_existing_artifacts(store)
    (store.root / slug / "index.html").write_text("<html><body><script>boom()</script></body></html>")

    lint = tool_handlers.lint_changed_artifact_files(store, before)

    assert lint.messages == [f"{slug}/index.html — console_error: boom"]
    assert lint.page_checks == {(slug, "index.html"): "failed"}
