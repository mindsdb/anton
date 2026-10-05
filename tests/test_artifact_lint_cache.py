"""Per-file lint results are reused while the artifact folder is unchanged.

The checkers can run LibreOffice or a headless browser, so a second lint of
the same folder (exec, then open) must not run them again. A change to any
file in the folder, or a checker that could not run at all, means a re-check:
a page's result depends on the scripts and assets next to it.
"""

from __future__ import annotations

import os
from pathlib import Path

import openpyxl
import pytest

from anton.core.tools import tool_handlers as th
from tests.ooxml_fixtures import write_minimal_pptx


class _FakeStore:
    def __init__(self, root: Path) -> None:
        self.root = root


@pytest.fixture(autouse=True)
def _empty_cache():
    th._lint_cache.clear()
    yield
    th._lint_cache.clear()


@pytest.fixture
def store(tmp_path: Path) -> _FakeStore:
    folder = tmp_path / "page-abc12345"
    folder.mkdir()
    (folder / "metadata.json").write_text("{}")
    return _FakeStore(tmp_path)


def _counting_linter(monkeypatch, result: th._LintResult | None) -> list[Path]:
    calls: list[Path] = []

    def _lint(path: Path) -> th._LintResult | None:
        calls.append(path)
        return result

    monkeypatch.setattr(th, "_artifact_linters", lambda: {".html": _lint})
    return calls


def test_unchanged_file_is_checked_once(store: _FakeStore, monkeypatch):
    (store.root / "page-abc12345" / "index.html").write_text("<p>hi</p>")
    calls = _counting_linter(monkeypatch, th._LintResult(["console error"]))

    first = th.lint_artifact_files(store, "page-abc12345")
    second = th.lint_artifact_files(store, "page-abc12345")

    assert first == second == ["page-abc12345/index.html — console error"]
    assert len(calls) == 1


def test_clean_result_is_reused_too(store: _FakeStore, monkeypatch):
    (store.root / "page-abc12345" / "index.html").write_text("<p>hi</p>")
    calls = _counting_linter(monkeypatch, th._LintResult([]))

    assert th.lint_artifact_files(store, "page-abc12345") == []
    assert th.lint_artifact_files(store, "page-abc12345") == []
    assert len(calls) == 1


def test_rewritten_file_is_checked_again(store: _FakeStore, monkeypatch):
    page = store.root / "page-abc12345" / "index.html"
    page.write_text("<p>hi</p>")
    calls = _counting_linter(monkeypatch, th._LintResult([]))

    th.lint_artifact_files(store, "page-abc12345")
    page.write_text("<p>yo</p>")  # same size: only the mtime tells it apart
    st = page.stat()
    os.utime(page, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    th.lint_artifact_files(store, "page-abc12345")

    assert len(calls) == 2


def _page_needs_app_js(monkeypatch) -> None:
    def _lint(path: Path) -> th._LintResult:
        if (path.parent / "app.js").exists():
            return th._LintResult([])
        return th._LintResult(["failed_request: app.js"])

    monkeypatch.setattr(th, "_artifact_linters", lambda: {".html": _lint})


def test_adding_a_missing_sibling_asset_clears_the_finding(store: _FakeStore, monkeypatch):
    folder = store.root / "page-abc12345"
    (folder / "index.html").write_text('<script src="app.js"></script>')
    _page_needs_app_js(monkeypatch)

    assert th.lint_artifact_files(store, "page-abc12345") == [
        "page-abc12345/index.html — failed_request: app.js"
    ]
    (folder / "app.js").write_text("console.log(1)")

    assert th.lint_artifact_files(store, "page-abc12345") == []


def test_deleting_a_sibling_asset_brings_the_finding_back(store: _FakeStore, monkeypatch):
    folder = store.root / "page-abc12345"
    (folder / "index.html").write_text('<script src="app.js"></script>')
    (folder / "app.js").write_text("console.log(1)")
    _page_needs_app_js(monkeypatch)

    assert th.lint_artifact_files(store, "page-abc12345") == []
    (folder / "app.js").unlink()

    assert th.lint_artifact_files(store, "page-abc12345") == [
        "page-abc12345/index.html — failed_request: app.js"
    ]


def test_could_not_check_is_retried(store: _FakeStore, monkeypatch):
    (store.root / "page-abc12345" / "index.html").write_text("<p>hi</p>")
    calls = _counting_linter(monkeypatch, None)

    th.lint_artifact_files(store, "page-abc12345")
    th.lint_artifact_files(store, "page-abc12345")

    assert len(calls) == 2


def test_partial_check_is_reported_but_retried(store: _FakeStore, monkeypatch):
    (store.root / "page-abc12345" / "index.html").write_text("<p>hi</p>")
    calls = _counting_linter(monkeypatch, th._LintResult(["structural issue"], complete=False))

    first = th.lint_artifact_files(store, "page-abc12345")
    th.lint_artifact_files(store, "page-abc12345")

    assert first == ["page-abc12345/index.html — structural issue"]
    assert len(calls) == 2


def test_xlsx_without_office_is_a_partial_check(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("ANTON_XLSX_LINT_OFFICE", raising=False)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    workbook = tmp_path / "book.xlsx"
    wb = openpyxl.Workbook()
    wb.active["A1"] = "=1+1"
    wb.save(workbook)

    result = th._artifact_linters()[".xlsx"](workbook)

    assert result == th._LintResult([], complete=False)


def test_pptx_without_office_is_a_partial_check(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("ANTON_XLSX_LINT_OFFICE", raising=False)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    deck = tmp_path / "deck.pptx"
    write_minimal_pptx(deck)

    result = th._artifact_linters()[".pptx"](deck)

    assert result == th._LintResult([], complete=False)


def test_cache_drops_oldest_entry_past_its_bound(store: _FakeStore, monkeypatch):
    folder = store.root / "page-abc12345"
    for name in ("a.html", "b.html", "c.html"):
        (folder / name).write_text(name)
    _counting_linter(monkeypatch, th._LintResult([]))
    monkeypatch.setattr(th, "_LINT_CACHE_MAX_ENTRIES", 2)

    th.lint_artifact_files(store, "page-abc12345")

    assert len(th._lint_cache) == 2
