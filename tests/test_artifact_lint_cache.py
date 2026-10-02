"""Per-file lint results are reused while the file is unchanged.

The checkers can run LibreOffice or a headless browser, so a second lint of
the same bytes (exec, then open) must not run them again. A file that
changed, or a checker that could not run at all, is checked again.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from anton.core.tools import tool_handlers as th


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


def _counting_linter(monkeypatch, result: list[str] | None) -> list[Path]:
    calls: list[Path] = []

    def _lint(path: Path) -> list[str] | None:
        calls.append(path)
        return result

    monkeypatch.setattr(th, "_artifact_linters", lambda: {".html": _lint})
    return calls


def test_unchanged_file_is_checked_once(store: _FakeStore, monkeypatch):
    (store.root / "page-abc12345" / "index.html").write_text("<p>hi</p>")
    calls = _counting_linter(monkeypatch, ["console error"])

    first = th.lint_artifact_files(store, "page-abc12345")
    second = th.lint_artifact_files(store, "page-abc12345")

    assert first == second == ["page-abc12345/index.html — console error"]
    assert len(calls) == 1


def test_clean_result_is_reused_too(store: _FakeStore, monkeypatch):
    (store.root / "page-abc12345" / "index.html").write_text("<p>hi</p>")
    calls = _counting_linter(monkeypatch, [])

    assert th.lint_artifact_files(store, "page-abc12345") == []
    assert th.lint_artifact_files(store, "page-abc12345") == []
    assert len(calls) == 1


def test_rewritten_file_is_checked_again(store: _FakeStore, monkeypatch):
    page = store.root / "page-abc12345" / "index.html"
    page.write_text("<p>hi</p>")
    calls = _counting_linter(monkeypatch, [])

    th.lint_artifact_files(store, "page-abc12345")
    page.write_text("<p>hello again</p>")
    st = page.stat()
    os.utime(page, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    th.lint_artifact_files(store, "page-abc12345")

    assert len(calls) == 2


def test_could_not_check_is_retried(store: _FakeStore, monkeypatch):
    (store.root / "page-abc12345" / "index.html").write_text("<p>hi</p>")
    calls = _counting_linter(monkeypatch, None)

    th.lint_artifact_files(store, "page-abc12345")
    th.lint_artifact_files(store, "page-abc12345")

    assert len(calls) == 2


def test_cache_drops_oldest_entry_past_its_bound(store: _FakeStore, monkeypatch):
    folder = store.root / "page-abc12345"
    for name in ("a.html", "b.html", "c.html"):
        (folder / name).write_text(name)
    _counting_linter(monkeypatch, [])
    monkeypatch.setattr(th, "_LINT_CACHE_MAX_ENTRIES", 2)

    th.lint_artifact_files(store, "page-abc12345")

    assert len(th._lint_cache) == 2
