"""Coverage for the exec-time artifact lint hook's html path (ENG-1204 Fix 3).

Mirrors `test_tool_handlers_xlsx_lint.py` — same `_FakeStore`/`_make_artifact`
shape, exercising `_lint_changed_artifact_files` directly rather than the
standalone `lint_html` (see `test_html_lint.py` for that).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from anton.core.tools.tool_handlers import _lint_changed_artifact_files

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


@pytest.fixture
def store(tmp_path: Path) -> _FakeStore:
    return _FakeStore(tmp_path)


def _make_artifact(store: _FakeStore, slug: str) -> Path:
    folder = store.root / slug
    folder.mkdir()
    (folder / "metadata.json").write_text("{}")
    return folder


def test_no_browser_configured_is_silent(monkeypatch, store: _FakeStore):
    monkeypatch.delenv("ANTON_HTML_LINT_BROWSER", raising=False)
    folder = _make_artifact(store, "dash-abc12345")
    (folder / "dash.html").write_text(_BROKEN_HTML)
    before = {"dash-abc12345": 0.0}

    assert _lint_changed_artifact_files(store, before) == []


@requires_browser
def test_flags_a_changed_html_artifact(store: _FakeStore):
    folder = _make_artifact(store, "dash-abc12345")
    (folder / "dash.html").write_text(_BROKEN_HTML)
    before = {"dash-abc12345": 0.0}

    messages = _lint_changed_artifact_files(store, before)

    assert any("dash-abc12345/dash.html" in m for m in messages)


@requires_browser
def test_skips_artifacts_the_cell_did_not_touch(store: _FakeStore):
    """A broken page already on disk before this cell ran must not be
    re-flagged every unrelated turn — only an mtime bump counts."""
    folder = _make_artifact(store, "dash-abc12345")
    page = folder / "dash.html"
    page.write_text(_BROKEN_HTML)
    current_mtime = page.stat().st_mtime
    before = {"dash-abc12345": current_mtime}

    assert _lint_changed_artifact_files(store, before) == []
