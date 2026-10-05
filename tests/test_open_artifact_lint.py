"""`open_artifact` re-checks the artifact's files and reports what it finds.

The exec-time lint shows a finding once, in the result of the cell that wrote
the file. These pin the second chance: a broken file written earlier (another
cell, another turn) is flagged again whenever the agent opens the artifact.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import openpyxl
import pytest

from anton.core.artifacts import ArtifactStore
from anton.core.tools import tool_handlers as th
from anton.core.tools.tool_handlers import handle_open_artifact, lint_artifact_files

_LINT_HEADER = "\n\n[artifact lint]\n"


class _FakeWorkspace:
    def __init__(self, root: Path) -> None:
        self.artifacts_dir = root


class _FakeSession:
    """The handful of ChatSession attributes the artifact handlers read."""

    def __init__(self, root: Path) -> None:
        self._workspace = _FakeWorkspace(root)
        self._session_id = "conv-1"
        self._turn_count = 0
        self._artifacts_touched: set[str] = set()
        self._data_vault = None


@pytest.fixture(autouse=True)
def _no_office_and_empty_cache(monkeypatch):
    monkeypatch.delenv("ANTON_XLSX_LINT_OFFICE", raising=False)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    th._lint_cache.clear()
    yield
    th._lint_cache.clear()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


@pytest.fixture
def session(root: Path) -> _FakeSession:
    return _FakeSession(root)


def _create(root: Path) -> tuple[str, Path]:
    artifact = ArtifactStore(root).create(name="Forecast", description="d", type="document")
    return artifact.slug, root / artifact.slug


def _write_workbook(path: Path, sheets: dict[str, dict[str, str]]) -> None:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, cells in sheets.items():
        ws = wb.create_sheet(title=name)
        for ref, formula in cells.items():
            ws[ref] = f"={formula}"
    wb.save(path)


async def test_broken_workbook_written_earlier_is_flagged_on_open(session, root):
    slug, folder = _create(root)
    _write_workbook(
        folder / "forecast.xlsx",
        {"Actuals": {"E6": "100"}, "Assumptions": {"E6": "SLOPE(E6:J6,{1,2,3,4,5,6})"}},
    )

    outcome = await handle_open_artifact(session, {"slug": slug})

    assert outcome.ok is True
    assert _LINT_HEADER in outcome.content
    lint_text = outcome.content.split(_LINT_HEADER, 1)[1]
    assert f"{slug}/forecast.xlsx" in lint_text
    assert "E6" in lint_text


async def test_clean_workbook_returns_the_plain_descriptor(session, root):
    slug, folder = _create(root)
    _write_workbook(
        folder / "forecast.xlsx",
        {"Actuals": {"E6": "100"}, "Assumptions": {"E6": "SLOPE(Actuals!E6:J6,{1,2,3,4,5,6})"}},
    )

    outcome = await handle_open_artifact(session, {"slug": slug})

    assert "[artifact lint]" not in outcome.content
    assert json.loads(outcome.content)["slug"] == slug


async def test_result_with_findings_still_starts_with_the_descriptor(session, root, monkeypatch):
    slug, folder = _create(root)
    (folder / "index.html").write_text("<p>hi</p>")
    monkeypatch.setattr(
        th, "_artifact_linters", lambda: {".html": lambda p: th._LintResult(["console error"])}
    )

    outcome = await handle_open_artifact(session, {"slug": slug})

    descriptor = json.loads(outcome.content.split(_LINT_HEADER, 1)[0])
    assert descriptor["slug"] == slug
    assert descriptor["path"] == str(folder)


async def test_html_finding_is_surfaced_on_open(session, root, monkeypatch):
    slug, folder = _create(root)
    (folder / "index.html").write_text("<script>boom()</script>")
    monkeypatch.setattr(
        th, "_artifact_linters",
        lambda: {".html": lambda p: th._LintResult(["console error: ReferenceError: boom is not defined"])},
    )

    outcome = await handle_open_artifact(session, {"slug": slug})

    assert f"{slug}/index.html — console error: ReferenceError" in outcome.content


async def test_raising_checker_does_not_fail_open(session, root, monkeypatch):
    slug, folder = _create(root)
    (folder / "index.html").write_text("<p>hi</p>")

    def _boom(_path: Path) -> th._LintResult:
        raise RuntimeError("checker crashed")

    monkeypatch.setattr(th, "_artifact_linters", lambda: {".html": _boom})

    outcome = await handle_open_artifact(session, {"slug": slug})

    assert outcome.ok is True
    descriptor, lint_text = outcome.content.split(_LINT_HEADER, 1)
    assert json.loads(descriptor)["slug"] == slug
    assert lint_text == f"{slug}/index.html — not checked: checker error"


async def test_failing_checker_setup_does_not_fail_open(session, root, monkeypatch):
    slug, folder = _create(root)
    (folder / "index.html").write_text("<p>hi</p>")

    def _no_linters():
        raise ImportError("checker module missing")

    monkeypatch.setattr(th, "_artifact_linters", _no_linters)

    outcome = await handle_open_artifact(session, {"slug": slug})

    assert outcome.ok is True
    assert json.loads(outcome.content)["slug"] == slug


async def test_raising_checker_keeps_other_files_findings(session, root, monkeypatch):
    slug, folder = _create(root)
    (folder / "index.html").write_text("<p>hi</p>")
    _write_workbook(
        folder / "forecast.xlsx",
        {"Actuals": {"E6": "100"}, "Assumptions": {"E6": "SLOPE(E6:J6,{1,2,3,4,5,6})"}},
    )
    real_linters = th._artifact_linters()

    def _boom(_path: Path) -> th._LintResult:
        raise RuntimeError("checker crashed")

    monkeypatch.setattr(
        th, "_artifact_linters", lambda: {".html": _boom, ".xlsx": real_linters[".xlsx"]}
    )

    outcome = await handle_open_artifact(session, {"slug": slug})

    lint_text = outcome.content.split(_LINT_HEADER, 1)[1]
    assert f"{slug}/index.html — not checked: checker error" in lint_text
    assert f"{slug}/forecast.xlsx" in lint_text


async def test_lint_on_open_runs_off_the_event_loop(session, root, monkeypatch):
    """The checkers shell out with multi-second timeouts; run inline they
    would stall the shared loop (and the cloud turn's heartbeat with it)."""
    slug, _ = _create(root)
    to_thread_funcs = []
    real_to_thread = asyncio.to_thread

    async def spying_to_thread(func, *args, **kwargs):
        to_thread_funcs.append(func)
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", spying_to_thread)

    await handle_open_artifact(session, {"slug": slug})

    assert lint_artifact_files in to_thread_funcs


def test_files_past_the_budget_are_named_as_not_checked(root, monkeypatch):
    slug, folder = _create(root)
    (folder / "a.html").write_text("a")
    (folder / "b.html").write_text("b")
    checked: list[str] = []

    def _lint(path: Path) -> th._LintResult:
        checked.append(path.name)
        return th._LintResult(["console error"])

    monkeypatch.setattr(th, "_artifact_linters", lambda: {".html": _lint})
    # deadline computed at 0, first file starts at 0, second at 20
    clock = iter([0.0, 0.0, 20.0])
    monkeypatch.setattr(th, "time", SimpleNamespace(monotonic=lambda: next(clock)))

    messages = lint_artifact_files(SimpleNamespace(root=root), slug, budget_seconds=10.0)

    assert len(checked) == 1
    skipped = [m for m in messages if "not checked" in m]
    assert len(skipped) == 1
    assert checked[0] not in skipped[0]


def test_remembered_results_are_reported_even_with_no_budget_left(root, monkeypatch):
    slug, folder = _create(root)
    (folder / "a.html").write_text("a")
    (folder / "b.html").write_text("b")
    monkeypatch.setattr(
        th, "_artifact_linters", lambda: {".html": lambda p: th._LintResult(["console error"])}
    )
    store = SimpleNamespace(root=root)
    lint_artifact_files(store, slug)

    messages = lint_artifact_files(store, slug, budget_seconds=0.0)

    assert sorted(messages) == [
        f"{slug}/a.html — console error",
        f"{slug}/b.html — console error",
    ]
