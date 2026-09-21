"""Coverage for the exec-time artifact lint hook.

`lint_changed_artifact_files` dispatches by file suffix (see
`_artifact_linters`) and wires it into the scratchpad exec path — scoped to
artifact folders whose content mtime moved since the cell started, so an
untouched broken workbook sitting in the workspace isn't re-flagged on
every turn.

`.xlsx` always runs both the LibreOffice oracle and the structural lint
(see `test_xlsx_office_check.py` for the oracle itself, and the merge
tests below for how the two combine). Most tests here force the oracle
unavailable (no office on this "host") so they exercise the
dispatch/scoping logic deterministically, regardless of whether the
machine running them happens to have LibreOffice installed.
"""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pytest

from anton.core.artifacts.xlsx_office_check import FormulaErrorFinding
from anton.core.tools.tool_handlers import lint_changed_artifact_files


@pytest.fixture(autouse=True)
def _no_office(monkeypatch):
    monkeypatch.delenv("ANTON_XLSX_LINT_OFFICE", raising=False)
    monkeypatch.setattr("shutil.which", lambda _name: None)


def _write_workbook(path: Path, sheets: dict[str, dict[str, str]]) -> None:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, cells in sheets.items():
        ws = wb.create_sheet(title=name)
        for ref, formula in cells.items():
            ws[ref] = f"={formula}"
    wb.save(path)


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


def test_flags_circular_refs_in_a_changed_artifact(store: _FakeStore):
    folder = _make_artifact(store, "forecast-abc12345")
    _write_workbook(
        folder / "forecast.xlsx",
        {
            "Actuals": {"E6": "100"},
            "Assumptions": {"E6": "SLOPE(E6:J6,{1,2,3,4,5,6})"},  # missing sheet prefix
        },
    )
    before = {"forecast-abc12345": 0.0}  # older than the mtime the write above just set

    messages = lint_changed_artifact_files(store, before)

    assert len(messages) == 1
    assert "forecast-abc12345/forecast.xlsx" in messages[0]
    assert "E6" in messages[0]


def test_clean_workbook_produces_no_messages_from_the_structural_lint(store: _FakeStore):
    """No LibreOffice on this 'host', so only the structural lint runs. A
    workbook it finds nothing wrong with produces no messages at all."""
    folder = _make_artifact(store, "clean-abc12345")
    _write_workbook(
        folder / "clean.xlsx",
        {
            # SLOPE needs 6 paired points to actually compute — a single
            # value here would recalculate to #DIV/0! and (correctly) trip
            # the LibreOffice oracle even though the reference is qualified.
            "Actuals": {f"{col}6": str(100 + i * 10) for i, col in enumerate("EFGHIJ")},
            "Assumptions": {"E6": "SLOPE('Actuals'!E6:J6,{1,2,3,4,5,6})"},
        },
    )
    before = {"clean-abc12345": 0.0}

    assert lint_changed_artifact_files(store, before) == []


def test_both_checkers_agree_a_clean_workbook_is_clean(monkeypatch, store: _FakeStore):
    """The reverse of the module-level fixture: the oracle actually ran
    (mocked here, since real LibreOffice may not be on this host) and
    agrees with the structural lint — still no messages."""
    monkeypatch.setattr(
        "anton.core.artifacts.xlsx_office_check.check_xlsx_via_office",
        lambda _path: [],
    )
    folder = _make_artifact(store, "clean-abc12345")
    _write_workbook(folder / "clean.xlsx", {"Sheet1": {"A1": "1+1"}})
    before = {"clean-abc12345": 0.0}

    assert lint_changed_artifact_files(store, before) == []


def test_structural_message_wins_when_both_flag_the_same_cell(monkeypatch, store: _FakeStore):
    """The oracle names the symptom (#VALUE!), the structural lint names
    the fix (missing sheet prefix) — for the same cell, only the more
    actionable structural message should reach the agent, not both."""
    monkeypatch.setattr(
        "anton.core.artifacts.xlsx_office_check.check_xlsx_via_office",
        lambda _path: [
            FormulaErrorFinding(
                sheet="Assumptions", cell="E6",
                formula="SLOPE(E6:J6,{1,2,3,4,5,6})", error="#VALUE!",
            )
        ],
    )
    folder = _make_artifact(store, "forecast-abc12345")
    _write_workbook(
        folder / "forecast.xlsx",
        {
            "Actuals": {"E6": "100"},
            "Assumptions": {"E6": "SLOPE(E6:J6,{1,2,3,4,5,6})"},  # missing sheet prefix
        },
    )
    before = {"forecast-abc12345": 0.0}

    messages = lint_changed_artifact_files(store, before)

    assert len(messages) == 1
    assert "missing cross-sheet prefix" in messages[0]
    assert "#VALUE!" not in messages[0]


def test_oracle_only_finding_still_surfaces(monkeypatch, store: _FakeStore):
    """A real formula error unrelated to same-sheet circularity — the
    structural lint has no way to see this, so losing the oracle's finding
    here would silently drop real coverage."""
    monkeypatch.setattr(
        "anton.core.artifacts.xlsx_office_check.check_xlsx_via_office",
        lambda _path: [
            FormulaErrorFinding(
                sheet="Sheet1", cell="B2", formula="A1/0", error="#DIV/0!",
            )
        ],
    )
    folder = _make_artifact(store, "clean-abc12345")
    _write_workbook(folder / "clean.xlsx", {"Sheet1": {"A1": "1", "B2": "A1/0"}})
    before = {"clean-abc12345": 0.0}

    messages = lint_changed_artifact_files(store, before)

    assert len(messages) == 1
    assert "#DIV/0!" in messages[0]


def test_skips_artifacts_the_cell_did_not_touch(store: _FakeStore):
    """A broken workbook already on disk before this cell ran must not be
    re-flagged on every unrelated turn — only an mtime bump counts as
    'this cell touched it'."""
    folder = _make_artifact(store, "forecast-abc12345")
    workbook = folder / "forecast.xlsx"
    _write_workbook(
        workbook,
        {
            "Actuals": {"E6": "100"},
            "Assumptions": {"E6": "SLOPE(E6:J6,{1,2,3,4,5,6})"},
        },
    )
    # `before` mtime equal to (not older than) the file's current mtime: no
    # edit happened during this cell's window.
    current_mtime = workbook.stat().st_mtime
    before = {"forecast-abc12345": current_mtime}

    assert lint_changed_artifact_files(store, before) == []
