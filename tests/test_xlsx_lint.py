"""Coverage for the xlsx structural lint (ENG-1204, Fix 2)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import openpyxl
import pytest

from anton.core.artifacts.xlsx_lint import lint_xlsx


def _write_workbook(path: Path, sheets: dict[str, dict[str, str]]) -> None:
    """`sheets`: ordered {sheet_name: {cell_ref: formula_without_leading_equals}}."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, cells in sheets.items():
        ws = wb.create_sheet(title=name)
        for ref, formula in cells.items():
            ws[ref] = f"={formula}"
    wb.save(path)


@pytest.fixture
def broken_forecast(tmp_path: Path) -> Path:
    """Mirrors the customer's `2026 Forecast (7).xlsx`: growth-rate
    formulas on 'Forecast Assumptions' meant to read from 'Actuals
    Jan-Jun' but missing the sheet prefix, so the range loops back
    onto its own row. 15 circular cells, plus one correctly-qualified
    same-sheet self-reference that must NOT be flagged."""
    path = tmp_path / "2026 Forecast.xlsx"
    paired_rows = [6, 7, 8, 12, 26]
    solo_rows = [18, 19, 21, 23, 24]
    cells = {
        # Correctly qualified — same sheet, but explicit prefix. Not circular.
        "E5": "'Forecast Assumptions'!$C$2",
    }
    for row in paired_rows:
        cells[f"E{row}"] = f"SLOPE(E{row}:J{row},{{1,2,3,4,5,6}})"
        cells[f"F{row}"] = f"INTERCEPT(E{row}:J{row},{{1,2,3,4,5,6}})"
    for row in solo_rows:
        cells[f"E{row}"] = f"AVERAGE(E{row}:J{row})"
    _write_workbook(
        path,
        {
            "Actuals Jan-Jun": {"E6": "100"},
            "Forecast Assumptions": cells,
        },
    )
    return path


@pytest.fixture
def clean_forecast(tmp_path: Path) -> Path:
    """Same shape, but every growth-rate formula correctly crosses to
    the actuals sheet — no same-sheet range contains its own cell."""
    path = tmp_path / "clean.xlsx"
    cells = {
        "E5": "'Forecast Assumptions'!$C$2",
        "E6": "SLOPE('Actuals Jan-Jun'!E6:J6,{1,2,3,4,5,6})",
        "F6": "INTERCEPT('Actuals Jan-Jun'!E6:J6,{1,2,3,4,5,6})",
        "E18": "AVERAGE('Actuals Jan-Jun'!E18:J18)",
    }
    _write_workbook(
        path,
        {
            "Actuals Jan-Jun": {"E6": "100"},
            "Forecast Assumptions": cells,
        },
    )
    return path


def test_flags_all_15_circular_cells(broken_forecast: Path):
    findings = lint_xlsx(broken_forecast)
    flagged = {(f.sheet, f.cell) for f in findings}
    paired_rows = [6, 7, 8, 12, 26]
    solo_rows = [18, 19, 21, 23, 24]
    expected = {("Forecast Assumptions", f"E{r}") for r in solo_rows}
    for row in paired_rows:
        expected.add(("Forecast Assumptions", f"E{row}"))
        expected.add(("Forecast Assumptions", f"F{row}"))
    assert flagged == expected
    assert len(expected) == 15


def test_does_not_flag_correctly_qualified_self_reference(broken_forecast: Path):
    findings = lint_xlsx(broken_forecast)
    assert not any(f.cell == "E5" for f in findings)


def test_clean_workbook_has_no_findings(clean_forecast: Path):
    assert lint_xlsx(clean_forecast) == []


def test_unparseable_file_yields_none_not_an_exception(tmp_path: Path):
    """None means 'could not check', distinct from `[]` ('checked, clean') —
    a parse failure must not be reported as if it verified anything."""
    junk = tmp_path / "not_really.xlsx"
    junk.write_bytes(b"this is not a zip file at all")
    assert lint_xlsx(junk) is None


def test_message_names_sheet_cell_and_formula(broken_forecast: Path):
    findings = lint_xlsx(broken_forecast)
    e6 = next(f for f in findings if f.cell == "E6")
    msg = e6.message()
    assert "Forecast Assumptions" in msg
    assert "E6" in msg
    assert "SLOPE" in msg


# ─── Shared formulas ──────────────────────────────────────────────────────
#
# openpyxl's public write API always writes one explicit formula per cell —
# it never emits Excel's `<f t="shared">` optimization. Reproducing a real
# shared formula (as a real Excel-authored workbook would carry, e.g. one a
# user attaches to the chat) needs a hand-built OOXML part.

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
</Types>"""

_ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""

_WORKBOOK = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>"""

_WORKBOOK_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"""


@pytest.fixture
def shared_formula_circular(tmp_path: Path) -> Path:
    """Master cell A1 = `A1:A3` (self-containing range); A2/A3 are shared
    followers with no formula text of their own, translated by openpyxl to
    `A2:A3`/`A3:A3` on read — each still contains its own cell."""
    path = tmp_path / "shared.xlsx"
    sheet_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<sheetData>
<row r="1"><c r="A1"><f t="shared" ref="A1:A3" si="0">SUM(A1:A3)</f></c></row>
<row r="2"><c r="A2"><f t="shared" si="0"/></c></row>
<row r="3"><c r="A3"><f t="shared" si="0"/></c></row>
</sheetData></worksheet>"""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", _CONTENT_TYPES)
        zf.writestr("_rels/.rels", _ROOT_RELS)
        zf.writestr("xl/workbook.xml", _WORKBOOK)
        zf.writestr("xl/_rels/workbook.xml.rels", _WORKBOOK_RELS)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)
    return path


def test_flags_shared_formula_follower_cells(shared_formula_circular: Path):
    """Regression for the gap a raw-XML parser has and openpyxl doesn't:
    follower cells carry no `<f>` text of their own in the file, only
    `t="shared"` — openpyxl translates them, so all three rows must flag."""
    findings = lint_xlsx(shared_formula_circular)
    flagged = {f.cell for f in findings}
    assert flagged == {"A1", "A2", "A3"}
