"""Structural lint for `.xlsx` artifacts — same-sheet circular refs.

Fix 2 from ENG-1204: a formula that means `SLOPE('Actuals'!E6:J6, ...)`
but drops the sheet prefix becomes `SLOPE(E6:J6, ...)` inside E6 itself
— a range that eats its own cell. Caught structurally, no evaluator.

Uses openpyxl rather than parsing the OOXML zip by hand, since a
hand-rolled parser misses shared formulas (`<f t="shared">` follower
cells carry no formula text of their own; openpyxl resolves them).

Known limitation: doesn't resolve defined names — one that looks like
a cell reference (e.g. `TAX10`) can false-positive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# A sheet-qualified reference: 'Quoted Name'!A1 or Name!A1:B2. Matched and
# stripped first so its cell/range portion isn't mistaken for a same-sheet one.
_CROSS_SHEET_REF = re.compile(
    r"(?:'[^']+'|[A-Za-z_][A-Za-z0-9_.]*)!\$?[A-Z]{1,3}\$?\d+(?::\$?[A-Z]{1,3}\$?\d+)?"
)

# A bare cell or range reference left after cross-sheet refs are stripped.
# The lookbehind/lookahead avoid matching mid-identifier (e.g. a defined
# name fragment) or a function name that happens to end in digits (LOG10().
_SAME_SHEET_REF = re.compile(
    r"(?<![A-Za-z0-9_!'])\$?([A-Z]{1,3})\$?(\d+)(?::\$?([A-Z]{1,3})\$?(\d+))?(?!\()"
)


@dataclass(frozen=True)
class CircularRefFinding:
    """A formula whose own cell falls inside one of its same-sheet ranges."""

    sheet: str
    cell: str
    formula: str

    def message(self) -> str:
        return (
            f"{self.sheet}!{self.cell}: formula '{self.formula}' references "
            "a same-sheet range that contains its own cell — likely a "
            "missing cross-sheet prefix."
        )


def lint_xlsx(path: Path) -> list[CircularRefFinding] | None:
    """Flag same-sheet formula ranges that contain their own cell.

    Returns `None` when the file could not actually be checked (corrupt,
    not really an xlsx, an openpyxl-unsupported feature, encrypted, ...) —
    distinct from `[]`, which means it parsed fine and nothing was flagged.
    Never raises either way, so a checker failure can't fail the artifact
    read/cell that triggered it.
    """

    try:
        return _lint_xlsx(path)
    except Exception:
        return None


def _lint_xlsx(path: Path) -> list[CircularRefFinding]:
    import openpyxl

    findings: list[CircularRefFinding] = []
    wb = openpyxl.load_workbook(path, data_only=False)
    try:
        for ws in wb.worksheets:
            for row in ws.iter_rows():
                for cell in row:
                    formula = _formula_text(cell)
                    if formula is None:
                        continue
                    remaining = _CROSS_SHEET_REF.sub(" ", formula)
                    if _contains_own_cell(remaining, (cell.column, cell.row)):
                        findings.append(
                            CircularRefFinding(
                                sheet=ws.title, cell=cell.coordinate, formula=formula
                            )
                        )
    finally:
        wb.close()
    return findings


def _formula_text(cell) -> str | None:
    """The formula string for a formula cell, or None for anything else.

    Handles both plain formula cells (`cell.value` is `"=..."`) and
    array-entered ones, which openpyxl surfaces as an `ArrayFormula`
    object with a `.text` attribute instead of a bare string.
    """
    from openpyxl.worksheet.formula import ArrayFormula
    
    if cell.data_type != "f":
        return None
    value = cell.value
    text = value.text if isinstance(value, ArrayFormula) else value
    if not isinstance(text, str):
        return None
    return text[1:] if text.startswith("=") else text


def _contains_own_cell(formula: str, own: tuple[int, int]) -> bool:
    own_col, own_row = own
    for match in _SAME_SHEET_REF.finditer(formula):
        start_col, start_row, end_col, end_row = match.groups()
        c1 = _col_to_num(start_col)
        r1 = int(start_row)
        c2, r2 = (c1, r1) if end_col is None else (_col_to_num(end_col), int(end_row))
        if min(c1, c2) <= own_col <= max(c1, c2) and min(r1, r2) <= own_row <= max(r1, r2):
            return True
    return False


def _col_to_num(col: str) -> int:
    n = 0
    for ch in col:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n
