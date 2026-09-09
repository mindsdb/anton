"""LibreOffice-backed recalculation check for `.xlsx` artifacts (ENG-1204).

`xlsx_lint.py` catches same-sheet circular refs structurally, without an
evaluator. This module is the oracle path: when LibreOffice is present, use
it to actually recalculate the workbook, then read back whatever error
values that produced. Catches more than circularity — any formula error
(`#REF!`, `#DIV/0!`, ...) the structural lint has no way to see.

`check_xlsx_via_office` returns `None` when the oracle couldn't run at all
(no LibreOffice, crash, timeout) so the caller knows to fall back to the
structural lint instead of treating "didn't run" as "found nothing".
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from anton.core.artifacts.xlsx_lint import _formula_text

# Headless conversion writes into a scratch profile per call, so parallel
# runs never contend for the same lock file (the LibreOffice "document
# already in use" dialog is otherwise a self-inflicted hang). Circular refs
# don't hang headless soffice — it writes `Err:522` into the cell instead
# of raising the interactive dialog — but a cold profile can still be slow
# the very first time a host builds its font cache, hence the generous cap.
_TIMEOUT_SECONDS = 25

# Any Excel/LibreOffice error literal a formula can evaluate to.
_ERROR_PREFIXES = ("#", "Err:")


@dataclass(frozen=True)
class FormulaErrorFinding:
    """A formula that LibreOffice itself could not evaluate cleanly."""

    sheet: str
    cell: str
    formula: str
    error: str

    def message(self) -> str:
        return (
            f"{self.sheet}!{self.cell}: formula '{self.formula}' recalculated "
            f"to {self.error} in LibreOffice."
        )


def check_xlsx_via_office(path: Path) -> list[FormulaErrorFinding] | None:
    """Recalculate `path` with LibreOffice and flag formulas that error out.

    Returns `None` when the oracle isn't usable right now (no LibreOffice on
    this host, the conversion crashed or timed out, or the output couldn't be
    read back) — never an empty list for "couldn't check", so a caller can
    tell "verified clean" apart from "unverified" and fall back accordingly.
    """
    try:
        return _check_via_office(path)
    except Exception:
        return None


def _check_via_office(path: Path) -> list[FormulaErrorFinding] | None:
    office = _discover_office()
    if office is None:
        return None

    with tempfile.TemporaryDirectory(prefix="anton-xlsx-lint-") as tmp:
        profile_dir = Path(tmp) / "profile"
        outdir = Path(tmp) / "out"
        outdir.mkdir()
        recalculated = _convert(office, path, outdir, profile_dir)
        if recalculated is None:
            return None
        return _diff_formula_errors(path, recalculated)


def _discover_office() -> str | None:
    override = os.environ.get("ANTON_XLSX_LINT_OFFICE")
    if override:
        if os.path.isfile(override) and os.access(override, os.X_OK):
            return override
        return shutil.which(override)
    return shutil.which("soffice") or shutil.which("libreoffice")


def _convert(office: str, path: Path, outdir: Path, profile_dir: Path) -> Path | None:
    cmd = [
        office,
        "--headless",
        "--norestore",
        "--nologo",
        "--nofirststartwizard",
        f"-env:UserInstallation=file://{profile_dir}",
        "--convert-to",
        "xlsx",
        "--outdir",
        str(outdir),
        str(path),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,  # own process group, so a timeout can kill the whole tree
    )
    try:
        proc.communicate(timeout=_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()
        return None
    if proc.returncode != 0:
        return None

    out_path = outdir / f"{path.stem}.xlsx"
    return out_path if out_path.is_file() else None


def _diff_formula_errors(original: Path, recalculated: Path) -> list[FormulaErrorFinding]:
    import openpyxl
    
    findings: list[FormulaErrorFinding] = []
    formulas_wb = openpyxl.load_workbook(original, data_only=False)
    values_wb = openpyxl.load_workbook(recalculated, data_only=True)
    try:
        for ws in formulas_wb.worksheets:
            if ws.title not in values_wb.sheetnames:
                continue
            values_ws = values_wb[ws.title]
            for row in ws.iter_rows():
                for cell in row:
                    formula = _formula_text(cell)
                    if formula is None:
                        continue
                    value = values_ws[cell.coordinate].value
                    if isinstance(value, str) and value.startswith(_ERROR_PREFIXES):
                        findings.append(
                            FormulaErrorFinding(
                                sheet=ws.title,
                                cell=cell.coordinate,
                                formula=formula,
                                error=value,
                            )
                        )
    finally:
        formulas_wb.close()
        values_wb.close()
    return findings
