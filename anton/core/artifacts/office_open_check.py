"""LibreOffice "does it open" oracle for `.pptx`/`.docx` artifacts (ENG-2175).

The `.xlsx` oracle (`xlsx_office_check.py`) recalculates a workbook; a deck
or document has nothing to recalculate, so the question here is only
whether LibreOffice can load it and export it to PDF. Reuses that module's
binary discovery (same `ANTON_XLSX_LINT_OFFICE` override) and conversion.

Same honest contract: `None` when the oracle could not run (no
LibreOffice, timeout), `[]` when it opened and exported the file, and a
`FileLoadFinding` when it ran and refused the file.

LibreOffice is more forgiving than PowerPoint and Word, so `[]` here is
evidence, not proof, that the target application opens the file. The
structural lint (`ooxml_lint.py`) runs first and does not depend on it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from anton.core.artifacts.xlsx_office_check import FileLoadFinding, _convert, _discover_office

# A PDF export renders every slide, so a long deck on a cold profile can
# take longer than a recalculation; a timeout is reported as "could not check".
_TIMEOUT_SECONDS = 30


def check_opens_via_office(path: Path) -> list[FileLoadFinding] | None:
    """Open `path` in headless LibreOffice by exporting it to PDF.

    Never raises: any failure of the oracle itself is `None`, never `[]`.
    """
    try:
        return _check(path)
    except Exception:
        return None


def _check(path: Path) -> list[FileLoadFinding] | None:
    office = _discover_office()
    if office is None:
        return None

    with tempfile.TemporaryDirectory(prefix="anton-office-open-") as tmp:
        profile_dir = Path(tmp) / "profile"
        outdir = Path(tmp) / "out"
        outdir.mkdir()
        exported, reject_detail = _convert(
            office, path, outdir, profile_dir, target="pdf", timeout=_TIMEOUT_SECONDS
        )
        if exported is not None:
            return []
        if reject_detail is not None:
            return [FileLoadFinding(detail=reject_detail)]
        return None
