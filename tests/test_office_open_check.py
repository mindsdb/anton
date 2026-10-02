"""Coverage for the LibreOffice "does it open" oracle used for `.pptx`/`.docx`.

Same honest contract as `check_xlsx_via_office`: `None` when the oracle
could not run (no LibreOffice, timeout), `[]` when LibreOffice opened and
exported the file, and a `FileLoadFinding` when it ran and refused it.

Fake `soffice` scripts stand in for LibreOffice so the three outcomes are
tested on any host; the `requires_office` tests run the real binary when
one is installed.
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

import pytest

from anton.core.artifacts import office_open_check
from anton.core.artifacts.office_open_check import check_opens_via_office
from anton.core.artifacts.xlsx_office_check import FileLoadFinding
from tests.ooxml_fixtures import write_minimal_pptx

requires_office = pytest.mark.skipif(
    not (os.environ.get("ANTON_XLSX_LINT_OFFICE") or shutil.which("soffice") or shutil.which("libreoffice")),
    reason="no soffice/libreoffice binary found on PATH or via ANTON_XLSX_LINT_OFFICE",
)

# Mimics `soffice --convert-to pdf --outdir DIR FILE`: writes DIR/<stem>.pdf.
_FAKE_OK = """#!/bin/sh
outdir=""; src=""
while [ $# -gt 0 ]; do
  case "$1" in --outdir) outdir="$2"; shift ;; *) src="$1" ;; esac
  shift
done
name=$(basename "$src"); echo "%PDF" > "$outdir/${name%.*}.pdf"
"""

# LibreOffice's real refusal shape: exit 0, a message, no output file.
_FAKE_REJECT = """#!/bin/sh
echo "Error: source file could not be loaded" >&2
exit 0
"""

_FAKE_HANG = """#!/bin/sh
sleep 30
"""


def _fake_office(tmp_path: Path, monkeypatch, script: str) -> None:
    exe = tmp_path / "fake-soffice"
    exe.write_text(script)
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("ANTON_XLSX_LINT_OFFICE", str(exe))


@pytest.fixture
def deck(tmp_path: Path) -> Path:
    return write_minimal_pptx(tmp_path / "deck.pptx")


def test_no_office_configured_yields_none(monkeypatch, deck: Path):
    monkeypatch.delenv("ANTON_XLSX_LINT_OFFICE", raising=False)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    assert check_opens_via_office(deck) is None


def test_office_path_that_does_not_exist_yields_none(monkeypatch, deck: Path):
    monkeypatch.setenv("ANTON_XLSX_LINT_OFFICE", "/no/such/binary-xyz")
    assert check_opens_via_office(deck) is None


def test_file_office_opens_has_no_findings(tmp_path: Path, monkeypatch, deck: Path):
    _fake_office(tmp_path, monkeypatch, _FAKE_OK)
    assert check_opens_via_office(deck) == []


def test_file_office_refuses_is_a_finding(tmp_path: Path, monkeypatch, deck: Path):
    _fake_office(tmp_path, monkeypatch, _FAKE_REJECT)

    findings = check_opens_via_office(deck)

    assert findings is not None and len(findings) == 1
    assert isinstance(findings[0], FileLoadFinding)
    assert "could not be loaded" in findings[0].message()


def test_timeout_yields_none(tmp_path: Path, monkeypatch, deck: Path):
    """A timeout says nothing about the file, so it must not read as
    'refused' and must not read as 'clean' either."""
    _fake_office(tmp_path, monkeypatch, _FAKE_HANG)
    monkeypatch.setattr(office_open_check, "_TIMEOUT_SECONDS", 1)

    assert check_opens_via_office(deck) is None


@requires_office
def test_real_office_opens_the_minimal_deck(deck: Path):
    assert check_opens_via_office(deck) == []
