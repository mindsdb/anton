"""Coverage for `.pptx`/`.docx` in the exec-time artifact lint hook (ENG-2175).

Mirrors `test_tool_handlers_xlsx_lint.py`: findings reach the agent as
`<slug>/<file> — <message>` lines in the cell's tool result, so a broken
deck is named before the agent can call it validated. Most tests force the
LibreOffice oracle unavailable so they run the same on any host.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anton.core.artifacts.xlsx_office_check import FileLoadFinding
from anton.core.tools.tool_handlers import lint_changed_artifact_files
from tests.ooxml_fixtures import minimal_pptx_parts, write_minimal_docx, write_minimal_pptx, write_package


@pytest.fixture(autouse=True)
def _no_office(monkeypatch):
    monkeypatch.delenv("ANTON_XLSX_LINT_OFFICE", raising=False)
    monkeypatch.setattr("shutil.which", lambda _name: None)


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


def test_truncated_deck_in_a_changed_artifact_is_flagged(store: _FakeStore, tmp_path: Path):
    good = write_minimal_pptx(tmp_path / "good.pptx").read_bytes()
    folder = _make_artifact(store, "synthesis-abc12345")
    (folder / "synthesis.pptx").write_bytes(good[: len(good) // 2])

    messages = lint_changed_artifact_files(store, {"synthesis-abc12345": 0.0})

    assert len(messages) == 1
    assert messages[0].startswith("synthesis-abc12345/synthesis.pptx")
    assert "not a valid" in messages[0]


def test_missing_slide_part_is_flagged(store: _FakeStore):
    parts = minimal_pptx_parts()
    del parts["ppt/slides/slide1.xml"]
    del parts["ppt/slides/_rels/slide1.xml.rels"]
    folder = _make_artifact(store, "deck-abc12345")
    write_package(folder / "deck.pptx", parts)

    messages = lint_changed_artifact_files(store, {"deck-abc12345": 0.0})

    assert messages
    assert all(m.startswith("deck-abc12345/deck.pptx") for m in messages)
    assert any("ppt/slides/slide1.xml" in m for m in messages)


def test_valid_deck_with_no_office_produces_no_messages(store: _FakeStore):
    """Structure checked and clean; the oracle could not run. Nothing to
    report, and nothing claims the oracle passed."""
    folder = _make_artifact(store, "deck-abc12345")
    write_minimal_pptx(folder / "deck.pptx")

    assert lint_changed_artifact_files(store, {"deck-abc12345": 0.0}) == []


def test_office_refusal_of_a_well_formed_deck_still_surfaces(monkeypatch, store: _FakeStore):
    monkeypatch.setattr(
        "anton.core.artifacts.office_open_check.check_opens_via_office",
        lambda _path: [FileLoadFinding(detail="general input/output error")],
    )
    folder = _make_artifact(store, "deck-abc12345")
    write_minimal_pptx(folder / "deck.pptx")

    messages = lint_changed_artifact_files(store, {"deck-abc12345": 0.0})

    assert len(messages) == 1
    assert "LibreOffice could not open this file" in messages[0]


def test_oracle_is_skipped_when_the_structure_is_already_broken(monkeypatch, store: _FakeStore):
    """A package that fails the structural check is already reported; a
    15-second LibreOffice run would only repeat it."""
    calls: list[Path] = []
    monkeypatch.setattr(
        "anton.core.artifacts.office_open_check.check_opens_via_office",
        lambda path: calls.append(path) or [],
    )
    folder = _make_artifact(store, "deck-abc12345")
    (folder / "deck.pptx").write_text("not a zip")

    assert lint_changed_artifact_files(store, {"deck-abc12345": 0.0})
    assert calls == []


def test_broken_docx_is_flagged_too(store: _FakeStore):
    folder = _make_artifact(store, "report-abc12345")
    (folder / "report.docx").write_text("not a zip")

    messages = lint_changed_artifact_files(store, {"report-abc12345": 0.0})

    assert len(messages) == 1
    assert messages[0].startswith("report-abc12345/report.docx")


def test_valid_docx_produces_no_messages(store: _FakeStore):
    folder = _make_artifact(store, "report-abc12345")
    write_minimal_docx(folder / "report.docx")

    assert lint_changed_artifact_files(store, {"report-abc12345": 0.0}) == []


def test_skips_decks_the_cell_did_not_touch(store: _FakeStore):
    folder = _make_artifact(store, "deck-abc12345")
    deck = folder / "deck.pptx"
    deck.write_text("not a zip")

    assert lint_changed_artifact_files(store, {"deck-abc12345": deck.stat().st_mtime}) == []
