"""Coverage for the `.pptx`/`.docx` structural lint (ENG-2175).

A deck that will not open is the failure this guards: the agent wrote
one, called it validated, and nothing checked. `[]` means the package is
well formed; a non-empty list names what is broken; `None` means the file
could not even be read, so nothing is claimed either way.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anton.core.artifacts.ooxml_lint import lint_docx, lint_pptx
from tests.ooxml_fixtures import (
    minimal_docx_parts,
    minimal_pptx_parts,
    write_minimal_docx,
    write_minimal_pptx,
    write_package,
)


def _messages(findings) -> str:
    assert findings, "expected at least one finding"
    return "\n".join(f.message() for f in findings)


def test_minimal_valid_deck_is_clean(tmp_path: Path):
    assert lint_pptx(write_minimal_pptx(tmp_path / "deck.pptx")) == []


def test_truncated_deck_is_reported(tmp_path: Path):
    """The likeliest real-world shape: a chunked binary write that stopped
    early, so the zip's central directory never made it to disk."""
    good = write_minimal_pptx(tmp_path / "good.pptx").read_bytes()
    deck = tmp_path / "deck.pptx"
    deck.write_bytes(good[: len(good) // 2])

    assert "not a valid" in _messages(lint_pptx(deck))


def test_non_zip_is_reported(tmp_path: Path):
    deck = tmp_path / "deck.pptx"
    deck.write_text("<html><body>slides</body></html>")

    assert "not a valid" in _messages(lint_pptx(deck))


def test_empty_file_is_reported(tmp_path: Path):
    deck = tmp_path / "deck.pptx"
    deck.write_bytes(b"")

    assert lint_pptx(deck)


def test_corrupt_member_is_reported(tmp_path: Path):
    """The zip directory is intact but a member's bytes are not: flip a
    byte inside the compressed slide so its CRC no longer matches."""
    deck = write_minimal_pptx(tmp_path / "deck.pptx")
    data = bytearray(deck.read_bytes())
    offset = data.find(b"ppt/slides/slide1.xml") + len("ppt/slides/slide1.xml") + 4
    data[offset] ^= 0xFF
    deck.write_bytes(bytes(data))

    assert lint_pptx(deck)


def test_missing_content_types_is_reported(tmp_path: Path):
    parts = minimal_pptx_parts()
    del parts["[Content_Types].xml"]
    deck = write_package(tmp_path / "deck.pptx", parts)

    assert "[Content_Types].xml" in _messages(lint_pptx(deck))


def test_missing_package_relationships_is_reported(tmp_path: Path):
    parts = minimal_pptx_parts()
    del parts["_rels/.rels"]
    deck = write_package(tmp_path / "deck.pptx", parts)

    assert "_rels/.rels" in _messages(lint_pptx(deck))


def test_slide_referenced_but_missing_is_reported(tmp_path: Path):
    """A hand-rolled deck that lists a slide in the relationships but
    never writes the slide part. PowerPoint refuses the whole file."""
    parts = minimal_pptx_parts()
    del parts["ppt/slides/slide1.xml"]
    del parts["ppt/slides/_rels/slide1.xml.rels"]
    deck = write_package(tmp_path / "deck.pptx", parts)

    assert "ppt/slides/slide1.xml" in _messages(lint_pptx(deck))


def test_malformed_xml_part_is_reported(tmp_path: Path):
    parts = minimal_pptx_parts()
    parts["ppt/slides/slide1.xml"] = parts["ppt/slides/slide1.xml"].replace("</p:sld>", "")
    deck = write_package(tmp_path / "deck.pptx", parts)

    assert "ppt/slides/slide1.xml" in _messages(lint_pptx(deck))


def test_dangling_relationship_id_is_reported(tmp_path: Path):
    """presentation.xml points at `rId9`, which its .rels never defines."""
    parts = minimal_pptx_parts()
    parts["ppt/presentation.xml"] = parts["ppt/presentation.xml"].replace(
        'r:id="rId2"', 'r:id="rId9"'
    )
    deck = write_package(tmp_path / "deck.pptx", parts)

    assert "rId9" in _messages(lint_pptx(deck))


def test_part_without_a_content_type_is_reported(tmp_path: Path):
    parts = minimal_pptx_parts()
    parts["ppt/media/image1.png"] = b"\x89PNG\r\n\x1a\n"
    deck = write_package(tmp_path / "deck.pptx", parts)

    assert "ppt/media/image1.png" in _messages(lint_pptx(deck))


def test_word_document_named_pptx_is_reported(tmp_path: Path):
    """A valid package of the wrong kind still will not open as a deck."""
    deck = write_package(tmp_path / "deck.pptx", minimal_docx_parts())

    assert "presentation" in _messages(lint_pptx(deck))


def test_external_relationship_is_not_a_missing_part(tmp_path: Path):
    parts = minimal_pptx_parts()
    parts["ppt/slides/_rels/slide1.xml.rels"] = parts["ppt/slides/_rels/slide1.xml.rels"].replace(
        "</Relationships>",
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
        'relationships/hyperlink" Target="https://example.com/a%20b" TargetMode="External"/>'
        "</Relationships>",
    )
    deck = write_package(tmp_path / "deck.pptx", parts)

    assert lint_pptx(deck) == []


def test_unreadable_file_means_could_not_check(tmp_path: Path):
    assert lint_pptx(tmp_path / "does-not-exist.pptx") is None


def test_minimal_valid_document_is_clean(tmp_path: Path):
    assert lint_docx(write_minimal_docx(tmp_path / "report.docx")) == []


def test_deck_named_docx_is_reported(tmp_path: Path):
    doc = write_minimal_pptx(tmp_path / "report.docx")

    assert "document" in _messages(lint_docx(doc))


@pytest.mark.parametrize("lint", [lint_pptx, lint_docx])
def test_many_findings_are_capped(tmp_path: Path, lint):
    """A badly broken package must not flood the agent's tool result."""
    parts = minimal_pptx_parts()
    for i in range(60):
        parts[f"ppt/media/extra{i}.bin"] = b"x"
    findings = lint(write_package(tmp_path / "x.pptx", parts))

    assert findings is not None
    assert len(findings) <= 21
