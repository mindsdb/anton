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


def test_corrupt_non_xml_member_is_reported(tmp_path: Path):
    """Only the zip checksum pass sees a damaged image: nothing parses it.
    Stored, not deflated, so a flipped byte breaks the CRC and nothing else."""
    import zipfile

    parts = minimal_pptx_parts()
    parts["[Content_Types].xml"] = parts["[Content_Types].xml"].replace(
        "<Default Extension=\"xml\"",
        '<Default Extension="png" ContentType="image/png"/><Default Extension="xml"',
    )
    deck = tmp_path / "deck.pptx"
    with zipfile.ZipFile(deck, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in parts.items():
            zf.writestr(name, data)
        zf.writestr("ppt/media/image1.png", b"\x89PNG\r\n\x1a\n" + b"pixels" * 20, zipfile.ZIP_STORED)
    data = bytearray(deck.read_bytes())
    data[data.find(b"pixels")] ^= 0xFF
    deck.write_bytes(bytes(data))

    assert "checksum" in _messages(lint_pptx(deck))


def test_oversized_required_part_means_could_not_check(tmp_path: Path, monkeypatch):
    """A part too big to parse is unchecked, not clean: skipping a broken
    `[Content_Types].xml` must not return `[]`."""
    import anton.core.artifacts.ooxml_lint as ooxml_lint

    parts = minimal_pptx_parts()
    parts["[Content_Types].xml"] = parts["[Content_Types].xml"].replace("</Types>", "") + " " * 4096
    deck = write_package(tmp_path / "deck.pptx", parts)
    monkeypatch.setattr(ooxml_lint, "_MAX_XML_PART_BYTES", 1024)

    assert lint_pptx(deck) is None


def test_package_too_large_in_total_means_could_not_check(tmp_path: Path, monkeypatch):
    """Many parts each under the per-part cap must not decompress without
    bound on the agent's turn."""
    import anton.core.artifacts.ooxml_lint as ooxml_lint

    deck = write_minimal_pptx(tmp_path / "deck.pptx")
    monkeypatch.setattr(ooxml_lint, "_MAX_TOTAL_UNCOMPRESSED_BYTES", 1024)

    assert lint_pptx(deck) is None


_STRICT_REL = "http://purl.oclc.org/ooxml/officeDocument/relationships"


def _strict(parts: dict[str, str]) -> dict[str, str]:
    """Strict OOXML swaps the relationship type and `r:` namespace URIs."""
    return {
        name: text.replace(
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships", _STRICT_REL
        )
        for name, text in parts.items()
    }


def test_strict_ooxml_deck_is_clean(tmp_path: Path):
    deck = write_package(tmp_path / "deck.pptx", _strict(minimal_pptx_parts()))

    assert lint_pptx(deck) == []


def test_strict_ooxml_dangling_relationship_id_is_reported(tmp_path: Path):
    parts = _strict(minimal_pptx_parts())
    parts["ppt/presentation.xml"] = parts["ppt/presentation.xml"].replace('r:id="rId2"', 'r:id="rId9"')
    deck = write_package(tmp_path / "deck.pptx", parts)

    assert "rId9" in _messages(lint_pptx(deck))


def test_part_names_match_case_insensitively(tmp_path: Path):
    """OPC part names compare ASCII case-insensitively, so a relationship to
    `slides/Slide1.XML` and an override for `/PPT/Slides/slide1.xml` both
    name the zip entry `ppt/slides/slide1.xml`."""
    parts = minimal_pptx_parts()
    parts["ppt/_rels/presentation.xml.rels"] = parts["ppt/_rels/presentation.xml.rels"].replace(
        'Target="slides/slide1.xml"', 'Target="slides/Slide1.XML"'
    )
    parts["[Content_Types].xml"] = parts["[Content_Types].xml"].replace(
        'PartName="/ppt/slides/slide1.xml"', 'PartName="/PPT/Slides/slide1.xml"'
    )
    deck = write_package(tmp_path / "deck.pptx", parts)

    assert lint_pptx(deck) == []
