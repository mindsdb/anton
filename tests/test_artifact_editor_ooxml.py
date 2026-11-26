"""Word, Excel and PowerPoint artifacts through the artifact editor (ENG-4687).

The documents are built here: an .xlsx with openpyxl (an anton dependency), a
.docx with python-docx when it is installed (the sandbox group) and by hand
otherwise, and a .pptx by hand, as the smallest zip PowerPoint opens. Each is
edited through the same workspace, editor loop and checkers as a web page.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import openpyxl
import pytest

import anton.core.artifact_editor as artifact_editor
from anton.core.artifact_editor import prompts
from anton.core.artifact_editor.checkers import (
    ChainChecker,
    PackageChecker,
    StaticChecker,
    package_findings,
    select_checker,
)
from anton.core.artifact_editor.editor import ArtifactEditor
from anton.core.artifact_editor.quick import run_quick_edit
from anton.core.artifact_editor.records import LOG_ENV_VAR
from anton.core.artifact_editor.router import describe_artifact, route_request
from anton.core.artifact_editor.workspace import (
    ArtifactWorkspace,
    WorkspaceError,
    _part_unview,
    _part_view,
)
from tests.test_artifact_editor_loop import (
    COLOR_PATCH,
    PAGE,
    ScriptedChecker,
    ScriptedLLM,
    Trail,
    diff,
    reply,
    tool_results,
)

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
DECL = "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>\n"


@pytest.fixture(autouse=True)
def _record(tmp_path, monkeypatch):
    monkeypatch.setenv(LOG_ENV_VAR, str(tmp_path / "record.jsonl"))
    monkeypatch.delenv("ANTON_RENDER_CHECK_URL", raising=False)


def _zip(members: dict[str, str | bytes]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, body in members.items():
            zf.writestr(name, body)
    return out.getvalue()


def _members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        return {info.filename: zf.read(info) for info in zf.infolist()}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── documents built in the test ──────────────────────────────────────


def _docx_by_hand() -> bytes:
    return _zip({
        "[Content_Types].xml": DECL + (
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>"
        ),
        "_rels/.rels": DECL + (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="{R}/officeDocument" Target="word/document.xml"/></Relationships>'
        ),
        "word/document.xml": DECL + (
            f'<w:document xmlns:w="{W}"><w:body>'
            "<w:p><w:r><w:t>Quarterly report</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>Revenue grew.</w:t></w:r></w:p>"
            "<w:sectPr/></w:body></w:document>"
        ),
    })


def _make_docx(folder: Path, name: str = "report.docx") -> Path:
    """A Word document with a title and one paragraph: python-docx's when installed."""
    path = folder / name
    try:
        import docx
    except ImportError:
        path.write_bytes(_docx_by_hand())
        return path
    document = docx.Document()
    document.add_paragraph("Quarterly report")
    document.add_paragraph("Revenue grew.")
    document.save(path)
    return path


def _make_xlsx(folder: Path, name: str = "model.xlsx") -> Path:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sales"
    sheet["A1"], sheet["B1"] = "Region", "Revenue"
    sheet["A2"], sheet["B2"] = "North", 120
    path = folder / name
    workbook.save(path)
    return path


P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"


def _make_pptx(folder: Path, name: str = "deck.pptx") -> Path:
    """The smallest presentation package: one slide with a title shape."""
    ct = "application/vnd.openxmlformats-officedocument.presentationml"
    path = folder / name
    path.write_bytes(_zip({
        "[Content_Types].xml": DECL + (
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            f'<Override PartName="/ppt/presentation.xml" ContentType="{ct}.presentation.main+xml"/>'
            f'<Override PartName="/ppt/slides/slide1.xml" ContentType="{ct}.slide+xml"/>'
            "</Types>"
        ),
        "_rels/.rels": DECL + (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="{R}/officeDocument" Target="ppt/presentation.xml"/></Relationships>'
        ),
        "ppt/presentation.xml": DECL + (
            f'<p:presentation xmlns:p="{P}" xmlns:r="{R}"><p:sldIdLst><p:sldId id="256" r:id="rId2"/>'
            '</p:sldIdLst><p:sldSz cx="9144000" cy="6858000"/></p:presentation>'
        ),
        "ppt/_rels/presentation.xml.rels": DECL + (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId2" Type="{R}/slide" Target="slides/slide1.xml"/></Relationships>'
        ),
        "ppt/slides/slide1.xml": DECL + (
            f'<p:sld xmlns:p="{P}" xmlns:a="{A}"><p:cSld><p:spTree>'
            '<p:sp><p:nvSpPr><p:cNvPr id="2" name="Title"/><p:cNvSpPr/><p:nvPr><p:ph type="title"/></p:nvPr>'
            "</p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:p><a:r><a:t>Q3 review</a:t></a:r></a:p></p:txBody>"
            "</p:sp></p:spTree></p:cSld></p:sld>"
        ),
        "ppt/media/image1.png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 32,
    }))
    return path


@pytest.fixture
def doc_folder(tmp_path) -> Path:
    folder = tmp_path / "artifacts" / "report"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text('{"name": "Report", "primary": "report.docx"}', encoding="utf-8")
    _make_docx(folder)
    return folder


def _has_docx() -> bool:
    try:
        import docx  # noqa: F401
    except ImportError:
        return False
    return True


def _word_texts(path: Path) -> list[str]:
    import docx

    return [p.text for p in docx.Document(str(path)).paragraphs]


# ── the flag ─────────────────────────────────────────────────────────


def test_the_package_exposes_the_ooxml_edits_flag():
    assert artifact_editor.OOXML_EDITS is True
    assert "OOXML_EDITS" in artifact_editor.__all__


# ── the view ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("xml", [
    DECL + f'<w:document xmlns:w="{W}"><w:body><w:p><w:r><w:t>Hi</w:t></w:r></w:p></w:body></w:document>',
    "<a><w:t></w:t><w:t>\n</w:t><w:t xml:space=\"preserve\"> x </w:t></a>",
    "<a>\n\n<b/>\n  <c/></a>",
    "<x/><y/>",
    "<si><t>line one\nline two</t></si>",
])
def test_the_part_view_breaks_lines_between_tags_and_undoes_exactly(xml):
    view = _part_view(xml)
    assert _part_unview(view) == xml
    assert "<w:t>\n" not in view.replace("<w:t>\n</w:t>", "")  # a text element is never broken


# ── listing and reading ──────────────────────────────────────────────


def test_a_package_is_listed_as_its_content_parts_never_whole(doc_folder):
    _make_xlsx(doc_folder)
    _make_pptx(doc_folder)
    (doc_folder / "scan.pdf").write_bytes(b"%PDF-1.7\n%\xe2\xe3\n")
    paths = [entry.path for entry in ArtifactWorkspace(doc_folder, kind="document").map()]

    assert "report.docx/word/document.xml" in paths
    assert "model.xlsx/xl/worksheets/sheet1.xml" in paths
    assert "deck.pptx/ppt/slides/slide1.xml" in paths
    assert "deck.pptx/ppt/presentation.xml" in paths
    assert not any(p.endswith((".rels", "[Content_Types].xml", ".png")) for p in paths)
    assert not any(p in ("report.docx", "model.xlsx", "deck.pptx", "scan.pdf") for p in paths)


def test_reading_a_part_unpacks_it_one_tag_per_line(doc_folder):
    ws = ArtifactWorkspace(doc_folder, kind="document", primary="report.docx")
    view = ws.read("report.docx/word/document.xml")
    assert "<w:t>Quarterly report</w:t>" in view.split("\n")
    assert ws.read("report.docx/word/nothing.xml") is None
    # Relationship parts are not listed, but a change that needs one can read it.
    assert "officeDocument" in ws.read("report.docx/_rels/.rels")


def test_the_package_itself_and_its_media_and_a_pdf_are_refused_with_a_reason(doc_folder):
    _make_pptx(doc_folder)
    (doc_folder / "scan.pdf").write_bytes(b"%PDF-1.7\n")
    ws = ArtifactWorkspace(doc_folder, kind="document")

    reason = ws.refuse_reason("report.docx", create=False)
    assert reason and "report.docx/word/document.xml" in reason
    assert "binary part" in ws.refuse_reason("deck.pptx/ppt/media/image1.png", create=False)
    assert "PDF" in ws.refuse_reason("scan.pdf", create=False)
    assert "no part" in ws.refuse_reason("report.docx/word/missing.xml", create=False)
    assert "no document" in ws.refuse_reason("other.docx/word/document.xml", create=True)
    with pytest.raises(WorkspaceError, match="Word document"):
        ws.read("report.docx")


def test_search_finds_text_inside_a_part(doc_folder):
    ws = ArtifactWorkspace(doc_folder, kind="document")
    (hit,) = ws.search("Revenue grew")
    assert hit.path == "report.docx/word/document.xml"
    assert ws.read(hit.path).split("\n")[hit.line - 1].strip() == hit.text


def test_preload_takes_the_content_parts_not_styles_or_themes(doc_folder):
    _make_xlsx(doc_folder)
    preload = ArtifactWorkspace(doc_folder, kind="document").preload()
    assert preload is not None
    assert "report.docx/word/document.xml" in preload
    assert "model.xlsx/xl/worksheets/sheet1.xml" in preload
    assert not any("styles" in p or "theme" in p for p in preload)


# ── writing: one part, repacked ──────────────────────────────────────


def test_a_docx_part_is_patched_and_the_package_repacked(doc_folder):
    path = doc_folder / "report.docx"
    before = _members(path)
    ws = ArtifactWorkspace(doc_folder, kind="document", primary="report.docx")
    rel = "report.docx/word/document.xml"
    ws.write({rel: ws.read(rel).replace("<w:t>Quarterly report</w:t>", "<w:t>Annual report</w:t>")})

    after = _members(path)
    assert list(after) == list(before)  # same members, same order: [Content_Types].xml first
    assert [n for n in after if after[n] != before[n]] == ["word/document.xml"]
    assert b"<w:t>Annual report</w:t>" in after["word/document.xml"]
    # Nothing else in the part moved: only the text changed.
    assert after["word/document.xml"] == before["word/document.xml"].replace(b"Quarterly", b"Annual")
    assert not list(doc_folder.glob("*.anton-tmp"))
    if _has_docx():
        assert _word_texts(path)[:2] == ["Annual report", "Revenue grew."]


def test_an_xlsx_cell_is_patched_and_openpyxl_reads_the_new_values(tmp_path):
    path = _make_xlsx(tmp_path)
    ws = ArtifactWorkspace(tmp_path, kind="document")
    rel = "model.xlsx/xl/worksheets/sheet1.xml"
    view = ws.read(rel)
    assert "<v>120</v>" in view and "<t>North</t>" in view
    ws.write({rel: view.replace("<v>120</v>", "<v>150</v>").replace("<t>North</t>", "<t>South</t>")})

    sheet = openpyxl.load_workbook(path)["Sales"]
    assert (sheet["A2"].value, sheet["B2"].value) == ("South", 150)


def test_a_pptx_slide_is_patched_and_the_package_still_opens(tmp_path):
    path = _make_pptx(tmp_path)
    ws = ArtifactWorkspace(tmp_path, kind="document")
    rel = "deck.pptx/ppt/slides/slide1.xml"
    ws.write({rel: ws.read(rel).replace("<a:t>Q3 review</a:t>", "<a:t>Q4 review</a:t>")})

    members = _members(path)
    assert b"<a:t>Q4 review</a:t>" in members["ppt/slides/slide1.xml"]
    assert members["ppt/media/image1.png"].startswith(b"\x89PNG")
    assert package_findings("deck.pptx", path.read_bytes(), ["ppt/slides/slide1.xml"]) == []
    try:
        import pptx
    except ImportError:
        return
    title = pptx.Presentation(str(path)).slides[0].shapes.title
    assert title.text == "Q4 review"


def test_a_new_part_is_added_and_an_unchanged_write_changes_nothing(doc_folder):
    path = doc_folder / "report.docx"
    ws = ArtifactWorkspace(doc_folder, kind="document")
    rel = "report.docx/word/document.xml"
    digest = _digest(path)
    ws.write({rel: ws.read(rel)})
    assert _digest(path) == digest  # a no-op is not a repack

    ws.write({"report.docx/word/extra.xml": "<extra/>"})
    assert _members(path)["word/extra.xml"] == b"<extra/>"


def test_text_files_and_parts_land_in_one_transaction(doc_folder, monkeypatch):
    from anton.core.artifact_editor import workspace as ws_module

    (doc_folder / "notes.md").write_text("old\n", encoding="utf-8")
    path = doc_folder / "report.docx"
    digest = _digest(path)
    ws = ArtifactWorkspace(doc_folder, kind="document")
    rel = "report.docx/word/document.xml"
    calls = {"n": 0}
    real = ws_module._replace_at

    def flaky(src, dst, dir_fd):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(28, "No space left on device")
        real(src, dst, dir_fd)

    monkeypatch.setattr(ws_module, "_replace_at", flaky)
    with pytest.raises(WorkspaceError, match="put back"):
        ws.write({"notes.md": "new\n", rel: ws.read(rel).replace("Revenue grew.", "Revenue fell.")})
    assert (doc_folder / "notes.md").read_text() == "old\n"
    assert _digest(path) == digest


def test_a_symlinked_package_is_refused(tmp_path):
    (tmp_path / "elsewhere").mkdir()
    real = _make_xlsx(tmp_path / "elsewhere")
    folder = tmp_path / "art"
    folder.mkdir()
    (folder / "model.xlsx").symlink_to(real)
    ws = ArtifactWorkspace(folder, kind="document")
    assert ws.map() == []
    assert "symlink" in ws.refuse_reason("model.xlsx/xl/worksheets/sheet1.xml", create=False)


# ── snapshot and restore ─────────────────────────────────────────────


def test_restoring_a_part_snapshot_puts_the_package_back_byte_for_byte(doc_folder):
    path = doc_folder / "report.docx"
    original = path.read_bytes()
    ws = ArtifactWorkspace(doc_folder, kind="document")
    rel = "report.docx/word/document.xml"
    snapshot = ws.snapshot([rel, "report.docx/word/styles.xml"])
    assert snapshot.paths == ["report.docx"]

    ws.write({rel: ws.read(rel).replace("Revenue grew.", "Revenue fell.")})
    assert path.read_bytes() != original
    snapshot.restore()
    assert path.read_bytes() == original


# ── the package check ────────────────────────────────────────────────


async def test_the_package_checker_passes_a_sound_edit_and_refuses_broken_xml(doc_folder):
    ws = ArtifactWorkspace(doc_folder, kind="document")
    rel = "report.docx/word/document.xml"
    ws.write({rel: ws.read(rel).replace("Revenue grew.", "Revenue fell.")})
    passed = await PackageChecker().check(ws, [rel])
    assert passed.status == "passed" and passed.channel == "package"

    ws.write({rel: ws.read(rel).replace("</w:body>", "")})
    failed = await PackageChecker().check(ws, [rel])
    assert failed.status == "failed"
    assert failed.findings[0].kind == "invalid_xml"
    assert rel in failed.findings[0].message


async def test_the_package_checker_refuses_a_package_that_is_not_a_zip(doc_folder):
    (doc_folder / "report.docx").write_bytes(b"not a zip at all")
    findings = package_findings("report.docx", b"not a zip at all", ["word/document.xml"])
    assert findings and findings[0].kind == "broken_document"


def test_the_package_check_asks_python_docx_when_it_is_installed():
    pytest.importorskip("docx")
    # Well-formed XML everywhere, but no officeDocument relationship: Word's
    # library cannot find the document's body.
    broken = _zip({
        "[Content_Types].xml": DECL + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>',
        "word/document.xml": DECL + f'<w:document xmlns:w="{W}"/>',
    })
    findings = package_findings("report.docx", broken, ["word/document.xml"])
    assert findings and findings[0].kind == "broken_document" and "Word document" in findings[0].message


async def test_the_package_checker_is_silent_for_an_edit_with_no_document(tmp_path):
    (tmp_path / "index.html").write_text(PAGE, encoding="utf-8")
    ws = ArtifactWorkspace(tmp_path, kind="html-app")
    result = await PackageChecker().check(ws, ["index.html"])
    assert (result.status, result.note) == ("unverified", "")
    # The chain then answers exactly as it did before documents existed.
    chained = await select_checker("html-app").check(ws, ["index.html"])
    assert chained.channel == "static" and chained.status == "unverified"
    assert "no browser check" in chained.note


def test_select_checker_leads_with_the_package_check():
    assert select_checker("document").checkers[0].channel == "package"


# ── the editor loop on a document ────────────────────────────────────

DOC_PATCH = "@report.docx/word/document.xml\n-<w:t>Quarterly report</w:t>\n+<w:t>Annual report</w:t>\n"
BROKEN_PATCH = "@report.docx/word/document.xml\n-</w:body>\n"


def _doc_editor(folder: Path, llm, checker=None, **kw) -> ArtifactEditor:
    ws = ArtifactWorkspace(folder, kind="document", primary="report.docx")
    checker = checker or ChainChecker([PackageChecker(), StaticChecker(node=None)])
    return ArtifactEditor(ws, llm, checker, Trail(), slug=folder.name, name="Report", **kw)


async def test_the_editor_changes_a_word_title_in_one_call(doc_folder):
    llm = ScriptedLLM([reply(diff(DOC_PATCH, note="Retitled the report."))])
    outcome = await _doc_editor(doc_folder, llm).edit("change the title to Annual report")

    assert outcome.status == "applied"
    assert outcome.files == ["report.docx/word/document.xml"]
    assert outcome.check.status == "passed" and outcome.check.channel == "package"
    assert llm.calls and "DOCUMENTS" in llm.calls[0]["system"]
    first = tool_results(llm.calls[0]["messages"])
    assert any("<w:t>Quarterly report</w:t>" in r for r in first)  # the main part was preloaded
    assert b"Annual report" in _members(doc_folder / "report.docx")["word/document.xml"]
    if _has_docx():
        assert _word_texts(doc_folder / "report.docx")[0] == "Annual report"


async def test_a_broken_package_goes_back_once_then_is_restored_byte_identical(doc_folder):
    path = doc_folder / "report.docx"
    original = path.read_bytes()
    llm = ScriptedLLM([
        reply(diff(BROKEN_PATCH)),
        reply(diff("@report.docx/word/document.xml\n-</w:document>\n")),
    ])
    outcome = await _doc_editor(doc_folder, llm).edit("tidy the document")

    assert outcome.status == "restored"
    assert outcome.check is not None and outcome.check.channel == "package"
    assert outcome.check.findings[0].kind == "invalid_xml"
    assert path.read_bytes() == original
    # The first failure went back to the model, with what was wrong.
    assert any("not well-formed XML" in t for m in llm.calls[1]["messages"]
               for t in ([m["content"]] if isinstance(m["content"], str) else
                         [b.get("text", "") for b in m["content"] if isinstance(b, dict)]))


async def test_a_quick_edit_on_a_docx_lands_one_change(doc_folder):
    llm = ScriptedLLM([reply(diff(DOC_PATCH, note="Retitled the report."))])
    result = await run_quick_edit(doc_folder, "change the title to Annual report", kind="document", llm=llm,
                                  checker=None)
    assert result.status == "applied"
    assert result.changed == ["report.docx/word/document.xml"]
    assert result.check is not None and result.check.status == "passed"


# ── PDFs go to the agent ─────────────────────────────────────────────


async def test_a_pdf_only_artifact_is_routed_to_the_agent_without_a_call():
    decision = await route_request("make the title red", files=["report.pdf", "metadata.json"], kind="document",
                                   base_url="https://hub.example/v1", api_key="k-test", http=object())
    assert (decision.route, decision.reason) == ("agent", "pdf_only")


async def test_a_quick_edit_on_a_pdf_escalates_before_any_model_call(tmp_path):
    folder = tmp_path / "scan"
    folder.mkdir()
    (folder / "report.pdf").write_bytes(b"%PDF-1.7\n%\xe2\xe3\n")
    llm = ScriptedLLM([])
    result = await run_quick_edit(folder, "make the title red", kind="document", llm=llm,
                                  checker=ScriptedChecker())
    assert result.status == "escalate"
    assert "PDF" in (result.escalate_reason or "")
    assert llm.calls == []
    assert (folder / "report.pdf").read_bytes().startswith(b"%PDF")


def test_the_router_outline_reads_a_documents_text(doc_folder):
    _make_pptx(doc_folder)
    outline = describe_artifact(doc_folder)["outline"]
    assert "report.docx" in outline and "deck.pptx" in outline
    assert "Quarterly report" in outline and "Q3 review" in outline


# ── text files are unchanged ─────────────────────────────────────────


async def test_a_page_next_to_a_document_is_edited_exactly_as_before(tmp_path):
    folder = tmp_path / "sales"
    folder.mkdir()
    (folder / "index.html").write_text(PAGE, encoding="utf-8")
    _make_docx(folder)
    llm = ScriptedLLM([reply(diff(COLOR_PATCH))])
    ws = ArtifactWorkspace(folder, kind="html-app", primary="index.html")
    outcome = await ArtifactEditor(ws, llm, ScriptedChecker(), Trail(), slug="sales").edit("navy heading")

    assert outcome.status == "applied" and outcome.files == ["index.html"]
    assert "color: navy" in (folder / "index.html").read_text()
    assert prompts.APPLY_DIFF in [t["name"] for t in llm.calls[0]["tools"]]
