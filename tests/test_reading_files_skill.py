"""The `reading-files` skill's code runs against the packages the sandbox image ships.

The skill is instructions the agent copies into scratchpad cells, so a snippet
that does not run costs the user a round on every file. These execute each
snippet as written, against the image's `scratchpad` dependency group. CI runs
this file in its own job with that group installed; elsewhere, it skips.
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

# The CI job that installs the group sets this, so a missing package fails
# there instead of skipping into a green check.
if not os.environ.get("ANTON_REQUIRE_SANDBOX_PACKAGES"):
    for _module in ("pdfplumber", "pypdfium2", "docx", "bs4", "requests"):
        pytest.importorskip(_module)
from PIL import Image, ImageDraw  # noqa: E402

from anton.core.memory import skills as skills_mod  # noqa: E402

SKILL = Path(skills_mod.__file__).parent / "builtin_skills" / "reading-files" / "SKILL.md"


def _blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", SKILL.read_text(), re.S)


def _block(marker: str) -> str:
    (code,) = [b for b in _blocks() if marker in b]
    return code


def _text_pdf(path: Path) -> None:
    # pypdf writes a real text layer without needing a font toolkit.
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=200)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)}),
    })
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 14 Tf 20 150 Td (Quarterly revenue grew 12 percent) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as fh:
        writer.write(fh)


def _scanned_pdf(path: Path, pages: int) -> None:
    images = []
    for i in range(pages):
        img = Image.new("RGB", (1240, 1754), "white")  # A4 at 150 dpi; rendered at 2x
        ImageDraw.Draw(img).text((20, 20), f"scanned page {i + 1}", fill="black")
        images.append(img)
    images[0].save(path, "PDF", save_all=True, append_images=images[1:])


class _FakeLLM:
    """Stands in for get_llm(): records each image it was sent and answers per page.

    Pages in `fail` raise, pages in `empty` get a blank reply, as a gateway
    error or a model that returns nothing would.
    """

    def __init__(self, fail=(), empty=(), stop=None):
        self.images: list[Image.Image] = []
        self.asked: list[str] = []
        self.fail, self.empty = set(fail), set(empty)
        self.stop = stop or {}  # page prompt -> stop_reason, as a cut-off or refused reply

    async def complete_async(self, *, system, messages, max_tokens):
        text_part, image_part = messages[0]["content"]
        assert image_part["type"] == "image"
        assert image_part["source"]["media_type"] == "image/jpeg"
        raw = base64.b64decode(image_part["source"]["data"])
        self.images.append(Image.open(io.BytesIO(raw)))
        self.asked.append(text_part["text"])
        if text_part["text"] in self.fail:
            raise RuntimeError("400 image rejected")
        if text_part["text"] in self.empty:
            return SimpleNamespace(content="  ", stop_reason="end_turn")
        return SimpleNamespace(content=f"text of {text_part['text']}",
                               stop_reason=self.stop.get(text_part["text"], "end_turn"))


def _out_dir(tmp_path: Path, stem: str) -> Path:
    # Saved pages live in one folder per file content: <stem>-<hash>.
    (found,) = (tmp_path / ".anton" / "output" / "extracted_text").glob(f"{stem}-*")
    return found


def _image_pdf(path: Path, img: Image.Image) -> None:
    img.save(path, "PDF")


def _with_text_layer(scan: Path, text_pdf: Path, out: Path) -> None:
    # A scan whose text layer holds only some text: a stamp, or a poor OCR layer.
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter(clone_from=scan)
    writer.pages[0].merge_page(PdfReader(text_pdf).pages[0])
    with out.open("wb") as fh:
        writer.write(fh)


def test_the_scan_section_says_to_copy_the_cell_not_write_an_async_loop():
    # In live Web turns the agent wrote its own async loop around get_llm()
    # instead of copying read_pages, and lost a cell each time to a top-level
    # `await` or `asyncio.run(asyncio.gather(...))`. The instruction sits right
    # above the code, and the code it points at is the one that runs (tests below).
    skill = SKILL.read_text()
    section = skill[skill.index("IMAGES, PHOTOS AND SCANNED PAGES"):skill.index("```python", skill.index("IMAGES, PHOTOS"))]
    assert "Copy this cell as it is" in section
    assert "top-level `await` is a SyntaxError" in section
    assert "asyncio.run(asyncio.gather(...))` fails" in section
    assert "asyncio.run(read_pages(" in _block("read_pages")


def test_pdf_snippet_extracts_text_and_flags_scanned_pages(tmp_path):
    text_pdf, scanned_pdf = tmp_path / "text.pdf", tmp_path / "scan.pdf"
    _text_pdf(text_pdf)
    _scanned_pdf(scanned_pdf, 2)

    ns = {"path": str(text_pdf)}
    exec(_block("pdfplumber.open"), ns)
    assert "Quarterly revenue grew 12 percent" in ns["pages"][0]
    assert ns["scanned"] == []

    ns = {"path": str(scanned_pdf)}
    exec(_block("pdfplumber.open"), ns)
    assert ns["scanned"] == [0, 1]


def _run_scan_snippet(tmp_path, monkeypatch, pages, llm):
    scanned_pdf = tmp_path / "scan.pdf"
    _scanned_pdf(scanned_pdf, pages)
    monkeypatch.chdir(tmp_path)
    progress, printed = [], []
    ns = {"get_llm": lambda: llm, "progress": progress.append, "path": str(scanned_pdf),
          "print": lambda *a: printed.append(" ".join(map(str, a)))}
    # Runs the snippet as written, including the call it tells the agent to make.
    exec(_block("read_pages"), ns)
    return ns, scanned_pdf, _out_dir(tmp_path, "scan"), progress, printed


def test_scan_snippet_reads_every_page_through_the_model_and_resumes(tmp_path, monkeypatch):
    llm = _FakeLLM()
    ns, scanned_pdf, out_dir, progress, printed = _run_scan_snippet(tmp_path, monkeypatch, 3, llm)

    assert sorted(p.name for p in out_dir.iterdir()) == ["page_0001.txt", "page_0002.txt", "page_0003.txt"]
    assert (out_dir / "page_0002.txt").read_text() == "text of Page 2."
    assert len(llm.images) == 3
    assert all(max(img.size) <= 1600 for img in llm.images)
    assert progress == ["Read 3 of 3 pages"]
    assert printed == ["pages not read: none"]

    # A cut-off job is resumed by the same call: saved pages are not sent again.
    (out_dir / "page_0003.txt").unlink()
    assert asyncio.run(ns["read_pages"](str(scanned_pdf), [0, 1, 2], out_dir, batch=2)) == {}
    assert len(llm.images) == 4
    assert (out_dir / "page_0003.txt").exists()
    assert progress[-1] == "Read 1 of 1 pages"


def test_only_scanned_pages_of_a_mixed_pdf_reach_the_model(tmp_path, monkeypatch):
    # Every page sent is a model call against the person's usage, and the
    # ask-above-20 rule counts only scanned pages, so the snippet as written
    # must not send the text pages too.
    from pypdf import PdfReader, PdfWriter

    text_pdf, scan_pdf, mixed = tmp_path / "text.pdf", tmp_path / "scan.pdf", tmp_path / "mixed.pdf"
    _text_pdf(text_pdf)
    _scanned_pdf(scan_pdf, 2)
    writer = PdfWriter()
    for _ in range(5):
        writer.add_page(PdfReader(text_pdf).pages[0])
    for page in PdfReader(scan_pdf).pages:
        writer.add_page(page)
    with mixed.open("wb") as fh:
        writer.write(fh)
    monkeypatch.chdir(tmp_path)
    llm = _FakeLLM()
    printed = []
    ns = {"get_llm": lambda: llm, "progress": lambda message: None, "path": str(mixed),
          "print": lambda *a: printed.append(" ".join(map(str, a)))}

    exec(_block("read_pages"), ns)

    assert llm.asked == ["Page 6.", "Page 7."]
    assert printed == ["pages not read: none"]


def test_the_scan_step_reads_the_current_files_pages_not_a_previous_ones(tmp_path, monkeypatch):
    # A scratchpad keeps its variables between cells. Reading file A with the
    # PDF step and then file B with only the scan step must send B's scanned
    # pages, not A's leftover list.
    from pypdf import PdfReader, PdfWriter

    def mixed(name: str, text_pages: int, scanned_pages: int) -> Path:
        text_pdf, scan_pdf, out = tmp_path / f"{name}-t.pdf", tmp_path / f"{name}-s.pdf", tmp_path / f"{name}.pdf"
        _text_pdf(text_pdf)
        _scanned_pdf(scan_pdf, scanned_pages)
        writer = PdfWriter()
        for _ in range(text_pages):
            writer.add_page(PdfReader(text_pdf).pages[0])
        for page in PdfReader(scan_pdf).pages:
            writer.add_page(page)
        with out.open("wb") as fh:
            writer.write(fh)
        return out

    file_a, file_b = mixed("a", 3, 2), mixed("b", 1, 6)
    monkeypatch.chdir(tmp_path)
    llm = _FakeLLM()
    ns = {"get_llm": lambda: llm, "progress": lambda message: None, "print": lambda *a: None}

    ns["path"] = str(file_a)
    exec(_block("pdfplumber.open"), ns)
    exec(_block("read_pages"), ns)
    assert llm.asked == ["Page 4.", "Page 5."]

    ns["path"] = str(file_b)
    exec(_block("read_pages"), ns)
    assert llm.asked[2:] == [f"Page {n}." for n in range(2, 8)]


def test_a_page_that_always_fails_does_not_stop_the_rest(tmp_path, monkeypatch):
    llm = _FakeLLM(fail={"Page 3."})
    ns, scanned_pdf, out_dir, progress, printed = _run_scan_snippet(tmp_path, monkeypatch, 20, llm)

    # One pass reads every other page and names the one it could not read.
    assert len(list(out_dir.iterdir())) == 19
    assert not (out_dir / "page_0003.txt").exists()
    assert printed == ["pages not read: {3: 'RuntimeError: 400 image rejected'}"]
    assert len(llm.asked) == 20

    # Running it again retries only that page.
    failed = asyncio.run(ns["read_pages"](str(scanned_pdf), range(20), out_dir))
    assert failed == {3: "RuntimeError: 400 image rejected"}
    assert llm.asked[20:] == ["Page 3."]


def test_a_page_that_cannot_be_rendered_does_not_stop_the_rest(tmp_path):
    scanned_pdf = tmp_path / "scan.pdf"
    _scanned_pdf(scanned_pdf, 20)
    llm = _FakeLLM()
    ns = {"get_llm": lambda: llm, "progress": lambda message: None}
    # The snippet's helpers without its final call, so one renderer can be swapped in.
    exec(_block("read_pages").split("# Call it with asyncio.run")[0], ns)
    real_render = ns["render"]

    def render(pdf, i):
        if i == 2:
            raise RuntimeError("page could not be rendered")
        return real_render(pdf, i)

    ns["render"] = render
    out_dir = tmp_path / "out"

    failed = asyncio.run(ns["read_pages"](str(scanned_pdf), range(20), out_dir))

    assert failed == {3: "RuntimeError: page could not be rendered"}
    assert len(list(out_dir.iterdir())) == 19
    assert "Page 3." not in llm.asked


def test_an_empty_reply_is_a_failed_page_not_a_saved_one(tmp_path, monkeypatch):
    llm = _FakeLLM(empty={"Page 2."})
    ns, scanned_pdf, out_dir, progress, printed = _run_scan_snippet(tmp_path, monkeypatch, 3, llm)

    assert sorted(p.name for p in out_dir.iterdir()) == ["page_0001.txt", "page_0003.txt"]
    assert printed == ["pages not read: {2: 'ValueError: the model returned no text'}"]


def test_word_snippet_reads_paragraphs_and_tables(tmp_path):
    import docx

    path = tmp_path / "memo.docx"
    d = docx.Document()
    d.add_paragraph("Board memo")
    table = d.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Region", "North"
    d.save(path)

    ns = {"path": str(path)}
    exec(_block("docx.Document"), ns)

    assert "Board memo" in ns["text"]
    assert ns["tables"] == [[["Region", "North"]]]


def test_web_snippet_returns_page_text_without_scripts(monkeypatch):
    import requests

    html = "<html><script>var x=1;</script><nav>Menu</nav><h1>Prices</h1><p>Gold 2400</p></html>"
    seen = {}

    def fake_get(url, **kwargs):
        seen.update(url=url, **kwargs)
        return SimpleNamespace(text=html, raise_for_status=lambda: None)

    monkeypatch.setattr(requests, "get", fake_get)
    ns = {"url": "https://example.com/prices"}
    exec(_block("requests.get"), ns)

    assert ns["text"].split("\n") == ["Prices", "Gold 2400"]
    assert seen["timeout"] == 30


def test_every_snippet_is_covered():
    # A new snippet must get a test here, or it can rot unnoticed.
    markers = ("pdfplumber.open", "read_pages", "docx.Document", "requests.get")
    assert len(_blocks()) == len(markers)
    for block in _blocks():
        assert any(m in block for m in markers)


def _scan_cell(limit=None) -> str:
    # The cell as the agent runs it, with LIMIT set the way the skill tells it to.
    code = _block("read_pages")
    if limit is not None:
        code = code.replace("LIMIT = None", f"LIMIT = {limit}", 1)
    return code


def test_a_revised_file_with_the_same_name_is_read_again(tmp_path, monkeypatch):
    # Saved pages are found by file content, not name: an updated contract.pdf
    # uploaded in the same conversation must not get the old file's text back.
    path = tmp_path / "contract.pdf"
    monkeypatch.chdir(tmp_path)
    llm = _FakeLLM()
    ns = {"get_llm": lambda: llm, "progress": lambda m: None, "print": lambda *a: None, "path": str(path)}

    _scanned_pdf(path, 2)
    exec(_scan_cell(), ns)
    assert len(llm.asked) == 2

    images = [Image.new("RGB", (1240, 1754), "white") for _ in range(2)]
    ImageDraw.Draw(images[0]).text((20, 20), "revised terms", fill="black")
    images[0].save(path, "PDF", save_all=True, append_images=images[1:])
    exec(_scan_cell(), ns)
    assert len(llm.asked) == 3  # page 1 again; page 2 is blank and needs no call
    assert len(list((tmp_path / ".anton" / "output" / "extracted_text").iterdir())) == 2


def test_choosing_the_first_20_pages_reads_only_20(tmp_path, monkeypatch):
    scanned_pdf = tmp_path / "scan.pdf"
    _scanned_pdf(scanned_pdf, 25)
    monkeypatch.chdir(tmp_path)
    llm, printed = _FakeLLM(), []
    ns = {"get_llm": lambda: llm, "progress": lambda m: None, "path": str(scanned_pdf),
          "print": lambda *a: printed.append(" ".join(map(str, a)))}

    exec(_scan_cell(limit=20), ns)

    assert llm.asked == [f"Page {n}." for n in range(1, 21)]
    assert printed == ["pages not read: none", "5 more scanned pages not read yet"]
    # Running it again with the same choice reads nothing new.
    exec(_scan_cell(limit=20), ns)
    assert len(llm.asked) == 20


def test_a_scan_with_a_stamp_or_ocr_text_layer_is_still_read_by_the_model(tmp_path, monkeypatch):
    # The text layer has more than 20 characters, but an image covers the page.
    text_pdf, scan, stamped = tmp_path / "text.pdf", tmp_path / "scan.pdf", tmp_path / "stamped.pdf"
    _text_pdf(text_pdf)
    _scanned_pdf(scan, 1)
    _with_text_layer(scan, text_pdf, stamped)
    monkeypatch.chdir(tmp_path)
    llm = _FakeLLM()
    ns = {"get_llm": lambda: llm, "progress": lambda m: None, "print": lambda *a: None, "path": str(stamped)}

    exec(_block("pdfplumber.open"), ns)
    assert "Quarterly revenue" in ns["pages"][0]
    assert ns["scanned"] == [0]  # the PDF step and the scan step agree
    exec(_scan_cell(), ns)
    assert ns["scanned"] == [0]
    assert llm.asked == ["Page 1."]

    # A text page with no covering image still isn't sent.
    ns["path"] = str(text_pdf)
    exec(_block("pdfplumber.open"), ns)
    assert ns["scanned"] == []
    exec(_scan_cell(), ns)
    assert llm.asked == ["Page 1."]


def test_a_blank_page_is_saved_as_blank_without_a_model_call(tmp_path, monkeypatch):
    from pypdf import PdfWriter

    white_scan, empty_page = tmp_path / "white.pdf", tmp_path / "empty.pdf"
    _image_pdf(white_scan, Image.new("RGB", (1240, 1754), "white"))
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    with empty_page.open("wb") as fh:
        writer.write(fh)
    monkeypatch.chdir(tmp_path)
    llm, printed = _FakeLLM(), []
    ns = {"get_llm": lambda: llm, "progress": lambda m: None,
          "print": lambda *a: printed.append(" ".join(map(str, a)))}

    for path in (white_scan, empty_page):
        ns["path"] = str(path)
        exec(_scan_cell(), ns)
        assert (_out_dir(tmp_path, path.stem) / "page_0001.txt").read_text() == "[blank page]"
    assert llm.asked == []
    assert printed == ["pages not read: none"] * 2


def test_a_very_large_page_renders_to_about_1600_pixels(tmp_path):
    # At a fixed scale a poster-size page renders to gigabytes and kills the pad.
    from pypdf import PdfWriter

    path = tmp_path / "poster.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=4000, height=3000)
    with path.open("wb") as fh:
        writer.write(fh)
    ns = {"get_llm": lambda: None}
    exec(_block("read_pages").split("# Call it with asyncio.run")[0], ns)

    import pypdfium2 as pdfium
    img = ns["render"](pdfium.PdfDocument(str(path)), 0)
    assert 1590 <= max(img.size) <= 1610


@pytest.mark.parametrize("stop", ["max_tokens", "length", "refusal", "content_filter"])
def test_a_cut_off_or_refused_reply_is_a_failed_page_not_a_saved_one(tmp_path, monkeypatch, stop):
    llm = _FakeLLM(stop={"Page 2.": stop})
    ns, scanned_pdf, out_dir, progress, printed = _run_scan_snippet(tmp_path, monkeypatch, 3, llm)

    assert sorted(p.name for p in out_dir.iterdir()) == ["page_0001.txt", "page_0003.txt"]
    assert "2:" in printed[0] and "ValueError" in printed[0]
    # It is retried on the next run instead of being skipped for good.
    llm.stop = {}
    assert asyncio.run(ns["read_pages"](str(scanned_pdf), [0, 1, 2], out_dir)) == {}
    assert llm.asked[3:] == ["Page 2."]
