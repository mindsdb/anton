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

    def __init__(self, fail=(), empty=()):
        self.images: list[Image.Image] = []
        self.asked: list[str] = []
        self.fail, self.empty = set(fail), set(empty)

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
            return SimpleNamespace(content="  ")
        return SimpleNamespace(content=f"text of {text_part['text']}")


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
    return ns, scanned_pdf, tmp_path / ".anton" / "output" / "extracted_text" / "scan", progress, printed


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
