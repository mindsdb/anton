---
name: reading-files
description: 'Recall BEFORE reading a web page, PDF, Word file, image, photo or
  scanned document in the scratchpad. Which libraries to use, how to read a
  scanned page or a photo of text with the model, and how to handle large
  files without losing the work done so far.'
metadata:
  display_name: Reading web pages, documents and scans
  provenance: builtin
---
READING WEB PAGES, DOCUMENTS AND SCANS

Libraries: `requests`, `beautifulsoup4` with `lxml`, `pypdf`, `pdfplumber` (with `pypdfium2`), `python-docx` and `pillow`. Cowork on the web has all of them; on a person's own computer some may be missing. Put every one of these libraries that a cell imports in that cell's `packages` array: one already installed is skipped, and a missing one is installed before the cell runs. An import alone never installs anything.

Use these Python libraries, not command-line tools such as pdftotext, pdftoppm, qpdf, tesseract or LibreOffice. The web sandbox has none of them and a person's computer usually doesn't either, so trying them costs a step.

Keep text out of the conversation. Write extracted text to a file under `.anton/output` and print only a short excerpt and its length; printed output stops at 10,000 characters. Search or read the file for what you need.

WEB PAGE
```python
import requests
from bs4 import BeautifulSoup
r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
r.raise_for_status()
soup = BeautifulSoup(r.text, "lxml")
for tag in soup(["script", "style", "nav", "footer"]):
    tag.decompose()
text = soup.get_text("\n", strip=True)
```
A page that comes back nearly empty is usually built by JavaScript. Say so and try another source rather than parsing it again.

PDF
```python
import pdfplumber
with pdfplumber.open(path) as pdf:
    pages = [(p.extract_text() or "") for p in pdf.pages]
scanned = [i for i, t in enumerate(pages) if len(t.strip()) < 20]
```
Tables: `page.extract_tables()`. Pages in `scanned` have no text layer; read them as images (below).

WORD
```python
import docx
d = docx.Document(path)
text = "\n".join(p.text for p in d.paragraphs)
tables = [[[c.text for c in row.cells] for row in t.rows] for t in d.tables]
```
Only `.docx` can be read. For an old `.doc` file, tell the person plainly and ask for a `.docx` or PDF copy.

IMAGES, PHOTOS AND SCANNED PAGES
Read them with the model. Render each page, then send it to `get_llm()` as an image:
```python
import asyncio, base64, io, pathlib
import pypdfium2 as pdfium
from PIL import Image

llm = get_llm()
SYSTEM = ("Transcribe all text in the image exactly, in reading order. Keep the original "
          "language. Write [illegible] for anything you cannot read; never guess.")

def render(pdf, i):
    return pdf[i].render(scale=2).to_pil()

def to_block(img):
    img = img.convert("RGB")
    img.thumbnail((1600, 1600))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    data = base64.b64encode(buf.getvalue()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}

async def read_page(pdf, i, out_dir):
    # Rendered here, inside the page's own task, so a page that cannot be
    # rendered fails alone instead of stopping the batch.
    block = to_block(render(pdf, i))
    r = await llm.complete_async(system=SYSTEM, messages=[{"role": "user", "content": [
        {"type": "text", "text": f"Page {i + 1}."}, block]}], max_tokens=4000)
    text = (r.content or "").strip()
    if not text:
        raise ValueError("the model returned no text")
    (out_dir / f"page_{i + 1:04d}.txt").write_text(text)

async def read_pages(pdf_path, pages, out_dir, batch=8):
    """Read the pages not saved yet. Returns {page number: error} for pages that failed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(pdf_path)
    todo = [i for i in pages if not (out_dir / f"page_{i + 1:04d}.txt").exists()]
    failed = {}
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        results = await asyncio.gather(*(read_page(pdf, i, out_dir) for i in chunk), return_exceptions=True)
        for i, result in zip(chunk, results):
            if isinstance(result, Exception):
                failed[i + 1] = f"{type(result).__name__}: {result}"[:200]
        progress(f"Read {start + len(chunk)} of {len(todo)} pages")
    return failed

# Call it with asyncio.run: a cell has no running event loop, so a bare `await` is a SyntaxError.
# Read only this file's pages with no text layer: each page sent is a model call.
# Found here for `path` every time, never from an earlier cell or another file.
# (For an image file, there are no pages: send the image itself, as below.)
doc = pdfium.PdfDocument(path)
scanned = [i for i in range(len(doc)) if len(doc[i].get_textpage().get_text_range().strip()) < 20]
out_dir = pathlib.Path(".anton/output/extracted_text") / pathlib.Path(path).stem
failed = asyncio.run(read_pages(path, scanned, out_dir))
print("pages not read:", failed or "none")
```
For a photo or image file, open it with `Image.open(path)` and send it the same way.

Large jobs:
- Count the pages you will read with the model across all files first: scanned pages, photos and images. Text you extract with the libraries above does not count.
- Up to 20 pages: go ahead, and say the number in your progress message.
- More than 20 pages: before reading any, tell the person how many pages there are, that each page is one model call that counts against their usage, and offer to read all of them or start with the first 20. Wait for their answer. Do not ask again if they already approved a job this size in this conversation.
- Set the step's estimated time from the page count: about 2 seconds a page, plus 30. The step is stopped when its time budget runs out.
- Each page is saved as it finishes and skipped if it already exists. A page that fails is reported in `failed` and the rest carry on. If the step is cut off, run the same call again: it continues where it stopped and retries the failed pages once.
- Work from the saved files: search them, then read only the pages you need.
- If the work cannot finish, deliver what the saved pages already show and say which pages are missing. Never end with nothing when some pages were read.

If pages still fail after one retry, or your model cannot take images, deliver what was read and tell the person which pages are missing and why. Never fill in text you did not read.
