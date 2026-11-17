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

Libraries: `requests`, `beautifulsoup4` with `lxml`, `pypdf`, `pdfplumber` (with `pypdfium2`), `python-docx` and `pillow`. If an import reports a package as not installed, run the cell again with it in the `packages` array; an import alone never installs anything. There are no command-line tools for this: no pdftotext, pdftoppm, qpdf, tesseract or LibreOffice. Do not try them.

Keep text out of the conversation. Write extracted text to a file in the output folder and print only a short excerpt and its length; printed output stops at 10,000 characters. Search or read the file for what you need.

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

def page_images(pdf_path, pages):
    pdf = pdfium.PdfDocument(pdf_path)
    for i in pages:
        yield i, pdf[i].render(scale=2).to_pil()

def to_block(img):
    img = img.convert("RGB")
    img.thumbnail((1600, 1600))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    data = base64.b64encode(buf.getvalue()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}

async def read_page(i, img, out_dir):
    r = await llm.complete_async(system=SYSTEM, messages=[{"role": "user", "content": [
        {"type": "text", "text": f"Page {i + 1}."}, to_block(img)]}], max_tokens=4000)
    (out_dir / f"page_{i + 1:04d}.txt").write_text(r.content)

async def read_pages(pdf_path, pages, out_dir, batch=8):
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in pages if not (out_dir / f"page_{i + 1:04d}.txt").exists()]
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        await asyncio.gather(*(read_page(i, img, out_dir) for i, img in page_images(pdf_path, chunk)))
        progress(f"Read {start + len(chunk)} of {len(todo)} pages")

# Call it with asyncio.run: a cell has no running event loop, so a bare `await` is a SyntaxError.
out_dir = pathlib.Path("extracted_text") / pathlib.Path(path).stem
asyncio.run(read_pages(path, range(len(pdfium.PdfDocument(path))), out_dir))
```
For a photo or image file, open it with `Image.open(path)` and send it the same way.

Large jobs:
- Count the pages across all files first, and say the number in your progress message. Each page is one model call, so a large job takes time and uses the person's allowance; do not stop to ask, but do not hide it either.
- Each page is saved as it finishes and skipped if it already exists. If the step is cut off, run the same call again and it continues where it stopped.
- Work from the saved files: search them, then read only the pages you need.
- If the work cannot finish, deliver what the saved pages already show and say which pages are missing. Never end with nothing when some pages were read.

If the model cannot read a page, or your model cannot take images, tell the person which pages and why. Never fill in text you did not read.
