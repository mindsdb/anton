"""World Atlas 2.0.2's 110m countries and land, embedded into a published HTML page that downloads
them, so the page draws its map where it may download nothing from another host.

The tested map recipe (generate_artifact's prompts and the build-html-dashboard skill) loads
https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-110m.json with `d3.json`. The app preview
allows that download, but a published link's viewer may not: MindsHub's sends
`connect-src 'self'`, so the map and the globe come back empty there. The files here are the
package's own at 2.0.2, the bytes jsDelivr serves, under the ISC license in LICENSE beside them.
"""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path

FILES = ("countries-110m.json", "land-110m.json")
URL = "https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/{}"
# The id of the script element that carries the files, which also marks a page that has them.
MARKER = "anton-world-atlas"

_HERE = Path(__file__).parent
# Where the files go: first in <head>, else first in <html>, else after the doctype, so they come
# before every script of the page's own.
_PLACES = (
    re.compile(r"<head\b[^>]*>", re.IGNORECASE),
    re.compile(r"<html\b[^>]*>", re.IGNORECASE),
    re.compile(r"<!doctype[^>]*>", re.IGNORECASE),
)
# Answers the page's fetch of an embedded file's address, d3.json's included, from the page's copy,
# and hands every other request to the browser's fetch.
_ANSWER = (
    "<script>(()=>{const atlas=JSON.parse(document.getElementById(%s).textContent);"
    "const download=window.fetch.bind(window);"
    "window.fetch=(input,init)=>{const url=typeof input==='string'?input:input instanceof URL?input.href:input&&input.url;"
    "return Object.prototype.hasOwnProperty.call(atlas,url)"
    "?Promise.resolve(new Response(JSON.stringify(atlas[url]),{status:200,headers:{'Content-Type':'application/json'}}))"
    ":download(input,init);};})();</script>"
)


@cache
def _atlas(name: str) -> object:
    return json.loads((_HERE / name).read_text(encoding="utf-8"))


def embed(html: str) -> str:
    """HTML with each World Atlas file whose address it names embedded, and its fetch of each
    answered from that copy. HTML comes back as it was when it names none, or already has them."""
    named = [URL.format(name) for name in FILES if URL.format(name) in html]
    if not named or f'id="{MARKER}"' in html:
        return html
    # "</" escaped, so no "</script>" in the data ends its element early.
    data = json.dumps({url: _atlas(url.rsplit("/", 1)[1]) for url in named}, separators=(",", ":")).replace("</", "<\\/")
    block = f'<script type="application/json" id="{MARKER}">{data}</script>' + _ANSWER % json.dumps(MARKER)
    for place in _PLACES:
        if (found := place.search(html)) is not None:
            return html[: found.end()] + block + html[found.end():]
    return block + html
