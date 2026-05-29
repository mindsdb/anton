"""Handler-dispatched fallbacks for ``web_search`` / ``web_fetch``.

These tools are registered on the session's ``ToolRegistry`` only when the
active LLM provider does *not* expose the equivalent capability natively
(see ``LLMProvider.native_web_tools()``). On Anthropic BYOK, OpenAI BYOK, and
the mdb.ai passthrough the model uses the provider's server-side tools and
this module is dormant.

For generic OpenAI-compatible third-party endpoints (Case 3 in the design):

- ``web_search`` is dispatched to Exa.ai or Brave Search using a key the user
  configured via ``anton setup search``. Without a configured key the handler
  returns a clear error message pointing at that command.
- ``web_fetch`` always works — it is a stdlib-style HTTP GET (via httpx, which
  Anton already depends on transitively through the LLM SDKs) plus a
  lightweight HTML→text stripper, so it does not need a third-party key.

Future enhancement (intentionally deferred from v1): when
``external_search_provider == "exa"`` and ``exa_api_key`` is set, ``web_fetch``
could route through Exa's ``/contents`` endpoint instead of stdlib HTTP for
higher-quality extraction (handles paywalls, JS-rendered nav, ad/boilerplate
stripping). Held back for now to keep behavior uniform across Exa, Brave, and
unconfigured users — the swap is local to ``handle_web_fetch_fallback``.
"""

from __future__ import annotations

import html
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any

import httpx

from anton.core.tools.tool_defs import ToolDef

if TYPE_CHECKING:
    from anton.core.session import ChatSession


# ─────────────────────────────────────────────────────────────────────────────
# External search provider adapters
# ─────────────────────────────────────────────────────────────────────────────

EXA_SEARCH_ENDPOINT = "https://api.exa.ai/search"
BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

_HTTP_TIMEOUT = 30.0


async def _search_exa(query: str, api_key: str, max_results: int) -> str:
    """Hit Exa's ``/search`` endpoint and format hits as markdown."""
    payload: dict[str, Any] = {
        "query": query,
        "num_results": max_results,
        # Include a short excerpt with each result so the model can answer
        # many questions without a follow-up fetch round-trip.
        "contents": {"text": {"max_characters": 600}},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        resp = await client.post(EXA_SEARCH_ENDPOINT, json=payload, headers=headers)
        if resp.status_code != 200:
            return f"Exa search failed ({resp.status_code}): {resp.text[:500]}"
        data = resp.json()

    results = data.get("results") or []
    if not results:
        return f"No results for query: {query!r}"
    lines = [f"Web search results for: {query!r} (Exa, {len(results)} hits)\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title") or r.get("url") or "(untitled)"
        url = r.get("url") or ""
        snippet = (r.get("text") or "").strip()
        if len(snippet) > 600:
            snippet = snippet[:600] + "…"
        lines.append(f"{i}. **{title}**\n   {url}")
        if snippet:
            lines.append(f"   {snippet}")
    return "\n".join(lines)


async def _search_brave(query: str, api_key: str, max_results: int) -> str:
    """Hit Brave Search's web endpoint and format hits as markdown."""
    headers = {
        "X-Subscription-Token": api_key,
        "Accept": "application/json",
    }
    params = {"q": query, "count": max_results}
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        resp = await client.get(BRAVE_SEARCH_ENDPOINT, headers=headers, params=params)
        if resp.status_code != 200:
            return f"Brave search failed ({resp.status_code}): {resp.text[:500]}"
        data = resp.json()

    web = (data.get("web") or {}).get("results") or []
    if not web:
        return f"No results for query: {query!r}"
    lines = [f"Web search results for: {query!r} (Brave, {len(web)} hits)\n"]
    for i, r in enumerate(web, 1):
        title = r.get("title") or r.get("url") or "(untitled)"
        url = r.get("url") or ""
        snippet = (r.get("description") or "").strip()
        lines.append(f"{i}. **{title}**\n   {url}")
        if snippet:
            lines.append(f"   {snippet}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Stdlib HTTP fetch + lightweight HTML extraction
# ─────────────────────────────────────────────────────────────────────────────


class _TextExtractor(HTMLParser):
    """Tiny stdlib-only HTML→text converter.

    Skips ``script``/``style``/``noscript`` content, decodes character refs,
    and normalizes whitespace. Good enough for the model to read article-style
    pages; will produce noisy output for heavily JS-driven SPAs (acceptable
    for v1 — the future Exa ``/contents`` enhancement covers that case).
    """

    _SKIP_TAGS = {"script", "style", "noscript", "head"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        # Block-level tags get an implicit newline so paragraphs don't smush.
        if tag in ("p", "br", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"):
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    def text(self) -> str:
        raw = "".join(self._chunks)
        # Collapse runs of whitespace; preserve paragraph breaks.
        lines = [line.strip() for line in raw.splitlines()]
        return "\n".join(line for line in lines if line)


def _strip_html(body: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(body)
    except Exception:
        # Bail out to a minimal "decode entities" fallback if the parser barfs.
        return html.unescape(body)
    return parser.text()


async def _fetch_url(url: str, max_chars: int) -> str:
    """GET a URL and return its text content, truncated to ``max_chars``."""
    try:
        async with httpx.AsyncClient(
            timeout=_HTTP_TIMEOUT, follow_redirects=True
        ) as client:
            resp = await client.get(url, headers={"User-Agent": "AntonBot/1.0"})
    except httpx.TimeoutException:
        return f"Fetch timed out after {_HTTP_TIMEOUT}s for {url}"
    except httpx.HTTPError as exc:
        return f"Fetch failed for {url}: {exc}"

    if resp.status_code >= 400:
        return f"Fetch returned HTTP {resp.status_code} for {url}"

    content_type = (resp.headers.get("content-type") or "").lower()
    body = resp.text

    if "html" in content_type or body.lstrip().startswith("<"):
        text = _strip_html(body)
    else:
        text = body

    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True

    header = f"Fetched {url} (HTTP {resp.status_code}, {len(resp.content)} bytes)"
    suffix = "\n... [truncated]" if truncated else ""
    return f"{header}\n\n{text}{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# Handlers + ToolDefs
# ─────────────────────────────────────────────────────────────────────────────


_NO_PROVIDER_MSG = (
    "No search provider configured for this LLM endpoint. "
    "Run `anton setup search` to configure Exa.ai or Brave Search."
)


async def handle_web_search_fallback(session: "ChatSession", tc_input: dict) -> str:
    query = (tc_input.get("query") or "").strip()
    if not query:
        return "web_search requires a non-empty `query`."
    max_results = int(tc_input.get("max_results") or 5)
    max_results = max(1, min(max_results, 20))

    settings = session._settings
    provider = (getattr(settings, "external_search_provider", None) or "").lower()

    if provider == "exa":
        key = getattr(settings, "exa_api_key", None)
        if not key:
            return _NO_PROVIDER_MSG
        return await _search_exa(query, key, max_results)
    if provider == "brave":
        key = getattr(settings, "brave_api_key", None)
        if not key:
            return _NO_PROVIDER_MSG
        return await _search_brave(query, key, max_results)

    return _NO_PROVIDER_MSG


async def handle_web_fetch_fallback(session: "ChatSession", tc_input: dict) -> str:
    del session  # unused — fetch needs no settings
    url = (tc_input.get("url") or "").strip()
    if not url:
        return "web_fetch requires a `url`."
    if not (url.startswith("http://") or url.startswith("https://")):
        return f"web_fetch only supports http(s) URLs; got: {url!r}"
    max_chars = int(tc_input.get("max_chars") or 20000)
    max_chars = max(500, min(max_chars, 200_000))
    return await _fetch_url(url, max_chars)


WEB_SEARCH_FALLBACK_TOOL = ToolDef(
    name="web_search",
    description=(
        "Search the web for up-to-date information. Returns a ranked list of "
        "results with title, URL, and a short excerpt. Use this when you need "
        "facts that may have changed recently, breaking news, or to discover "
        "URLs to fetch in detail. Backed by Exa.ai or Brave Search."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return (1-20, default 5).",
            },
        },
        "required": ["query"],
    },
    handler=handle_web_search_fallback,
)


WEB_FETCH_FALLBACK_TOOL = ToolDef(
    name="web_fetch",
    description=(
        "Fetch a URL and return its text content. Strips HTML markup; works "
        "best on article-style pages. Use this after web_search when you need "
        "the full body of a result, or directly when the user provides a URL."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Absolute http(s) URL to fetch.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters to return (default 20000, max 200000).",
            },
        },
        "required": ["url"],
    },
    handler=handle_web_fetch_fallback,
)
