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

import asyncio
import html
import ipaddress
import logging
import os
import socket
import ssl
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx

from anton.core.tools.tool_defs import ToolDef

if TYPE_CHECKING:
    from anton.core.session import ChatSession

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# External search provider adapters
# ─────────────────────────────────────────────────────────────────────────────

EXA_SEARCH_ENDPOINT = "https://api.exa.ai/search"
BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

_HTTP_TIMEOUT = 30.0

# ─────────────────────────────────────────────────────────────────────────────
# Retry policy (transient failures only)
# ─────────────────────────────────────────────────────────────────────────────

# web_fetch is GET-only, so retries are always idempotent. We retry ONLY
# transient failures and fail fast on permanent ones:
# - retry:    transient DNS (EAI_AGAIN), connect/read errors, timeouts, HTTP 5xx
# - no retry: NXDOMAIN, SSL/certificate errors, HTTP 4xx
# 2 attempts (1 retry): a transient blip almost always clears on the first
# retry, and a second retry mostly adds latency — up to _HTTP_TIMEOUT per extra
# attempt on a hang. Only ~6/42 observed failures were transient at all.
_MAX_FETCH_ATTEMPTS = 2
_FETCH_BACKOFF_BASE_S = 0.5  # single 0.5s pause before the retry

# Only these 5xx codes are transient. Others (501 Not Implemented, 505 HTTP
# Version Not Supported, 511 Network Authentication Required, ...) are permanent
# and must fail fast — retrying them just adds latency.
_RETRYABLE_STATUS = frozenset({500, 502, 503, 504})

# glibc EAI_AGAIN ("Temporary failure in name resolution") is a transient DNS
# error worth retrying; EAI_NONAME ("Name or service not known") is a permanent
# NXDOMAIN and must fail fast. getattr keeps import safe on platforms lacking it.
_TRANSIENT_GAI_ERRNOS = {getattr(socket, "EAI_AGAIN", -3)}


class _TransientFetchError(Exception):
    """Retryable fetch failure: transient DNS, connect/read error, timeout, or 5xx.

    The message is caller-facing — it is surfaced verbatim once retries are
    exhausted, so it must read as a normal ``web_fetch`` error string.
    """


@dataclass(frozen=True)
class _FetchResult:
    """Result of a single ``_fetch_once`` attempt, carrying audit metadata.

    ``status`` is the HTTP status code for any received response, or a short
    label ("blocked", "transport_error") for failures that never got one.
    """

    text: str  # caller-facing content or error message
    status: int | str
    num_bytes: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# SSRF guard
# ─────────────────────────────────────────────────────────────────────────────

# Private/loopback/link-local/cloud-metadata ranges that must never be fetched
# server-side.  Cloud instance metadata (169.254.169.254) lives in link-local.
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),      # loopback
    ipaddress.ip_network("::1/128"),           # IPv6 loopback
    ipaddress.ip_network("10.0.0.0/8"),        # RFC1918 private
    ipaddress.ip_network("172.16.0.0/12"),     # RFC1918 private
    ipaddress.ip_network("192.168.0.0/16"),    # RFC1918 private
    ipaddress.ip_network("169.254.0.0/16"),    # link-local / cloud metadata
    ipaddress.ip_network("fe80::/10"),         # IPv6 link-local
    ipaddress.ip_network("fc00::/7"),          # IPv6 unique-local
    ipaddress.ip_network("100.64.0.0/10"),     # carrier-grade NAT / GCP metadata
    ipaddress.ip_network("0.0.0.0/8"),         # unspecified
]


def _is_blocked_ip(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return True  # unparseable address → block
    return any(ip in net for net in _BLOCKED_NETWORKS)


def _check_url_ssrf(url: str) -> str | None:
    """Return an error string if *url* targets a private/internal host, else None.

    Resolves the hostname to its IP addresses and rejects any that fall in
    private/loopback/link-local/cloud-metadata ranges.  This prevents both
    direct private-IP requests and 302-to-internal redirect bypasses (each
    redirect target is checked before following).

    Set ANTON_ALLOW_PRIVATE_FETCH=1 to disable (self-hosted / LAN use only).
    """
    if os.environ.get("ANTON_ALLOW_PRIVATE_FETCH") == "1":
        return None

    try:
        parsed = urlparse(url)
        host = parsed.hostname
        if not host:
            return f"Invalid URL — could not parse hostname: {url!r}"

        # Resolve all A/AAAA records and reject if any land in a blocked range.
        # Using all records (not just the first) guards against DNS round-robin
        # where one record is public and another is internal.
        try:
            infos = socket.getaddrinfo(host, None)
        except socket.gaierror as exc:
            if exc.errno in _TRANSIENT_GAI_ERRNOS:
                raise _TransientFetchError(
                    f"DNS temporarily failed for {host!r}: {exc}"
                ) from exc
            return f"Could not resolve host {host!r}: {exc}"

        addrs = {info[4][0] for info in infos}
        for addr in addrs:
            if _is_blocked_ip(addr):
                return (
                    f"Fetch blocked: {host!r} resolves to a private or "
                    f"reserved address ({addr}). "
                    "Set ANTON_ALLOW_PRIVATE_FETCH=1 to allow fetching from "
                    "private/LAN addresses (self-hosted deployments only)."
                )
    except _TransientFetchError:
        raise  # let the retry loop handle transient DNS failures
    except Exception as exc:
        return f"SSRF pre-flight check failed for {url!r}: {exc}"

    return None


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


async def _fetch_once(url: str, max_chars: int) -> _FetchResult:
    """One fetch attempt.

    Raises ``_TransientFetchError`` on retryable failures (transient DNS,
    connect/read errors, timeouts, HTTP 5xx); returns a ``_FetchResult`` on
    success or on any permanent failure (SSRF block, NXDOMAIN, SSL, HTTP 4xx).
    """
    # SSRF guard: resolve the initial URL before opening any connection.
    if err := _check_url_ssrf(url):
        return _FetchResult(err, status="blocked")

    try:
        # follow_redirects=False so we can inspect each redirect target before
        # following it — prevents a public URL redirecting to an internal one.
        async with httpx.AsyncClient(
            timeout=_HTTP_TIMEOUT, follow_redirects=False
        ) as client:
            resp = await client.get(url, headers={"User-Agent": "AntonBot/1.0"})

            # Manually follow redirects, checking each destination for SSRF.
            hops = 0
            while resp.is_redirect and hops < 10:
                location = resp.headers.get("location", "")
                if not location:
                    break
                # Resolve relative redirects against the current URL.
                next_url = str(resp.next_request.url) if resp.next_request else location
                if err := _check_url_ssrf(next_url):
                    return _FetchResult(err, status="blocked")
                resp = await client.send(resp.next_request)
                hops += 1
    except httpx.TimeoutException as exc:
        raise _TransientFetchError(
            f"Fetch timed out after {_HTTP_TIMEOUT}s for {url}"
        ) from exc
    except httpx.HTTPError as exc:
        # A ConnectError wrapping an SSL/certificate failure is a permanent
        # config problem; every other transport error is a transient network
        # blip worth retrying.
        is_ssl = isinstance(exc, httpx.ConnectError) and isinstance(
            exc.__cause__, ssl.SSLError
        )
        if isinstance(exc, httpx.TransportError) and not is_ssl:
            raise _TransientFetchError(f"Fetch failed for {url}: {exc}") from exc
        return _FetchResult(f"Fetch failed for {url}: {exc}", status="transport_error")

    if resp.status_code in _RETRYABLE_STATUS:
        raise _TransientFetchError(f"Fetch returned HTTP {resp.status_code} for {url}")
    if resp.status_code >= 400:
        return _FetchResult(
            f"Fetch returned HTTP {resp.status_code} for {url}", status=resp.status_code
        )

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

    num_bytes = len(resp.content)
    header = f"Fetched {url} (HTTP {resp.status_code}, {num_bytes} bytes)"
    suffix = "\n... [truncated]" if truncated else ""
    return _FetchResult(
        f"{header}\n\n{text}{suffix}",
        status=resp.status_code,
        num_bytes=num_bytes,
    )


def _redact_url(url: str) -> str:
    """Reduce a URL to scheme://host[:port]/path for logging.

    A model-supplied URL can carry credentials in the query (``?api_key=…``) or
    in userinfo (``user:pass@host``); dropping both — plus the fragment — keeps
    the audit line useful without leaking them.
    """
    try:
        parts = urlparse(url)
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
    except ValueError:
        return "<unparseable-url>"
    base = f"{parts.scheme}://{host}{parts.path}"
    return f"{base}?<redacted>" if parts.query else base


def _log_fetch(
    url: str,
    status: object,
    num_bytes: int,
    attempts: int,
    started: float,
    *,
    level: int,
) -> None:
    """Emit one structured audit line per web_fetch call."""
    logger.log(
        level,
        "web_fetch url=%s method=GET status=%s bytes=%d attempts=%d elapsed_ms=%d",
        _redact_url(url),
        status,
        num_bytes,
        attempts,
        int((time.monotonic() - started) * 1000),
    )


async def _fetch_url(url: str, max_chars: int) -> str:
    """GET a URL with bounded retry on transient failures; return text content.

    GET is idempotent, so retrying is safe. Permanent failures (4xx, NXDOMAIN,
    SSL) short-circuit inside ``_fetch_once`` and never reach a second attempt.
    Emits exactly one structured log line per call, whatever the outcome.
    """
    started = time.monotonic()
    last_error = ""
    for attempt in range(1, _MAX_FETCH_ATTEMPTS + 1):
        try:
            outcome = await _fetch_once(url, max_chars)
        except _TransientFetchError as exc:
            last_error = str(exc)
            if attempt < _MAX_FETCH_ATTEMPTS:
                await asyncio.sleep(_FETCH_BACKOFF_BASE_S * 2 ** (attempt - 1))
            continue
        _log_fetch(
            url, outcome.status, outcome.num_bytes, attempt, started, level=logging.INFO
        )
        return outcome.text

    _log_fetch(
        url, "transient_giveup", 0, _MAX_FETCH_ATTEMPTS, started, level=logging.WARNING
    )
    return f"{last_error} (gave up after {_MAX_FETCH_ATTEMPTS} attempts)"


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
