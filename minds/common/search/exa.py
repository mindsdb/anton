"""Exa.ai implementation of :class:`SearchProvider`.

Wraps the official async client (``exa_py.AsyncExa``); all of its methods are
awaitable coroutines, so no thread-pool offload is needed. The ``exa_py``
import is kept inside this module so other search providers don't drag the
dependency into their import graph.

Exa's modern ``search`` returns text contents inline via the ``contents``
argument (``search_and_contents`` is deprecated), and ``get_contents`` carries
a per-URL ``statuses`` list we check before trusting the extracted text.
"""

from __future__ import annotations

from typing import Any

from exa_py import AsyncExa

from minds.common.logger import setup_logging
from minds.common.search.base import FetchedContent, SearchResult

logger = setup_logging()


class ExaProvider:
    """:class:`SearchProvider` backed by the Exa.ai API."""

    def __init__(self, api_key: str, *, max_results: int = 5) -> None:
        self._exa = AsyncExa(api_key=api_key)
        self._max_results = max_results

    async def search(self, query: str, *, num_results: int = 5) -> list[SearchResult]:
        n = min(num_results, self._max_results)
        # Request only ``highlights`` (short, query-relevant excerpts) for the
        # snippet — NOT full page ``text``. Exa bills text and highlights as
        # separate content types, and full-page reads are served by the
        # explicit ``fetch_url`` tool (Exa get_contents), so fetching text here
        # would be paid-for content we never surface.
        resp = await self._exa.search(query, num_results=n, type="auto", contents={"highlights": True})
        return [self._to_result(r) for r in (resp.results or [])]

    async def fetch(self, url: str, *, char_limit: int = 20000) -> FetchedContent:
        resp = await self._exa.get_contents([url], text=True, summary=True)
        results = resp.results or []
        # Per-URL status lives on ``statuses``; bail to an explicit empty
        # result rather than surfacing a half-crawled page as if it succeeded.
        statuses = getattr(resp, "statuses", None) or []
        if statuses and getattr(statuses[0], "status", "success") != "success":
            err = getattr(statuses[0], "error", None)
            logger.warning("Exa fetch did not succeed", extra={"url": url, "error": str(err)})
            return FetchedContent(url=url, title=None, text="", truncated=False)
        if not results:
            return FetchedContent(url=url, title=None, text="", truncated=False)

        r = results[0]
        text = r.text or r.summary or ""
        truncated = len(text) > char_limit
        return FetchedContent(
            url=getattr(r, "url", url),
            title=getattr(r, "title", None),
            text=text[:char_limit],
            truncated=truncated,
        )

    @staticmethod
    def _to_result(r: Any) -> SearchResult:
        """Map an Exa result object to a vendor-neutral :class:`SearchResult`.

        The snippet is the joined ``highlights`` (short, query-relevant
        excerpts). We don't request full page ``text`` for search results, so
        if a result has no highlights the snippet is empty — the model still
        gets the title + url and can read the page in full via ``fetch_url``.
        """
        highlights = getattr(r, "highlights", None) or []
        snippet = " … ".join(highlights)
        return SearchResult(
            title=getattr(r, "title", None) or getattr(r, "url", ""),
            url=getattr(r, "url", ""),
            snippet=snippet,
            published_date=getattr(r, "published_date", None),
            score=getattr(r, "score", None),
        )
