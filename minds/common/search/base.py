"""Provider-agnostic web-search abstraction.

The passthrough agent (and potentially other agents) need to run web
searches on a model's behalf when the upstream provider has no hosted
search index — most notably the Fireworks-hosted models, which route
through the Anthropic-shape Messages API but cannot search the web
themselves. Rather than couple that loop to a single vendor, callers
depend on the :class:`SearchProvider` protocol here; concrete providers
(Exa today, Tavily/Serper later) live in sibling modules and are selected
at runtime via :func:`minds.common.search.registry.get_search_provider`.

Result types are deliberately small and vendor-neutral: only the fields an
LLM actually benefits from seeing in a tool result (title, url, a snippet,
recency, relevance). Provider-specific extras are dropped at the provider
boundary so the loop and its callers never grow vendor knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SearchResult:
    """One web-search hit, normalized across providers.

    ``snippet`` is the provider's best short, LLM-ready excerpt — highlights
    when available, otherwise a truncated text/summary. ``published_date``
    and ``score`` are optional because not every provider returns them.
    """

    title: str
    url: str
    snippet: str
    published_date: str | None = None
    score: float | None = None


@dataclass(frozen=True)
class FetchedContent:
    """The extracted contents of a single URL.

    ``truncated`` is True when ``text`` was cut to a caller-supplied
    character budget, so the model can be told it's seeing a prefix.
    """

    url: str
    title: str | None
    text: str
    truncated: bool


@runtime_checkable
class SearchProvider(Protocol):
    """A pluggable web-search backend.

    Implementations must be safe to call from an asyncio event loop without
    blocking it (use an async HTTP client, not a sync one). Both methods may
    raise on transport/credential errors; the tool-execution loop catches
    those and feeds the error back to the model as a recoverable tool
    result, so providers don't need their own retry/swallow logic.
    """

    async def search(self, query: str, *, num_results: int = 5) -> list[SearchResult]:
        """Return up to ``num_results`` hits for ``query``."""
        ...

    async def fetch(self, url: str, *, char_limit: int = 20000) -> FetchedContent:
        """Fetch and extract ``url``, truncating text to ``char_limit`` chars."""
        ...
