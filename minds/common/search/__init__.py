"""Provider-agnostic web search for providers without native search."""

from minds.common.search.base import FetchedContent, SearchProvider, SearchResult
from minds.common.search.registry import get_search_provider, supported_search_providers

__all__ = [
    "FetchedContent",
    "SearchProvider",
    "SearchResult",
    "get_search_provider",
    "supported_search_providers",
]
