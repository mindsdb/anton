"""Search-provider selection.

Maps the ``SEARCH__PROVIDER`` setting to a concrete :class:`SearchProvider`.
Follows the same no-silent-fallback discipline as the passthrough alias
resolver (``minds.inference.model_resolver.ModelResolver._resolve_alias``): an
unknown or misconfigured provider raises rather than quietly degrading to "no
search", so a typo'd env var surfaces loudly instead of silently disabling the
feature.

Adding a provider is a one-line entry in ``_BUILDERS`` plus its module — no
changes at the call sites, which depend only on the :class:`SearchProvider`
protocol.
"""

from __future__ import annotations

from collections.abc import Callable

from minds.common.search.base import SearchProvider
from minds.common.settings.app_settings import AppSettings


def _build_exa(settings: AppSettings) -> SearchProvider:
    # Imported lazily so the exa_py dependency is only required when the Exa
    # provider is actually selected.
    from minds.common.search.exa import ExaProvider

    search = settings.search
    if not search.exa_api_key:
        raise ValueError("SEARCH__PROVIDER=exa but SEARCH__EXA_API_KEY is not set")
    return ExaProvider(api_key=search.exa_api_key, max_results=search.max_results)


_BUILDERS: dict[str, Callable[[AppSettings], SearchProvider]] = {
    "exa": _build_exa,
}


def supported_search_providers() -> list[str]:
    """Names of the search providers this build knows how to construct."""
    return sorted(_BUILDERS)


def get_search_provider(settings: AppSettings, *, provider_name: str | None = None) -> SearchProvider:
    """Return a search provider, raising on an unknown name.

    ``provider_name`` overrides ``settings.search.provider`` (used to honor a
    per-user Statsig selection); the rest of the search settings (api key,
    max_results) still come from ``settings``. With no override the behavior is
    unchanged. The no-silent-fallback discipline is preserved: an unknown name
    raises rather than quietly degrading to "no search".
    """
    name = provider_name or settings.search.provider
    builder = _BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"Unknown search provider {name!r}. Supported: {sorted(_BUILDERS)}")
    return builder(settings)
