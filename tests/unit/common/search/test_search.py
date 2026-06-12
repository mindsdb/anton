"""Unit tests for the provider-agnostic search abstraction (Exa + registry).

No network: the Exa SDK constructor is monkeypatched at
``minds.common.search.exa.AsyncExa`` with a fake whose ``search`` /
``get_contents`` are AsyncMocks returning deterministic objects — mirroring
the SDK-constructor patching used in
``tests/unit/agents/passthrough_agent/test_web_tools.py``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from minds.common.search.base import FetchedContent, SearchProvider, SearchResult
from minds.common.search.registry import get_search_provider


def _fake_search_response(results):
    return SimpleNamespace(results=results, statuses=None)


def _fake_result(**kw):
    base = dict(
        url="https://example.com/a",
        title="A title",
        text="full text body",
        summary=None,
        highlights=["hl one", "hl two"],
        published_date="2026-01-01",
        score=0.42,
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def patched_exa(monkeypatch):
    """Patch AsyncExa with a fake; return the fake instance for assertions."""
    fake = SimpleNamespace(
        search=AsyncMock(),
        get_contents=AsyncMock(),
        client=SimpleNamespace(aclose=AsyncMock()),
    )
    monkeypatch.setattr("minds.common.search.exa.AsyncExa", lambda **_: fake)
    return fake


async def test_exa_search_maps_results(patched_exa):
    from minds.common.search.exa import ExaProvider

    patched_exa.search.return_value = _fake_search_response([_fake_result()])
    provider = ExaProvider(api_key="k", max_results=5)

    results = await provider.search("hello", num_results=3)

    assert isinstance(results, list) and len(results) == 1
    r = results[0]
    assert isinstance(r, SearchResult)
    assert r.url == "https://example.com/a"
    assert r.title == "A title"
    assert r.snippet == "hl one … hl two"  # highlights joined
    assert r.published_date == "2026-01-01"
    assert r.score == 0.42
    # num_results is clamped to the provider's max_results.
    assert patched_exa.search.await_args.kwargs["num_results"] == 3
    # Search requests only highlights, not full-page text (fetch_url does that).
    assert patched_exa.search.await_args.kwargs["contents"] == {"highlights": True}


async def test_exa_search_empty_snippet_when_no_highlights(patched_exa):
    from minds.common.search.exa import ExaProvider

    # No highlights → empty snippet (no text fetched for search results); the
    # model still gets title + url and can read the page via fetch_url.
    patched_exa.search.return_value = _fake_search_response([_fake_result(highlights=None)])
    provider = ExaProvider(api_key="k")

    results = await provider.search("q")
    assert results[0].snippet == ""


async def test_exa_search_clamps_to_max_results(patched_exa):
    from minds.common.search.exa import ExaProvider

    patched_exa.search.return_value = _fake_search_response([])
    provider = ExaProvider(api_key="k", max_results=2)

    await provider.search("q", num_results=50)
    assert patched_exa.search.await_args.kwargs["num_results"] == 2


async def test_exa_fetch_truncates(patched_exa):
    from minds.common.search.exa import ExaProvider

    long_text = "x" * 100
    patched_exa.get_contents.return_value = SimpleNamespace(
        results=[_fake_result(text=long_text)], statuses=[SimpleNamespace(status="success", error=None)]
    )
    provider = ExaProvider(api_key="k")

    content = await provider.fetch("https://example.com/a", char_limit=10)
    assert isinstance(content, FetchedContent)
    assert content.text == "x" * 10
    assert content.truncated is True


async def test_exa_fetch_handles_error_status(patched_exa):
    from minds.common.search.exa import ExaProvider

    patched_exa.get_contents.return_value = SimpleNamespace(
        results=[], statuses=[SimpleNamespace(status="error", error="boom")]
    )
    provider = ExaProvider(api_key="k")

    content = await provider.fetch("https://example.com/missing")
    assert content.text == ""
    assert content.truncated is False


async def test_exa_provider_satisfies_protocol(patched_exa):
    from minds.common.search.exa import ExaProvider

    assert isinstance(ExaProvider(api_key="k"), SearchProvider)


# --- registry ---------------------------------------------------------------


def _settings_with_search(**overrides):
    fields = dict(provider="exa", exa_api_key="key", max_results=5, fetch_char_limit=20000)
    fields.update(overrides)
    return SimpleNamespace(search=SimpleNamespace(**fields))


def test_registry_resolves_exa(patched_exa):
    provider = get_search_provider(_settings_with_search())
    assert isinstance(provider, SearchProvider)


def test_registry_raises_on_unknown_provider():
    with pytest.raises(ValueError, match="Unknown search provider"):
        get_search_provider(_settings_with_search(provider="bogus"))


def test_registry_raises_when_exa_key_missing():
    with pytest.raises(ValueError, match="SEARCH__EXA_API_KEY"):
        get_search_provider(_settings_with_search(exa_api_key=""))
