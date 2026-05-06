"""Tests for the handler-dispatched web_search/web_fetch fallbacks and the
session-side routing decision (native vs handler-dispatched).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from anton.core.tools.web_tools import (
    WEB_FETCH_FALLBACK_TOOL,
    WEB_SEARCH_FALLBACK_TOOL,
    _strip_html,
    handle_web_fetch_fallback,
    handle_web_search_fallback,
)


def _session_with_settings(**fields):
    """Build a stand-in session object exposing only ._settings."""
    settings = SimpleNamespace(
        external_search_provider=fields.get("external_search_provider"),
        exa_api_key=fields.get("exa_api_key"),
        brave_api_key=fields.get("brave_api_key"),
    )
    return SimpleNamespace(_settings=settings)


# ─────────────────────────────────────────────────────────────────────────────
# web_search fallback — Exa
# ─────────────────────────────────────────────────────────────────────────────


class TestWebSearchFallbackExa:
    async def test_returns_no_provider_message_when_unconfigured(self):
        session = _session_with_settings()
        result = await handle_web_search_fallback(session, {"query": "anything"})
        assert "anton setup search" in result
        assert "No search provider" in result

    async def test_returns_no_provider_when_provider_set_but_no_key(self):
        session = _session_with_settings(external_search_provider="exa")
        result = await handle_web_search_fallback(session, {"query": "x"})
        assert "anton setup search" in result

    async def test_empty_query_short_circuits(self):
        session = _session_with_settings(
            external_search_provider="exa", exa_api_key="k"
        )
        result = await handle_web_search_fallback(session, {"query": "  "})
        assert "non-empty" in result.lower()

    async def test_calls_exa_endpoint_with_bearer_auth(self):
        session = _session_with_settings(
            external_search_provider="exa", exa_api_key="exa-key-xyz"
        )

        # Capture the outgoing request, return a canned response.
        captured: dict = {}

        async def _post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            request = httpx.Request("POST", url)
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Result A",
                            "url": "https://a.example",
                            "text": "snippet A " * 5,
                        },
                        {
                            "title": "Result B",
                            "url": "https://b.example",
                            "text": "snippet B",
                        },
                    ]
                },
                request=request,
            )

        with patch.object(httpx.AsyncClient, "post", new=_post):
            out = await handle_web_search_fallback(
                session, {"query": "what is anton", "max_results": 2}
            )

        assert captured["url"] == "https://api.exa.ai/search"
        assert captured["headers"]["Authorization"] == "Bearer exa-key-xyz"
        assert captured["json"]["query"] == "what is anton"
        assert captured["json"]["num_results"] == 2
        # Output is markdown-ish with both results. Assert the URL appears as
        # an exact formatted line ("   <url>") rather than via substring `in`
        # — the latter would also pass for "https://a.example.evil.com" and
        # CodeQL's incomplete-URL-substring-sanitization rule (correctly)
        # warns on that pattern even in tests.
        out_lines = out.splitlines()
        assert "Result A" in out
        assert "   https://a.example" in out_lines
        assert "Result B" in out

    async def test_exa_non_200_response_returns_error_string(self):
        session = _session_with_settings(
            external_search_provider="exa", exa_api_key="k"
        )

        async def _post(self, url, json=None, headers=None):
            return httpx.Response(
                401, text="bad key", request=httpx.Request("POST", url)
            )

        with patch.object(httpx.AsyncClient, "post", new=_post):
            out = await handle_web_search_fallback(session, {"query": "x"})
        assert "Exa search failed" in out
        assert "401" in out

    async def test_caps_max_results_to_safe_range(self):
        session = _session_with_settings(
            external_search_provider="exa", exa_api_key="k"
        )

        captured: dict = {}

        async def _post(self, url, json=None, headers=None):
            captured["json"] = json
            return httpx.Response(
                200, json={"results": []}, request=httpx.Request("POST", url)
            )

        with patch.object(httpx.AsyncClient, "post", new=_post):
            await handle_web_search_fallback(
                session, {"query": "x", "max_results": 999}
            )
        # 999 is clamped to 20 (the upper bound).
        assert captured["json"]["num_results"] == 20


# ─────────────────────────────────────────────────────────────────────────────
# web_search fallback — Brave
# ─────────────────────────────────────────────────────────────────────────────


class TestWebSearchFallbackBrave:
    async def test_calls_brave_endpoint_with_subscription_token(self):
        session = _session_with_settings(
            external_search_provider="brave", brave_api_key="brv-key"
        )
        captured: dict = {}

        async def _get(self, url, headers=None, params=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["params"] = params
            return httpx.Response(
                200,
                json={
                    "web": {
                        "results": [
                            {
                                "title": "Brave hit",
                                "url": "https://b.example",
                                "description": "A hit.",
                            }
                        ]
                    }
                },
                request=httpx.Request("GET", url),
            )

        with patch.object(httpx.AsyncClient, "get", new=_get):
            out = await handle_web_search_fallback(session, {"query": "anton"})

        assert captured["url"] == "https://api.search.brave.com/res/v1/web/search"
        assert captured["headers"]["X-Subscription-Token"] == "brv-key"
        assert captured["params"] == {"q": "anton", "count": 5}
        assert "Brave hit" in out
        assert "A hit." in out

    async def test_brave_no_results(self):
        session = _session_with_settings(
            external_search_provider="brave", brave_api_key="k"
        )

        async def _get(self, url, headers=None, params=None):
            return httpx.Response(
                200, json={"web": {"results": []}}, request=httpx.Request("GET", url)
            )

        with patch.object(httpx.AsyncClient, "get", new=_get):
            out = await handle_web_search_fallback(session, {"query": "obscure"})
        assert "No results" in out


# ─────────────────────────────────────────────────────────────────────────────
# web_fetch fallback
# ─────────────────────────────────────────────────────────────────────────────


class TestWebFetchFallback:
    async def test_rejects_non_http_urls(self):
        out = await handle_web_fetch_fallback(None, {"url": "ftp://x.example"})
        assert "http(s)" in out

    async def test_empty_url(self):
        out = await handle_web_fetch_fallback(None, {"url": "   "})
        assert "requires" in out

    async def test_strips_html_to_text(self):
        async def _get(self, url, headers=None):
            return httpx.Response(
                200,
                text=(
                    "<html><head><title>T</title>"
                    "<script>var x = 1;</script></head>"
                    "<body><p>Hello, <b>world</b>!</p>"
                    "<p>Second para.</p></body></html>"
                ),
                headers={"content-type": "text/html"},
                request=httpx.Request("GET", url),
            )

        with patch.object(httpx.AsyncClient, "get", new=_get):
            out = await handle_web_fetch_fallback(
                None, {"url": "https://example.com"}
            )

        # Body text is preserved, script and tags are stripped.
        assert "Hello" in out
        assert "world" in out
        assert "Second para" in out
        assert "<script>" not in out
        assert "var x = 1" not in out

    async def test_truncates_to_max_chars(self):
        big = "<html><body><p>" + ("x" * 5000) + "</p></body></html>"

        async def _get(self, url, headers=None):
            return httpx.Response(
                200,
                text=big,
                headers={"content-type": "text/html"},
                request=httpx.Request("GET", url),
            )

        with patch.object(httpx.AsyncClient, "get", new=_get):
            out = await handle_web_fetch_fallback(
                None, {"url": "https://example.com", "max_chars": 500}
            )

        assert "[truncated]" in out
        # max_chars caps the body text we return; the header line is separate.
        assert out.count("x") <= 600

    async def test_returns_error_for_4xx(self):
        async def _get(self, url, headers=None):
            return httpx.Response(
                404, text="missing", request=httpx.Request("GET", url)
            )

        with patch.object(httpx.AsyncClient, "get", new=_get):
            out = await handle_web_fetch_fallback(
                None, {"url": "https://example.com/missing"}
            )
        assert "404" in out

    async def test_handles_timeout(self):
        async def _get(self, url, headers=None):
            raise httpx.TimeoutException("slow")

        with patch.object(httpx.AsyncClient, "get", new=_get):
            out = await handle_web_fetch_fallback(
                None, {"url": "https://example.com"}
            )
        assert "timed out" in out.lower()


class TestStripHtml:
    def test_drops_script_and_style(self):
        html = (
            "<style>p{color:red}</style>"
            "<script>alert('x')</script>"
            "<p>Visible.</p>"
        )
        assert _strip_html(html).strip() == "Visible."

    def test_decodes_entities(self):
        assert "you & me" in _strip_html("<p>you &amp; me</p>")

    def test_block_tags_get_newline_separation(self):
        html = "<p>one</p><p>two</p>"
        out = _strip_html(html)
        assert "one" in out and "two" in out
        # Some kind of separator between paragraphs (newline or blank line).
        assert "\n" in out


# ─────────────────────────────────────────────────────────────────────────────
# Session-side resolution: native vs fallback by provider
# ─────────────────────────────────────────────────────────────────────────────


class TestSessionWebToolResolution:
    """ChatSession.__init__ must resolve the per-session web tool plan correctly:

    - When the planning provider claims a capability natively, it goes into
      ``_native_web_tools`` and the fallback ToolDef is NOT registered.
    - When the provider does not, the capability goes into ``_fallback_web_tools``
      and the corresponding ToolDef IS registered.
    """

    def _build_session(self, *, provider_native: set[str], cfg_kwargs: dict | None = None):
        from anton.core.session import ChatSession, ChatSessionConfig
        from anton.core.llm.provider import ProviderConnectionInfo

        mock_llm = AsyncMock()
        mock_llm.coding_provider = MagicMock()
        mock_llm.coding_provider.export_connection_info = MagicMock(
            return_value=ProviderConnectionInfo(provider="x", api_key="k")
        )
        mock_llm.coding_model = "x"
        mock_llm.planning_provider = MagicMock()
        mock_llm.planning_provider.native_web_tools = MagicMock(
            return_value=provider_native
        )
        cfg = ChatSessionConfig(llm_client=mock_llm, **(cfg_kwargs or {}))
        return ChatSession(cfg)

    def test_anthropic_style_native_provider_uses_no_fallback(self):
        session = self._build_session(provider_native={"web_search", "web_fetch"})
        assert session._native_web_tools == {"web_search", "web_fetch"}
        assert session._fallback_web_tools == set()

    def test_generic_provider_routes_both_to_fallback(self):
        session = self._build_session(provider_native=set())
        assert session._native_web_tools == set()
        assert session._fallback_web_tools == {"web_search", "web_fetch"}

    def test_disabled_search_drops_from_both_sets(self):
        session = self._build_session(
            provider_native={"web_search", "web_fetch"},
            cfg_kwargs={"web_search_enabled": False},
        )
        assert "web_search" not in session._native_web_tools
        assert "web_search" not in session._fallback_web_tools
        assert "web_fetch" in session._native_web_tools

    def test_fallback_toolDefs_registered_when_provider_lacks_native(self):
        session = self._build_session(provider_native=set())
        # Trigger lazy build of the registry.
        tools = session._build_tools()
        names = {t["name"] for t in tools}
        assert "web_search" in names
        assert "web_fetch" in names

    def test_fallback_toolDefs_not_registered_when_provider_is_native(self):
        session = self._build_session(provider_native={"web_search", "web_fetch"})
        tools = session._build_tools()
        names = {t["name"] for t in tools}
        # web tools are server-side on the provider; they should NOT appear in
        # the registry — the model invokes them through the provider directly.
        assert "web_search" not in names
        assert "web_fetch" not in names


class TestNativeWebToolsForwarded:
    """plan_with_recovery / plan_stream_with_recovery must forward the resolved
    native_web_tools set to the LLM client without each call site needing to
    remember it."""

    async def test_plan_with_recovery_forwards_native_set(self):
        from anton.core.session import ChatSession, ChatSessionConfig
        from anton.core.llm.provider import LLMResponse, ProviderConnectionInfo, Usage

        mock_llm = AsyncMock()
        mock_llm.coding_provider = MagicMock()
        mock_llm.coding_provider.export_connection_info = MagicMock(
            return_value=ProviderConnectionInfo(provider="x", api_key="k")
        )
        mock_llm.coding_model = "x"
        mock_llm.planning_provider = MagicMock()
        mock_llm.planning_provider.native_web_tools = MagicMock(
            return_value={"web_search", "web_fetch"}
        )
        mock_llm.plan = AsyncMock(
            return_value=LLMResponse(content="ok", usage=Usage())
        )

        session = ChatSession(ChatSessionConfig(llm_client=mock_llm))
        await session.plan_with_recovery(system="sys")

        kwargs = mock_llm.plan.call_args.kwargs
        assert kwargs["native_web_tools"] == {"web_search", "web_fetch"}

    async def test_plan_with_recovery_omits_kwarg_when_no_native(self):
        from anton.core.session import ChatSession, ChatSessionConfig
        from anton.core.llm.provider import LLMResponse, ProviderConnectionInfo, Usage

        mock_llm = AsyncMock()
        mock_llm.coding_provider = MagicMock()
        mock_llm.coding_provider.export_connection_info = MagicMock(
            return_value=ProviderConnectionInfo(provider="x", api_key="k")
        )
        mock_llm.coding_model = "x"
        mock_llm.planning_provider = MagicMock()
        mock_llm.planning_provider.native_web_tools = MagicMock(return_value=set())
        mock_llm.plan = AsyncMock(
            return_value=LLMResponse(content="ok", usage=Usage())
        )

        session = ChatSession(ChatSessionConfig(llm_client=mock_llm))
        await session.plan_with_recovery(system="sys")

        kwargs = mock_llm.plan.call_args.kwargs
        # When the provider has no native web tools, the kwarg is left out
        # entirely so it doesn't even appear in older mocks' call_args.
        assert "native_web_tools" not in kwargs


class TestToolDefShapes:
    def test_search_tool_schema_requires_query(self):
        assert "query" in WEB_SEARCH_FALLBACK_TOOL.input_schema["required"]

    def test_fetch_tool_schema_requires_url(self):
        assert "url" in WEB_FETCH_FALLBACK_TOOL.input_schema["required"]

    def test_tool_names_match_native_capability_strings(self):
        # The fallback names MUST match the native capability strings so that
        # provider-side execution and handler-side execution feel identical to
        # the agent. If these drift, tools registered conditionally won't line
        # up with the native_web_tools set.
        assert WEB_SEARCH_FALLBACK_TOOL.name == "web_search"
        assert WEB_FETCH_FALLBACK_TOOL.name == "web_fetch"
