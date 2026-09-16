"""Tests for the handler-dispatched web_search/web_fetch fallbacks and the
session-side routing decision (native vs handler-dispatched).
"""

from __future__ import annotations

import logging
import socket
import ssl
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2 as httpx
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


def _text(out):
    """The caller-facing text of a `web_search` / `web_fetch` result.

    Since ENG-2677 both handlers return a `ToolOutcome` carrying the verdict on
    the provider/fetch paths; the argument-validation and no-provider branches
    still return a bare `str`.
    """
    return out.content if hasattr(out, "content") else out


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
        out_lines = _text(out).splitlines()
        assert "Result A" in _text(out)
        assert "   https://a.example" in out_lines
        assert "Result B" in _text(out)

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
        assert "Exa search failed" in _text(out)
        assert "401" in _text(out)

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
        assert "Brave hit" in _text(out)
        assert "A hit." in _text(out)

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
        assert "No results" in _text(out)


# ─────────────────────────────────────────────────────────────────────────────
# web_fetch fallback
# ─────────────────────────────────────────────────────────────────────────────


class TestWebFetchFallback:
    async def test_rejects_non_http_urls(self):
        out = await handle_web_fetch_fallback(None, {"url": "ftp://x.example"})
        assert "http(s)" in _text(out)

    async def test_empty_url(self):
        out = await handle_web_fetch_fallback(None, {"url": "   "})
        assert "requires" in _text(out)

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
        assert "Hello" in _text(out)
        assert "world" in _text(out)
        assert "Second para" in _text(out)
        assert "<script>" not in _text(out)
        assert "var x = 1" not in _text(out)

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

        assert "[truncated]" in _text(out)
        # max_chars caps the body text we return; the header line is separate.
        assert _text(out).count("x") <= 600

    async def test_returns_error_for_4xx(self):
        async def _get(self, url, headers=None):
            return httpx.Response(
                404, text="missing", request=httpx.Request("GET", url)
            )

        with patch.object(httpx.AsyncClient, "get", new=_get):
            out = await handle_web_fetch_fallback(
                None, {"url": "https://example.com/missing"}
            )
        assert "404" in _text(out)

    async def test_handles_timeout(self):
        calls = {"n": 0}

        async def _get(self, url, headers=None):
            calls["n"] += 1
            raise httpx.TimeoutException("slow")

        with patch.object(httpx.AsyncClient, "get", new=_get), patch(
            "anton.core.tools.web_tools._FETCH_BACKOFF_BASE_S", 0
        ):
            out = await handle_web_fetch_fallback(
                None, {"url": "https://example.com"}
            )
        # Timeout is transient → retried up to the attempt cap before giving up.
        assert calls["n"] == 2
        assert "timed out" in _text(out).lower()


class TestWebFetchRetry:
    """Narrow retry: transient failures are retried with backoff; permanent ones
    (4xx, NXDOMAIN, SSL) fail fast on the first attempt."""

    async def test_retries_5xx_then_succeeds(self):
        calls = {"n": 0}

        async def _get(self, url, headers=None):
            calls["n"] += 1
            if calls["n"] < 2:
                return httpx.Response(
                    503, text="busy", request=httpx.Request("GET", url)
                )
            return httpx.Response(
                200,
                text="<p>ok</p>",
                headers={"content-type": "text/html"},
                request=httpx.Request("GET", url),
            )

        with patch.object(httpx.AsyncClient, "get", new=_get), patch(
            "anton.core.tools.web_tools._FETCH_BACKOFF_BASE_S", 0
        ):
            out = await handle_web_fetch_fallback(None, {"url": "https://example.com"})
        assert calls["n"] == 2
        assert "ok" in _text(out)

    async def test_5xx_exhausts_retries(self):
        calls = {"n": 0}

        async def _get(self, url, headers=None):
            calls["n"] += 1
            return httpx.Response(500, text="boom", request=httpx.Request("GET", url))

        with patch.object(httpx.AsyncClient, "get", new=_get), patch(
            "anton.core.tools.web_tools._FETCH_BACKOFF_BASE_S", 0
        ):
            out = await handle_web_fetch_fallback(None, {"url": "https://example.com"})
        assert calls["n"] == 2
        assert "500" in _text(out) and "gave up" in _text(out)

    async def test_does_not_retry_non_retryable_5xx(self):
        # 511 Network Authentication Required is a permanent 5xx → no retry.
        calls = {"n": 0}

        async def _get(self, url, headers=None):
            calls["n"] += 1
            return httpx.Response(511, text="portal", request=httpx.Request("GET", url))

        with patch.object(httpx.AsyncClient, "get", new=_get), patch(
            "anton.core.tools.web_tools._FETCH_BACKOFF_BASE_S", 0
        ):
            out = await handle_web_fetch_fallback(None, {"url": "https://example.com"})
        assert calls["n"] == 1
        assert "511" in _text(out)

    async def test_does_not_retry_4xx(self):
        calls = {"n": 0}

        async def _get(self, url, headers=None):
            calls["n"] += 1
            return httpx.Response(404, text="nope", request=httpx.Request("GET", url))

        with patch.object(httpx.AsyncClient, "get", new=_get), patch(
            "anton.core.tools.web_tools._FETCH_BACKOFF_BASE_S", 0
        ):
            out = await handle_web_fetch_fallback(None, {"url": "https://example.com"})
        assert calls["n"] == 1
        assert "404" in _text(out)

    async def test_retries_connect_error(self):
        calls = {"n": 0}

        async def _get(self, url, headers=None):
            calls["n"] += 1
            raise httpx.ConnectError("all connection attempts failed")

        with patch.object(httpx.AsyncClient, "get", new=_get), patch(
            "anton.core.tools.web_tools._FETCH_BACKOFF_BASE_S", 0
        ):
            out = await handle_web_fetch_fallback(None, {"url": "https://example.com"})
        assert calls["n"] == 2
        assert "gave up" in _text(out)

    async def test_does_not_retry_ssl_error(self):
        calls = {"n": 0}

        async def _get(self, url, headers=None):
            calls["n"] += 1
            raise httpx.ConnectError("cert fail") from ssl.SSLCertVerificationError(
                "bad cert"
            )

        with patch.object(httpx.AsyncClient, "get", new=_get), patch(
            "anton.core.tools.web_tools._FETCH_BACKOFF_BASE_S", 0
        ):
            out = await handle_web_fetch_fallback(None, {"url": "https://example.com"})
        assert calls["n"] == 1
        assert "cert fail" in _text(out)

    async def test_retries_transient_dns(self):
        # EAI_AGAIN from the SSRF preflight's getaddrinfo → transient → retried.
        calls = {"n": 0}

        def _gai(host, *args, **kwargs):
            calls["n"] += 1
            raise socket.gaierror(
                socket.EAI_AGAIN, "Temporary failure in name resolution"
            )

        with patch(
            "anton.core.tools.web_tools.socket.getaddrinfo", new=_gai
        ), patch("anton.core.tools.web_tools._FETCH_BACKOFF_BASE_S", 0):
            out = await handle_web_fetch_fallback(None, {"url": "https://example.com"})
        assert calls["n"] == 2
        assert "gave up" in _text(out)

    async def test_does_not_retry_nxdomain(self):
        # EAI_NONAME → permanent NXDOMAIN → single attempt, no retry.
        calls = {"n": 0}

        def _gai(host, *args, **kwargs):
            calls["n"] += 1
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

        with patch(
            "anton.core.tools.web_tools.socket.getaddrinfo", new=_gai
        ), patch("anton.core.tools.web_tools._FETCH_BACKOFF_BASE_S", 0):
            out = await handle_web_fetch_fallback(None, {"url": "https://example.com"})
        assert calls["n"] == 1
        assert "Could not resolve" in _text(out)


class TestWebFetchLogging:
    """Exactly one structured audit line per web_fetch call."""

    _LOGGER = "anton.core.tools.web_tools"

    async def test_success_logs_single_info_line(self, caplog):
        async def _get(self, url, headers=None):
            return httpx.Response(
                200,
                text="<p>hi</p>",
                headers={"content-type": "text/html"},
                request=httpx.Request("GET", url),
            )

        with patch.object(httpx.AsyncClient, "get", new=_get), caplog.at_level(
            logging.INFO, logger=self._LOGGER
        ):
            await handle_web_fetch_fallback(None, {"url": "https://example.com"})

        recs = [r for r in caplog.records if r.name == self._LOGGER]
        assert len(recs) == 1
        assert recs[0].levelno == logging.INFO
        msg = recs[0].getMessage()
        assert "method=GET" in msg
        assert "status=200" in msg
        assert "attempts=1" in msg
        assert "bytes=" in msg and "elapsed_ms=" in msg

    async def test_url_query_string_is_redacted_in_log(self, caplog):
        async def _get(self, url, headers=None):
            return httpx.Response(
                200,
                text="ok",
                headers={"content-type": "text/plain"},
                request=httpx.Request("GET", url),
            )

        secret_url = "https://user:pass@example.com/data?api_key=SECRET123&x=1"
        with patch.object(httpx.AsyncClient, "get", new=_get), caplog.at_level(
            logging.INFO, logger=self._LOGGER
        ):
            await handle_web_fetch_fallback(None, {"url": secret_url})

        msg = next(r for r in caplog.records if r.name == self._LOGGER).getMessage()
        assert "SECRET123" not in msg
        assert "api_key" not in msg
        assert "pass" not in msg  # userinfo credentials stripped too
        assert "url=https://example.com/data?<redacted>" in msg

    async def test_giveup_logs_single_warning_line(self, caplog):
        async def _get(self, url, headers=None):
            raise httpx.ConnectError("all connection attempts failed")

        with patch.object(httpx.AsyncClient, "get", new=_get), patch(
            "anton.core.tools.web_tools._FETCH_BACKOFF_BASE_S", 0
        ), caplog.at_level(logging.INFO, logger=self._LOGGER):
            await handle_web_fetch_fallback(None, {"url": "https://example.com"})

        recs = [r for r in caplog.records if r.name == self._LOGGER]
        assert len(recs) == 1
        assert recs[0].levelno == logging.WARNING
        msg = recs[0].getMessage()
        assert "status=transient_giveup" in msg
        assert "attempts=2" in msg


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
        assert "one" in _text(out) and "two" in _text(out)
        # Some kind of separator between paragraphs (newline or blank line).
        assert "\n" in _text(out)


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

    def _settings_with_credential(self):
        from anton.config.settings import AntonSettings

        return AntonSettings(external_search_provider="exa", exa_api_key="k")

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
        # web_search additionally needs a usable Exa/Brave credential; without
        # one it must not be registered even though the provider isn't native.
        session = self._build_session(
            provider_native=set(),
            cfg_kwargs={"settings": self._settings_with_credential()},
        )
        tools = session._build_tools()
        names = {t["name"] for t in tools}
        assert "web_search" in names
        assert "web_fetch" in names

    def test_fallback_web_search_not_registered_without_credential(self):
        # A model offered a tool that can only fail (no Exa/Brave key
        # configured) will call it anyway — don't register it at all.
        session = self._build_session(provider_native=set())
        tools = session._build_tools()
        names = {t["name"] for t in tools}
        assert "web_search" not in names
        # web_fetch needs no credential, so it's unaffected.
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


# ─────────────────────────────────────────────────────────────────────────────
# web_fetch verdicts (ENG-2677)
# ─────────────────────────────────────────────────────────────────────────────


class TestWebFetchVerdict:
    """`ok` on the fetch path, and what it does to the per-tool error streak.

    The bug this closes: on success the handler returns the PAGE CONTENT, and
    the ENG-1276 substring fallback scanned that content for "failed" /
    "timed out" / "[error]". Ordinary articles therefore counted as failed tool
    calls, and five in a row told the agent to stop retrying an approach that
    was working.
    """

    # A public address, so the SSRF pre-flight passes without a real lookup.
    # Stubbing this is load-bearing, not tidiness: `_fetch_once` runs
    # `socket.getaddrinfo` before any HTTP call, and on a resolver failure it
    # returns status="blocked" -> ok=None. `test_4xx_...` would then be GREEN
    # without ever exercising a 404 (verified by running it with the resolver
    # patched to raise).
    _PUBLIC_DNS = [(2, 1, 6, "", ("93.184.216.34", 0))]

    async def _fetch(self, status, body="<p>hi</p>"):
        async def _get(self, url, headers=None):
            return httpx.Response(
                status, text=body,
                headers={"content-type": "text/html"},
                request=httpx.Request("GET", url),
            )
        with patch.object(httpx.AsyncClient, "get", new=_get), patch(
            "anton.core.tools.web_tools._FETCH_BACKOFF_BASE_S", 0
        ), patch(
            "anton.core.tools.web_tools.socket.getaddrinfo",
            new=lambda *a, **k: self._PUBLIC_DNS,
        ):
            return await handle_web_fetch_fallback(None, {"url": "https://example.com"})

    async def test_2xx_is_an_explicit_success(self):
        out = await self._fetch(200)
        assert out.ok is True

    async def test_4xx_is_left_unverdicted_on_purpose(self):
        # Tier 3: the tool worked, the answer was negative. Same category as
        # recall_skill's NO MATCH family (ENG-2248). `ok=False` here would be a
        # behaviour change with its own before/after, and gets its own ticket.
        out = await self._fetch(404)
        assert out.ok is None

    async def test_no_branch_ever_declares_failure(self):
        """The change can only REMOVE a nudge/breaker firing, never add one."""
        for status in (200, 301, 403, 404, 410, 418, 451, 501):
            out = await self._fetch(status)
            assert out.ok is not False, f"status {status} newly declares failure"


class TestWebFetchDoesNotPoisonTheErrorStreak:
    """What the verdict DOES once the handler declares it.

    These feed `_apply_error_tracking` directly, so they document the
    consequence rather than guard the handler — reverting the handler to a bare
    `str` leaves them all green (verified by mutation). The guard on the handler
    itself is `TestWebFetchVerdict`; those three fail on that mutation.

    Their job is to pin BOTH states so neither can drift unnoticed: what the bug
    looked like (`..._unverdicted_are_what_the_bug_looked_like`, streak 5) and
    what correct looks like (streak 0), plus the two paths ENG-2677
    deliberately leaves alone.
    """

    # Real prose that happens to contain a marker word. All of these are
    # successful HTTP 200 fetches.
    PAGES = [
        "Q3 results: the merger failed to clear regulatory review, revenue rose 12%",
        "Postmortem: the deployment timed out after thirty minutes",
        "Refunds apply whenever a payment failed, for any reason",
        "Apollo 13's oxygen tank failed en route to the Moon",
        "Troubleshooting: if the connection timed out, check your firewall",
    ]

    @staticmethod
    def _session():
        from types import SimpleNamespace
        from anton.core.llm.prompts import RESILIENCE_NUDGE
        return SimpleNamespace(
            _resilience_nudge_at=2,
            _max_consecutive_errors=5,
            _select_resilience_nudge=lambda name, text: RESILIENCE_NUDGE,
        )

    def _run(self, results):
        """Feed (text, ok) pairs through the REAL streak tracker."""
        from anton.core.session import ChatSession
        from anton.core.llm.prompts import RESILIENCE_NUDGE
        sess, streak, nudged = self._session(), {}, set()
        nudge = breaker = False
        for text, ok in results:
            out = ChatSession._apply_error_tracking(
                sess, text, "web_fetch", streak, nudged, ok=ok
            )
            nudge = nudge or RESILIENCE_NUDGE in _text(out)
            breaker = breaker or "SYSTEM: The 'web_fetch' tool has failed" in _text(out)
        return streak.get("web_fetch", 0), nudge, breaker

    def test_successful_pages_mentioning_failure_do_not_climb(self):
        streak, nudge, breaker = self._run([(p, True) for p in self.PAGES])
        assert (streak, nudge, breaker) == (0, False, False)

    def test_the_same_pages_unverdicted_are_what_the_bug_looked_like(self):
        """Pin the old behaviour so the fix cannot be silently reverted."""
        streak, nudge, breaker = self._run([(p, None) for p in self.PAGES])
        assert (streak, nudge, breaker) == (5, True, True)

    def test_timeouts_still_count_exactly_as_before(self):
        # Left ok=None, and "timed out" matches the legacy markers — so this
        # path is unchanged by ENG-2677 and must stay that way.
        msg = "Fetch timed out after 30.0s for https://x.example (gave up after 2 attempts)"
        assert self._run([(msg, None)] * 5) == (5, True, True)

    def test_4xx_still_does_not_count(self):
        # Unchanged too: no marker word, left unverdicted. Whether it SHOULD
        # count is the deferred question.
        msg = "Fetch returned HTTP 403 for https://x.example"
        assert self._run([(msg, None)] * 6) == (0, False, False)


class TestWebFetchVerdictOnNonHttpFailures:
    """The paths where no HTTP status is ever obtained (F4).

    `TestWebFetchVerdict` only covers integer statuses. These cover the
    `status="transient_giveup"` and `status="blocked"` branches, so a later
    change that made a timeout or a resolver failure `ok=False` — a real
    behaviour change — cannot pass unnoticed.
    """

    async def test_exhausted_timeout_is_left_unverdicted(self):
        async def _get(self, url, headers=None):
            raise httpx.TimeoutException("slow")

        with patch.object(httpx.AsyncClient, "get", new=_get), patch(
            "anton.core.tools.web_tools._FETCH_BACKOFF_BASE_S", 0
        ), patch(
            "anton.core.tools.web_tools.socket.getaddrinfo",
            new=lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))],
        ):
            out = await handle_web_fetch_fallback(None, {"url": "https://x.example"})
        assert out.ok is None
        assert "timed out" in out.content.lower()

    async def test_unresolvable_host_is_left_unverdicted(self):
        def _boom(*a, **k):
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

        with patch("anton.core.tools.web_tools.socket.getaddrinfo", new=_boom):
            out = await handle_web_fetch_fallback(None, {"url": "https://gone.example"})
        assert out.ok is None
        assert "Could not resolve host" in out.content


class TestWebSearchVerdict:
    """`ok` on the search path — the same defect as web_fetch had (F5).

    On success the results block is up to 20 snippets of 600 characters of
    arbitrary web prose, which the substring fallback scanned for "failed".
    """

    @staticmethod
    def _session():
        return _session_with_settings(
            external_search_provider="exa", exa_api_key="k"
        )

    async def _search(self, response):
        async def _post(self, url, json=None, headers=None):
            return response(httpx.Request("POST", url))

        with patch.object(httpx.AsyncClient, "post", new=_post):
            return await handle_web_search_fallback(self._session(), {"query": "q"})

    async def test_hits_are_an_explicit_success(self):
        out = await self._search(lambda req: httpx.Response(
            200,
            json={"results": [{
                "title": "Why the merger failed",
                "url": "https://a.example",
                "text": "The deal timed out before regulatory review closed.",
            }]},
            request=req,
        ))
        # Both marker words appear in the RESULTS — the exact shape that used
        # to make a working search count as a failed tool call.
        assert "failed" in out.content and "timed out" in out.content
        assert out.ok is True

    async def test_no_results_is_left_unverdicted(self):
        out = await self._search(lambda req: httpx.Response(
            200, json={"results": []}, request=req))
        assert out.ok is None
        assert "No results" in out.content

    async def test_provider_error_is_left_unverdicted(self):
        out = await self._search(lambda req: httpx.Response(
            401, text="bad key", request=req))
        assert out.ok is None
        # Contains "failed", so the substring fallback still counts it — the
        # behaviour this change deliberately leaves alone.
        assert "Exa search failed" in out.content

    async def test_no_configured_provider_still_returns_a_plain_string(self):
        out = await handle_web_search_fallback(
            _session_with_settings(), {"query": "q"}
        )
        assert isinstance(out, str)
        assert "anton setup search" in out
