"""Live integration tests for the native web tool paths.

These tests make real API calls — they exercise the wire format end-to-end
(tool spec serialization, server-side execution, response parsing) instead of
just checking what we send. They auto-skip when the corresponding API key is
not in the environment, so CI without keys is unaffected.

Loads ``.env`` from the project root once at import time so a developer who
keeps their keys in ``.env`` (the standard pattern for this repo) doesn't need
to ``source`` anything before running ``pytest``.

Coverage map:

- ``TestAnthropicLive`` — ``AnthropicProvider`` with ``native_web_tools``
  resolving to ``web_search_20250305`` and ``web_fetch_20250910`` server tools.
  Hits the Messages API directly.
- ``TestOpenAIBYOKLive`` — ``OpenAIProvider(flavor="openai")`` with
  ``native_web_tools`` routing through the Responses API
  (``client.responses.create``). The whole BYOK OpenAI path runs through
  Responses now, so this also validates non-tool calls along the way.
- ``TestMindsPassthroughLive`` — flavor=``"minds-passthrough"``, base_url=
  ``https://mdb.ai/api/v1``, model=``"_reason_"``. Currently skipped because
  the mdb.ai ``passthrough_agent`` web-tools translation lands in a separate
  PR; the scaffolding here means we just remove the skip mark when it ships.

Cost note: each test uses small ``max_tokens`` to keep the bill negligible.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Load .env once so plain os.environ reads pick up keys the developer put in
# the repo-root .env (matches AntonSettings' env-file precedence).
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except Exception:
    # python-dotenv is a transitive dep through pydantic-settings; if it's
    # missing for any reason, fall back to whatever's already in os.environ.
    pass


def _have(key: str) -> bool:
    return bool(os.environ.get(key))


def _has_https_url_line(text: str) -> bool:
    """Return True if any line of ``text`` is a formatted URL row.

    The web_search formatter emits URLs on their own indented line — see
    ``anton.core.tools.web_tools._search_exa`` / ``_search_brave``. Asserting
    against an exact line beginning is both stricter than ``"https://" in out``
    (which would also pass for ``"foo https://x evil"``) and avoids tripping
    CodeQL's ``py/incomplete-url-substring-sanitization`` rule, which
    correctly flags the substring pattern even in test contexts.
    """
    return any(line.lstrip().startswith("https://") for line in text.splitlines())


anthropic_only = pytest.mark.skipif(
    not _have("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — live test skipped",
)
openai_only = pytest.mark.skipif(
    not _have("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — live test skipped",
)
minds_only = pytest.mark.skipif(
    not _have("MINDS_API_KEY"),
    reason="MINDS_API_KEY not set — live test skipped",
)
exa_only = pytest.mark.skipif(
    not _have("EXA_API_KEY"),
    reason="EXA_API_KEY not set — live test skipped",
)
brave_only = pytest.mark.skipif(
    not _have("BRAVE_API_KEY"),
    reason="BRAVE_API_KEY not set — live test skipped",
)


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic BYOK — native web_search and web_fetch
# ─────────────────────────────────────────────────────────────────────────────


@anthropic_only
class TestAnthropicLive:
    """Real calls to Anthropic with the native server-side web tools.

    On success the model emits some combination of ``server_tool_use`` /
    ``web_search_tool_result`` / ``text`` blocks; the existing extraction loop
    in ``AnthropicProvider.complete`` already filters down to text blocks (and
    real ``tool_use`` blocks for function tools), so the model's natural
    response — which incorporates the search/fetch result — flows back as
    ``LLMResponse.content``. The assertions are deliberately loose: we only
    care that the call succeeds and returns plausible content; exact
    summarization is the model's job, not our wire format's.
    """

    @pytest.mark.asyncio
    async def test_complete_with_native_web_search(self):
        from anton.core.llm.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = await provider.complete(
            model="claude-sonnet-4-6",
            system="Use web_search if you need current information. Be brief.",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Search the web for 'Anthropic Claude' and tell me one "
                        "fact in a single sentence."
                    ),
                }
            ],
            native_web_tools={"web_search"},
            max_tokens=512,
        )

        assert response.content, "expected non-empty model response"
        assert len(response.content) > 20
        # The query forces a search-shaped answer; "Anthropic" or "Claude"
        # should land in the text either way.
        lowered = response.content.lower()
        assert "anthropic" in lowered or "claude" in lowered

    @pytest.mark.asyncio
    async def test_complete_with_native_web_fetch(self):
        from anton.core.llm.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = await provider.complete(
            model="claude-sonnet-4-6",
            system=(
                "Use the web_fetch tool to retrieve the URL the user provides "
                "and quote one short phrase from the page. Be brief."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Fetch https://example.com and tell me what the page says."
                    ),
                }
            ],
            native_web_tools={"web_fetch"},
            max_tokens=512,
        )

        assert response.content, "expected non-empty model response"
        # example.com's signature phrase — the model should surface it after
        # the server-side fetch lands.
        assert "example" in response.content.lower()

    @pytest.mark.asyncio
    async def test_complete_with_both_native_tools(self):
        """Both server tools wired in the same call — exercises the merged
        tools array + the beta header co-existing with non-beta tooling."""
        from anton.core.llm.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = await provider.complete(
            model="claude-sonnet-4-6",
            system="Use whichever web tool fits. Keep your answer short.",
            messages=[
                {
                    "role": "user",
                    "content": "What is on https://example.com? One sentence.",
                }
            ],
            native_web_tools={"web_search", "web_fetch"},
            max_tokens=512,
        )

        assert response.content, "expected non-empty model response"

    @pytest.mark.asyncio
    async def test_complete_without_web_tools_still_works(self):
        """Sanity: opting out (``native_web_tools=None``) must not regress
        the existing chat-only path."""
        from anton.core.llm.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = await provider.complete(
            model="claude-sonnet-4-6",
            system="Reply with exactly: pong",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=16,
        )
        assert "pong" in response.content.lower()


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI BYOK — Responses API path with native web_search
# ─────────────────────────────────────────────────────────────────────────────


@openai_only
class TestOpenAIBYOKLive:
    """Real calls to ``client.responses.create`` for ``flavor="openai"``.

    Validates that the entire Responses API translation (input shape, tools,
    instructions, output_text/function_call extraction) lines up with what
    the live API expects. The earlier mocked tests cover the request shape
    going out; these confirm it actually works against the real endpoint.
    """

    @pytest.mark.asyncio
    async def test_responses_api_basic_call(self):
        """No web tools — just confirm the Responses API transport works for
        the simple text path. If this fails, every other BYOK OpenAI test
        is also broken."""
        from anton.core.llm.openai import OpenAIProvider

        provider = OpenAIProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            flavor=OpenAIProvider.FLAVOR_OPENAI,
        )
        response = await provider.complete(
            model="gpt-5",
            system="Reply with exactly: pong",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=512,
        )
        assert response.content, "expected non-empty Responses API output_text"
        assert "pong" in response.content.lower()
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    @pytest.mark.asyncio
    async def test_responses_api_with_native_web_search(self):
        from anton.core.llm.openai import OpenAIProvider

        provider = OpenAIProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            flavor=OpenAIProvider.FLAVOR_OPENAI,
        )
        # gpt-5 is a reasoning model: reasoning tokens + the (often large)
        # web_search result payload share the ``max_output_tokens`` budget,
        # so a tight cap can leave nothing for the final text. 4096 is
        # comfortable headroom for a one-sentence answer over a search.
        response = await provider.complete(
            model="gpt-5",
            system="Use web_search if you need current information. Be brief.",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Search for 'OpenAI Responses API' and summarize one "
                        "thing about it in a single sentence."
                    ),
                }
            ],
            native_web_tools={"web_search"},
            max_tokens=4096,
        )

        assert response.content, (
            f"expected non-empty model response but got stop_reason="
            f"{response.stop_reason!r} (input_tokens={response.usage.input_tokens}, "
            f"output_tokens={response.usage.output_tokens})"
        )
        assert len(response.content) > 20
        lowered = response.content.lower()
        assert "openai" in lowered or "responses" in lowered or "api" in lowered

    @pytest.mark.asyncio
    async def test_responses_api_with_function_tool_round_trip(self):
        """Forced function-tool call through the Responses API.

        Confirms the flat function-tool shape (`{"type": "function", "name": ...}`),
        the ``tool_choice`` translation, and the ``call_id`` round-trip all
        work against the live endpoint. This is the same path
        ``generate_object*`` uses, so a regression here would cascade.

        Note on ``max_tokens``: gpt-5 is a reasoning model and its reasoning
        tokens count against ``max_output_tokens``. A low cap can leave the
        model with no budget to emit the function call (``stop_reason=
        "incomplete"``), so we use a generous 4096 here. Still pennies per run.
        """
        from anton.core.llm.openai import OpenAIProvider

        provider = OpenAIProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            flavor=OpenAIProvider.FLAVOR_OPENAI,
        )
        response = await provider.complete(
            model="gpt-5",
            system="Call the answer tool to provide your reply.",
            messages=[{"role": "user", "content": "What is 6 times 7?"}],
            tools=[
                {
                    "name": "answer",
                    "description": "Provide the numeric answer.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "integer"},
                            "explanation": {"type": "string"},
                        },
                        "required": ["value"],
                    },
                }
            ],
            tool_choice={"type": "tool", "name": "answer"},
            max_tokens=4096,
        )

        assert response.tool_calls, (
            f"expected forced tool call but got stop_reason={response.stop_reason!r} "
            f"with content={response.content!r}"
        )
        tc = response.tool_calls[0]
        assert tc.name == "answer"
        # call_id is the canonical id we'll reference in any follow-up
        # function_call_output items.
        assert tc.id
        assert tc.input.get("value") == 42

    @pytest.mark.asyncio
    async def test_responses_api_streaming(self):
        """Quick smoke of the streaming path. Streaming has its own
        per-event translation (output_text.delta, function_call_arguments.*,
        completed) that the non-streaming test doesn't exercise."""
        from anton.core.llm.openai import OpenAIProvider
        from anton.core.llm.provider import StreamComplete, StreamTextDelta

        provider = OpenAIProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            flavor=OpenAIProvider.FLAVOR_OPENAI,
        )

        text_chunks: list[str] = []
        final_response = None
        async for event in provider.stream(
            model="gpt-5",
            system="Reply with exactly: pong",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=512,
        ):
            if isinstance(event, StreamTextDelta):
                text_chunks.append(event.text)
            elif isinstance(event, StreamComplete):
                final_response = event.response

        joined = "".join(text_chunks).lower()
        assert "pong" in joined
        assert final_response is not None
        assert final_response.content == "".join(text_chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Minds passthrough — same path as OpenAI-compatible chat.completions
# ─────────────────────────────────────────────────────────────────────────────


@minds_only
@pytest.mark.skip(
    reason=(
        "mdb.ai passthrough_agent web-tools translation lives in a separate "
        "backend PR. Scaffolding is in place — remove this skip when the "
        "passthrough side ships."
    )
)
class TestMindsPassthroughLive:
    """Native web tools through mdb.ai (chat.completions transport with
    ``{"type": "web_search"}`` / ``{"type": "fetch"}`` appended raw).

    The wire format on our end is finalized — this suite already passes
    against the local mock — but the upstream ``passthrough_agent`` doesn't
    translate the web tool entries to the underlying provider yet, so a real
    call returns either a 4xx or a no-op completion. Tests are skipped at
    the class level until the backend lands, so any future change to the
    passthrough path that breaks our wire format will surface here on the
    first un-skipped run.
    """

    @pytest.mark.asyncio
    async def test_complete_with_native_web_search(self):
        from anton.core.llm.openai import OpenAIProvider

        provider = OpenAIProvider(
            api_key=os.environ["MINDS_API_KEY"],
            base_url="https://mdb.ai/api/v1",
            flavor=OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH,
            supports_vision=False,
        )
        response = await provider.complete(
            model="_reason_",
            system="Use web_search if you need current information. Be brief.",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Search for 'Anthropic' and tell me one fact in a sentence."
                    ),
                }
            ],
            native_web_tools={"web_search"},
            max_tokens=512,
        )
        assert response.content
        assert len(response.content) > 20

    @pytest.mark.asyncio
    async def test_complete_with_native_fetch(self):
        from anton.core.llm.openai import OpenAIProvider

        provider = OpenAIProvider(
            api_key=os.environ["MINDS_API_KEY"],
            base_url="https://mdb.ai/api/v1",
            flavor=OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH,
            supports_vision=False,
        )
        response = await provider.complete(
            model="_reason_",
            system="Use the fetch tool. Be brief.",
            messages=[
                {
                    "role": "user",
                    "content": "Fetch https://example.com and tell me what's there.",
                }
            ],
            native_web_tools={"web_fetch"},
            max_tokens=512,
        )
        assert response.content
        assert "example" in response.content.lower()

    @pytest.mark.asyncio
    async def test_complete_without_web_tools_still_works(self):
        """Mind-passthrough chat.completions without web tools — sanity
        check that our flavor flag doesn't break the baseline chat call."""
        from anton.core.llm.openai import OpenAIProvider

        provider = OpenAIProvider(
            api_key=os.environ["MINDS_API_KEY"],
            base_url="https://mdb.ai/api/v1",
            flavor=OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH,
            supports_vision=False,
        )
        response = await provider.complete(
            model="_reason_",
            system="Reply with exactly: pong",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=16,
        )
        assert "pong" in response.content.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Case 3 — generic OpenAI-compatible fallback: Exa.ai & Brave Search
# ─────────────────────────────────────────────────────────────────────────────
#
# The Exa/Brave adapters live in ``anton/core/tools/web_tools.py``. Mocked
# tests in ``test_web_tools.py`` already cover the request shape going out;
# these confirm the live endpoints accept our auth + payload + still return
# the response shape we parse. They also implicitly validate the setup probe
# in ``cli._setup_exa`` / ``_setup_brave``, which uses the same auth +
# endpoint pair.


def _settings_with(**fields):
    """Tiny stand-in for AntonSettings — only the attrs the handlers read."""
    from types import SimpleNamespace

    return SimpleNamespace(
        external_search_provider=fields.get("external_search_provider"),
        exa_api_key=fields.get("exa_api_key"),
        brave_api_key=fields.get("brave_api_key"),
    )


def _session_with(settings):
    from types import SimpleNamespace

    return SimpleNamespace(_settings=settings)


@exa_only
class TestExaLive:
    """Real calls to Exa.ai's ``/search`` endpoint."""

    @pytest.mark.asyncio
    async def test_search_returns_real_results(self):
        """Direct adapter call — the format helper formats real hits."""
        from anton.core.tools.web_tools import _search_exa

        out = await _search_exa(
            query="Anthropic Claude",
            api_key=os.environ["EXA_API_KEY"],
            max_results=3,
        )

        assert "Web search results for: 'Anthropic Claude'" in out
        # At least one https:// URL should appear in the formatted output.
        assert _has_https_url_line(out)
        # And the markdown numbering means we got real hits, not the "no
        # results" branch.
        assert "1. **" in out

    @pytest.mark.asyncio
    async def test_handler_dispatch_via_session(self):
        """The full path the agent actually uses: session settings →
        ``handle_web_search_fallback`` → ``_search_exa`` → real network."""
        from anton.core.tools.web_tools import handle_web_search_fallback

        session = _session_with(
            _settings_with(
                external_search_provider="exa",
                exa_api_key=os.environ["EXA_API_KEY"],
            )
        )
        out = await handle_web_search_fallback(
            session, {"query": "Anthropic Claude", "max_results": 2}
        )
        assert _has_https_url_line(out)
        assert "Anthropic Claude" in out  # query echoed in the header

    @pytest.mark.asyncio
    async def test_setup_probe_endpoint_contract(self):
        """The setup probe in ``cli._setup_exa`` posts the same payload to
        the same URL with the same auth header. This test validates that
        contract against the live API — if Exa changes their endpoint or
        auth shape, both setup AND runtime would break, and this would
        catch it on the next live run."""
        import httpx as _httpx

        # Exact same shape ``cli._setup_exa._test`` uses internally.
        resp = await _httpx.AsyncClient(timeout=15.0).post(
            "https://api.exa.ai/search",
            headers={"Authorization": f"Bearer {os.environ['EXA_API_KEY']}"},
            json={"query": "anton ping", "num_results": 1},
        )
        assert resp.status_code == 200, (
            f"setup probe contract broken: HTTP {resp.status_code} — {resp.text[:200]}"
        )


@brave_only
class TestBraveLive:
    """Real calls to Brave Search's web endpoint."""

    @pytest.mark.asyncio
    async def test_search_returns_real_results(self):
        from anton.core.tools.web_tools import _search_brave

        out = await _search_brave(
            query="Anthropic Claude",
            api_key=os.environ["BRAVE_API_KEY"],
            max_results=3,
        )

        assert "Web search results for: 'Anthropic Claude'" in out
        assert _has_https_url_line(out)
        assert "1. **" in out

    @pytest.mark.asyncio
    async def test_handler_dispatch_via_session(self):
        from anton.core.tools.web_tools import handle_web_search_fallback

        session = _session_with(
            _settings_with(
                external_search_provider="brave",
                brave_api_key=os.environ["BRAVE_API_KEY"],
            )
        )
        out = await handle_web_search_fallback(
            session, {"query": "Anthropic Claude", "max_results": 2}
        )
        assert _has_https_url_line(out)
        assert "Anthropic Claude" in out

    @pytest.mark.asyncio
    async def test_setup_probe_endpoint_contract(self):
        """Mirror of the Exa probe-contract test for Brave (matches
        ``cli._setup_brave._test``)."""
        import httpx as _httpx

        resp = await _httpx.AsyncClient(timeout=15.0).get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "X-Subscription-Token": os.environ["BRAVE_API_KEY"],
                "Accept": "application/json",
            },
            params={"q": "anton ping", "count": 1},
        )
        assert resp.status_code == 200, (
            f"setup probe contract broken: HTTP {resp.status_code} — {resp.text[:200]}"
        )


class TestWebFetchLive:
    """Real ``handle_web_fetch_fallback`` against a stable known URL.

    No API key needed — fetch is the always-on Case 3 capability. ``example.com``
    is operated by IANA and has a stable, well-formed signature page (``Example
    Domain`` heading) which makes this assertion stable enough to live in CI.
    """

    @pytest.mark.asyncio
    async def test_fetches_example_dot_com(self):
        from anton.core.tools.web_tools import handle_web_fetch_fallback

        out = await handle_web_fetch_fallback(
            None, {"url": "https://example.com", "max_chars": 5000}
        )
        # The header line includes status + byte count.
        assert "HTTP 200" in out
        # Signature text from the canonical example.com page.
        assert "Example Domain" in out
        # Confirms the HTML stripper actually ran (the live page has
        # <html>/<body>/<a> tags that should not survive in our output).
        assert "<html" not in out.lower()
        assert "<body" not in out.lower()
