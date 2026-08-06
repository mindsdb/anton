"""Per-turn token cost accounting (ENG-1288).

Covers the Done-when list:
- the accumulator counts every call the LLMClient makes (planning stream,
  coding, router, and the verifier's structured calls — including failed
  ones, which still cost money);
- the running total is readable mid-turn (ENG-1286's ceiling surface);
- the books reset per turn on a long-lived session;
- ended_by resolution per terminal path;
- the analytics event carries numbers/names/IDs only, via send_event.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anton.core.llm.client import LLMClient
from anton.core.llm.provider import (
    LLMResponse,
    StreamComplete,
    StreamTextDelta,
    Usage,
)
from anton.core.session import ChatSession
from anton.core.turn_cost import TurnCost


def _usage(inp=100, out=10, cr=0, cc=0) -> Usage:
    return Usage(
        input_tokens=inp,
        output_tokens=out,
        cache_read_tokens=cr,
        cache_creation_tokens=cc,
    )


class TestTurnCostAccumulator:
    def test_add_sums_components_and_counts_calls(self):
        tc = TurnCost()
        tc.add("planning", "sonnet", _usage(100, 10, 500, 20))
        tc.add("coding", "haiku", _usage(50, 5))
        assert tc.input_tokens == 150
        assert tc.output_tokens == 15
        assert tc.cache_read_tokens == 500
        assert tc.cache_creation_tokens == 20
        assert tc.llm_calls == 2
        assert tc.total_tokens == 150 + 15 + 500 + 20

    def test_peak_context_includes_cache_components(self):
        # A warm-cache call has tiny fresh input but a huge cached prompt —
        # peak context must reflect what the call actually carried, or a
        # 190k-context call reads as ~2k (the cache-blindness misread).
        tc = TurnCost()
        tc.add("planning", "sonnet", _usage(2_000, 10, 180_000, 8_000))
        tc.add("planning", "sonnet", _usage(50_000, 10))
        assert tc.peak_context_tokens == 190_000

    def test_usage_context_tokens_property(self):
        assert _usage(100, 99, 400, 25).context_tokens == 525


class TestClientUsageListener:
    def _client(self, provider: AsyncMock) -> LLMClient:
        return LLMClient(
            planning_provider=provider,
            planning_model="planner",
            coding_provider=provider,
            coding_model="coder",
        )

    @pytest.mark.asyncio
    async def test_plan_and_code_and_summarize_notify(self):
        provider = AsyncMock()
        provider.complete.return_value = LLMResponse(content="ok", usage=_usage())
        client = self._client(provider)
        seen: list[tuple[str, str]] = []
        client.usage_listener = lambda role, model, usage: seen.append((role, model))

        await client.plan(system="s", messages=[])
        await client.code(system="s", messages=[])
        await client.summarize(system="s", messages=[])
        await client.gate(system="s", messages=[])

        assert seen == [
            ("planning", "planner"),
            ("coding", "coder"),
            ("router", "coder"),   # router defaults to the coding role
            ("router", "coder"),
        ]

    @pytest.mark.asyncio
    async def test_plan_stream_notifies_on_stream_complete(self):
        provider = AsyncMock()

        async def _stream(**kwargs):
            yield StreamTextDelta(text="hi")
            yield StreamComplete(
                response=LLMResponse(content="hi", usage=_usage(inp=42))
            )

        provider.stream = MagicMock(side_effect=lambda **kw: _stream(**kw))
        client = self._client(provider)
        tc = TurnCost()
        client.usage_listener = tc.add

        async for _ in client.plan_stream(system="s", messages=[]):
            pass

        assert tc.llm_calls == 1
        assert tc.input_tokens == 42

    @pytest.mark.asyncio
    async def test_failed_structured_call_still_counts(self):
        # The verifier's verdict call costs tokens even when the model never
        # reaches the forced tool call (ENG-1081's failure shape). Counting
        # happens BEFORE the no-tool-call raise.
        from pydantic import BaseModel

        class _Schema(BaseModel):
            x: str

        provider = AsyncMock()
        provider.complete.return_value = LLMResponse(
            content="narration, no tool call", usage=_usage(inp=300, out=256)
        )
        client = self._client(provider)
        tc = TurnCost()
        client.usage_listener = tc.add

        with pytest.raises(Exception):
            await client.generate_object_code(_Schema, system="s", messages=[])

        assert tc.llm_calls == 1
        assert tc.output_tokens == 256

    @pytest.mark.asyncio
    async def test_listener_exception_never_breaks_the_call(self):
        provider = AsyncMock()
        provider.complete.return_value = LLMResponse(content="ok", usage=_usage())
        client = self._client(provider)

        def _boom(role, model, usage):
            raise RuntimeError("broken accumulator")

        client.usage_listener = _boom
        response = await client.plan(system="s", messages=[])
        assert response.content == "ok"


def _bare_session(**overrides) -> ChatSession:
    """Minimal object carrying exactly what _emit_turn_cost reads."""
    s = ChatSession.__new__(ChatSession)
    s._turn_cost = overrides.pop("turn_cost", TurnCost())
    s._llm = MagicMock()
    s._llm.planning_model = "planner"
    s._llm.coding_model = "coder"
    s._session_id = "conv-123"
    s._harness = "cowork"
    s._turn_count = 4
    s._cancel_event = overrides.pop("cancel_event", MagicMock(is_set=lambda: False))
    s._settings = overrides.pop("settings", None)
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


class TestEmitTurnCost:
    def test_emits_log_and_event_with_join_keys(self, caplog):
        s = _bare_session()
        s._turn_cost.add("planning", "planner", _usage(100, 10, 50, 5))
        with patch("anton.analytics.send_event") as send:
            with caplog.at_level(logging.INFO, logger="anton.core.session"):
                s._emit_turn_cost()
        assert any("turn_cost" in r.message for r in caplog.records)
        assert send.called
        kwargs = send.call_args.kwargs
        assert kwargs["conversation_id"] == "conv-123"
        assert kwargs["turn_index"] == "5"
        assert kwargs["tokens_total"] == str(100 + 10 + 50 + 5)
        assert kwargs["ended_by"] == "completed"
        assert kwargs["planning_model"] == "planner"
        # Numbers, names, and IDs only — nothing that could carry content.
        for value in kwargs.values():
            assert isinstance(value, str) and len(value) < 200

    def test_books_close_and_listener_disarms(self):
        s = _bare_session()
        with patch("anton.analytics.send_event"):
            s._emit_turn_cost()
        assert s._turn_cost is None
        assert s._llm.usage_listener is None
        # Second emit is a no-op, not a duplicate event.
        with patch("anton.analytics.send_event") as send:
            s._emit_turn_cost()
        assert not send.called

    def test_explicit_terminal_marks_survive_clean_exit(self):
        s = _bare_session()
        s._turn_cost.ended_by = "handback_stuck"
        with patch("anton.analytics.send_event") as send:
            s._emit_turn_cost()
        assert send.call_args.kwargs["ended_by"] == "handback_stuck"

    def test_inflight_exception_resolves_to_error(self):
        s = _bare_session()
        with patch("anton.analytics.send_event") as send:
            try:
                raise ValueError("boom")
            except ValueError:
                s._emit_turn_cost()
        assert send.call_args.kwargs["ended_by"] == "error"

    def test_cancel_event_resolves_to_cancelled(self):
        s = _bare_session(cancel_event=MagicMock(is_set=lambda: True))
        s._turn_cost.ended_by = "round_cap"  # cancel wins over explicit marks
        with patch("anton.analytics.send_event") as send:
            s._emit_turn_cost()
        assert send.call_args.kwargs["ended_by"] == "cancelled"

    def test_settings_without_analytics_fields_falls_back(self):
        # A host passing bare CoreSettings (no analytics_* fields) must not
        # crash emission — the tools.py fallback pattern resolves a fresh
        # AntonSettings (whose analytics_enabled env guard then applies).
        s = _bare_session(settings=object())
        with patch("anton.analytics.send_event") as send:
            s._emit_turn_cost()
        assert send.called


class TestTurnResetsBooks:
    @pytest.mark.asyncio
    async def test_each_turn_starts_fresh_books(self, monkeypatch):
        # Long-lived session (CLI): turn N+1 must not inherit turn N's totals.
        # Drive the real turn() twice with a text-only response and compare
        # the emitted totals.
        from tests.conftest import make_mock_llm

        llm = make_mock_llm()
        llm.plan.return_value = LLMResponse(content="hi", usage=_usage(inp=100))

        from anton.core.session import ChatSessionConfig

        session = ChatSession(ChatSessionConfig(llm_client=llm))
        monkeypatch.setattr(session, "_router_enabled", False, raising=False)

        books: list[TurnCost] = []
        totals: list[int] = []
        real_emit = session._emit_turn_cost

        def _spy():
            if session._turn_cost is not None:
                books.append(session._turn_cost)
                # Simulate this turn having counted something, so cross-turn
                # leakage would be visible in the next turn's starting total
                # (the mocked client never fires the listener itself).
                session._turn_cost.add("planning", "planner", _usage(inp=100))
                totals.append(session._turn_cost.total_tokens)
            real_emit()

        monkeypatch.setattr(session, "_emit_turn_cost", _spy)
        await session.turn("one")
        await session.turn("two")

        assert len(books) == 2
        assert books[0] is not books[1], "turn 2 must open fresh books"
        assert totals[0] == totals[1], "totals must not accumulate across turns"


class TestCacheSplitAcrossProviders:
    """The two headline invariants of the cache split (#309 review: the
    no-subtract mutation survived all 1474 tests, so both were unpinned).

    1. ``input_tokens`` means FRESH prompt tokens on every provider — the
       OpenAI dialect's cache-inclusive ``prompt_tokens`` gets both cached
       buckets subtracted out.
    2. ``context_pressure`` keeps its exact pre-split value: all prompt-side
       tokens, cached or not. Getting this wrong triggers premature
       compaction on gateway traffic.
    """

    def _chat_usage(self, prompt=1000, cached=800, written=0, completion=50):
        return SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=cached, cache_write_tokens=written
            ),
        )

    def test_chat_shape_subtracts_cached_share(self):
        from anton.core.llm.openai import _split_cached_input

        assert _split_cached_input(self._chat_usage()) == (200, 800, 0)

    def test_gateway_shape_splits_reads_and_writes(self):
        # mindshub_inference publishes both buckets and composes
        # prompt_tokens = fresh + reads + writes.
        from anton.core.llm.openai import _split_cached_input

        usage = self._chat_usage(prompt=100, cached=30, written=10)
        assert _split_cached_input(usage) == (60, 30, 10)

    def test_responses_api_shape_uses_input_token_names(self):
        from anton.core.llm.openai import _split_cached_input

        usage = SimpleNamespace(
            input_tokens=1000,
            output_tokens=10,
            input_tokens_details=SimpleNamespace(
                cached_tokens=800, cache_write_tokens=50
            ),
        )
        assert _split_cached_input(usage) == (150, 800, 50)

    def test_no_details_means_all_fresh(self):
        from anton.core.llm.openai import _split_cached_input

        assert _split_cached_input(SimpleNamespace(prompt_tokens=500)) == (500, 0, 0)

    def test_malformed_payload_never_yields_negative_fresh_input(self):
        from anton.core.llm.openai import _split_cached_input

        usage = self._chat_usage(prompt=100, cached=900, written=900)
        fresh, read, write = _split_cached_input(usage)
        assert fresh == 0 and read + write <= 100

    @pytest.mark.asyncio
    async def test_provider_reports_split_and_unchanged_context_pressure(self):
        # End to end through the real provider: components split, and
        # context_pressure computed on ALL prompt tokens (1000), which is
        # exactly what it was before the split existed.
        from anton.core.llm.openai import OpenAIProvider
        from anton.core.llm.provider import compute_context_pressure

        provider = OpenAIProvider(api_key="k")
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="hi", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=self._chat_usage(prompt=1000, cached=750, written=50),
        )
        provider._client = MagicMock()
        provider._client.chat = MagicMock()
        provider._client.chat.completions = MagicMock()
        provider._client.chat.completions.create = AsyncMock(return_value=completion)

        response = await provider.complete(
            model="sonnet", system="s", messages=[{"role": "user", "content": "hi"}]
        )
        assert response.usage.input_tokens == 200
        assert response.usage.cache_read_tokens == 750
        assert response.usage.cache_creation_tokens == 50
        assert response.usage.context_tokens == 1000
        assert response.usage.context_pressure == compute_context_pressure("sonnet", 1000)


class TestListenerCapturedAtIssueTime:
    @pytest.mark.asyncio
    async def test_background_call_started_before_disarm_books_nowhere(self):
        # End-of-turn background work (cerebellum flush, identity update)
        # shares the client. A call issued during turn N that lands after
        # turn N+1 armed its books must NOT land in turn N+1 (#309 review).
        release = asyncio.Event()
        provider = AsyncMock()

        async def _slow_complete(**kwargs):
            await release.wait()
            return LLMResponse(content="late", usage=_usage(inp=999))

        provider.complete = AsyncMock(side_effect=_slow_complete)
        client = LLMClient(
            planning_provider=provider,
            planning_model="planner",
            coding_provider=provider,
            coding_model="coder",
        )

        turn_one = TurnCost()
        client.usage_listener = turn_one.add
        task = asyncio.create_task(client.code(system="s", messages=[]))
        await asyncio.sleep(0)  # let the call reach the await

        # Turn 1 ends: books close, listener disarms. Turn 2 opens its own.
        client.usage_listener = None
        turn_two = TurnCost()
        client.usage_listener = turn_two.add

        release.set()
        await task

        assert turn_two.llm_calls == 0, "late call must not book into the next turn"
        assert turn_one.llm_calls == 1, "it belongs to the turn that issued it"
