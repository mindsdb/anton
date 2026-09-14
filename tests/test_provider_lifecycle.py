"""Provider/client shutdown: who gets closed, once, and what cleanup may swallow."""

from __future__ import annotations

import asyncio
import gc

import pytest

from anton.core.llm import provider as provider_mod
from anton.core.llm.client import LLMClient
from anton.core.llm.openai import _aclose_stream
from anton.core.llm.provider import (
    close_live_providers,
    register_provider,
    unregister_provider,
)


class FakeProvider:
    """Minimal stand-in: records closes, optionally raises."""

    def __init__(self, raises: BaseException | None = None) -> None:
        self.closes = 0
        self._raises = raises

    async def aclose(self) -> None:
        self.closes += 1
        if self._raises is not None:
            raise self._raises


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = list(provider_mod._LIVE_PROVIDERS)
    provider_mod._LIVE_PROVIDERS.clear()
    yield
    provider_mod._LIVE_PROVIDERS.clear()
    for p in saved:
        provider_mod._LIVE_PROVIDERS.add(p)


def _client(planning, coding, router=None) -> LLMClient:
    return LLMClient(
        planning_provider=planning,
        planning_model="p",
        coding_provider=coding,
        coding_model="c",
        router_provider=router,
        router_model="r" if router is not None else None,
    )


class TestLLMClientAclose:
    async def test_closes_all_three_roles(self):
        planning, coding, router = FakeProvider(), FakeProvider(), FakeProvider()
        await _client(planning, coding, router).aclose()
        assert (planning.closes, coding.closes, router.closes) == (1, 1, 1)

    async def test_shared_provider_closed_once(self):
        shared = FakeProvider()
        await _client(shared, shared, shared).aclose()
        assert shared.closes == 1

    async def test_router_defaults_to_coding_and_is_not_double_closed(self):
        planning, coding = FakeProvider(), FakeProvider()
        await _client(planning, coding).aclose()
        assert coding.closes == 1

    async def test_one_failure_does_not_stop_the_others(self):
        planning = FakeProvider(raises=RuntimeError("boom"))
        coding, router = FakeProvider(), FakeProvider()
        await _client(planning, coding, router).aclose()
        assert (coding.closes, router.closes) == (1, 1)

    async def test_cancellation_is_not_swallowed(self):
        planning = FakeProvider(raises=asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await _client(planning, FakeProvider(), FakeProvider()).aclose()


class TestRegistry:
    async def test_close_live_providers_closes_and_empties(self):
        p = FakeProvider()
        register_provider(p)
        await close_live_providers()
        assert p.closes == 1
        assert len(provider_mod._LIVE_PROVIDERS) == 0

    def test_registry_does_not_keep_providers_alive(self):
        # Weak refs: in a long-lived host that never drains (cowork-server
        # builds providers per turn), registration must not become a leak.
        register_provider(FakeProvider())
        gc.collect()
        assert len(provider_mod._LIVE_PROVIDERS) == 0

    async def test_unregister_stops_a_second_close(self):
        p = FakeProvider()
        register_provider(p)
        unregister_provider(p)
        await close_live_providers()
        assert p.closes == 0

    def test_unregister_is_safe_when_absent(self):
        unregister_provider(FakeProvider())  # must not raise

    async def test_one_failure_does_not_strand_the_rest(self):
        bad, good = FakeProvider(raises=RuntimeError("boom")), FakeProvider()
        register_provider(bad)
        register_provider(good)
        await close_live_providers()
        assert good.closes == 1


class TestAcloseStream:
    async def test_plain_async_generator_uses_aclose(self):
        closed = []

        class Gen:
            async def aclose(self):
                closed.append(True)

        await _aclose_stream(Gen())
        assert closed == [True]

    async def test_prefers_aclose_over_a_sync_close(self):
        # A sync close() beside an async aclose() must not win the getattr
        # race: awaiting its return value would raise, be swallowed, and leak.
        class Stream:
            aclosed = False

            def close(self):
                raise AssertionError("picked sync close over async aclose")

            async def aclose(self):
                self.aclosed = True

        s = Stream()
        await _aclose_stream(s)
        assert s.aclosed

    async def test_sync_only_close_is_still_called(self):
        class Stream:
            closed = False

            def close(self):
                self.closed = True

        s = Stream()
        await _aclose_stream(s)
        assert s.closed

    async def test_none_is_a_noop(self):
        await _aclose_stream(None)

    async def test_close_failure_is_swallowed(self):
        class Stream:
            async def close(self):
                raise RuntimeError("boom")

        await _aclose_stream(Stream())  # must not raise

    async def test_cancellation_is_not_swallowed(self):
        class Stream:
            async def close(self):
                raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await _aclose_stream(Stream())


class _StubSDKClient:
    def __init__(self) -> None:
        self.closed = 0

    async def close(self) -> None:
        self.closed += 1


class TestRealProvidersAclose:
    """The wiring the registry drain depends on: __init__ registers, aclose
    awaits the SDK client's close() and unregisters."""

    async def test_openai_provider(self):
        from anton.core.llm.openai import OpenAIProvider

        p = OpenAIProvider(api_key="test")
        assert p in provider_mod._LIVE_PROVIDERS
        stub = _StubSDKClient()
        p._client = stub
        await p.aclose()
        assert stub.closed == 1
        assert p not in provider_mod._LIVE_PROVIDERS

    async def test_anthropic_provider(self):
        from anton.core.llm.anthropic import AnthropicProvider

        p = AnthropicProvider(api_key="test")
        assert p in provider_mod._LIVE_PROVIDERS
        stub = _StubSDKClient()
        p._client = stub
        await p.aclose()
        assert stub.closed == 1
        assert p not in provider_mod._LIVE_PROVIDERS


class TestAsyncgenNoiseFilter:
    """The exit-noise filter must match exactly the known upstream error."""

    @staticmethod
    def _context(filename="site-packages/httpcore2/_async/http11.py",
                 exc=RuntimeError("generator didn't stop after athrow()")):
        class Code:
            co_filename = filename

        class Agen:
            ag_code = Code()

        return {"exception": exc, "asyncgen": Agen()}

    def test_matches_the_upstream_error(self):
        assert provider_mod._is_upstream_asyncgen_noise(self._context())

    def test_other_exceptions_pass_through(self):
        ctx = self._context(exc=RuntimeError("something real broke"))
        assert not provider_mod._is_upstream_asyncgen_noise(ctx)

    def test_our_own_generators_pass_through(self):
        ctx = self._context(filename="anton/core/session.py")
        assert not provider_mod._is_upstream_asyncgen_noise(ctx)

    def test_context_without_asyncgen_passes_through(self):
        ctx = {"exception": RuntimeError("generator didn't stop after athrow()")}
        assert not provider_mod._is_upstream_asyncgen_noise(ctx)

    async def test_install_respects_an_existing_handler(self):
        loop = asyncio.get_running_loop()
        sentinel = lambda loop, context: None  # noqa: E731
        loop.set_exception_handler(sentinel)
        try:
            provider_mod.install_asyncgen_noise_filter()
            assert loop.get_exception_handler() is sentinel
        finally:
            loop.set_exception_handler(None)

    async def test_install_filters_noise_and_forwards_the_rest(self):
        loop = asyncio.get_running_loop()
        assert loop.get_exception_handler() is None
        try:
            provider_mod.install_asyncgen_noise_filter()
            handler = loop.get_exception_handler()
            forwarded = []
            loop.default_exception_handler = lambda ctx: forwarded.append(ctx)
            handler(loop, self._context())  # dropped
            real = {"exception": RuntimeError("boom"), "message": "x"}
            handler(loop, real)  # forwarded
            assert forwarded == [real]
        finally:
            loop.set_exception_handler(None)
