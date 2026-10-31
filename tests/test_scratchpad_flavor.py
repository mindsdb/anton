"""Flavor resolution for the scratchpad's injected web_search() helper.

Regression guard for the minds gateway case: OpenAIProvider.name is always
"openai" even when the base URL is the minds gateway, so keying the flavor off
the provider name alone picked the Responses-API flavor and dispatched
web_search to a Responses endpoint the minds gateway does not implement — it
silently returned empty. The host must win over the name.
"""

from __future__ import annotations

import pytest

from anton.core.llm.openai import OpenAIProvider

resolve = OpenAIProvider.resolve_web_flavor


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.mindshub.ai/v1",
        "https://api.staging.mindshub.ai/v1",
        "https://mdb.ai/api/v1",
        "HTTPS://API.MINDSHUB.AI/v1",  # case-insensitive
    ],
)
def test_minds_host_uses_passthrough_even_with_openai_name(base_url):
    # provider name is "openai" (the class default) yet the host is minds ->
    # must be the chat.completions passthrough, NOT the Responses API.
    assert resolve("openai", base_url) == OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH


def test_direct_openai_uses_responses_api():
    assert resolve("openai", "https://api.openai.com/v1") == OpenAIProvider.FLAVOR_OPENAI
    assert resolve("openai", "") == OpenAIProvider.FLAVOR_OPENAI
    assert resolve("openai", None) == OpenAIProvider.FLAVOR_OPENAI


def test_other_compatible_endpoint_is_generic():
    assert (
        resolve("openai-compatible", "https://my-proxy.internal/v1")
        == OpenAIProvider.FLAVOR_OPENAI_COMPATIBLE_GENERIC
    )


def test_openai_compatible_minds_base_still_passthrough():
    # An "openai-compatible" provider pointed at minds keeps passthrough too.
    assert (
        resolve("openai-compatible", "https://api.mindshub.ai/v1")
        == OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH
    )


# ── The agent's own flavor reaches the pad ────────────────────────────


from anton.core.backends.manager import ScratchpadManager  # noqa: E402


@pytest.mark.parametrize(
    "flavor",
    [
        OpenAIProvider.FLAVOR_OPENAI,
        OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH,
        OpenAIProvider.FLAVOR_OPENAI_COMPATIBLE_GENERIC,
    ],
)
def test_export_connection_info_carries_the_providers_flavor(flavor):
    provider = OpenAIProvider(api_key="k", base_url="http://model-route:8000/v1", flavor=flavor)
    assert provider.export_connection_info().flavor == flavor


class _Pad:
    async def start(self):
        pass


def _manager(factory, flavor):
    return ScratchpadManager(
        runtime_factory=factory,
        coding_provider="openai",
        coding_model="m",
        coding_api_key="k",
        coding_base_url="http://model-route:8000/v1",
        coding_flavor=flavor,
    )


async def test_the_manager_passes_the_flavor_to_a_factory_that_takes_it():
    seen = {}

    def factory(*, name, cells, coding_provider, coding_model, coding_api_key,
                coding_base_url, workspace_path, coding_flavor=None):
        seen["coding_flavor"] = coding_flavor
        return _Pad()

    await _manager(factory, OpenAIProvider.FLAVOR_OPENAI_COMPATIBLE_GENERIC).get_or_create("p")
    assert seen == {"coding_flavor": OpenAIProvider.FLAVOR_OPENAI_COMPATIBLE_GENERIC}


async def test_the_manager_leaves_the_flavor_off_a_factory_that_predates_it():
    """A factory written against the previous signature still builds its pad."""

    def factory(*, name, cells, coding_provider, coding_model, coding_api_key,
                coding_base_url, workspace_path):
        return _Pad()

    pad = await _manager(factory, OpenAIProvider.FLAVOR_OPENAI_COMPATIBLE_GENERIC).get_or_create("p")
    assert isinstance(pad, _Pad)
