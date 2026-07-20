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
