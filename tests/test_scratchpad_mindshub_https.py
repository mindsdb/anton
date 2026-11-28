"""A saved http:// MindsHub URL reaches the CLI's scratchpad over https.

``AntonSettings`` loads such a URL as https, but the CLI's pads keep the server
environment (``inherit_server_env=True``), and code in the pad reads
``ANTON_MINDS_URL``, ``ANTON_OPENAI_BASE_URL`` and ``OPENAI_BASE_URL`` as they stand.
The runtime moves them to https before it spawns the pad, so code in the pad never
sends the key in clear. Other URLs reach the pad as saved.
"""

from __future__ import annotations

import asyncio

import pytest

from anton.core.backends.local import LocalScratchpadRuntime


class _Spawned(Exception):
    """Stops the start at the spawn, once the pad's environment is known."""


async def _env_the_pad_gets(
    tmp_path, monkeypatch, *, coding_provider: str = "anthropic", coding_base_url: str = ""
) -> dict[str, str]:
    seen: dict[str, str] = {}

    async def spawn(*_argv, env, **_kwargs):
        seen.update(env)
        raise _Spawned

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    pad = LocalScratchpadRuntime(
        name="mindshub-https",
        coding_provider=coding_provider,
        coding_model="",
        coding_api_key="",
        coding_base_url=coding_base_url,
        workspace_path=tmp_path,
        inherit_server_env=True,
    )
    with pytest.raises(_Spawned):
        await pad.start()
    return seen


@pytest.mark.parametrize(
    ("saved", "seen"),
    [
        ("http://api.mindshub.ai", "https://api.mindshub.ai"),
        ("HTTP://api.mindshub.ai", "https://api.mindshub.ai"),
        ("http://localhost:11434", "http://localhost:11434"),
        ("http://api.example.com", "http://api.example.com"),
    ],
)
async def test_the_pad_gets_mindshub_urls_over_https(tmp_path, monkeypatch, saved, seen):
    monkeypatch.setenv("ANTON_MINDS_URL", saved)
    monkeypatch.setenv("ANTON_OPENAI_BASE_URL", f"{saved}/v1")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    env = await _env_the_pad_gets(tmp_path, monkeypatch)

    assert env["ANTON_MINDS_URL"] == seen
    assert env["ANTON_OPENAI_BASE_URL"] == f"{seen}/v1"
    assert env["OPENAI_BASE_URL"] == f"{seen}/v1"


@pytest.mark.parametrize(
    ("coding_base_url", "seen"),
    [
        ("http://api.mindshub.ai/v1", "https://api.mindshub.ai/v1"),
        ("http://localhost:11434/v1", "http://localhost:11434/v1"),
    ],
)
async def test_the_runtimes_own_coding_base_url_reaches_the_pad_over_https(
    tmp_path, monkeypatch, coding_base_url, seen
):
    """An in-session setup can leave an http:// MindsHub URL that never went through
    AntonSettings. The runtime writes it into the pad's environment last, so the
    rewrite has to come after that write."""
    for name in ("ANTON_MINDS_URL", "ANTON_OPENAI_BASE_URL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)

    env = await _env_the_pad_gets(
        tmp_path, monkeypatch, coding_provider="openai-compatible", coding_base_url=coding_base_url
    )

    assert env["OPENAI_BASE_URL"] == seen
    assert env["ANTON_OPENAI_BASE_URL"] == seen
