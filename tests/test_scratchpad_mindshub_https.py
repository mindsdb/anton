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


async def _env_the_pad_gets(tmp_path, monkeypatch) -> dict[str, str]:
    seen: dict[str, str] = {}

    async def spawn(*_argv, env, **_kwargs):
        seen.update(env)
        raise _Spawned

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    pad = LocalScratchpadRuntime(
        name="mindshub-https",
        coding_provider="anthropic",
        coding_model="",
        coding_api_key="",
        coding_base_url="",
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
