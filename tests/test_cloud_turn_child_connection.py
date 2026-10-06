"""The scratchpad child of a cloud turn receives the turn's own inference connection.

The session refuses a datasource turn unless ``settings.openai_api_key`` and
``settings.openai_base_url`` are the turn's; the datasource helper then reads
the child's ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL``. This proves the second
pair is the first, whatever plain ``OPENAI_*`` values the pod itself carries.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

import anton.core.backends.local as local
from anton.config.settings import AntonSettings
from anton.core.llm.client import LLMClient

TURN_KEY = "mdb_turn.secret"
TURN_BASE = "https://inference.internal/v1"


class _Stop(Exception):
    """Raised in place of spawning the child, once its environment is built."""


def test_the_child_gets_the_connection_the_session_checked(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "pod-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://pod.invalid/v1")
    monkeypatch.delenv("ANTON_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTON_OPENAI_BASE_URL", raising=False)
    settings = AntonSettings(
        _env_file=None,
        planning_provider="minds-cloud",
        coding_provider="minds-cloud",
        minds_api_key=TURN_KEY,
        minds_url=TURN_BASE,
    )
    connection = LLMClient.from_settings(settings).coding_provider.export_connection_info()
    runtime = local.LocalScratchpadRuntime(
        "pad",
        coding_provider=connection.provider,
        coding_model="model",
        coding_api_key=connection.api_key or "",
        coding_base_url=connection.base_url or "",
        workspace_path=tmp_path,
        workspace_env_overlay={"ANTON_CLOUD_TURN": "1"},
    )
    monkeypatch.setattr(runtime, "_ensure_venv", lambda: None)
    runtime._venv_python = sys.executable
    seen: dict[str, str] = {}

    async def spawn(*_args, env=None, **_kwargs):
        seen.update(env or {})
        raise _Stop

    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(_Stop):
        asyncio.run(runtime.start())

    assert (settings.openai_api_key, settings.openai_base_url) == (TURN_KEY, TURN_BASE)
    assert seen["OPENAI_API_KEY"] == settings.openai_api_key
    assert seen["OPENAI_BASE_URL"] == settings.openai_base_url
