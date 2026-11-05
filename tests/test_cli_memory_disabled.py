"""`ANTON_MEMORY_ENABLED=false` turns the CLI's cortex off.

The banner says "memory disabled" for that setting, so the cortex must write
nothing: no cerebellum lesson, no consolidation, no identity extraction. Every
place the CLI sets the cortex's mode from settings reads
`AntonSettings.effective_memory_mode`, which is "off" when memory is disabled.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from anton.config.settings import AntonSettings
from tests.conftest import make_mock_llm


def _settings(tmp_path: Path, *, memory_enabled: bool, memory_mode: str = "autopilot") -> AntonSettings:
    settings = AntonSettings(
        memory_enabled=memory_enabled,
        memory_mode=memory_mode,
        episodic_memory=False,
        _env_file=None,
    )
    settings.resolve_workspace(str(tmp_path))
    return settings


@pytest.mark.parametrize(
    ("memory_enabled", "memory_mode", "expected"),
    [
        (True, "autopilot", "autopilot"),
        (True, "copilot", "copilot"),
        (True, "off", "off"),
        (False, "autopilot", "off"),
        (False, "copilot", "off"),
    ],
)
def test_effective_memory_mode(tmp_path, memory_enabled, memory_mode, expected):
    settings = _settings(tmp_path, memory_enabled=memory_enabled, memory_mode=memory_mode)
    assert settings.effective_memory_mode == expected


async def test_the_cli_builds_its_cortex_off_when_memory_is_disabled(tmp_path, monkeypatch):
    import anton.chat as chat
    import anton.chat_session as chat_session

    # The CLI reads global memory from the home directory; keep it in tmp.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    captured: dict = {}

    class Built(Exception):
        """Stops _chat_loop once it has built its session."""

    def fake_session(config):
        captured["config"] = config
        raise Built

    vault = MagicMock()
    vault.list_connections.return_value = []
    with (
        patch("anton.core.llm.client.LLMClient.from_settings", return_value=make_mock_llm()),
        patch.object(chat, "LocalDataVault", return_value=vault),
        patch.object(chat, "ChatSession", fake_session),
        patch.object(chat, "build_runtime_context", lambda _s: ""),
        patch.object(chat_session, "get_runtime_factory", lambda *_a, **_k: MagicMock()),
        pytest.raises(Built),
    ):
        await chat._chat_loop(MagicMock(), _settings(tmp_path, memory_enabled=False))

    assert captured["config"].cortex.mode == "off"


async def test_a_rebuilt_session_keeps_the_cortex_off_when_memory_is_disabled(tmp_path):
    import anton.chat_session as chat_session

    cortex = MagicMock(mode="off")
    with (
        patch("anton.chat.ChatSession", lambda _config: MagicMock()),
        patch("anton.core.llm.client.LLMClient.from_settings", return_value=make_mock_llm()),
        patch.object(chat_session, "get_runtime_factory", lambda *_a, **_k: MagicMock()),
        patch.object(chat_session, "refresh_knowledge", lambda *a, **k: None),
        patch.object(chat_session, "build_runtime_context", lambda _s: ""),
    ):
        await chat_session.rebuild_session(
            settings=_settings(tmp_path, memory_enabled=False),
            state={},
            self_awareness=None,
            cortex=cortex,
            workspace=MagicMock(),
            console=MagicMock(),
        )

    assert cortex.mode == "off"


async def test_picking_a_memory_mode_keeps_the_cortex_off_when_memory_is_disabled(
    tmp_path, monkeypatch
):
    import anton.memory.manage as manage

    async def pick_copilot(*_args, **_kwargs):
        return "2"

    monkeypatch.setattr(manage, "prompt_or_cancel", pick_copilot)
    settings = _settings(tmp_path, memory_enabled=False)
    cortex = MagicMock(mode="off")
    workspace = MagicMock()

    await manage.handle_setup_memory(MagicMock(), settings, workspace, cortex)

    # The choice is saved for when memory is enabled again.
    assert settings.memory_mode == "copilot"
    workspace.set_secret.assert_called_once_with("ANTON_MEMORY_MODE", "copilot")
    assert cortex.mode == "off"
