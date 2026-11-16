"""`tool_messages` is a capability the host declares: whether it renders a
tool's message to the user (`StreamToolResult(action="message")`) as an agent
message. The relay is pinned in test_tool_registry_streaming.py."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from anton.config.settings import AntonSettings
from tests.conftest import make_mock_llm


def _cli_settings(tmp_path: Path) -> AntonSettings:
    settings = AntonSettings(episodic_memory=False, _env_file=None)
    settings.resolve_workspace(str(tmp_path))
    return settings


async def cli_chat_loop_config(tmp_path: Path, monkeypatch):
    """The ChatSessionConfig the CLI's `_chat_loop` builds for a fresh session."""
    import anton.chat as chat
    import anton.chat_session as chat_session

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
        await chat._chat_loop(MagicMock(), _cli_settings(tmp_path))
    return captured["config"]


def cloud_session_config(tmp_path: Path, monkeypatch, **request_fields):
    """The ChatSessionConfig the pod builds for a request with `request_fields`."""
    import anton.core.llm.client as llm_client_mod
    import anton.core.session as session_mod
    from anton.cloud_turn.contract import TurnRequestV1
    from anton.cloud_turn.session import _WORKSPACE_PATH_ENV, build_cloud_chat_session

    captured: dict = {}
    monkeypatch.setattr(session_mod, "ChatSession", lambda config: captured.setdefault("config", config))
    monkeypatch.setattr(
        llm_client_mod.LLMClient, "from_settings", classmethod(lambda cls, settings: object()),
    )
    monkeypatch.setenv(_WORKSPACE_PATH_ENV, str(tmp_path))
    build_cloud_chat_session(
        TurnRequestV1(protocol_version=1, conversation_id="c", input="hi", **request_fields)
    )
    return captured["config"]


def test_the_flag_defaults_off_and_lands_on_the_session(make_session):
    from anton.core.session import ChatSessionConfig

    assert ChatSessionConfig.__dataclass_fields__["tool_messages"].default is False
    assert make_session().tool_messages is False
    assert make_session(tool_messages=True).tool_messages is True


async def test_both_cli_session_builders_declare_it(tmp_path, monkeypatch):
    import anton.chat_session as chat_session

    assert (await cli_chat_loop_config(tmp_path, monkeypatch)).tool_messages is True

    captured: dict = {}
    with (
        patch("anton.chat.ChatSession", lambda config: captured.setdefault("config", config)),
        patch("anton.core.llm.client.LLMClient.from_settings", return_value=make_mock_llm()),
        patch.object(chat_session, "get_runtime_factory", lambda *_a, **_k: MagicMock()),
        patch.object(chat_session, "refresh_knowledge", lambda *a, **k: None),
        patch.object(chat_session, "build_runtime_context", lambda _s: ""),
    ):
        await chat_session.rebuild_session(
            settings=_cli_settings(tmp_path), state={}, self_awareness=None,
            cortex=None, workspace=MagicMock(), console=MagicMock(),
        )
    assert captured["config"].tool_messages is True


def test_the_cloud_request_carries_it():
    from anton.cloud_turn.contract import TurnRequestV1

    base = {"protocol_version": 1, "conversation_id": "c", "input": "hi"}
    assert TurnRequestV1.from_json(json.dumps(base)).tool_messages is False
    assert TurnRequestV1.from_json(json.dumps({**base, "tool_messages": True})).tool_messages is True
    assert TurnRequestV1.from_json(json.dumps({**base, "tool_messages": "true"})).tool_messages is False


def test_the_pod_passes_it_to_the_session(tmp_path, monkeypatch):
    assert cloud_session_config(tmp_path, monkeypatch).tool_messages is False
    assert cloud_session_config(tmp_path, monkeypatch, tool_messages=True).tool_messages is True


def test_the_cloud_contract_documents_the_message_result():
    from anton.cloud_turn import contract

    doc = contract.__doc__ or ""
    assert '`action: "message"`' in doc
    assert "tool_messages" in doc


def test_a_message_reaches_the_user_only_with_both_the_flag_and_an_emitter():
    from types import SimpleNamespace

    from anton.core.tools.progress import renders_tool_messages

    assert renders_tool_messages(SimpleNamespace(tool_messages=True, emitter=object())) is True
    assert renders_tool_messages(SimpleNamespace(tool_messages=True, emitter=None)) is False
    assert renders_tool_messages(SimpleNamespace(tool_messages=False, emitter=object())) is False
    assert renders_tool_messages(SimpleNamespace()) is False


def test_the_session_exposes_its_act_first_setting(make_session):
    assert make_session().act_first is True
    assert make_session(act_first=False).act_first is False
