"""A cloud turn's custom API connections in the pod: never fetched, passed to the scratchpad by id only."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import anton.core.llm.client as llm_client_mod
import anton.core.session as session_mod
from anton.cloud_turn.contract import TurnRequestV1
from anton.cloud_turn.session import _WORKSPACE_PATH_ENV, build_cloud_chat_session

_FIXTURE = Path(__file__).parent / "fixtures" / "turn_custom_api_block.json"


def _request(**fields) -> TurnRequestV1:
    """A turn request parsed from the wire, carrying the producer's recorded custom API block."""
    body = {
        "protocol_version": 1,
        "conversation_id": "conv_1",
        "input": "hello",
        "custom_api": json.loads(_FIXTURE.read_text(encoding="utf-8"))["custom_api"],
    }
    body.update(fields)
    return TurnRequestV1.from_json(json.dumps(body))


def _captured_config(tmp_path, monkeypatch, request: TurnRequestV1):
    """The ChatSessionConfig the cloud builder hands to ChatSession for ``request``."""
    captured: dict = {}
    monkeypatch.setattr(
        session_mod, "ChatSession", lambda config: captured.setdefault("config", config)
    )
    monkeypatch.setattr(
        llm_client_mod.LLMClient,
        "from_settings",
        classmethod(lambda cls, settings: object()),
    )
    monkeypatch.setenv(_WORKSPACE_PATH_ENV, str(tmp_path))
    build_cloud_chat_session(request)
    return captured["config"]


def test_a_custom_api_turn_opens_no_connection_and_builds_no_vault(
    tmp_path, monkeypatch
):
    """The pod never asks auth, or anyone, for a custom connection's values."""
    attempts: list = []

    def refuse_connect(sock, address):
        attempts.append(address)
        raise OSError("no network in this test")

    monkeypatch.setattr(socket.socket, "connect", refuse_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse_connect)

    config = _captured_config(tmp_path, monkeypatch, _request())

    assert attempts == []
    assert config.data_vault is None
