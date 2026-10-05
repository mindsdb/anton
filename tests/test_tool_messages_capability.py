"""`tool_messages` is a capability the host declares: whether it renders a
tool's message to the user (`StreamToolResult(action="message")`) as an agent
message. The relay is pinned in test_tool_registry_streaming.py."""
from __future__ import annotations

import inspect
import json


def test_the_flag_defaults_off_and_lands_on_the_session(make_session):
    from anton.core.session import ChatSessionConfig

    assert ChatSessionConfig.__dataclass_fields__["tool_messages"].default is False
    assert make_session().tool_messages is False
    assert make_session(tool_messages=True).tool_messages is True


def test_both_cli_session_builders_declare_it():
    from anton import chat, chat_session

    assert "tool_messages=True" in inspect.getsource(chat)
    assert "tool_messages=True" in inspect.getsource(chat_session)


def test_the_cloud_request_carries_it():
    from anton.cloud_turn.contract import TurnRequestV1

    base = {"protocol_version": 1, "conversation_id": "c", "input": "hi"}
    assert TurnRequestV1.from_json(json.dumps(base)).tool_messages is False
    assert TurnRequestV1.from_json(json.dumps({**base, "tool_messages": True})).tool_messages is True
    assert TurnRequestV1.from_json(json.dumps({**base, "tool_messages": "true"})).tool_messages is False


def test_the_pod_passes_it_to_the_session():
    from anton.cloud_turn import session as cloud_session

    assert "tool_messages=request.tool_messages" in inspect.getsource(cloud_session)


def test_the_cloud_contract_documents_the_message_result():
    from anton.cloud_turn import contract

    doc = contract.__doc__ or ""
    assert '`action: "message"`' in doc
    assert "tool_messages" in doc
