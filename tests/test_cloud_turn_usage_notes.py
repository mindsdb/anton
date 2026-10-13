"""A cloud turn's connector usage notes reach the assembled system prompt, for
the connections the turn actually has and no others."""
from __future__ import annotations

import sys
import types

import pytest

import anton.core.llm.client as llm_client_mod
from anton.cloud_turn.contract import TurnRequestV1
from anton.cloud_turn.session import _WORKSPACE_PATH_ENV, build_cloud_chat_session
from anton.core.datasources.data_vault import TurnKeyDataVault
from tests.conftest import make_mock_llm

DRIVE_MARKER = "DRIVE-USAGE-MARKER"
GMAIL_MARKER = "GMAIL-USAGE-MARKER"


@pytest.fixture(autouse=True)
def _offline_turn(tmp_path, monkeypatch):
    monkeypatch.setenv(_WORKSPACE_PATH_ENV, str(tmp_path))
    monkeypatch.setattr(
        llm_client_mod.LLMClient, "from_settings",
        classmethod(lambda cls, settings: make_mock_llm()),
    )
    # MCP discovery is not under test here; stubbing the module also keeps the
    # test runnable in an env that lacks the `mcp` runtime dependency.
    wiring = types.ModuleType("anton.core.mcp.wiring")
    wiring.discover_mcp_tools = lambda vault, connections: ([], [])
    monkeypatch.setitem(sys.modules, "anton.core.mcp.wiring", wiring)
    monkeypatch.setattr(
        TurnKeyDataVault, "_fetch",
        lambda self, engine, name: {
            "access_token": "t", "account_email": "u@example.com", "auth_type": "oauth",
        },
    )


async def _system_prompt(**req_overrides) -> str:
    body = dict(protocol_version=1, conversation_id="conv_1", input="hi")
    body.update(req_overrides)
    session = build_cloud_chat_session(TurnRequestV1(**body))
    return await session._build_system_prompt()


async def test_a_connected_engine_note_reaches_the_system_prompt():
    prompt = await _system_prompt(
        oauth={
            "turn_key": "tk",
            "connections": [{"engine": "google_drive", "name": "primary"}],
        },
        connectors={"google_drive": {"usage_notes": DRIVE_MARKER}},
    )

    assert prompt.count(DRIVE_MARKER) == 1
    assert prompt.index("### Slug: `google_drive-primary`") < prompt.index(DRIVE_MARKER)


async def test_a_note_for_an_engine_the_turn_does_not_have_is_left_out():
    prompt = await _system_prompt(
        oauth={
            "turn_key": "tk",
            "connections": [{"engine": "google_drive", "name": "primary"}],
        },
        connectors={
            "google_drive": {"usage_notes": DRIVE_MARKER},
            "gmail": {"usage_notes": GMAIL_MARKER},
        },
    )

    assert DRIVE_MARKER in prompt
    assert GMAIL_MARKER not in prompt
