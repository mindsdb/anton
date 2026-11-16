"""`tool_peek` is a capability the host declares, not a phase every wire
carries. The relay itself is pinned in
test_tool_registry_streaming.py; this file pins the three ends around it.
"""
from __future__ import annotations

import dataclasses

from tests.test_tool_messages_capability import cli_chat_loop_config, cloud_session_config


def test_the_flag_defaults_off_and_lands_on_the_session(make_session):
    from anton.core.session import ChatSessionConfig

    assert ChatSessionConfig.__dataclass_fields__["live_tool_peek"].default is False
    assert make_session().live_tool_peek is False
    assert make_session(live_tool_peek=True).live_tool_peek is True


async def test_the_cli_is_the_host_that_declares_it(tmp_path, monkeypatch):
    """The CLI footer renders the tail (`chat_ui.update_progress`, phase
    `tool_peek`); it is the only first-party host that does, so it is the
    only one that opts in. cowork-server builds its config in its own repo
    and the cloud pod keeps it off whatever the request asks for."""
    from anton.cloud_turn.contract import TurnRequestV1

    assert (await cli_chat_loop_config(tmp_path, monkeypatch)).live_tool_peek is True

    every_flag_on = {f.name: True for f in dataclasses.fields(TurnRequestV1) if f.type in (bool, "bool")}
    assert every_flag_on
    assert cloud_session_config(tmp_path, monkeypatch).live_tool_peek is False
    assert cloud_session_config(tmp_path, monkeypatch, **every_flag_on).live_tool_peek is False


def test_the_cloud_contract_names_the_tool_phases():
    """The pod's JSONL is read by scratchpad-controller and cowork-server;
    a phase they can meet has to be spelled out there, including the one
    that is kept off the wire and why."""
    from anton.cloud_turn import contract

    doc = contract.__doc__ or ""
    for marker in ('phase: "tool_progress"', 'phase: "tool_done"', 'phase: "tool_peek"',
                   "live_tool_peek", "does NOT appear on this wire"):
        assert marker in doc, marker
