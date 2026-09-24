"""`tool_peek` is a capability the host declares, not a phase every wire
carries (review of PR #335). The relay itself is pinned in
test_tool_registry_streaming.py; this file pins the three ends around it.
"""
from __future__ import annotations

import inspect


def test_the_flag_defaults_off_and_lands_on_the_session(make_session):
    from anton.core.session import ChatSessionConfig

    assert ChatSessionConfig.__dataclass_fields__["live_tool_peek"].default is False
    assert make_session().live_tool_peek is False
    assert make_session(live_tool_peek=True).live_tool_peek is True


def test_the_cli_is_the_host_that_declares_it():
    """The CLI footer renders the tail (`chat_ui.update_progress`, phase
    `tool_peek`); it is the only first-party host that does, so it is the
    only one that opts in. cowork-server builds its config in its own repo
    and the cloud pod's builder must stay silent, which the default covers."""
    from anton import chat
    from anton.cloud_turn import session as cloud_session

    assert "live_tool_peek=True" in inspect.getsource(chat)
    assert "live_tool_peek" not in inspect.getsource(cloud_session)


def test_the_cloud_contract_names_the_tool_phases():
    """The pod's JSONL is read by scratchpad-controller and cowork-server;
    a phase they can meet has to be spelled out there, including the one
    that is kept off the wire and why."""
    from anton.cloud_turn import contract

    doc = contract.__doc__ or ""
    for marker in ('phase: "tool_progress"', 'phase: "tool_done"', 'phase: "tool_peek"',
                   "live_tool_peek", "does NOT appear on this wire"):
        assert marker in doc, marker
