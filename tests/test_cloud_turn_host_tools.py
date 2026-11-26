"""Tools the host runs, called from a cloud turn: an event out, stdin lines back."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import anton.cloud_turn.host_tools as host_tools_mod
import anton.core.llm.client as llm_client_mod
import anton.core.session as session_mod
from anton.cloud_turn.contract import HostToolSpecV1, TurnRequestV1
from anton.cloud_turn.host_tools import (
    NO_ANSWER_TEXT,
    PROGRESS_KIND,
    RESULT_KIND,
    CloudHostTools,
    proxy_tool_defs,
)
from anton.cloud_turn.session import (
    CLOUD_TOOL_ALLOWLIST,
    _CORE_TOOL_NAMES,
    _WORKSPACE_PATH_ENV,
    build_cloud_chat_session,
)
from anton.core.llm.provider import StreamTaskProgress, StreamTextDelta
from anton.core.tools.progress import ToolProgress
from anton.core.tools.registry import ToolOutcome, ToolRegistry
from anton.core.tools.tool_defs import ToolDef
from tests.test_cloud_turn_entrypoint import _FakeSession
from tests.test_cloud_turn_session import _mock_llm

NO_ANSWER = ToolOutcome(content=NO_ANSWER_TEXT, ok=False, reason="host_no_answer")
SPEC = HostToolSpecV1(name="read_note", description="Read a note.", input_schema={"type": "object"},
                      prompt="Read notes with read_note.", timeout_s=60)


async def _collect(gen):
    return [item async for item in gen]


async def _first_event(events: list[dict]) -> dict:
    for _ in range(100):
        if events:
            return events[0]
        await asyncio.sleep(0)
    raise AssertionError("no event was emitted")


async def _settle():
    for _ in range(5):
        await asyncio.sleep(0)


def _part(call_id: str, seq: int, text: str, *, last: bool, ok: bool = True) -> dict:
    return {"kind": RESULT_KIND, "call_id": call_id, "seq": seq, "last": last, "ok": ok, "text": text}


# ── CloudHostTools ───────────────────────────────────────────────────


async def test_a_call_emits_the_event_and_returns_the_result():
    events: list[dict] = []
    bridge = CloudHostTools(events.append)
    task = asyncio.ensure_future(_collect(bridge.call("read_note", {"id": "n1"}, 60)))
    event = await _first_event(events)
    assert event["kind"] == "host_tool_call"
    assert re.fullmatch(r"htool:[0-9a-f]{32}", event["call_id"])
    assert event["tool"] == "read_note" and event["input"] == {"id": "n1"}
    assert set(event) == {"kind", "call_id", "tool", "input"}
    bridge.deliver(_part(event["call_id"], 0, "hello", last=True))
    assert await task == [ToolOutcome(content="hello", ok=True)]
    assert bridge.pending == 0


async def test_progress_lines_come_out_as_tool_progress_before_the_result():
    events: list[dict] = []
    bridge = CloudHostTools(events.append)
    task = asyncio.ensure_future(_collect(bridge.call("read_note", {}, 60)))
    call_id = (await _first_event(events))["call_id"]
    bridge.deliver({"kind": PROGRESS_KIND, "call_id": call_id, "text": "Reading"})
    bridge.deliver({"kind": PROGRESS_KIND, "call_id": call_id, "text": "Rendering"})
    bridge.deliver(_part(call_id, 0, "hello", last=True))
    assert await task == [ToolProgress(text="Reading"), ToolProgress(text="Rendering"),
                          ToolOutcome(content="hello", ok=True)]


async def test_parts_join_in_seq_order_once_all_arrived():
    events: list[dict] = []
    bridge = CloudHostTools(events.append)
    task = asyncio.ensure_future(_collect(bridge.call("read_note", {}, 60)))
    call_id = (await _first_event(events))["call_id"]
    bridge.deliver(_part(call_id, 2, "c", last=True))
    bridge.deliver(_part(call_id, 0, "a", last=False))
    await _settle()
    assert not task.done()  # part 1 is still missing
    bridge.deliver(_part(call_id, 1, "b", last=False))
    assert await task == [ToolOutcome(content="abc", ok=True)]


async def test_repeated_and_out_of_range_parts_are_ignored():
    events: list[dict] = []
    bridge = CloudHostTools(events.append)
    task = asyncio.ensure_future(_collect(bridge.call("read_note", {}, 60)))
    call_id = (await _first_event(events))["call_id"]
    bridge.deliver(_part(call_id, 2, "z", last=False))   # beyond the last part named below
    bridge.deliver(_part(call_id, 1, "b", last=True))
    bridge.deliver(_part(call_id, 1, "y", last=False))   # a repeat of part 1
    bridge.deliver(_part(call_id, 3, "w", last=True))    # a second last part
    await _settle()
    assert not task.done()
    bridge.deliver(_part(call_id, 0, "a", last=False))
    assert await task == [ToolOutcome(content="ab", ok=True)]


async def test_a_failed_result_keeps_its_text_and_reports_not_ok():
    events: list[dict] = []
    bridge = CloudHostTools(events.append)
    task = asyncio.ensure_future(_collect(bridge.call("read_note", {}, 60)))
    call_id = (await _first_event(events))["call_id"]
    bridge.deliver(_part(call_id, 0, "Refused ", last=False, ok=False))
    bridge.deliver(_part(call_id, 1, "(not_owner)", last=True, ok=False))
    assert await task == [ToolOutcome(content="Refused (not_owner)", ok=False, reason="host_tool_failed")]


async def test_no_complete_answer_within_the_timeout_is_a_failure():
    events: list[dict] = []
    bridge = CloudHostTools(events.append)
    started = asyncio.get_running_loop().time()
    task = asyncio.ensure_future(_collect(bridge.call("read_note", {}, 0.05)))
    call_id = (await _first_event(events))["call_id"]
    bridge.deliver(_part(call_id, 1, "b", last=True))  # part 0 never comes
    assert await task == [NO_ANSWER]
    assert asyncio.get_running_loop().time() - started < 2
    assert bridge.pending == 0


async def test_a_result_that_landed_as_the_timer_fired_still_wins(monkeypatch):
    events: list[dict] = []
    bridge = CloudHostTools(events.append)
    real_wait_for = asyncio.wait_for

    async def racing_wait_for(awaitable, timeout):
        awaitable.close()
        call_id = events[0]["call_id"]
        bridge.deliver({"kind": PROGRESS_KIND, "call_id": call_id, "text": "step"})
        bridge.deliver(_part(call_id, 0, "late", last=True))
        raise TimeoutError

    monkeypatch.setattr(host_tools_mod.asyncio, "wait_for", racing_wait_for)
    try:
        items = await _collect(bridge.call("read_note", {}, 60))
    finally:
        monkeypatch.setattr(host_tools_mod.asyncio, "wait_for", real_wait_for)
    assert items == [ToolProgress(text="step"), ToolOutcome(content="late", ok=True)]
    assert bridge.pending == 0


async def test_closing_answers_pending_calls_and_later_calls_send_nothing():
    events: list[dict] = []
    bridge = CloudHostTools(events.append)
    task = asyncio.ensure_future(_collect(bridge.call("read_note", {}, 60)))
    await _first_event(events)
    bridge.close()
    assert await task == [NO_ANSWER]
    assert await _collect(bridge.call("read_note", {}, 60)) == [NO_ANSWER]
    assert len(events) == 1  # no event for a call made after the close


async def test_a_complete_result_queued_before_the_close_still_wins():
    events: list[dict] = []
    bridge = CloudHostTools(events.append)
    task = asyncio.ensure_future(_collect(bridge.call("read_note", {}, 60)))
    call_id = (await _first_event(events))["call_id"]
    bridge.deliver(_part(call_id, 0, "hello", last=True))
    bridge.close()
    assert await task == [ToolOutcome(content="hello", ok=True)]


async def test_lines_for_no_pending_call_are_ignored():
    events: list[dict] = []
    bridge = CloudHostTools(events.append)
    bridge.deliver(_part("htool:" + "0" * 32, 0, "late", last=True))
    task = asyncio.ensure_future(_collect(bridge.call("read_note", {}, 60)))
    call_id = (await _first_event(events))["call_id"]
    bridge.deliver(_part("htool:" + "1" * 32, 0, "someone else", last=True))
    await _settle()
    assert not task.done()
    bridge.deliver(_part(call_id, 0, "mine", last=True))
    assert await task == [ToolOutcome(content="mine", ok=True)]
    bridge.deliver(_part(call_id, 0, "repeat after the end", last=True))  # no error


async def test_a_call_that_cannot_be_sent_fails_without_waiting():
    def broken_emit(event):
        raise BrokenPipeError("stdout closed")

    bridge = CloudHostTools(broken_emit)
    assert await asyncio.wait_for(_collect(bridge.call("read_note", {}, 60)), 2) == [NO_ANSWER]
    assert bridge.pending == 0


async def test_a_cancelled_call_leaves_nothing_pending():
    events: list[dict] = []
    bridge = CloudHostTools(events.append)
    task = asyncio.ensure_future(_collect(bridge.call("read_note", {}, 60)))
    await _first_event(events)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert bridge.pending == 0


# ── proxy tools ──────────────────────────────────────────────────────


def test_a_proxy_carries_the_spec_and_skips_taken_names(caplog):
    taken_spec = HostToolSpecV1(name="scratchpad", description="Host version.", input_schema={"type": "object"})
    with caplog.at_level("WARNING", logger="anton.cloud_turn.host_tools"):
        tools = proxy_tool_defs(CloudHostTools(lambda e: None), [taken_spec, SPEC], {"scratchpad"})
    [tool] = tools
    assert (tool.name, tool.description, tool.input_schema, tool.prompt, tool.unlock_skill) == (
        "read_note", "Read a note.", {"type": "object"}, "Read notes with read_note.", None)
    assert any("skipped host tool" in r.getMessage() for r in caplog.records)


def test_a_proxy_waits_for_the_skill_its_spec_names():
    spec = HostToolSpecV1(name="edit_doc", description="Edit a doc.", input_schema={"type": "object"},
                          unlock_skill="docs")
    [tool] = proxy_tool_defs(CloudHostTools(lambda e: None), [spec], set(), skills={"docs"})
    assert tool.unlock_skill == "docs"


def test_a_proxy_whose_skill_is_not_in_the_turn_is_offered_at_once(caplog):
    spec = HostToolSpecV1(name="edit_doc", description="Edit a doc.", input_schema={"type": "object"},
                          unlock_skill="docs")
    with caplog.at_level("WARNING", logger="anton.cloud_turn.host_tools"):
        [tool] = proxy_tool_defs(CloudHostTools(lambda e: None), [spec], set(), skills={"notes"})
    assert tool.unlock_skill is None
    assert [r.reason for r in caplog.records] == ["skill_missing"]


async def test_progress_reaches_the_session_with_the_tool_call_id():
    events: list[dict] = []
    bridge = CloudHostTools(events.append)
    registry = ToolRegistry()
    for tool in proxy_tool_defs(bridge, [SPEC], set()):
        registry.register_tool(tool)

    class _Emitter:
        def __init__(self):
            self.events = []

        async def emit(self, event):
            self.events.append(event)

    session = SimpleNamespace(emitter=_Emitter(), tool_messages=False, live_tool_peek=False)

    async def host():
        call_id = (await _first_event(events))["call_id"]
        bridge.deliver({"kind": PROGRESS_KIND, "call_id": call_id, "text": "Reading"})
        bridge.deliver(_part(call_id, 0, "hello", last=True))

    outcome, _ = await asyncio.gather(
        registry.dispatch_tool(session, "read_note", {"id": "n1"}, tool_call_id="toolu_1"), host())
    assert outcome == ToolOutcome(content="hello", ok=True)
    [progress] = session.emitter.events
    assert isinstance(progress, StreamTaskProgress)
    assert (progress.phase, progress.message, progress.id) == ("tool_progress", "Reading", "toolu_1")


def test_the_proxy_code_names_no_host_feature():
    """The proxy is generic: it knows tool names only from the request."""
    source = Path(host_tools_mod.__file__).read_text(encoding="utf-8")
    assert not re.search(r"(?i)\bdoc|artifact", source)


# ── the session ──────────────────────────────────────────────────────

NOTE_TOOL = {"name": "read_note", "description": "Read a note.", "input_schema": {"type": "object"},
             "prompt": "HOST TOOL RULE: read notes with read_note before you quote them."}
ROUTER = {"base_url": "https://app.example.test/browser", "profile": "t-1"}


def _request(**over) -> TurnRequestV1:
    body = {"protocol_version": 1, "conversation_id": "c", "input": "hi", **over}
    return TurnRequestV1.from_json(json.dumps(body))


def _config(tmp_path, monkeypatch, *, bridge, elicitor=None, **over):
    """The ChatSessionConfig the builder hands to ChatSession, for a request read off the wire."""
    captured = {}

    def fake_chat_session(config):
        captured["config"] = config
        return config

    monkeypatch.setattr(session_mod, "ChatSession", fake_chat_session)
    monkeypatch.setattr(llm_client_mod.LLMClient, "from_settings", classmethod(lambda cls, settings: object()))
    monkeypatch.setenv(_WORKSPACE_PATH_ENV, str(tmp_path))
    build_cloud_chat_session(_request(**over), elicitor=elicitor, host_tools=bridge)
    return captured["config"]


def _real_session(tmp_path, monkeypatch, *, bridge, elicitor=None, **over):
    """A REAL ChatSession off the cloud builder: the registry is built lazily, so only this proves it."""
    monkeypatch.setenv(_WORKSPACE_PATH_ENV, str(tmp_path))
    monkeypatch.setattr(llm_client_mod.LLMClient, "from_settings", classmethod(lambda cls, settings: _mock_llm()))
    session = build_cloud_chat_session(_request(**over), elicitor=elicitor, host_tools=bridge)
    session._scratchpads = MagicMock(available_packages=[])
    return session


def test_a_host_tool_goes_into_both_tools_and_the_allowlist(tmp_path, monkeypatch):
    cfg = _config(tmp_path, monkeypatch, bridge=CloudHostTools(lambda e: None), host_tools=[NOTE_TOOL])
    assert [t.name for t in cfg.tools] == ["read_note"]
    assert cfg.tool_allowlist == CLOUD_TOOL_ALLOWLIST | {"read_note"}


def test_without_a_bridge_no_proxy_is_registered(tmp_path, monkeypatch):
    cfg = _config(tmp_path, monkeypatch, bridge=None, host_tools=[NOTE_TOOL])
    assert cfg.tools == []
    assert cfg.tool_allowlist == CLOUD_TOOL_ALLOWLIST


def test_a_request_without_host_tools_builds_as_before(tmp_path, monkeypatch):
    cfg = _config(tmp_path, monkeypatch, bridge=CloudHostTools(lambda e: None))
    assert cfg.tools == []
    assert cfg.tool_allowlist == CLOUD_TOOL_ALLOWLIST


def _with_mcp_tool(monkeypatch):
    import anton.core.mcp.wiring as wiring
    import anton.utils.datasources as datasources

    mcp_tool = ToolDef(name="mcp_lookup", description="MCP lookup.", input_schema={"type": "object"},
                       handler=lambda session, tc_input: None)
    monkeypatch.setattr(wiring, "discover_mcp_tools", lambda vault, connections: ([mcp_tool], []))
    monkeypatch.setattr(datasources, "restore_namespaced_env", lambda vault: None)
    return {"oauth": {"turn_key": "tk", "connections": [{"engine": "google_drive", "name": "primary"}]}}


@pytest.mark.parametrize("clash,extra", [
    ("scratchpad", "none"),            # the static allowlist
    ("generate_image", "none"),        # allowlisted, even where it is withheld
    ("ask_user", "elicitor"),
    ("browser", "browser"),
    ("create_connector_draft", "drafts"),
    ("mcp_lookup", "mcp"),
    ("read_image", "none"),            # a core tool the cloud withholds
    ("launch_backend", "none"),        # a core tool the cloud withholds
])
def test_a_host_tool_named_like_any_tool_of_the_turn_is_skipped(clash, extra, tmp_path, monkeypatch, caplog):
    over: dict = {}
    elicitor = None
    if extra == "elicitor":
        elicitor = object()
    elif extra == "browser":
        over["browser"] = ROUTER
    elif extra == "drafts":
        over["connector_drafts"] = True
    elif extra == "mcp":
        over.update(_with_mcp_tool(monkeypatch))
    bridge = CloudHostTools(lambda e: None)
    base = _config(tmp_path, monkeypatch, bridge=bridge, elicitor=elicitor, **over)
    host = dict(NOTE_TOOL, name=clash, description="Host version.")
    with caplog.at_level("WARNING", logger="anton.cloud_turn.host_tools"):
        cfg = _config(tmp_path, monkeypatch, bridge=bridge, elicitor=elicitor,
                      host_tools=[host, NOTE_TOOL], **over)
    assert [t.name for t in cfg.tools] == [t.name for t in base.tools] + ["read_note"]
    assert all(t.description != "Host version." for t in cfg.tools)
    assert cfg.tool_allowlist == base.tool_allowlist | {"read_note"}
    assert any("skipped host tool" in r.getMessage() for r in caplog.records)


async def test_a_real_session_offers_the_proxy_and_puts_its_prompt_in_the_system_prompt(tmp_path, monkeypatch):
    session = _real_session(tmp_path, monkeypatch, bridge=CloudHostTools(lambda e: None), host_tools=[NOTE_TOOL])
    prompt = await session._build_system_prompt()
    names = {t.name for t in session.tool_registry.get_tool_defs()}
    # `generate_image` is withheld where no images API serves it, as with this mock client,
    # and `inspect_artifact` outside a Cowork surface (ENG-4684); this session reports none.
    assert names == (set(CLOUD_TOOL_ALLOWLIST) - {"generate_image", "inspect_artifact"}) | {"read_note"}
    assert NOTE_TOOL["prompt"] in prompt


_NOTES_SKILL = (
    "---\nname: notes\ndescription: Load before editing notes.\n---\nEdit notes with edit_note.\n"
)


async def test_a_host_tool_that_waits_for_a_skill_registers_once_the_skill_is_recalled(tmp_path, monkeypatch):
    from anton.core.tools.recall_skill import handle_recall_skill

    waiting = dict(NOTE_TOOL, name="edit_note", unlock_skill="notes")
    session = _real_session(tmp_path, monkeypatch, bridge=CloudHostTools(lambda e: None),
                            host_tools=[NOTE_TOOL, waiting],
                            skills={"notes": {"files": {"SKILL.md": _NOTES_SKILL}}})
    session._build_tools()
    names = {t.name for t in session.tool_registry.get_tool_defs()}
    assert "read_note" in names and "edit_note" not in names
    out = await handle_recall_skill(session, {"label": "notes"})
    assert "Edit notes with edit_note." in str(getattr(out, "content", out))
    session._build_tools()
    assert "edit_note" in {t.name for t in session.tool_registry.get_tool_defs()}


async def test_a_host_tool_whose_skill_was_not_sent_is_registered_at_once(tmp_path, monkeypatch):
    waiting = dict(NOTE_TOOL, name="edit_note", unlock_skill="notes")
    session = _real_session(tmp_path, monkeypatch, bridge=CloudHostTools(lambda e: None),
                            host_tools=[NOTE_TOOL, waiting], skills={})
    session._build_tools()
    assert {"read_note", "edit_note"} <= {t.name for t in session.tool_registry.get_tool_defs()}


async def test_a_real_session_without_host_tools_has_no_host_prompt(tmp_path, monkeypatch):
    session = _real_session(tmp_path, monkeypatch, bridge=CloudHostTools(lambda e: None))
    prompt = await session._build_system_prompt()
    assert "HOST TOOL RULE" not in prompt


def test_every_tool_core_registers_in_a_cloud_turn_is_reserved(tmp_path, monkeypatch):
    """A core tool missing from both sets could be shadowed into the allowlist by a host tool of its name."""
    from anton.cloud_turn.elicitor import CloudElicitor

    session = _real_session(tmp_path, monkeypatch, bridge=None, elicitor=CloudElicitor(lambda e: None),
                            browser=ROUTER, connector_drafts=True)
    session._build_core_tools()
    names = {t.name for t in session.tool_registry.get_tool_defs()}
    assert names <= CLOUD_TOOL_ALLOWLIST | _CORE_TOOL_NAMES, sorted(names - CLOUD_TOOL_ALLOWLIST - _CORE_TOOL_NAMES)


# ── the turn ─────────────────────────────────────────────────────────

TURN = {"protocol_version": 1, "conversation_id": "c", "input": "hi", "host_tools": [NOTE_TOOL]}


class _HostToolSession(_FakeSession):
    """Calls its one host tool through the real proxy ToolDef, the way the session's dispatch would."""

    def __init__(self, *, in_background: bool = False):
        super().__init__()
        self.bridge = None
        self.tools: list = []
        self.items = None
        self.background = None
        self._in_background = in_background

    def builder(self, req, elicitor=None, preview_checker=None, host_tools=None):
        self.bridge = host_tools
        self.tools = proxy_tool_defs(host_tools, req.host_tools, set()) if host_tools is not None else []
        return self

    async def _call(self):
        return [item async for item in self.tools[0].handler(self, {"id": "n1"})]

    async def turn_stream(self, user_input, **kwargs):
        if self._in_background:
            # The call is still waiting when the turn ends.
            self.background = asyncio.ensure_future(self._call())
            while self.bridge.pending == 0:
                await asyncio.sleep(0)
        else:
            self.items = await self._call()
        yield StreamTextDelta(text="done")


def _run_turn(request: dict, builder, host=None):
    """Drive one turn over a pipe stdin. *host(event)* returns the stdin lines to write, or "close"."""
    from anton.cloud_turn.__main__ import stream_turn

    read_fd, write_fd = os.pipe()
    stdin = os.fdopen(read_fd, "rb")
    events: list[dict] = []
    write_open = [True]

    def emit(event):
        events.append(event)
        reply = host(event) if host is not None else None
        if reply == "close":
            os.close(write_fd)
            write_open[0] = False
        elif reply:
            os.write(write_fd, b"".join((json.dumps(line) + "\n").encode() for line in reply))

    try:
        asyncio.run(stream_turn(json.dumps(request), emit, session_builder=builder, stdin=stdin))
    finally:
        if write_open[0]:
            os.close(write_fd)
        stdin.close()
    return events


def test_a_turn_sends_the_call_and_takes_progress_and_parts_from_stdin():
    session = _HostToolSession()

    def host(event):
        if event["kind"] != "host_tool_call":
            return None
        call_id = event["call_id"]
        return [
            {"kind": "host_tool_progress", "call_id": call_id, "entry_id": call_id + ":0", "text": "Reading"},
            {"kind": "host_tool_result", "call_id": call_id, "entry_id": call_id + ":1",
             "seq": 1, "last": True, "ok": True, "text": "world"},
            {"kind": "host_tool_result", "call_id": call_id, "entry_id": call_id + ":0",
             "seq": 0, "last": False, "ok": True, "text": "hello "},
        ]

    events = _run_turn(TURN, session.builder, host)
    assert session.items == [ToolProgress(text="Reading"), ToolOutcome(content="hello world", ok=True)]
    call = next(e for e in events if e["kind"] == "host_tool_call")
    assert call["tool"] == "read_note" and call["input"] == {"id": "n1"}
    assert re.fullmatch(r"htool:[0-9a-f]{32}", call["call_id"])
    assert events[-1]["kind"] == "turn_completed"


def test_stdin_closing_answers_a_pending_call_as_unanswered():
    session = _HostToolSession()
    events = _run_turn(TURN, session.builder, lambda e: "close" if e["kind"] == "host_tool_call" else None)
    assert session.items == [NO_ANSWER]
    assert events[-1]["kind"] == "turn_completed"


def test_the_turn_ending_answers_a_pending_call_as_unanswered():
    from anton.cloud_turn.__main__ import stream_turn

    session = _HostToolSession(in_background=True)
    read_fd, write_fd = os.pipe()
    stdin = os.fdopen(read_fd, "rb")

    async def scenario():
        await stream_turn(json.dumps(TURN), lambda e: None, session_builder=session.builder, stdin=stdin)
        return await asyncio.wait_for(session.background, 5)

    try:
        assert asyncio.run(scenario()) == [NO_ANSWER]
    finally:
        os.close(write_fd)
        stdin.close()


def test_without_stdin_no_bridge_is_built():
    from anton.cloud_turn.__main__ import stream_turn

    built = {}

    def builder(req, elicitor=None, preview_checker=None, host_tools=None):
        built["host_tools"] = host_tools
        return _FakeSession()

    events: list[dict] = []
    asyncio.run(stream_turn(json.dumps(TURN), events.append, session_builder=builder, stdin=None))
    assert built["host_tools"] is None
    assert events[-1]["kind"] == "turn_completed"


def test_a_request_without_host_tools_builds_no_bridge():
    built = {}

    def builder(req, elicitor=None, preview_checker=None, host_tools=None):
        built["host_tools"] = host_tools
        return _FakeSession(deltas=["ok"])

    request = {k: v for k, v in TURN.items() if k != "host_tools"}
    events = _run_turn(request, builder)
    assert built["host_tools"] is None
    assert not any(e["kind"] == "host_tool_call" for e in events)
    assert events[-1]["kind"] == "turn_completed"


def test_a_builder_that_takes_no_bridge_still_runs():
    def builder(req, elicitor=None):
        return _FakeSession()

    assert _run_turn(TURN, builder)[-1]["kind"] == "turn_completed"
