"""Scenario — Browser Control (read-only) tool driven through anton-core.

These scenarios exercise WS3 of the Browser Control M1 plan against the local
stub LLM server. Unlike the subprocess-based scenarios in this package, they
drive ``ChatSession.turn_stream`` in-process for two reasons the subprocess
harness cannot satisfy:

1. The real ``browser_control`` tool is injected by the cowork-server host, not
   by anton-core. Anton stays host-agnostic, so to exercise the engine path we
   register our own FAKE ``browser_control`` ``ToolDef`` here (with a canned,
   observed-data handler) via ``ChatSessionConfig.tools`` — exactly how a host
   injects extra tools.
2. The assertion is on the *streamed progress events* — that the per-action
   ``StreamTaskProgress`` carries the human-readable ``progress_message`` the
   agent supplied (e.g. "Reading account list"), not the raw ``browser_control``
   tool name. Those events are only observable in-process.

The stub LLM HTTP server (``tests/e2e/stub_server.py``) is still the driver: we
queue scripted tool calls + final text the same way the subprocess scenarios
do. anton-core connects to it over HTTP via a real ``OpenAIProvider`` pointed
at ``stub.base_url``, so the whole turn loop (tool dispatch, progress
streaming, completion verification) runs for real.
"""

from __future__ import annotations

import json

import pytest

from anton.core.llm.client import LLMClient
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import StreamTaskProgress
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.tools.tool_defs import ToolDef
from tests.e2e.stub_server import StubServer

# The whole point is to observe the in-process progress stream + a fake tool
# handler, which the live provider path cannot script.
pytestmark = pytest.mark.stub_only


# Canned "observed" payloads a real bridge would return for each read-only
# action. Content-free-ness is a WS4 concern; here we only need enough to let
# the model cite an answer, mirroring the transient ``observed`` shape.
_ACCOUNT_LIST_OBSERVED = {
    "url": "https://reports.example.com/accounts",
    "title": "Accounts",
    "text": "Acme Inc, Globex, Initech. Monthly reports available.",
    "links": [
        {"text": "July report", "href": "https://reports.example.com/july"},
        {"text": "June report", "href": "https://reports.example.com/june"},
    ],
}
_JULY_REPORT_OBSERVED = {
    "url": "https://reports.example.com/july",
    "title": "July Report",
    "text": "July total revenue: $1,234,567.",
    "links": [],
}


class _FakeBrowserBridge:
    """A fake bridge/tool handler returning canned observed data per action.

    Records every call so a test can assert the sequence of actions the model
    drove. Never raises — mirrors the anton tool-handler contract (handlers
    return strings, error strings included).
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def handle(self, session, tc_input: dict) -> str:  # noqa: ANN001
        del session  # host-agnostic fake; no session state needed
        self.calls.append(dict(tc_input))
        action = tc_input.get("action")
        if action == "inspect" and not self.calls[:-1]:
            observed = _ACCOUNT_LIST_OBSERVED
        elif action == "follow_link":
            observed = _JULY_REPORT_OBSERVED
        elif action == "inspect":
            # Second inspect (after the follow_link) reads the July report.
            observed = _JULY_REPORT_OBSERVED
        else:
            return json.dumps(
                {"status": "unsupported_action", "observed": None, "citations": []}
            )
        return json.dumps(
            {
                "status": "ok",
                "observed": observed,
                "citations": [{"url": observed["url"], "title": observed["title"]}],
            }
        )


def _make_browser_tool(bridge: _FakeBrowserBridge) -> ToolDef:
    """A fake ``browser_control`` ToolDef matching the WS3 tool contract shape.

    Required fields ``action`` / ``reason`` / ``progress_message`` mirror the
    real tool so anton-core's ``browser_control`` special-case (which keys on
    the tool name + the presence of ``progress_message``) fires.
    """
    return ToolDef(
        name="browser_control",
        description=(
            "Read-only browser control. Prefer a connector via lookup_connector "
            "first; use this only when no connector satisfies the task."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["inspect", "follow_link", "scroll", "wait"],
                },
                "reason": {"type": "string"},
                "progress_message": {"type": "string"},
                "href": {"type": "string"},
                "direction": {"type": "string"},
            },
            "required": ["action", "reason", "progress_message"],
        },
        handler=bridge.handle,
    )


def _lookup_connector_tool(calls: list[dict]) -> ToolDef:
    """A minimal fake ``lookup_connector`` tool used by the connector-preferred
    scenario, so the model has a connector-first path to take."""

    async def handle(session, tc_input: dict) -> str:  # noqa: ANN001
        del session
        calls.append(dict(tc_input))
        return json.dumps(
            {
                "match": "reports_api",
                "confidence": 0.95,
                "label": "Reports API",
            }
        )

    return ToolDef(
        name="lookup_connector",
        description="Find a connector that satisfies the task.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=handle,
    )


def _make_session(stub: StubServer, tools: list[ToolDef]) -> ChatSession:
    """Build a real ChatSession wired to the stub LLM over HTTP.

    Web tools are disabled so the only injected tools are the browser/connector
    fakes — keeping the registry minimal and the assertions unambiguous.
    """
    provider = OpenAIProvider(
        api_key="test-key-e2e",
        base_url=stub.base_url,
        flavor=OpenAIProvider.FLAVOR_OPENAI_COMPATIBLE_GENERIC,
    )
    llm = LLMClient(
        planning_provider=provider,
        planning_model="gpt-test",
        coding_provider=provider,
        coding_model="gpt-test",
    )
    return ChatSession(
        ChatSessionConfig(
            llm_client=llm,
            tools=tools,
            web_search_enabled=False,
            web_fetch_enabled=False,
        )
    )


async def _drain(session: ChatSession, user_input: str):
    """Run a full turn, returning (all_events, task_progress_events)."""
    events = []
    progress: list[StreamTaskProgress] = []
    async for event in session.turn_stream(user_input):
        events.append(event)
        if isinstance(event, StreamTaskProgress):
            progress.append(event)
    return events, progress


async def test_three_step_read_only_task_streams_human_progress(cfg, stub):
    """Scenario 1 — a three-step read-only browser task.

    The model drives: inspect ("Reading account list") -> follow_link
    ("Opening July report") -> inspect ("Reading July report") -> a cited
    answer. Assert the streamed progress carries each human-readable
    ``progress_message`` (NOT the raw "browser_control" tool name).
    """
    bridge = _FakeBrowserBridge()

    # Script the stub: three browser_control tool calls, then a final cited
    # answer, then the completion-verification "STATUS: COMPLETE".
    stub.queue_tool_call(
        "browser_control",
        {
            "action": "inspect",
            "reason": "No connector exposes the accounts page.",
            "progress_message": "Reading account list",
        },
    )
    stub.queue_tool_call(
        "browser_control",
        {
            "action": "follow_link",
            "reason": "No connector exposes the July report.",
            "progress_message": "Opening July report",
            "href": "https://reports.example.com/july",
        },
    )
    stub.queue_tool_call(
        "browser_control",
        {
            "action": "inspect",
            "reason": "No connector exposes the July report.",
            "progress_message": "Reading July report",
        },
    )
    stub.queue_text(
        "July total revenue was $1,234,567 "
        "(source: https://reports.example.com/july). ANSWER_DONE"
    )
    stub.queue_verification_ok()

    session = _make_session(stub, [_make_browser_tool(bridge)])
    _events, progress = await _drain(
        session, "What was the July revenue on my reports site?"
    )

    # The fake bridge saw all three read-only actions in order.
    assert [c["action"] for c in bridge.calls] == ["inspect", "follow_link", "inspect"]

    browser_progress = [p for p in progress if p.phase == "browser_action"]
    messages = [p.message for p in browser_progress]

    # Every browser action surfaced the agent-supplied human message, and the
    # raw tool name never leaked into the progress stream.
    assert "Reading account list" in messages
    assert "Opening July report" in messages
    assert "Reading July report" in messages
    assert "browser_control" not in messages, (
        f"Progress must carry the human progress_message, not the raw tool "
        f"name. Got: {messages}"
    )

    # Progress events correlate back to their originating tool_use id.
    assert all(p.id is not None for p in browser_progress)

    # Stream shape: each action yields exactly two browser_action events —
    # a pre-dispatch one with no eta (running) and a post-dispatch completion
    # one carrying the elapsed. Hosts (CLI display, cowork-server formatter)
    # key completion off the eta-bearing event, so this shape is contractual.
    assert len(browser_progress) == 6  # 3 actions x (start + done)
    for start, done in zip(browser_progress[::2], browser_progress[1::2]):
        assert start.message == done.message
        assert start.id == done.id
        assert start.eta_seconds is None
        assert done.eta_seconds is not None
    # The generic tool_start/tool_done phases never fire for browser_control.
    assert not [
        p
        for p in progress
        if p.phase in ("tool_start", "tool_done") and p.message == "browser_control"
    ]

    # The cited answer made it into the assistant reply.
    reply = next(
        (m["content"] for m in reversed(session._history) if m["role"] == "assistant"),
        "",
    )
    assert "1,234,567" in str(reply)


async def test_connector_preferred_never_emits_browser_control(cfg, stub):
    """Scenario 2 — connector-first routing.

    When a connector satisfies the task, the model calls ``lookup_connector``
    and NEVER emits ``browser_control``. Assert no browser_action progress is
    streamed and the fake browser bridge is never touched.
    """
    connector_calls: list[dict] = []
    bridge = _FakeBrowserBridge()

    stub.queue_tool_call("lookup_connector", {"query": "reports api"})
    stub.queue_text(
        "I can use the Reports API connector for this — no browser needed. "
        "CONNECTOR_DONE"
    )
    stub.queue_verification_ok()

    session = _make_session(
        stub, [_lookup_connector_tool(connector_calls), _make_browser_tool(bridge)]
    )
    _events, progress = await _drain(session, "Pull the latest revenue numbers.")

    # Connector path was taken; the browser bridge was never invoked.
    assert connector_calls, "Expected lookup_connector to be called"
    assert bridge.calls == [], "browser_control must not be called on the connector path"

    # No browser progress was streamed.
    assert not [p for p in progress if p.phase == "browser_action"]
