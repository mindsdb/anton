"""The `delegate` tool end to end, over a real `LLMClient` and `ChatSession`.

The sidekick is a scripted provider: each round's reply is the next item of
its script. Jev is a fake `decide` on the session's policy.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import httpx2 as httpx
import pytest

from anton.core.chimera.config import ChimeraConfig
from anton.core.chimera.policy import ChimeraPolicy
from anton.core.chimera.tools import DELEGATE_TOOL, SIDEKICK_ALLOWLIST, SIDEKICK_NEVER
from anton.core.llm.client import LLMClient, routed_model
from anton.core.llm.jev import JevAnswer, JevDecision
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import LLMProvider, LLMResponse, StreamComplete, ToolCall, Usage
from anton.core.llm.tracing import TraceContext, get_trace_context, reset_trace_context, set_trace_context
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.tools.progress import ToolProgress
from anton.core.tools.registry import ToolOutcome
from anton.core.tools.tool_defs import ToolDef
from anton.core.turn_cost import TurnCost

from tests.conftest import make_mock_llm


class _Provider(LLMProvider):
    """Answers each stream from ``script`` (an LLMResponse, or a callable of
    the call's kwargs) and records the model and messages of every call."""

    name = "fake"

    def __init__(self, script=None, *, summary: str = "notes so far") -> None:
        self.script = list(script or [])
        self.models: list[str] = []
        self.calls: list[dict] = []
        self.summary = summary

    async def complete(self, **kw):
        self.models.append(kw["model"])
        return LLMResponse(content=self.summary, usage=Usage(input_tokens=5, output_tokens=5))

    async def stream(self, **kw):
        self.models.append(kw["model"])
        self.calls.append({**kw, "messages": json.loads(json.dumps(kw["messages"]))})
        step = self.script.pop(0) if self.script else _finish()
        response = step(kw) if callable(step) else step
        yield StreamComplete(response=response)


def _reply(*calls: ToolCall, text: str = "", pressure: float = 0.0) -> LLMResponse:
    return LLMResponse(content=text, tool_calls=list(calls),
                       usage=Usage(input_tokens=100, output_tokens=10, context_pressure=pressure))


_ids = iter(range(10_000))


def _call(name: str, **inp) -> ToolCall:
    return ToolCall(id=f"tc{next(_ids)}", name=name, input=inp)


def _finish(status="done", summary="Counted 42 parts.", **extra) -> LLMResponse:
    return _reply(_call("finish", status=status, summary=summary, **extra))


def _lookup_tool(seen: list) -> ToolDef:
    async def handler(session, tc_input):
        seen.append(("lookup", tc_input))
        return ToolOutcome(content=f"answer for {tc_input.get('q')}: 42", ok=True)

    return ToolDef(name="lookup", description="Look a value up.",
                   input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
                   handler=handler, sidekick_safe=True)


def _host_tool(name: str, seen: list, *, safe: bool) -> ToolDef:
    async def handler(session, tc_input):
        seen.append((name, tc_input))
        return "ran"

    return ToolDef(name=name, description=name, input_schema={"type": "object", "properties": {}},
                   handler=handler, sidekick_safe=safe)


def _session(script=None, *, chimera=None, sidekick_model="gpt-sol", tools=(), sidekick_state=None,
             planning=None, coding=None):
    sidekick = _Provider(script)
    planning = planning or _Provider()
    coding = coding or _Provider()
    llm = LLMClient(planning_provider=planning, planning_model="opus", coding_provider=coding,
                    coding_model="haiku", sidekick_provider=sidekick, sidekick_model=sidekick_model)
    session = ChatSession(ChatSessionConfig(
        llm_client=llm,
        chimera=chimera if chimera is not None else ChimeraConfig(enabled=True, jev_every=0),
        tools=list(tools),
        sidekick_state=sidekick_state,
    ))
    session._build_tools()
    return session, sidekick


def _with_jev(session, *decisions):
    """Answer the session's sidekick checks with ``decisions`` in turn
    (dicts of probabilities), then with nothing."""
    queue = list(decisions)
    asked: list = []

    async def decide(state, questions, metadata):
        asked.append(metadata)
        if not queue:
            return None
        ps = queue.pop(0)
        return JevDecision(answers={n: JevAnswer(type="noul", p=p) for n, p in ps.items()})

    session._chimera_policy = ChimeraPolicy(session._chimera, decide=decide)
    return asked


async def _delegate(session, **tc_input):
    tc_input.setdefault("brief", "Count the parts in the parts table.")
    tc_input.setdefault("done_when", "You have the count and checked it twice.")
    progress, result = [], None
    async for item in session.tool_registry.dispatch_tool_stream(session, "delegate", tc_input):
        if isinstance(item, ToolProgress):
            progress.append(item.text)
        else:
            result = item
    assert isinstance(result, ToolOutcome)
    return result, json.loads(result.content) if result.content.startswith("{") else result.content, progress


def _tool_results(messages: list[dict]) -> list:
    return [
        block["content"] for m in messages if isinstance(m.get("content"), list)
        for block in m["content"] if block.get("type") == "tool_result"
    ]


# -- registration -------------------------------------------------------------------


def _names(session) -> set[str]:
    return {t.name for t in session.tool_registry.get_tool_defs()}


def test_delegate_is_offered_when_chimera_is_on_and_a_sidekick_is_configured():
    session, _ = _session()
    assert "delegate" in _names(session)
    # Its prompt is the delegation policy, injected into the system prompt.
    delegate = next(t for t in session.tool_registry.get_tool_defs() if t.name == "delegate")
    assert delegate is DELEGATE_TOOL and "Always review the report" in delegate.prompt


@pytest.mark.parametrize("chimera,sidekick_model", [
    (ChimeraConfig(enabled=False), "gpt-sol"),
    (ChimeraConfig(enabled=True), None),
])
def test_delegate_is_not_offered_otherwise(chimera, sidekick_model):
    session, _ = _session(chimera=chimera, sidekick_model=sidekick_model)
    assert "delegate" not in _names(session)


def test_a_session_without_chimera_input_reads_it_from_settings_and_stays_off():
    session = ChatSession(ChatSessionConfig(llm_client=make_mock_llm()))
    session._build_tools()
    assert session._chimera == ChimeraConfig(enabled=False)
    assert session._chimera_policy is None
    assert session.sidekick_state is None
    assert "delegate" not in _names(session)


def test_a_test_double_client_never_counts_as_having_a_sidekick():
    # make_mock_llm is an AsyncMock: every attribute is truthy, none is True.
    session = ChatSession(ChatSessionConfig(llm_client=make_mock_llm(), chimera=ChimeraConfig(enabled=True)))
    session._build_tools()
    assert "delegate" not in _names(session)


def test_the_sidekicks_tools_are_the_allowlist_and_host_tools_marked_safe():
    seen: list = []
    session, _ = _session(tools=[
        _lookup_tool(seen), _host_tool("publish", seen, safe=True), _host_tool("connect", seen, safe=False),
    ])
    from anton.core.chimera.tools import sidekick_tool_defs

    names = {t.name for t in sidekick_tool_defs(session)}
    assert "lookup" in names
    assert {"publish", "connect", "delegate", "select_path", "recall_skill"}.isdisjoint(names)
    assert names - {"lookup"} <= SIDEKICK_ALLOWLIST
    assert "publish" in SIDEKICK_NEVER and "ask_user" in SIDEKICK_NEVER


# -- outcomes -------------------------------------------------------------------------


async def test_finish_reports_done_and_keeps_the_conversation():
    seen: list = []
    session, sidekick = _session([
        _reply(_call("lookup", q="parts")),
        _finish(outputs=["count=42"], evidence=["lookup returned 42 twice"]),
    ], tools=[_lookup_tool(seen)])

    outcome, report, progress = await _delegate(session, context="The table is parts_v2.")

    assert outcome.ok is True
    assert report["status"] == "done"
    assert report["summary"] == "Counted 42 parts."
    assert (report["outputs"], report["evidence"], report["rounds"]) == (
        ["count=42"], ["lookup returned 42 twice"], 2)
    assert report["tokens"] == 220  # two sidekick calls, outside a turn's books
    assert seen == [("lookup", {"q": "parts"})]
    assert sidekick.models == ["gpt-sol", "gpt-sol"]
    # The kickoff carries the brief, the done-when and the context; the tools
    # are the sidekick's own plus finish.
    kickoff = sidekick.calls[0]["messages"][0]["content"][0]["text"]
    assert "Count the parts" in kickoff and "checked it twice" in kickoff and "parts_v2" in kickoff
    tools = {t["name"] for t in sidekick.calls[0]["tools"]}
    assert {"lookup", "finish", "read_text_file"} <= tools
    assert "delegate" not in tools
    assert any("Sidekick" in line for line in progress)
    state = session.sidekick_state
    assert state["status"] == "idle"
    assert state["model"] == "gpt-sol"
    assert state["messages"][-1]["role"] == "user"
    assert "answer for parts: 42" in _tool_results(state["messages"])
    assert "Counted 42 parts." in state["summary"]


async def test_the_sidekick_may_hand_back_itself():
    session, _ = _session([_finish(status="needs_main", summary="Which year?", open_questions=["2025 or 2026?"])])
    _, report, _ = await _delegate(session)
    assert report["status"] == "needs_main"
    assert report["open_questions"] == ["2025 or 2026?"]
    assert session.sidekick_state["status"] == "open"


async def test_jev_take_back_ends_the_delegation_as_needs_main():
    seen: list = []
    session, sidekick = _session(
        [_reply(_call("lookup", q="a")), _reply(_call("lookup", q="b"))],
        chimera=ChimeraConfig(enabled=True, jev_every=1), tools=[_lookup_tool(seen)],
    )
    asked = _with_jev(session, {"take_back": 0.95, "stuck": 0.1, "off_scope": 0.0, "done": 0.0})

    outcome, report, _ = await _delegate(session)

    assert outcome.ok is True
    assert report["status"] == "needs_main"
    assert report["rounds"] == 1
    assert report["open_questions"]
    assert len(sidekick.calls) == 1
    assert asked[0]["jev_trigger"] == "every"
    assert asked[0]["sidekick_round"] == "1"
    state = session.sidekick_state
    assert (state["status"], state["signals"]) == ("open", 1)
    # Every tool call is answered, so the next delegation can continue.
    assert state["messages"][-1]["role"] == "user"


async def test_the_round_cap_ends_it_as_budget_exhausted():
    seen: list = []
    session, sidekick = _session(
        [_reply(_call("lookup", q=str(i)), text=f"step {i}") for i in range(10)], tools=[_lookup_tool(seen)],
    )

    outcome, report, _ = await _delegate(session, max_rounds=3)

    assert report["status"] == "budget_exhausted"
    assert report["rounds"] == 3
    assert "step 2" in report["summary"]
    assert len(sidekick.calls) == 3
    # The last rounds were told to wrap up.
    last_note = sidekick.calls[-1]["messages"][-1]["content"][-1]["text"]
    assert "wrap up NOW" in last_note
    assert session.sidekick_state["status"] == "open"


async def test_the_round_cap_never_exceeds_the_hard_cap():
    session, sidekick = _session([_reply(_call("lookup")) for _ in range(50)],
                                 chimera=ChimeraConfig(enabled=True, jev_every=0, max_rounds_cap=4),
                                 tools=[_lookup_tool([])])
    _, report, _ = await _delegate(session, max_rounds=40)
    assert report["rounds"] == 4


async def test_a_tool_outside_the_allowlist_is_refused_without_running():
    seen: list = []
    session, sidekick = _session(
        [_reply(_call("publish", target="web"), _call("memorize", text="x")), _finish()],
        tools=[_host_tool("publish", seen, safe=True)],
    )

    _, report, _ = await _delegate(session)

    assert report["status"] == "done"
    assert seen == []
    refusals = _tool_results(sidekick.calls[1]["messages"])
    assert len(refusals) == 2
    assert all("is not available to you" in r for r in refusals)


async def test_a_tools_subset_narrows_the_sidekick_and_unknown_names_are_an_error():
    seen: list = []
    session, sidekick = _session([_finish()], tools=[_lookup_tool(seen)])

    _, report, _ = await _delegate(session, tools=["lookup"])
    assert {t["name"] for t in sidekick.calls[0]["tools"]} == {"lookup", "finish"}

    outcome, content, _ = await _delegate(session, tools=["lookup", "ask_user"])
    assert outcome.ok is False
    assert "cannot use ask_user" in content


@pytest.mark.parametrize("missing", ["brief", "done_when"])
async def test_brief_and_done_when_are_required(missing):
    session, sidekick = _session()
    outcome, content, _ = await _delegate(session, **{missing: "  "})
    assert outcome.ok is False and f"`{missing}` is required" in content
    assert sidekick.calls == []


async def test_jev_refuses_a_finish_it_is_sure_is_not_done_once():
    session, sidekick = _session([_finish(), _finish(summary="Really counted.")])
    _with_jev(session, *[{"take_back": 0.0, "stuck": 0.0, "off_scope": 0.0, "done": 0.05}] * 2)

    _, report, _ = await _delegate(session)

    assert report["status"] == "done"
    assert report["summary"] == "Really counted."
    assert "Not accepted" in _tool_results(sidekick.calls[1]["messages"])[-1]


async def test_stuck_twice_hands_back():
    seen: list = []
    session, _ = _session([_reply(_call("lookup")) for _ in range(5)],
                          chimera=ChimeraConfig(enabled=True, jev_every=1), tools=[_lookup_tool(seen)])
    _with_jev(session, *[{"take_back": 0.1, "stuck": 0.9, "off_scope": 0.0, "done": 0.0}] * 2)

    _, report, _ = await _delegate(session)

    assert report["status"] == "needs_main"
    assert report["rounds"] == 2
    assert session.sidekick_state["signals"] == 2


async def test_a_provider_failure_is_a_failed_report():
    def boom(kw):
        raise RuntimeError("gateway down")

    session, _ = _session([boom])
    outcome, report, _ = await _delegate(session)
    assert outcome.ok is False
    assert report["status"] == "failed"
    assert "gateway down" in report["summary"]


# -- the persistent conversation -----------------------------------------------------


async def test_a_follow_up_within_five_minutes_continues_the_conversation():
    session, sidekick = _session([_finish(summary="first"), _finish(summary="second")])
    await _delegate(session, brief="First task.")
    first_id = session.sidekick_state["sidekick_id"]

    await _delegate(session, brief="Second task.")

    second_call = sidekick.calls[1]["messages"]
    assert "First task." in json.dumps(second_call)
    assert "Second task." in second_call[-1]["content"][-1]["text"]
    assert session.sidekick_state["sidekick_id"] == first_id


async def test_a_stale_conversation_restarts_from_its_summary():
    old = {
        "sidekick_id": "sk-old", "model": "gpt-sol", "summary": "Earlier: pulled parts_v2.",
        "messages": [{"role": "user", "content": "old task"}, {"role": "assistant", "content": "old"}],
        "status": "idle", "signals": 0,
        "last_call_at": (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat(),
    }
    session, sidekick = _session([_finish()], sidekick_state=old)
    assert session.sidekick_state == old  # loaded, not yet run

    await _delegate(session)

    messages = sidekick.calls[0]["messages"]
    assert len(messages) == 1
    assert "Earlier: pulled parts_v2." in messages[0]["content"][0]["text"]
    assert session.sidekick_state["sidekick_id"] == "sk-old"


async def test_fresh_starts_a_new_sidekick():
    session, sidekick = _session([_finish(), _finish()])
    await _delegate(session)
    first_id = session.sidekick_state["sidekick_id"]
    await _delegate(session, fresh=True)
    assert session.sidekick_state["sidekick_id"] != first_id
    assert len(sidekick.calls[1]["messages"]) == 1


async def test_compaction_may_move_up_the_ladder_once_earned():
    state = {"sidekick_id": "sk1", "model": "gpt-sol", "messages": [], "summary": "", "status": "open",
             "last_call_at": None, "signals": 2}
    coding = _Provider(summary="Looked up a and b.")
    session, sidekick = _session(
        [_reply(_call("lookup", q="a"), pressure=0.95), _finish()],
        chimera=ChimeraConfig(enabled=True, jev_every=0, upgrade_ladder=("gpt-sol", "kimi-k3")),
        tools=[_lookup_tool([])], sidekick_state=state, coding=coding,
    )

    _, report, progress = await _delegate(session)

    assert report["status"] == "done"
    assert sidekick.models == ["gpt-sol", "kimi-k3"]
    # The summary ran on the router role (the coding provider here).
    assert coding.models == ["haiku"]
    second = sidekick.calls[1]["messages"]
    assert len(second) == 1 and "Looked up a and b." in second[0]["content"][0]["text"]
    saved = session.sidekick_state
    assert (saved["model"], saved["signals"]) == ("kimi-k3", 0)
    assert "Sidekick moved up to kimi-k3" in progress


# -- generate_artifact on the sidekick's model ----------------------------------------


async def test_generate_artifact_from_the_sidekick_runs_on_the_sidekick_model(monkeypatch):
    monkeypatch.delenv("ANTON_ARTIFACT_MODEL", raising=False)
    planning = _Provider()
    seen: dict = {}

    async def fake_generate_artifact(session, tc_input):
        seen["handler"] = routed_model()

        async def pipeline():
            # The pipeline's own task, started from inside the handler, as
            # the real one is.
            seen["task"] = routed_model()
            async for _ in session._llm.plan_stream(system="s", messages=[]):
                pass

        task = asyncio.create_task(pipeline())
        yield ToolProgress("building")
        await task
        yield "built"

    tool = ToolDef(name="generate_artifact", description="g", input_schema={"type": "object"},
                   handler=fake_generate_artifact)
    session, _ = _session([_reply(_call("generate_artifact", slug="x")), _finish()],
                          tools=[tool], planning=planning)

    _, report, progress = await _delegate(session)

    assert report["status"] == "done"
    assert seen == {"handler": "gpt-sol", "task": "gpt-sol"}
    assert planning.models == ["gpt-sol"]
    assert "Sidekick · building" in progress
    # The agent's own calls afterwards keep the agent's model.
    await session._llm.plan(system="s", messages=[])
    assert planning.models == ["gpt-sol", "opus"]


async def test_the_real_generate_artifact_handler_builds_on_the_sidekick_model(tmp_path, monkeypatch):
    import anton.core.tools.generate_artifact as gen_pkg
    from anton.core.artifacts import ArtifactStore
    from anton.core.tools.tool_defs import GENERATE_ARTIFACT_TOOL

    monkeypatch.delenv("ANTON_ARTIFACT_MODEL", raising=False)
    slug = ArtifactStore(tmp_path / "artifacts").create(
        name="Clock", description="d", type="fullstack-stateless-app").slug
    planning, coding = _Provider(), _Provider()

    async def fake_generate(**kw):
        async for _ in kw["session"]._llm.plan_stream(system="s", messages=[]):
            pass
        async for _ in kw["session"]._llm.code_stream(system="s", messages=[]):
            pass
        return "Backend verification failed after retry: stop here"

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    session, _ = _session([
        _reply(_call("generate_artifact", slug=slug, user_request="a clock", agent_understanding="a clock",
                     user_language="English")),
        _finish(status="needs_main", summary="The build failed."),
    ], tools=[GENERATE_ARTIFACT_TOOL], planning=planning, coding=coding)
    session._workspace = type("W", (), {"artifacts_dir": tmp_path / "artifacts", "base": tmp_path})()

    _, report, _ = await _delegate(session)

    assert report["status"] == "needs_main"
    assert planning.models == ["gpt-sol"]
    assert coding.models == ["gpt-sol"]


# -- cancellation ----------------------------------------------------------------


def _slow_session():
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def slow(session, tc_input):
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "never"

    tool = ToolDef(name="slow", description="s", input_schema={"type": "object"}, handler=slow,
                   sidekick_safe=True)
    session, _ = _session([_reply(_call("slow"))], tools=[tool])
    return session, started, cancelled


async def test_a_stopped_turn_cancels_the_sidekick_inside_the_tool_call():
    # The session dispatches in a task it cancels on a Stop (`_run_draining`).
    session, started, cancelled = _slow_session()
    dispatch = asyncio.create_task(
        session.tool_registry.dispatch_tool(session, "delegate", {"brief": "b", "done_when": "d"}))
    await asyncio.wait_for(started.wait(), 5)

    dispatch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dispatch

    assert cancelled.is_set()
    assert session.sidekick_state is None  # nothing half-done is saved


async def test_closing_the_handler_cancels_the_sidekick_before_it_returns():
    from anton.core.chimera.tools import handle_delegate

    session, started, cancelled = _slow_session()
    stream = handle_delegate(session, {"brief": "b", "done_when": "d"})
    while not started.is_set():
        assert isinstance(await anext(stream), ToolProgress)

    await stream.aclose()

    assert cancelled.is_set()


# -- telemetry ----------------------------------------------------------------------


async def test_a_delegation_is_booked_on_the_turns_books_and_event():
    session, _ = _session([_reply(_call("lookup")), _finish()], tools=[_lookup_tool([])],
                          chimera=ChimeraConfig(enabled=True, jev_every=1))
    _with_jev(session, *[{"take_back": 0.1, "stuck": 0.1, "off_scope": 0.1, "done": 0.9}] * 2)
    books = TurnCost()
    session._turn_cost = books
    session._llm.usage_listener = books.add
    token = set_trace_context(TraceContext(session_id="conv-1", turn_id=1))
    traced: list = []
    original = session._llm.sidekick_stream

    def spy(**kw):
        traced.append(dict(get_trace_context().metadata or {}))
        return original(**kw)

    session._llm.sidekick_stream = spy
    try:
        _, report, _ = await _delegate(session)
    finally:
        reset_trace_context(token)

    assert report["tokens"] == books.by_role["sidekick"].tokens == 220
    assert books.sidekick_delegations == 1
    assert books.sidekick_rounds == 2
    assert books.sidekick_last_status == "done"
    assert books.sidekick_jev_checks == 2
    assert traced[0]["sidekick_round"] == "1" and traced[1]["sidekick_round"] == "2"
    assert traced[0]["delegation_id"] == traced[1]["delegation_id"]

    with patch("anton.analytics.send_event") as send:
        session._emit_turn_cost(books, exc=None)
    fields = send.call_args.kwargs
    assert fields["sidekick_delegations"] == "1"
    assert fields["sidekick_last_status"] == "done"
    assert (fields["sidekick_model"], fields["sidekick_tokens"]) == ("gpt-sol", "220")


def test_a_turn_without_chimera_reports_no_chimera_fields():
    session, _ = _session(chimera=ChimeraConfig(enabled=False))
    books = TurnCost()
    books.add("planning", "opus", Usage(input_tokens=1))
    session._turn_cost = books
    with patch("anton.analytics.send_event") as send:
        session._emit_turn_cost(books, exc=None)
    assert not any(k.startswith(("sidekick_", "decision_")) for k in send.call_args.kwargs)


async def test_a_sidekick_check_on_mindshub_is_traced_and_booked_as_a_decision():
    coding = OpenAIProvider(api_key="k", base_url="https://api.mindshub.ai/v1",
                            flavor=OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH)
    session, _ = _session(coding=coding)
    sent = {}

    def handler(request):
        sent["metadata"] = json.loads(request.headers["langfuse-metadata"])
        sent["session"] = request.headers.get("langfuse-session-id")
        return httpx.Response(200, json={"model": "jev-1.13.0", "usage": {"input_tokens": 800, "output_tokens": 0},
                                         "answers": {n: {"type": "noul", "noul": 0.1}
                                                     for n in ("take_back", "stuck", "off_scope", "done")}})

    session._jev_http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    books = TurnCost()
    session._turn_cost = books
    token = set_trace_context(TraceContext(session_id="conv-1", turn_id=1, harness="anton"))
    try:
        verdict = await session._chimera_policy.check(
            trigger="every", brief="b", done_when="d", transcript="t", metadata={"delegation_id": "d1"})
    finally:
        reset_trace_context(token)
        await session._jev_http.aclose()

    assert not verdict.take_back
    assert sent["session"] is None  # kept out of the conversation, like the verifier's checks
    assert sent["metadata"]["conversation_id"] == "conv-1"
    assert sent["metadata"]["role"] == "decision"
    assert (sent["metadata"]["delegation_id"], sent["metadata"]["jev_trigger"]) == ("d1", "every")
    assert books.by_role["decision"].tokens == 800


def test_main_compaction_reaches_the_policy(caplog):
    session, _ = _session()
    calls = []
    session._chimera_policy.on_main_compaction = lambda s: calls.append(s)
    session._on_main_compaction()
    assert calls == [session]
