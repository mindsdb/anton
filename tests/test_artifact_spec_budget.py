"""ENG-1116 / I-07: the two whole-document calls run with an explicit output
budget, retry once with more room when cut off, and fail loudly rather than
handing a truncated specification to the generators.

Before this, `make_tech_spec` checked only for an empty reply: a spec cut off at
the cap was written to `spec.md` and fed to backend and frontend generation with
nothing anywhere reporting the loss.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from anton.core.llm.provider import StreamComplete
from anton.core.tools.generate_artifact import engine, orchestrator, prompts
from anton.core.tools.generate_artifact.state import (
    SPEC_MAX_TOKENS,
    SPEC_MAX_TOKENS_RETRY,
    GenState,
)


def _response(content: str, *, output_tokens: int, stop_reason: str | None = None):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(output_tokens=output_tokens),
    )


async def _one_event_stream(response):
    yield StreamComplete(response=response)


def _stream_mock(*responses):
    """`plan_stream` fake: each call returns a fresh one-event stream, one
    queued response at a time (mirrors `AsyncMock(side_effect=[...])`)."""
    return Mock(side_effect=[_one_event_stream(r) for r in responses])


def _state(tmp_path, **kw):
    base = dict(
        session=AsyncMock(), artifact_type="html-app", artifact_path=tmp_path,
        slug="a", brief="Show orders", is_fullstack=False,
    )
    base.update(kw)
    return GenState(**base)


# ── budgets ─────────────────────────────────────────────────────────────────

def test_the_retry_budget_is_larger_than_the_first():
    assert SPEC_MAX_TOKENS_RETRY > SPEC_MAX_TOKENS


def test_budgets_stay_within_what_the_gateway_accepts():
    """Measured 2026-08-24 on `api.mindshub.ai/v1`, alias `opus`: 20480 answers,
    24576 and above return HTTP 500 — which anton classifies as a transient
    provider error, so an over-large budget does not fail fast, it burns the
    retry ladder on every single generation. Re-measure before raising."""
    assert SPEC_MAX_TOKENS_RETRY <= 20480


async def test_first_call_uses_the_spec_budget_not_the_client_default(tmp_path: Path):
    """8192 is the client default; a specification is among the longest single
    answers anton asks for, and reasoning models spend their thinking from the
    same budget."""
    session = AsyncMock()
    session._llm.plan_stream = _stream_mock(_response("# Spec", output_tokens=10))
    body, err = await engine._plan_whole_document(
        session, system="s", user="u", node_label="make_tech_spec",
    )
    assert (body, err) == ("# Spec", None)
    assert session._llm.plan_stream.call_args.kwargs["max_tokens"] == SPEC_MAX_TOKENS


async def test_an_untruncated_answer_makes_exactly_one_call(tmp_path: Path):
    session = AsyncMock()
    session._llm.plan_stream = _stream_mock(_response("# Spec", output_tokens=10))
    await engine._plan_whole_document(
        session, system="s", user="u", node_label="make_tech_spec",
    )
    assert session._llm.plan_stream.call_count == 1


# ── retry ───────────────────────────────────────────────────────────────────

async def test_a_cut_answer_is_retried_with_more_room_and_a_compact_nudge():
    """The re-ask must CHANGE the call: an identical re-issue dies identically
    (measured for the main loop's own recovery)."""
    session = AsyncMock()
    session._llm.plan_stream = _stream_mock(
        _response("# Spec (half", output_tokens=SPEC_MAX_TOKENS),
        _response("# Spec, complete", output_tokens=200),
    )
    body, err = await engine._plan_whole_document(
        session, system="s", user="the brief", node_label="make_tech_spec",
    )
    assert (body, err) == ("# Spec, complete", None)
    calls = session._llm.plan_stream.call_args_list
    assert calls[0].kwargs["max_tokens"] == SPEC_MAX_TOKENS
    assert calls[1].kwargs["max_tokens"] == SPEC_MAX_TOKENS_RETRY
    retry_user = calls[1].kwargs["messages"][0]["content"]
    assert "cut off by the output limit" in retry_user
    assert retry_user.startswith("the brief")


async def test_truncation_is_detected_by_stop_reason_too():
    """A cut that stops just under the cap is invisible to a token count; the
    gateway has reported `stop_reason` correctly since 2026-08-03."""
    session = AsyncMock()
    session._llm.plan_stream = _stream_mock(
        _response("# half", output_tokens=12, stop_reason="length"),
        _response("# whole", output_tokens=20),
    )
    body, err = await engine._plan_whole_document(
        session, system="s", user="u", node_label="make_tech_spec",
    )
    assert (body, err) == ("# whole", None)
    assert session._llm.plan_stream.call_count == 2


async def test_the_retry_is_announced_as_progress():
    """A second whole-document call at a 20k budget is minutes of silence
    otherwise."""
    session = AsyncMock()
    session._llm.plan_stream = _stream_mock(
        _response("half", output_tokens=SPEC_MAX_TOKENS),
        _response("whole", output_tokens=10),
    )
    seen = []
    await engine._plan_whole_document(
        session, system="s", user="u", node_label="make_tech_spec",
        on_retry=lambda: seen.append("retry"),
    )
    assert seen == ["retry"]


# ── loud failure ────────────────────────────────────────────────────────────

async def test_two_cut_answers_produce_an_error_not_a_body():
    session = AsyncMock()
    session._llm.plan_stream = _stream_mock(
        _response("# half", output_tokens=SPEC_MAX_TOKENS),
        _response("# still half", output_tokens=SPEC_MAX_TOKENS_RETRY),
    )
    body, err = await engine._plan_whole_document(
        session, system="s", user="u", node_label="make_tech_spec",
    )
    assert body == ""
    assert err is not None
    assert "output limit" in err and "make_tech_spec" in err
    assert str(SPEC_MAX_TOKENS_RETRY) in err


async def test_a_truncated_tech_spec_is_never_written_to_disk(tmp_path: Path):
    """The whole point: `_spec_context` reads spec.md back and hands it to both
    generators, so a half spec on disk means half a system built silently."""
    st = _state(tmp_path)
    st.session._llm.plan_stream = _stream_mock(
        _response("# half", output_tokens=SPEC_MAX_TOKENS),
        _response("# still half", output_tokens=SPEC_MAX_TOKENS_RETRY),
    )
    err = await orchestrator._write_tech_spec(st)
    assert err is not None and "output limit" in err
    assert not (tmp_path / "spec.md").exists()
    assert st.internal_files == []
    assert [(s.node, s.outcome) for s in st.trace] == [("make_tech_spec", "fail")]


async def test_a_truncated_api_spec_is_not_reported_as_invalid_json(tmp_path: Path):
    """A cut JSON document fails `json.loads`, so before the truncation check
    the run blamed the model's syntax for what was really an output-cap hit."""
    session = AsyncMock()
    session._llm.plan_stream = _stream_mock(
        _response('{"paths": {"/api/i', output_tokens=SPEC_MAX_TOKENS),
        _response('{"paths": {"/api/i', output_tokens=SPEC_MAX_TOKENS_RETRY),
    )
    out = await engine._generate_api_spec(session, "ctx")
    assert out.startswith("Error:")
    assert "output limit" in out
    assert "not valid JSON" not in out


GOOD_SPEC = '{"openapi": "3.1.0", "paths": {"/api/time": {"get": {"summary": "t", "responses": {"200": {"description": "ok"}}}}}}'


async def test_api_spec_hot_path_sends_the_instruction_alone():
    """The PRD and spec.md are already in the shared history as the model's
    own replies; restating them cost ~4 000 characters per run."""
    seen: list[dict] = []

    def _capture(*, system, messages, max_tokens=None, tools=None):
        seen.append({"system": system, "last": messages[-1]["content"], "n": len(messages)})
        return _one_event_stream(_response(GOOD_SPEC, output_tokens=50))

    session = SimpleNamespace(_llm=SimpleNamespace(plan_stream=Mock(side_effect=_capture)))
    history = [{"role": "user", "content": "kickoff"}, {"role": "assistant", "content": "# PRD"}]
    out = await engine._generate_api_spec(
        session, "## Product requirements\nCTX", stateless=True,
        messages=history, tools=[{"name": "x", "description": "d", "input_schema": {}}],
        system_override="pipeline-system",
    )
    assert not out.startswith("Error:")
    assert seen[0]["system"] == "pipeline-system"
    assert seen[0]["n"] == len(history) + 1
    assert seen[0]["last"] == prompts.build_api_spec_instruction(stateless=True)
    assert "CTX" not in seen[0]["last"]


async def test_api_spec_cold_path_carries_the_context_before_the_instruction():
    seen: dict = {}

    def _capture(*, system, messages, max_tokens=None, tools=None):
        seen["last"] = messages[-1]["content"]
        return _one_event_stream(_response(GOOD_SPEC, output_tokens=50))

    session = SimpleNamespace(_llm=SimpleNamespace(plan_stream=Mock(side_effect=_capture)))
    out = await engine._generate_api_spec(session, "CTX", stateless=False)
    assert not out.startswith("Error:")
    assert seen["last"].startswith("## Requirements\nCTX")
    assert seen["last"].endswith(prompts.build_api_spec_instruction(stateless=False))


async def test_a_fenced_api_spec_is_still_accepted():
    session = AsyncMock()
    session._llm.plan_stream = _stream_mock(
        _response("```json\n" + GOOD_SPEC + "\n```", output_tokens=50),
    )
    out = await engine._generate_api_spec(session, "ctx")
    assert '"/api/time"' in out and not out.startswith("Error:")


async def test_an_unusable_api_spec_is_asked_for_once_more_with_the_problem_named():
    """Before, "valid JSON" was the whole check: `{"paths": {}}` passed, the
    generators built from nothing, and `_probe_app` had nothing to probe."""
    calls: list[str] = []

    def _capture(*, system, messages, max_tokens=None, tools=None):
        calls.append(messages[-1]["content"])
        body = '{"openapi": "3.1.0", "paths": {}}' if len(calls) == 1 else GOOD_SPEC
        return _one_event_stream(_response(body, output_tokens=50))

    session = SimpleNamespace(_llm=SimpleNamespace(plan_stream=Mock(side_effect=_capture)))
    retries: list[int] = []
    out = await engine._generate_api_spec(session, "ctx", on_retry=lambda: retries.append(1))
    assert not out.startswith("Error:")
    assert len(calls) == 2
    assert "## Previous reply rejected" in calls[1]
    assert "`paths` is missing or empty" in calls[1]
    assert retries == [1]


async def test_a_second_unusable_api_spec_ends_the_node():
    bad = '{"openapi": "3.1.0", "paths": {"/time": {"get": {"responses": {"200": {}}}}}}'
    session = AsyncMock()
    session._llm.plan_stream = _stream_mock(
        _response(bad, output_tokens=50), _response(bad, output_tokens=50),
    )
    out = await engine._generate_api_spec(session, "ctx")
    assert out.startswith("Error: API spec is not a usable OpenAPI document")
    assert "path `/time` is not under `/api/`" in out


def test_api_spec_problem_names_each_shape_defect():
    p = engine._api_spec_problem
    assert p([]) == "the document is not a JSON object"
    assert p({"openapi": "3.1.0"}) == "`paths` is missing or empty"
    assert p({"paths": {"/api/x": {"get": {}}}}) == "`GET /api/x` has no `responses`"
    assert p({"paths": {"/api/x": {"post": {"responses": {"201": {}}}}}}) is None
    # Non-operation keys (parameters, summary) at path level are ignored.
    assert p({"paths": {"/api/x": {"parameters": [], "get": {"responses": {"200": {}}}}}}) is None


_BACKEND_SECTION = """\
## Technical specification
## Implementation notes
Polls `GET .../state` twice a second.

## Backend

Endpoints:
- `POST /api/rooms`: create a room.
- `GET /api/rooms/{code}/state`: authoritative snapshot.
- POST /api/rooms/{code}/action: apply one command.
### Health
- `GET /api/health`: liveness, added by the generator.

## Progress journal (steps completed so far)
- GET /api/not-an-endpoint appears outside the section.
"""


def test_declared_api_paths_come_from_the_backend_section_only():
    """Backticked or bare, `/api/health` dropped, `###` subsections kept,
    everything outside `## Backend` ignored — and nothing at all when the
    section is missing, so a spec without one skips the comparison."""
    assert engine._declared_api_paths(_BACKEND_SECTION) == {
        "/api/rooms": "/api/rooms",
        "/api/rooms/{}/state": "/api/rooms/{code}/state",
        "/api/rooms/{}/action": "/api/rooms/{code}/action",
    }
    assert engine._declared_api_paths("## Product requirements\nX") == {}


def _doc(*paths: str) -> dict:
    return {"paths": {p: {"get": {"responses": {"200": {}}}} for p in paths}}


def test_api_spec_problem_rejects_paths_that_differ_from_spec_md():
    """The twentieth live run turned `/api/rooms/{code}/state` into
    `/api/rooms/{code}`: the count matched, so the shape check let it
    through and `spec.md` stopped describing the app."""
    p = engine._api_spec_problem
    declared = engine._declared_api_paths(_BACKEND_SECTION)
    exact = _doc("/api/rooms", "/api/rooms/{code}/state", "/api/rooms/{code}/action")
    assert p(exact, declared) is None
    # Parameter names, a trailing slash and `/api/health` are not differences.
    assert p(_doc("/api/rooms/", "/api/rooms/{id}/state", "/api/rooms/{id}/action", "/api/health"), declared) is None
    renamed = _doc("/api/rooms", "/api/rooms/{code}", "/api/rooms/{code}/action")
    problem = p(renamed, declared)
    assert problem.startswith("`spec.md` lists `/api/rooms/{code}/state` but the document has no such path")
    added = _doc("/api/rooms", "/api/rooms/{code}/state", "/api/rooms/{code}/action", "/api/stats")
    assert p(added, declared).startswith("the document adds `/api/stats`, which `spec.md` does not list")
    # Without a declared list (no `## Backend` section) the paths are the model's.
    assert p(renamed) is None
    assert p(renamed, {}) is None


async def test_a_renamed_path_is_sent_back_with_spec_md_quoted():
    calls: list[str] = []

    def _capture(*, system, messages, max_tokens=None, tools=None):
        calls.append(messages[-1]["content"])
        paths = ("/api/rooms", "/api/rooms/{code}", "/api/rooms/{code}/action") if len(calls) == 1 else (
            "/api/rooms", "/api/rooms/{code}/state", "/api/rooms/{code}/action")
        return _one_event_stream(_response(json.dumps(_doc(*paths)), output_tokens=50))

    session = SimpleNamespace(_llm=SimpleNamespace(plan_stream=Mock(side_effect=_capture)))
    out = await engine._generate_api_spec(session, _BACKEND_SECTION)
    assert not out.startswith("Error:")
    assert len(calls) == 2
    assert "## Previous reply rejected" in calls[1]
    assert "`spec.md` lists `/api/rooms/{code}/state`" in calls[1]
    assert '"/api/rooms/{code}/state"' in out


async def test_a_recovered_tech_spec_is_written_normally(tmp_path: Path):
    st = _state(tmp_path)
    st.session._llm.plan_stream = _stream_mock(
        _response("# half", output_tokens=SPEC_MAX_TOKENS),
        _response("# Spec\nbody", output_tokens=100),
    )
    assert await orchestrator._write_tech_spec(st) is None
    assert (tmp_path / "spec.md").read_text(encoding="utf-8") == "# Spec\nbody"
    assert st.internal_files == ["spec.md"]


# ── The shared history reaches the spec nodes (design 2.1) ──────────────────


async def test_spec_call_continues_the_shared_history_when_given_one():
    """Phase D is the last node that sees the source material, so it runs on
    the shared message list rather than a fresh one-message conversation."""
    seen: dict = {}

    def _capture(*, system, messages, max_tokens=None, tools=None):
        seen["messages"] = messages
        seen["tools"] = tools
        return _one_event_stream(_response("# Spec", output_tokens=10))

    session = SimpleNamespace(_llm=SimpleNamespace(plan_stream=Mock(side_effect=_capture)))
    history = [
        {"role": "user", "content": "kickoff"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "1", "name": "scratchpad", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "1", "content": "ok"},
        ]},
    ]
    body, err = await engine._plan_whole_document(
        session, system="sys", user="write the spec", node_label="make_tech_spec",
        messages=history,
        tools=[{"name": "scratchpad", "description": "d", "input_schema": {}}],
    )
    assert err is None
    assert body == "# Spec"
    # The instruction is appended to the history, not sent on its own.
    assert len(seen["messages"]) == len(history) + 1
    assert seen["messages"][0] == history[0]
    assert seen["messages"][-1]["content"] == "write the spec"
    # A non-empty tools array is mandatory whenever the history carries
    # tool_use/tool_result blocks — the API rejects the request otherwise.
    assert seen["tools"]


async def test_spec_call_without_history_still_sends_one_user_message():
    """The cold-start path builds its context from disk, not from a history."""
    seen: dict = {}

    def _capture(*, system, messages, max_tokens=None, tools=None):
        seen["messages"] = messages
        return _one_event_stream(_response("# Spec", output_tokens=10))

    session = SimpleNamespace(_llm=SimpleNamespace(plan_stream=Mock(side_effect=_capture)))
    body, err = await engine._plan_whole_document(
        session, system="sys", user="write the spec", node_label="make_tech_spec",
    )
    assert err is None
    assert seen["messages"] == [{"role": "user", "content": "write the spec"}]


async def test_a_tool_call_on_a_spec_step_is_refused_once_then_fails_the_node():
    """The array is fixed for the whole region, so a spec node can be handed a
    call it must not run. One refusal, then the node fails: each further round
    re-sends the entire shared history, the most expensive round there is."""
    calls: list[dict] = []

    def _capture(*, system, messages, max_tokens=None, tools=None):
        calls.append(messages[-1]["content"])
        return _one_event_stream(SimpleNamespace(
            content="",
            tool_calls=[SimpleNamespace(id="1", name="scratchpad", input={})],
            stop_reason="tool_calls",
            usage=SimpleNamespace(output_tokens=10),
        ))

    session = SimpleNamespace(_llm=SimpleNamespace(plan_stream=Mock(side_effect=_capture)))
    body, err = await engine._plan_whole_document(
        session, system="sys", user="write the spec", node_label="make_tech_spec",
    )
    assert body == ""
    assert err is not None and "tools" in err
    assert len(calls) == 2  # the original ask, then one nudged retry
    assert "Do NOT call any tool" in calls[1]


async def test_a_refused_tool_call_does_not_consume_the_truncation_retry():
    """The budget ladder exists for cut answers. A tool-call refusal that ate
    a rung would leave a genuinely truncated spec with no room to retry."""
    budgets: list[int] = []

    responses = [
        SimpleNamespace(
            content="",
            tool_calls=[SimpleNamespace(id="1", name="scratchpad", input={})],
            stop_reason="tool_calls",
            usage=SimpleNamespace(output_tokens=10),
        ),
        _response("x" * 100, output_tokens=SPEC_MAX_TOKENS, stop_reason="length"),
        _response("# Spec", output_tokens=10),
    ]

    def _capture(*, system, messages, max_tokens=None, tools=None):
        budgets.append(max_tokens)
        return _one_event_stream(responses[len(budgets) - 1])

    session = SimpleNamespace(_llm=SimpleNamespace(plan_stream=Mock(side_effect=_capture)))
    body, err = await engine._plan_whole_document(
        session, system="sys", user="write the spec", node_label="make_tech_spec",
    )
    assert err is None
    assert body == "# Spec"
    assert budgets == [SPEC_MAX_TOKENS, SPEC_MAX_TOKENS, SPEC_MAX_TOKENS_RETRY]


async def test_the_spec_node_falls_back_to_the_assembled_context_without_history(tmp_path, monkeypatch):
    """Cold start: `messages` is empty, so the node must send the brief and
    the notes explicitly — not a lone instruction into the void."""
    seen: dict = {}

    async def _fake(session, *, system, user, node_label, trace=None, on_text=None,
                    on_retry=None, messages=None, tools=None):
        seen["messages"] = messages
        seen["user"] = user
        seen["system"] = system
        return "# Spec", None

    monkeypatch.setattr(engine, "_plan_whole_document", _fake)
    state = _state(tmp_path, brief="## Goal\nA dashboard.")
    state.data_notes = "rows = q()"
    await orchestrator._write_tech_spec(state)

    assert seen["messages"] is None
    assert "A dashboard." in seen["user"]
    assert "rows = q()" in seen["user"]


async def test_the_spec_node_continues_the_history_when_there_is_one(tmp_path, monkeypatch):
    seen: dict = {}

    async def _fake(session, *, system, user, node_label, trace=None, on_text=None,
                    on_retry=None, messages=None, tools=None):
        seen["messages"] = messages
        seen["user"] = user
        seen["system"] = system
        seen["tools"] = tools
        return "# Spec", None

    monkeypatch.setattr(engine, "_plan_whole_document", _fake)
    state = _state(tmp_path, brief="## Goal\nA dashboard.")
    state.messages = [{"role": "user", "content": "kickoff"}]
    await orchestrator._write_tech_spec(state)

    assert seen["messages"] is state.messages
    assert seen["system"] == state.pipeline_system
    assert seen["tools"] == state.pipeline_tools
    # No restating of a context the conversation already holds.
    assert "A dashboard." not in seen["user"]
    # ...but the step's own rules travel in the instruction: this is the only
    # step-specific text the hot path ever sees (seventh live run 2026-09-16
    # ran on a bare "write the specification now" and retold the PRD).
    assert seen["user"] == prompts.build_tech_spec_instruction(state)
    assert prompts._TECH_SPEC_STACK in seen["user"]
    assert "Implementation notes" in seen["user"]


def test_the_tech_spec_ask_demands_the_source_material_be_carried_forward():
    """After this node the shared history is dropped, so anything the
    generators need verbatim — figures, quotes, image URLs, the source link —
    has to be in `spec.md` or it is gone."""
    from anton.core.tools.generate_artifact.prompts import (
        TECH_SPEC_CARRY_FORWARD,
        build_tech_spec_instruction,
    )

    instruction = build_tech_spec_instruction(SimpleNamespace())
    assert TECH_SPEC_CARRY_FORWARD in instruction
    assert "LAST step" in TECH_SPEC_CARRY_FORWARD
    assert "verbatim" in TECH_SPEC_CARRY_FORWARD


def test_the_carry_forward_names_what_actually_reaches_the_build_step():
    """The old text said "nothing else from above reaches them". False —
    `_spec_context` hands the generators the brief, the PRD verbatim, the
    exec record and the web excerpts — and it pushed the model to carry the
    PRD forward word for word. The accurate list is what makes "carry forward
    only what they cannot get elsewhere" mean something."""
    from anton.core.tools.generate_artifact.discovery.notes import EXEC_OUTPUT_MAX
    from anton.core.tools.generate_artifact.prompts import TECH_SPEC_CARRY_FORWARD

    text = TECH_SPEC_CARRY_FORWARD
    assert "nothing else from above" not in text
    assert "the PRD verbatim" in text
    assert f"first {EXEC_OUTPUT_MAX} characters" in text
    assert "do NOT receive this conversation" in text
    assert "cannot get elsewhere" in text
