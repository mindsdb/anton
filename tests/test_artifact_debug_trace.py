"""debug_trace.py: GenTrace/NullTrace — the merged pipeline's step log.

One trace for the whole pipeline, reading the single
`ANTON_DEBUG_ARTIFACT_GENERATE_TOOL` env var: the discovery phases and the
generation FSM append to the same viewer-compatible file, which is what makes
a run readable end to end instead of as two disjoint logs."""
from __future__ import annotations

import json

from anton.core.llm.provider import LLMResponse, Usage
from anton.core.tools.generate_artifact import debug_trace


def _lines(path):
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in records:
        r.pop("ts")
    return records


def test_make_trace_returns_null_trace_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("ANTON_DEBUG_ARTIFACT_GENERATE_TOOL", raising=False)
    assert isinstance(debug_trace.make_trace(), debug_trace.NullTrace)


def test_make_trace_returns_prd_trace_when_env_var_set(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTON_DEBUG_ARTIFACT_GENERATE_TOOL", str(tmp_path / "trace.jsonl"))
    assert isinstance(debug_trace.make_trace(), debug_trace.GenTrace)


def test_null_trace_methods_are_all_no_ops():
    trace = debug_trace.NullTrace()
    trace.run_start(slug="s", artifact_type="html-app", user_request="x", agent_understanding="y")
    trace.node("n", "done")
    trace.llm_call(node="n", method="plan", system="s", messages=[])
    trace.verdict(node="n", schema="X", value={})
    trace.scratchpad(node="n", input={}, output="")
    trace.run_result(ok=True)


def test_run_start_records_the_call_fields_verbatim(tmp_path):
    """One run_start for the whole pipeline, fields passed through as given.

    The two tools used to fold their inputs into a single `brief` string so
    that both emitted the same record shape for the shared viewer. There is
    one tool now and one run_start, so the folding has nothing left to
    reconcile — and a debug log that reshapes its input is a debug log you
    cannot compare against the call that produced it.
    """
    trace = debug_trace.GenTrace(str(tmp_path / "trace.jsonl"))
    trace.run_start(
        slug="clock", artifact_type="html-app",
        user_request="build a clock", agent_understanding="an analog clock",
        known_data="none", user_preferences="dark mode",
    )
    rec = _lines(tmp_path / "trace.jsonl")[0]
    assert rec["event"] == "run_start"
    assert rec["slug"] == "clock"
    assert rec["artifact_type"] == "html-app"
    assert rec["user_request"] == "build a clock"
    assert rec["agent_understanding"] == "an analog clock"
    assert rec["known_data"] == "none"
    assert rec["user_preferences"] == "dark mode"


def test_run_start_records_only_the_fields_it_was_given(tmp_path):
    """Open-ended by design: the generation FSM passes `artifact_path` and
    `is_fullstack`, the discovery phases pass the call fields, and neither
    has to carry the other's keys."""
    trace = debug_trace.GenTrace(str(tmp_path / "trace.jsonl"))
    trace.run_start(
        slug="clock", artifact_type="html-app",
        user_request="build a clock", agent_understanding="an analog clock",
    )
    rec = _lines(tmp_path / "trace.jsonl")[0]
    assert "known_data" not in rec
    assert "user_preferences" not in rec


def test_run_start_stringifies_a_path(tmp_path):
    """`artifact_path` arrives as a Path and json.dumps cannot serialise one
    directly; the `default=str` fallback would hide the conversion, so it is
    explicit."""
    trace = debug_trace.GenTrace(str(tmp_path / "trace.jsonl"))
    trace.run_start(slug="clock", artifact_path=tmp_path, is_fullstack=False)
    rec = _lines(tmp_path / "trace.jsonl")[0]
    assert rec["artifact_path"] == str(tmp_path)


def test_node_writes_node_outcome_and_detail(tmp_path):
    trace = debug_trace.GenTrace(str(tmp_path / "trace.jsonl"))
    trace.node("gathering", "done", detail="finish_gathering: type=html-app")
    rec = _lines(tmp_path / "trace.jsonl")[0]
    assert rec == {
        "event": "node",
        "node": "gathering",
        "outcome": "done",
        "detail": "finish_gathering: type=html-app",
    }


def test_llm_call_with_response_uses_serialize_response(tmp_path):
    trace = debug_trace.GenTrace(str(tmp_path / "trace.jsonl"))
    response = LLMResponse(content="hello", tool_calls=[], usage=Usage(input_tokens=1, output_tokens=2))
    trace.llm_call(
        node="draft_brief", method="plan", system="sys",
        messages=[{"role": "user", "content": "hi"}], response=response,
    )
    rec = _lines(tmp_path / "trace.jsonl")[0]
    assert rec["event"] == "llm_call"
    assert rec["node"] == "draft_brief"
    assert rec["method"] == "plan"
    assert rec["response"]["content"] == "hello"
    assert rec["response"]["usage"]["output_tokens"] == 2
    assert rec["messages"] == [{"role": "user", "content": "hi"}]


def test_llm_call_with_structured_value_omits_response_shape(tmp_path):
    trace = debug_trace.GenTrace(str(tmp_path / "trace.jsonl"))
    trace.llm_call(
        node="classify_feedback", method="generate_object", system="sys",
        messages=[], value={"route": "revise_brief"},
    )
    rec = _lines(tmp_path / "trace.jsonl")[0]
    assert rec["response"] == {"structured": {"route": "revise_brief"}}


def test_verdict_writes_schema_and_value(tmp_path):
    trace = debug_trace.GenTrace(str(tmp_path / "trace.jsonl"))
    trace.verdict(node="classify_feedback", schema="FeedbackVerdict", value={"route": "revise_brief"})
    rec = _lines(tmp_path / "trace.jsonl")[0]
    assert rec["schema"] == "FeedbackVerdict"
    assert rec["value"] == {"route": "revise_brief"}


def test_scratchpad_writes_input_and_output(tmp_path):
    trace = debug_trace.GenTrace(str(tmp_path / "trace.jsonl"))
    trace.scratchpad(node="web_search", input={"query": "btc price"}, output="found 3 results")
    rec = _lines(tmp_path / "trace.jsonl")[0]
    assert rec["node"] == "web_search"
    assert rec["input"] == {"query": "btc price"}
    assert rec["output"] == "found 3 results"


def test_run_result_writes_ok_result_or_error(tmp_path):
    trace = debug_trace.GenTrace(str(tmp_path / "trace.jsonl"))
    trace.run_result(ok=True, result={"status": "prd_written"})
    trace.run_result(ok=False, error="boom")
    recs = _lines(tmp_path / "trace.jsonl")
    assert recs == [
        {"event": "run_result", "ok": True, "result": {"status": "prd_written"}, "error": None},
        {"event": "run_result", "ok": False, "result": None, "error": "boom"},
    ]


def test_emit_never_raises_when_the_path_is_unwritable(tmp_path):
    trace = debug_trace.GenTrace(str(tmp_path))  # a directory, not a file
    trace.node("gathering", "done")  # must not raise


def test_writes_append_rather_than_truncate(tmp_path):
    """The whole point of sharing generate_artifact's env var: a second
    trace instance opened later in the same run must add to the file, not
    wipe out what the first one already wrote."""
    path = tmp_path / "trace.jsonl"
    debug_trace.GenTrace(str(path)).node("gathering", "done")
    debug_trace.GenTrace(str(path)).node("draft_brief", "done")  # a second instance, same path
    assert [r["node"] for r in _lines(path)] == ["gathering", "draft_brief"]


def test_a_run_combining_both_tools_writes_one_interleaved_log(tmp_path, monkeypatch):
    """generate_prd and generate_artifact run back to back for a web
    artifact — this is what makes their two trace files actually become
    one, viewable together in artifact_trace_viewer.html."""
    from anton.core.tools.generate_artifact import debug_trace as artifact_debug_trace

    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("ANTON_DEBUG_ARTIFACT_GENERATE_TOOL", str(path))

    prd_trace = debug_trace.make_trace()
    prd_trace.run_start(slug="s", artifact_type="html-app", user_request="x", agent_understanding="y")
    prd_trace.run_result(ok=True, result={"status": "prd_written"})

    artifact_trace = artifact_debug_trace.make_trace()
    artifact_trace.run_start(slug="s", artifact_type="html-app", artifact_path="/tmp/s", brief="ctx", is_fullstack=False)

    events = [r["event"] for r in _lines(path)]
    assert events == ["run_start", "run_result", "run_start"]
