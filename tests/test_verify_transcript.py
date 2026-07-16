"""Unit tests for the completion-verifier's compact transcript view (ENG-716).

Guards the two review findings on PR #255: the verifier view must retain
referential task context from prior turns AND carry truncated tool-result
evidence (not just an ok/error flag), while omitting internal SYSTEM
injections.
"""

from __future__ import annotations

from anton.core.session import _render_verify_transcript


def test_keeps_prior_turn_context_for_referential_followups():
    history = [
        {"role": "user", "content": "clean up report_q1.csv"},
        {"role": "assistant", "content": "Done — deduped 12 rows."},
        {"role": "user", "content": "now do the same for the other file"},
        {"role": "assistant", "content": "Which file did you mean?"},
    ]
    out = _render_verify_transcript(history)
    # The current request is referential ("the same", "the other file"); the
    # prior turn must remain so the verifier can resolve it.
    assert "clean up report_q1.csv" in out
    assert "now do the same for the other file" in out


def test_includes_truncated_tool_result_evidence():
    history = [
        {"role": "user", "content": "how many open PRs?"},
        {"role": "assistant", "content": [{"type": "tool_use", "name": "scratchpad"}]},
        {"role": "user", "content": [{"type": "tool_result", "content": "[error]\nHTTP 403 Forbidden"}]},
        {"role": "assistant", "content": "There are 42 open PRs."},
    ]
    out = _render_verify_transcript(history)
    # The tool actually errored while the assistant claimed a number — the
    # verifier must see the error content to catch the mismatch.
    assert "TOOL RESULT: [error]" in out
    assert "HTTP 403 Forbidden" in out
    assert "There are 42 open PRs." in out


def test_omits_internal_system_injections():
    history = [
        {"role": "user", "content": "build the dashboard"},
        {"role": "user", "content": "SYSTEM: Task verification determined this task is not yet complete. Continue working."},
        {"role": "assistant", "content": "Working on it."},
    ]
    out = _render_verify_transcript(history)
    assert "SYSTEM:" not in out
    assert "Continue working" not in out
    assert "build the dashboard" in out


def test_truncates_large_tool_output():
    big = "x" * 5000
    history = [
        {"role": "user", "content": "run it"},
        {"role": "user", "content": [{"type": "tool_result", "content": big}]},
    ]
    out = _render_verify_transcript(history, tool_cap=400)
    # Tool output is capped so the verify call stays cheap.
    assert out.count("x") <= 400


def test_empty_history_is_safe():
    assert _render_verify_transcript([]) == "(no conversation)"
