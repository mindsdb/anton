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


def test_truncates_large_tool_output_keeping_both_ends():
    # Direction matters, not just the cap: a homogeneous fixture ("x" * 5000)
    # cannot tell head-truncation from tail-truncation, and head-truncation is
    # exactly the ENG-836 bug. Mark both ends and assert both survive.
    big = "HEAD_MARKER " + "x" * 5000 + " CAUSE_MARKER"
    history = [
        {"role": "user", "content": "run it"},
        {"role": "user", "content": [{"type": "tool_result", "content": big}]},
    ]
    out = _render_verify_transcript(history, tool_cap=400)
    # Tool output is capped so the verify call stays cheap...
    assert out.count("x") <= 400
    # ...but the clip keeps the head (what ran / the [error] marker) AND the
    # tail (the cause), eliding the middle.
    assert "HEAD_MARKER" in out
    assert "CAUSE_MARKER" in out
    assert "chars elided" in out


def test_traceback_cause_survives_truncation():
    # Regression for ENG-836: the verifier judged an unrecoverable environment
    # wall INCOMPLETE because head-truncation kept the traceback preamble and
    # discarded the final line naming the missing system library. The fixture
    # must exceed tool_cap with the cause at the END, shaped like the real
    # pyodbc failure — a short tidy error string would pass either way.
    frames = "".join(
        f'  File "/app/step_{i}.py", line {i * 7}, in run\n    connect()\n'
        for i in range(12)
    )
    traceback = (
        "Traceback (most recent call last):\n"
        + frames
        + "ImportError: libodbc.so.2: cannot open shared object file: "
        "No such file or directory"
    )
    assert len(traceback) > 400
    history = [
        {"role": "user", "content": "build the KPI dashboard against Azure SQL"},
        {"role": "user", "content": [{"type": "tool_result", "content": traceback}]},
    ]
    out = _render_verify_transcript(history, tool_cap=400)
    assert "libodbc.so.2: cannot open shared object file" in out
    assert "Traceback (most recent call last):" in out


def test_empty_history_is_safe():
    assert _render_verify_transcript([]) == "(no conversation)"


def test_long_tool_turn_keeps_request_and_antecedent():
    # A prior turn establishes context; the current turn is referential and runs
    # >8 tool rounds. Tool activity must NOT evict the conversational thread.
    history = [
        {"role": "user", "content": "clean up report_q1.csv"},
        {"role": "assistant", "content": "Done — deduped 12 rows in report_q1.csv."},
        {"role": "user", "content": "now do the same for the other file"},
    ]
    for i in range(10):  # 10 tool rounds, each with preamble text + a tool call
        history.append({"role": "assistant", "content": [
            {"type": "text", "text": f"Processing step {i}"},
            {"type": "tool_use", "name": "scratchpad"},
        ]})
        history.append({"role": "user", "content": [{"type": "tool_result", "content": f"row {i} cleaned"}]})
    history.append({"role": "assistant", "content": "All cleaned."})

    out = _render_verify_transcript(history)
    # Both the referential request and its antecedent survive the tool volume —
    # preamble text ("Processing step N") must not consume the conversation budget.
    assert "now do the same for the other file" in out
    assert "clean up report_q1.csv" in out
    assert "Done — deduped 12 rows in report_q1.csv." in out  # antecedent's answer
    # And tool evidence is still present.
    assert "TOOL RESULT:" in out


def test_multimodal_user_text_labeled_user_not_assistant():
    history = [
        {"role": "user", "content": [
            {"type": "text", "text": "describe this image and save a report"},
            {"type": "image", "source": {"type": "base64", "data": "A" * 500}},
        ]},
        {"role": "assistant", "content": "Here's the description."},
    ]
    out = _render_verify_transcript(history)
    assert "USER: describe this image and save a report" in out
    assert "ASSISTANT: describe this image" not in out
    assert "USER: [image]" in out
    assert "A" * 100 not in out  # no base64 leaked


def test_multimodal_tool_result_keeps_summary_drops_base64():
    history = [
        {"role": "user", "content": "read the chart"},
        {"role": "user", "content": [{"type": "tool_result", "content": [
            {"type": "image", "source": {"type": "base64", "data": "Z" * 5000}},
            {"type": "text", "text": "Chart saved: 12 monthly bars"},
        ]}]},
    ]
    out = _render_verify_transcript(history)
    assert "Chart saved: 12 monthly bars" in out
    assert "[image]" in out
    assert "Z" * 100 not in out  # base64 payload never serialized into the view


def test_system_injections_do_not_consume_budget():
    # SYSTEM injections are dropped before budgeting, so real turns aren't evicted
    # by internal noise.
    history = []
    for i in range(8):
        history.append({"role": "user", "content": f"SYSTEM: internal note {i}"})
    history.append({"role": "user", "content": "the actual request"})
    out = _render_verify_transcript(history, max_convo=3)
    assert "the actual request" in out
    assert "internal note" not in out
