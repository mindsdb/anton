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


def test_includes_tool_result_evidence_verbatim_when_short():
    # Short results (under tool_cap) reach the verifier untouched — this pins
    # the ENG-716 "content, not just ok/error" property. It does NOT exercise
    # truncation; direction-of-truncation coverage lives in
    # test_truncates_large_tool_output_keeping_both_ends.
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


def test_near_threshold_output_passes_through_whole():
    # A 401-char result at cap=400 must come back verbatim, not "clipped" into
    # ~428 chars with a "[... 1 chars elided ...]" in the middle — the marker
    # may never cost more than it removes (#305 review nit). The invariant:
    # clipping never returns more characters than it was given.
    near = "BEGIN " + "y" * 395 + " END"  # just over the 400 cap
    assert 400 < len(near) < 430
    history = [
        {"role": "user", "content": "run it"},
        {"role": "user", "content": [{"type": "tool_result", "content": near}]},
    ]
    out = _render_verify_transcript(history, tool_cap=400)
    assert near in out
    assert "chars elided" not in out


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


# ---------------------------------------------------------------------------
# ENG-1633 — the conversational path used a bare `text[:text_cap]`, so a reply
# over the cap reached the verifier ending mid-word with nothing saying it had
# been cut. Every case below fails on that code.
# ---------------------------------------------------------------------------


def test_judged_reply_under_final_cap_reaches_the_verifier_whole():
    # 3,000 chars: over `text_cap` (2,000), under `final_cap` (12,000). The
    # reply the verifier is judging must arrive intact — the whole bug is that
    # it did not. Asserts the CONCLUSION specifically, because that is the
    # evidence a completion verifier is actually looking for and it is exactly
    # what a head-slice discards.
    reply = "OPENING_MARKER. " + "The reconciliation matched every line. " * 74 + "CONCLUSION_MARKER."
    assert 2000 < len(reply) < 12000
    history = [
        {"role": "user", "content": "reconcile the ledger"},
        {"role": "assistant", "content": reply},
    ]
    out = _render_verify_transcript(history)
    assert reply in out
    assert "CONCLUSION_MARKER." in out
    assert "chars elided" not in out


def test_oversize_judged_reply_keeps_both_ends_and_marks_the_elision():
    # Past `final_cap` the reply is clipped, but never silently: the opening,
    # an explicit marker, and the closing sentence all survive. The last of
    # those is what stops the verifier reading the clip as the assistant
    # stopping mid-sentence.
    reply = "OPENING_MARKER. " + "The reconciliation matched every line. " * 400 + "CONCLUSION_MARKER."
    assert len(reply) > 12000
    history = [
        {"role": "user", "content": "reconcile the ledger"},
        {"role": "assistant", "content": reply},
    ]
    out = _render_verify_transcript(history)
    assert "OPENING_MARKER." in out
    assert "chars elided" in out
    assert "CONCLUSION_MARKER." in out
    # The rendered reply must not end mid-word — that is the literal trigger
    # the verifier quoted back ("cuts off mid-sentence").
    assert out.rstrip().endswith("CONCLUSION_MARKER.")


def test_buried_failure_admission_survives_in_the_judged_reply():
    # The regression guard for the fix that was ALMOST shipped. A flat both-ends
    # clip at 2,000 elides the middle of this reply, which is where the
    # assistant admits it could not do step 3 — measured live, that turns the
    # verdict into a false COMPLETE, i.e. the user is told the email was sent.
    # Fails on the shipped head-slice too (the admission sits past 2,000).
    head = "All three done — reconciliation, duplicates, and the summary is on its way.\n\n"
    filler = "Matched 47 invoices against the bank export, all reconciled. " * 40
    admission = "\n\nI could NOT send the email — there is no mail tool in this environment.\n\n"
    appendix = "Appendix: the full register follows, one row per invoice. " * 30
    reply = head + filler + admission + appendix
    assert 2000 < len(reply) < 12000
    history = [
        {"role": "user", "content": "reconcile, flag duplicates, email finance"},
        {"role": "assistant", "content": reply},
    ]
    out = _render_verify_transcript(history)
    assert "I could NOT send the email" in out


def test_prior_turns_keep_the_smaller_budget():
    # Only the message under judgment gets `final_cap`. An identically sized
    # EARLIER reply is still clipped — that asymmetry is what bounds the added
    # cost to one entry instead of `max_convo` of them.
    old = "OLD_HEAD. " + "Earlier turn body text goes here. " * 100 + "OLD_TAIL."
    new = "NEW_HEAD. " + "Current turn body text goes here. " * 100 + "NEW_TAIL."
    assert 2000 < len(old) < 12000 and 2000 < len(new) < 12000
    history = [
        {"role": "user", "content": "first request"},
        {"role": "assistant", "content": old},
        {"role": "user", "content": "second request"},
        {"role": "assistant", "content": new},
    ]
    out = _render_verify_transcript(history)
    assert new in out                 # judged reply: whole
    assert old not in out             # prior reply: clipped
    assert "chars elided" in out      # ...but marked, never silently
    assert "OLD_HEAD." in out and "OLD_TAIL." in out


def test_long_user_message_keeps_its_trailing_requirement():
    # Same bare slice clipped USER messages, so a long pasted request lost its
    # final requirement and the verifier judged completion against a truncated
    # ask. Only bites prior turns (the current request is also passed unclipped
    # in the header), but it is the same line and the same defect.
    ask = "Please do the following.\n" + "Requirement X must be satisfied. " * 100 + "\nIMPORTANT: finish with a summary."
    assert len(ask) > 2000
    history = [
        {"role": "user", "content": ask},
        {"role": "assistant", "content": "Working on it."},
    ]
    out = _render_verify_transcript(history)
    assert "IMPORTANT: finish with a summary." in out
    assert "chars elided" in out


def test_no_conversational_text_is_ever_clipped_without_a_marker():
    # The ticket's headline invariant, asserted directly: for every entry the
    # renderer shortens, the output says so. A speaker segment that is both
    # shorter than its source and unmarked is the bug.
    long_user = "U_HEAD " + "user text " * 400 + "U_TAIL"
    long_reply = "A_HEAD " + "assistant text " * 2000 + "A_TAIL"
    history = [
        {"role": "user", "content": long_user},
        {"role": "assistant", "content": long_reply},
    ]
    out = _render_verify_transcript(history)
    for source in (long_user, long_reply):
        if source not in out:
            assert "chars elided" in out
    # Neither original survives whole here, so the marker must be present.
    assert long_user not in out and long_reply not in out
    assert out.count("chars elided") == 2


def test_a_trailing_image_block_does_not_steal_the_judged_budget():
    # One assistant message contributes one entry per content block, so "the
    # last assistant entry" is not necessarily the reply: a trailing image
    # placeholder would take `final_cap` and drop the reply back to `text_cap`,
    # which is the flat-cap behaviour this function exists to avoid. Caught in
    # self-review on #364 — anton itself never builds this shape, but a host can
    # supply it through `initial_history` / a cloud-turn request.
    reply = "IMG_HEAD. " + "The answer body continues here. " * 100 + "IMG_TAIL."
    assert 2000 < len(reply) < 12000
    history = [
        {"role": "user", "content": "describe the chart and answer"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": reply}, {"type": "image"}],
        },
    ]
    out = _render_verify_transcript(history)
    assert reply in out
    assert "chars elided" not in out
