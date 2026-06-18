"""Scenario D — Loop safety and hang protection."""

from __future__ import annotations

import json
import pytest

from tests.e2e.harness import (
    assert_exit_ok, assert_not_output, assert_output, base_env, run_anton,
)


@pytest.mark.stub_only
def test_max_tool_rounds_circuit_breaker_fires(cfg, stub, tmp_path):
    # _MAX_TOOL_ROUNDS = 25; backstop fires at round 26 (> 25) — need 26 queued tool calls.
    for i in range(26):
        stub.queue_tool_call("scratchpad", {"action": "exec", "name": f"loop_{i}", "code": f"print({i})"})
    stub.queue_text("Summarising. CIRCUIT_FIRED")
    result = run_anton(["--folder", str(tmp_path)], ["run forever", "exit"],
                       env=base_env(stub), timeout=cfg.timeout(60))

    assert_exit_ok(result)
    assert_not_output(result, "Traceback (most recent call last)")
    assert_output(result, "CIRCUIT_FIRED")
    assert any(
        "You have used 25 tool-call rounds" in json.dumps(r.get("messages", []))
        for r in stub.requests
    ), f"Max-rounds message not found. Request count: {stub.request_count}"


@pytest.mark.stub_only
def test_continuation_limit_respected(cfg, stub, tmp_path):
    _tool = lambda i: {"action": "exec", "name": f"c{i}", "code": f"print({i})"}
    for i in range(3):
        stub.queue_tool_call("scratchpad", _tool(i))
        stub.queue_text(f"Round {i} done.")
        stub.queue_verification_incomplete(f"not done, attempt {i}")
    stub.queue_tool_call("scratchpad", _tool(3))
    stub.queue_text("Round 3 done.")
    stub.queue_text("BUDGET_EXHAUSTED")
    result = run_anton(["--folder", str(tmp_path)], ["do continuations", "exit"],
                       env=base_env(stub), timeout=cfg.timeout(60))

    assert_exit_ok(result)
    assert_not_output(result, "Traceback (most recent call last)")
    assert_output(result, "BUDGET_EXHAUSTED")
    assert any(
        "You have attempted to complete this task multiple times" in json.dumps(r.get("messages", []))
        for r in stub.requests
    ), f"Budget-exhausted message not found. Request count: {stub.request_count}"


def test_session_exits_within_timeout(cfg, stub, tmp_path):
    stub.queue_text("Quick reply. QUICK_EXIT")
    stub.queue_verification_ok()
    result = run_anton(["--folder", str(tmp_path)], ["quick question", "exit"],
                       env=base_env(stub), timeout=cfg.timeout(25))
    assert_exit_ok(result)
    assert_not_output(result, "Traceback (most recent call last)")
    if not cfg.live:
        assert_output(result, "QUICK_EXIT")


@pytest.mark.stub_only
def test_resilience_nudge_injected_after_two_errors(cfg, stub, tmp_path):
    # Reuse ONE scratchpad name: a realistic retry loop is the same cell
    # failing twice. (Distinct names would instead trip the single-scratchpad
    # guard, which is exercised separately.)
    bad_code = "def oops(:\n    pass"
    stub.queue_tool_call("scratchpad", {"action": "exec", "name": "bad", "code": bad_code})
    stub.queue_tool_call("scratchpad", {"action": "exec", "name": "bad", "code": bad_code})
    stub.queue_text("NUDGE_RECEIVED")
    stub.queue_verification_ok()
    result = run_anton(["--folder", str(tmp_path)], ["do bad stuff", "exit"],
                       env=base_env(stub), timeout=cfg.timeout(30))

    assert_exit_ok(result)
    assert_not_output(result, "Traceback (most recent call last)")
    assert_output(result, "NUDGE_RECEIVED")
    assert any(
        "failed twice in a row" in json.dumps(r.get("messages", []))
        for r in stub.requests
    ), f"Resilience nudge not found. Request count: {stub.request_count}"


@pytest.mark.stub_only
def test_circuit_breaker_fires_after_five_consecutive_errors(cfg, stub, tmp_path):
    # Reuse ONE scratchpad name so this exercises the consecutive-error
    # circuit breaker, not the single-scratchpad guard (distinct names would
    # trigger a guard challenge that resets the streak).
    bad_code = "def bad(:\n    pass"
    for i in range(5):
        stub.queue_tool_call("scratchpad", {"action": "exec", "name": "err", "code": bad_code})
    stub.queue_text("ERRORS_EXHAUSTED")
    stub.queue_verification_ok()
    result = run_anton(["--folder", str(tmp_path)], ["break everything", "exit"],
                       env=base_env(stub), timeout=cfg.timeout(45))

    assert_exit_ok(result)
    assert_not_output(result, "Traceback (most recent call last)")
    assert_output(result, "ERRORS_EXHAUSTED")
    assert any(
        "has failed 5 times in a row" in json.dumps(r.get("messages", []))
        for r in stub.requests
    ), f"Circuit-breaker message not found. Request count: {stub.request_count}"
