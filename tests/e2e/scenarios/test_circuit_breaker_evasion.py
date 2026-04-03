"""Scenario I — Circuit breaker evasion via alternating error/success."""

from __future__ import annotations

import json
import pytest

from scripts.e2e.harness import (
    assert_exit_ok, assert_not_output, assert_output, base_env, run_anton,
)


_BAD_CODE = "def broken(:\n    pass\n"
_GOOD_CODE = "print('ok')\n"


@pytest.mark.stub_only
def test_alternating_errors_evade_circuit_breaker(cfg, stub, tmp_path):
    """Alternating error/success keeps streak <=1 — MAX_TOOL_ROUNDS is the only backstop."""
    # _MAX_TOOL_ROUNDS = 25; backstop fires at round 26 (> 25).
    # 13 bad+good pairs = 26 tool-call slots, keeping the error streak at ≤1 per tool.
    for i in range(13):
        stub.queue_tool_call("scratchpad", {"action": "exec", "name": f"bad_{i}", "code": _BAD_CODE})
        stub.queue_tool_call("scratchpad", {"action": "exec", "name": f"good_{i}", "code": _GOOD_CODE})
    stub.queue_text("Max rounds hit. ROUNDS_EXHAUSTED")
    result = run_anton(["--folder", str(tmp_path)], ["keep trying", "exit"],
                       env=base_env(stub), timeout=cfg.timeout(60))

    assert_exit_ok(result)
    assert_not_output(result, "Traceback (most recent call last)")
    assert_output(result, "ROUNDS_EXHAUSTED")

    all_messages = json.dumps([r.get("messages", []) for r in stub.requests])
    assert "You have used 25 tool-call rounds" in all_messages, \
        f"Max-rounds message not found. Request count: {stub.request_count}"
    assert "has failed 5 times in a row" not in all_messages, \
        "Circuit breaker fired unexpectedly"
    assert "failed twice in a row" not in all_messages, \
        "Resilience nudge fired unexpectedly"
