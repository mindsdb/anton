"""Scenario I — Circuit breaker evasion via alternating error/success."""

from __future__ import annotations

import json
import pytest

from anton.core.session import _ROUND_CAP_GRACE_ROUNDS
from tests.e2e.harness import (
    assert_exit_ok, assert_not_output, assert_output, base_env, run_anton,
)


_BAD_CODE = "def broken(:\n    pass\n"
_GOOD_CODE = "print('ok')\n"
_MAX_TOOL_ROUNDS = 25
_EFFECTIVE_CAP = _MAX_TOOL_ROUNDS + _ROUND_CAP_GRACE_ROUNDS  # ENG-1893 grace


@pytest.mark.stub_only
def test_alternating_errors_evade_circuit_breaker(cfg, stub, tmp_path):
    """Alternating error/success keeps streak <=1 — MAX_TOOL_ROUNDS is the only backstop."""
    # Backstop fires at effective_cap + 1 (the one-time grace extension counts
    # too, see _EFFECTIVE_CAP) — exactly that many tool-call slots, or the
    # extra queued response gets consumed by the hand-back diagnosis call
    # instead of the text reply below. Pairs keep the error streak at ≤1 per
    # tool; a trailing odd slot rounds up to effective_cap + 1.
    pairs, odd_one = divmod(_EFFECTIVE_CAP + 1, 2)
    for i in range(pairs):
        stub.queue_tool_call("scratchpad", {"action": "exec", "name": f"bad_{i}", "code": _BAD_CODE})
        stub.queue_tool_call("scratchpad", {"action": "exec", "name": f"good_{i}", "code": _GOOD_CODE})
    if odd_one:
        stub.queue_tool_call("scratchpad", {"action": "exec", "name": f"bad_{pairs}", "code": _BAD_CODE})
    stub.queue_text("Max rounds hit. ROUNDS_EXHAUSTED")
    result = run_anton(["--folder", str(tmp_path)], ["keep trying", "exit"],
                       env=base_env(stub), timeout=cfg.timeout(60))

    assert_exit_ok(result)
    assert_not_output(result, "Traceback (most recent call last)")
    assert_output(result, "ROUNDS_EXHAUSTED")

    all_messages = json.dumps([r.get("messages", []) for r in stub.requests])
    assert f"You have used {_EFFECTIVE_CAP} tool-call rounds" in all_messages, \
        f"Max-rounds message not found. Request count: {stub.request_count}"
    assert "has failed 5 times in a row" not in all_messages, \
        "Circuit breaker fired unexpectedly"
    assert "failed twice in a row" not in all_messages, \
        "Resilience nudge fired unexpectedly"
