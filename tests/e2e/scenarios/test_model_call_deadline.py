"""Scenario: the model-call deadline in the real CLI.

`ANTON_MODEL_CALL_IDLE_TIMEOUT_S` bounds how long one model call may send
nothing. The stub holds a call silent with `: keepalive` comments, which the
OpenAI SDK drops, so only the deadline can end it. A call that keeps writing
past the deadline must answer.
"""

from __future__ import annotations

import pytest

from tests.e2e.harness import (
    assert_exit_fail, assert_exit_ok, assert_not_output, assert_output, base_env, run_anton,
)


def _env(stub) -> dict[str, str]:
    env = base_env(stub)
    env["ANTON_MODEL_CALL_IDLE_TIMEOUT_S"] = "1"
    return env


@pytest.mark.stub_only
def test_a_silent_call_ends_with_the_no_output_message(cfg, stub, tmp_path):
    stub.script(streamed=["hold:3:HELD_ANSWER_MARKER"], keepalive_s=0.2)

    result = run_anton(["--folder", str(tmp_path)], ["hello", "exit"],
                       env=_env(stub), timeout=cfg.timeout(30))

    assert_exit_fail(result)  # scripted stdin cannot answer the setup/retry prompt
    assert_output(result, "The model sent no output for 1 second")
    assert_output(result, "setup/retry")
    assert_not_output(result, "HELD_ANSWER_MARKER")
    assert_not_output(result, "Traceback (most recent call last)")
    streamed = [r for r in stub.request_records if r.stream]
    assert len(streamed) == 1, f"the turn re-sent the silent call: {streamed}"


@pytest.mark.stub_only
def test_a_call_that_keeps_writing_is_not_cut(cfg, stub, tmp_path):
    stub.script(streamed=["trickle:2:0.2"], keepalive_s=0)

    result = run_anton(["--folder", str(tmp_path)], ["hello", "exit"],
                       env=_env(stub), timeout=cfg.timeout(30))

    assert_exit_ok(result)
    assert_output(result, "tick 8")
    assert_not_output(result, "sent no output")
