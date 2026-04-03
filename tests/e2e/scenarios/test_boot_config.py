"""Scenario A — CLI boot and config."""

from __future__ import annotations

import pytest

from scripts.e2e.harness import (
    LiveProvider, assert_exit_fail, assert_exit_ok, assert_not_output,
    assert_output, base_env, run_anton,
)


def test_clean_boot(cfg, stub, tmp_path):
    # No stub response queued: "exit" breaks the loop before any LLM call is made.
    result = run_anton(["--folder", str(tmp_path)], ["exit"],
                       env=base_env(stub), timeout=cfg.timeout(20))
    assert_exit_ok(result)
    assert_output(result, "exit' to quit")
    assert_not_output(result, "Traceback (most recent call last)")


def test_sessions_subcommand_empty_listing(cfg, stub, tmp_path):
    # run_chat writes to HistoryStore, not SessionStore; the sessions command
    # reads SessionStore, so the listing is always empty under this harness.
    stub.queue_text("Hi there!")
    stub.queue_verification_ok()
    setup_result = run_anton(["--folder", str(tmp_path)], ["hello", "exit"],
                             env=base_env(stub), timeout=cfg.timeout(20))
    assert_exit_ok(setup_result)
    assert_not_output(setup_result, "Traceback (most recent call last)")

    result = run_anton(["--folder", str(tmp_path), "sessions"], [],
                       env=base_env(stub), timeout=cfg.timeout(10))
    assert_exit_ok(result)
    assert_output(result, "No sessions found.")
    assert_not_output(result, "Traceback (most recent call last)")


def test_version_subcommand(cfg):
    # version needs no provider — avoid spinning up a stub server
    result = run_anton(["version"], [], env=base_env(LiveProvider()), timeout=cfg.timeout(10))
    assert_exit_ok(result)
    assert_output(result, "Anton v")
    assert_not_output(result, "Traceback (most recent call last)")


def test_missing_api_key_triggers_onboarding(cfg, stub, tmp_path):
    """Without an API key the app shows the onboarding intro (non-TTY static fallback)."""
    env = base_env(stub)
    env["ANTON_OPENAI_API_KEY"] = ""
    env["ANTON_FIRST_RUN_DONE"] = "true"   # prevent _agent_zero from consuming stdin
    env.pop("OPENAI_API_KEY", None)
    result = run_anton(["--folder", str(tmp_path)], [""],
                       env=env, timeout=cfg.timeout(15))
    assert_exit_ok(result)
    assert_output(result, "Hi Boss!")
    assert_not_output(result, "Traceback (most recent call last)")
    assert stub.request_count == 0, \
        f"Expected no LLM requests during onboarding with blank input, got {stub.request_count}"


def test_invalid_provider_exits_with_error(cfg, stub, tmp_path):
    env = base_env(stub)
    env["ANTON_PLANNING_PROVIDER"] = "nonexistent-provider-xyz"
    result = run_anton(["--folder", str(tmp_path)], ["exit"],
                       env=env, timeout=cfg.timeout(15))
    assert_exit_fail(result)
    assert_output(result, "Unknown planning provider")
    assert_output(result, "nonexistent-provider-xyz")


@pytest.mark.stub_only
def test_planning_model_env_var_used(cfg, stub, tmp_path):
    """ANTON_PLANNING_MODEL is sent as the model field in the HTTP request."""
    stub.queue_text("All good.")
    env = base_env(stub)
    env["ANTON_PLANNING_MODEL"] = "my-custom-model-for-test"
    result = run_anton(["--folder", str(tmp_path)], ["hello", "exit"],
                       env=env, timeout=cfg.timeout(20))
    assert_exit_ok(result)
    assert_not_output(result, "Traceback (most recent call last)")
    assert stub.request_count >= 1, "Expected at least one request to reach the stub"
    assert any(r.get("model") == "my-custom-model-for-test" for r in stub.requests), \
        f"Expected model 'my-custom-model-for-test' in requests, got: {[r.get('model') for r in stub.requests]}"
