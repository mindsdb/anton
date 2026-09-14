"""The live env must carry CI's credentials past the per-test scrub."""

import os

from tests.e2e import harness
from tests.e2e.harness import LiveProvider, base_env


def test_live_env_carries_credentials_past_the_scrub(monkeypatch):
    monkeypatch.setattr(harness, "_LIVE_CREDENTIALS", {"ANTON_OPENAI_API_KEY": "set-before-collection"})
    # The autouse `_no_leaked_credentials` fixture has already run for this test.
    assert "ANTON_OPENAI_API_KEY" not in os.environ
    env = base_env(LiveProvider())
    assert env["ANTON_OPENAI_API_KEY"] == "set-before-collection"


def test_live_env_test_overrides_still_win(monkeypatch):
    monkeypatch.setattr(harness, "_LIVE_CREDENTIALS", {"ANTON_OPENAI_API_KEY": "k"})
    env = base_env(LiveProvider())
    assert env["ANTON_ANALYTICS_ENABLED"] == "false"
    assert env["ANTON_TERMS_CONSENT"] == "true"
