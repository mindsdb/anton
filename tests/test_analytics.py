"""Tests for the anonymous analytics layer — the ENG-385 ``is_ci`` cohort flag.

``send_event`` fires a daemon thread doing an HTTP GET; both are stubbed here so
the built URL can be inspected deterministically without network or threads.
"""

from __future__ import annotations

import urllib.parse

import anton.analytics as analytics


class _Settings:
    analytics_enabled = True
    analytics_url = "https://example.test/collect"


def _capture_url(monkeypatch) -> list[str]:
    """Run send_event's thread target synchronously and record the GET URL."""
    captured: list[str] = []

    class _SyncThread:
        def __init__(self, target=None, args=(), daemon=None):
            self._target = target
            self._args = args

        def start(self):
            if self._target:
                self._target(*self._args)

    monkeypatch.setattr(analytics.threading, "Thread", _SyncThread)
    monkeypatch.setattr(analytics, "_fire", captured.append)
    return captured


def _query(url: str) -> dict[str, str]:
    return dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))


def test_is_ci_true_when_ci_env_set(monkeypatch):
    monkeypatch.setattr(analytics, "_cached_is_ci", None)
    monkeypatch.setenv("CI", "true")
    assert analytics._is_ci() is True


def test_is_ci_false_without_ci_env(monkeypatch):
    monkeypatch.setattr(analytics, "_cached_is_ci", None)
    for var in analytics._CI_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    assert analytics._is_ci() is False


def test_send_event_stamps_is_ci_true(monkeypatch):
    monkeypatch.setattr(analytics, "_cached_is_ci", None)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    captured = _capture_url(monkeypatch)

    analytics.send_event(_Settings(), "anton_started")

    assert len(captured) == 1
    params = _query(captured[0])
    assert params["action"] == "anton_started"
    assert params["is_ci"] == "true"


def test_send_event_stamps_is_ci_false_and_preserves_extra(monkeypatch):
    monkeypatch.setattr(analytics, "_cached_is_ci", None)
    for var in analytics._CI_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    captured = _capture_url(monkeypatch)

    analytics.send_event(_Settings(), "anton_query", llm_provider="openai")

    assert len(captured) == 1
    params = _query(captured[0])
    assert params["is_ci"] == "false"
    assert params["llm_provider"] == "openai"
