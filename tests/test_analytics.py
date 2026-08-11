"""Tests for the anonymous analytics layer (ENG-385).

CI/automation traffic is dropped entirely rather than sent. ``send_event`` fires
a daemon thread doing an HTTP GET; both are stubbed so we can assert what (if
anything) would be sent, without network or threads.
"""

from __future__ import annotations

import urllib.parse

import anton.analytics as analytics

# Markers _is_ci() consults — cleared in tests so the suite's own environment
# (it may run under GitHub Actions) doesn't leak into assertions.
_CI_MARKERS = (
    "ANTON_IS_CI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "BUILDKITE",
    "CIRCLECI",
    "TF_BUILD",
    "JENKINS_URL",
)


class _Settings:
    analytics_enabled = True
    analytics_url = "https://example.test/collect"


def _clear_ci(monkeypatch):
    monkeypatch.setattr(analytics, "_cached_is_ci", None)
    for var in _CI_MARKERS:
        monkeypatch.delenv(var, raising=False)


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


def test_is_ci_true_with_explicit_anton_flag(monkeypatch):
    _clear_ci(monkeypatch)
    monkeypatch.setenv("ANTON_IS_CI", "true")
    assert analytics._is_ci() is True


def test_is_ci_true_with_github_actions(monkeypatch):
    _clear_ci(monkeypatch)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert analytics._is_ci() is True


def test_is_ci_ignores_bare_ci_false(monkeypatch):
    # A stray `CI=false` (or a leaked `CI`) must not classify as CI — the bare
    # `CI` var is intentionally not consulted.
    _clear_ci(monkeypatch)
    monkeypatch.setenv("CI", "false")
    assert analytics._is_ci() is False


def test_is_ci_false_without_markers(monkeypatch):
    _clear_ci(monkeypatch)
    assert analytics._is_ci() is False


def test_send_event_dropped_in_ci(monkeypatch):
    _clear_ci(monkeypatch)
    monkeypatch.setenv("ANTON_IS_CI", "true")
    captured = _capture_url(monkeypatch)

    analytics.send_event(_Settings(), "anton_started")

    assert captured == []  # CI traffic is dropped, never sent


def test_send_event_sends_when_not_ci(monkeypatch):
    _clear_ci(monkeypatch)
    captured = _capture_url(monkeypatch)

    analytics.send_event(_Settings(), "anton_query", llm_provider="openai")

    assert len(captured) == 1
    params = _query(captured[0])
    assert params["action"] == "anton_query"
    assert params["llm_provider"] == "openai"
    assert "is_ci" not in params  # flag removed; CI events aren't sent at all


def test_fire_sends_an_identifying_user_agent(monkeypatch):
    """urllib's default ``Python-urllib/3.x`` agent is answered with 403 by the
    bot rules on the mindshub.ai zone, and ``_fire`` throws its response away,
    so an event blocked for looking like a script would vanish with no trace.
    """
    seen: list[object] = []

    def _fake_urlopen(request, timeout=None):
        seen.append(request)
        return None

    monkeypatch.setattr(analytics.urllib.request, "urlopen", _fake_urlopen)

    analytics._fire("https://example.test/collect?action=anton_query")

    assert len(seen) == 1
    agent = seen[0].get_header("User-agent")
    assert agent == analytics._USER_AGENT
    assert "python-urllib" not in agent.lower()


def test_default_collector_url_is_a_host_we_control():
    """Guards the regression this replaced: the previous default was a raw
    ``*.execute-api.amazonaws.com`` id in an account nobody could deploy to, so
    the endpoint could not be changed without shipping a new release to every
    install.
    """
    from anton.config.settings import AntonSettings

    url = AntonSettings.model_fields["analytics_url"].default
    assert url.startswith("https://collect.")
    assert "execute-api" not in url
