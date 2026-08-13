"""Tests for the anonymous analytics layer (ENG-385).

CI/automation traffic is dropped entirely rather than sent. ``send_event`` fires
a daemon thread doing an HTTP GET; both are stubbed so we can assert what (if
anything) would be sent, without network or threads.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

import pytest

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


# ── The PostHog direct sink (ENG-1495) ──────────────────────────────
# `turn_completed` bypasses the collector and posts straight to PostHog. The
# collector allowlists five property names and relays only `anton_`/`ds_connect`
# actions, returning HTTP 200 either way, so every one of these assertions
# guards something that previously failed invisibly.


class _PosthogSettings(_Settings):
    posthog_host = "https://ph.example.test"
    posthog_key = "phc_test"


def _capture_posthog(monkeypatch) -> list[tuple[str, dict]]:
    """Run send_event's thread target synchronously; record (url, body).

    Also fails the test outright if the collector path is taken, so "went to
    the wrong sink" can never read as "sent nothing".
    """
    captured: list[tuple[str, dict]] = []

    class _SyncThread:
        def __init__(self, target=None, args=(), daemon=None):
            self._target = target
            self._args = args

        def start(self):
            if self._target:
                self._target(*self._args)

    monkeypatch.setattr(analytics.threading, "Thread", _SyncThread)
    monkeypatch.setattr(
        analytics,
        "_fire_posthog",
        lambda url, body: captured.append((url, json.loads(body))),
    )
    monkeypatch.setattr(
        analytics,
        "_fire",
        lambda url: pytest.fail(f"took the collector path instead: {url}"),
    )
    return captured


def test_turn_completed_goes_to_posthog_not_the_collector(monkeypatch):
    """The whole point: this event must not touch the dropping collector."""
    _clear_ci(monkeypatch)
    captured = _capture_posthog(monkeypatch)

    analytics.send_event(_PosthogSettings(), "turn_completed", tokens_total="24970")

    assert len(captured) == 1
    url, body = captured[0]
    assert url == "https://ph.example.test/capture/"
    assert body["api_key"] == "phc_test"
    assert body["event"] == "turn_completed"


def test_unregistered_events_still_use_the_collector(monkeypatch):
    """Scope guard. Only names in `_POSTHOG_EVENTS` move; nothing else does.

    Without this, a change that routed everything would look identical to a
    change that routed one event — and would drag the ~105k/30d ds_connect
    volume into the production project.
    """
    _clear_ci(monkeypatch)
    captured = _capture_url(monkeypatch)  # asserts on the collector GET
    monkeypatch.setattr(
        analytics,
        "_fire_posthog",
        lambda url, body: pytest.fail("ds_connect_success must stay on the collector"),
    )

    analytics.send_event(_PosthogSettings(), "ds_connect_success", engine="postgres")

    assert len(captured) == 1
    assert _query(captured[0])["action"] == "ds_connect_success"


def test_person_profile_is_disabled(monkeypatch):
    """Load-bearing: without this flag every install fingerprint becomes a
    PostHog *Person* in project 424726, which is keyed on the Keycloak `sub`.
    Machine fingerprints would then be counted alongside real identified users
    in every person metric and cohort there, and no later query can separate
    them again. Deliberately unquantified — the ratio this docstring used to
    give could not be reproduced from the description it carried."""
    _clear_ci(monkeypatch)
    captured = _capture_posthog(monkeypatch)

    analytics.send_event(_PosthogSettings(), "turn_completed")

    assert captured[0][1]["properties"]["$process_person_profile"] is False


def test_distinct_id_is_the_install_fingerprint(monkeypatch):
    _clear_ci(monkeypatch)
    monkeypatch.setattr(analytics, "_cached_aid", "abc123def456abcd")
    captured = _capture_posthog(monkeypatch)

    analytics.send_event(_PosthogSettings(), "turn_completed")

    _, body = captured[0]
    assert body["distinct_id"] == "abc123def456abcd"
    # Kept as a property too, so a query can group on the install without
    # reaching into person data.
    assert body["properties"]["aid"] == "abc123def456abcd"


def test_every_property_survives(monkeypatch):
    """The collector allowlisted five names and dropped the other 26. The
    direct path must carry all of them, so assert on ones the collector is
    known to have eaten (`tokens_total`, `ended_by`) alongside one it passed
    (`llm_provider`)."""
    _clear_ci(monkeypatch)
    captured = _capture_posthog(monkeypatch)

    analytics.send_event(
        _PosthogSettings(),
        "turn_completed",
        tokens_total="24970",
        ended_by="round_cap",
        llm_provider="anthropic",
        planning_model="haiku",
        harness="cli",
        conversation_id="conv-1",
        turn_index="3",
    )

    props = captured[0][1]["properties"]
    assert props["tokens_total"] == "24970"
    assert props["ended_by"] == "round_cap"      # the Phase 0 criterion-3 field
    assert props["llm_provider"] == "anthropic"
    assert props["planning_model"] == "haiku"
    assert props["harness"] == "cli"
    assert props["conversation_id"] == "conv-1"
    assert props["turn_index"] == "3"


def test_transport_noise_is_not_a_property(monkeypatch):
    """`_` is a GET cache buster and `action`/`timestamp` are top-level fields
    on the Capture API. Leaking them into properties would create three
    permanent junk property definitions in the project."""
    _clear_ci(monkeypatch)
    captured = _capture_posthog(monkeypatch)

    analytics.send_event(_PosthogSettings(), "turn_completed")

    _, body = captured[0]
    props = body["properties"]
    assert "_" not in props
    assert "action" not in props
    assert "timestamp" not in props
    # Promoted to PostHog's own field: the turn's end, not the daemon thread's
    # eventual send.
    assert body["timestamp"]
    assert body["properties"]["$lib"] == "anton-library"


def test_empty_key_disables_the_direct_sink_without_falling_back(monkeypatch):
    """`ANTON_POSTHOG_KEY=""` is the documented opt-out. It must not silently
    reroute the event onto the collector, which would drop it anyway."""
    _clear_ci(monkeypatch)

    class _NoKey(_PosthogSettings):
        posthog_key = ""

    captured = _capture_posthog(monkeypatch)  # fails the test on collector use

    analytics.send_event(_NoKey(), "turn_completed")

    assert captured == []


def test_turn_completed_still_dropped_in_ci(monkeypatch):
    """The CI gate has to apply to the new path too — it is how anton keeps
    test runs out of the funnel, and there is no `environment` tag to filter
    on afterwards."""
    _clear_ci(monkeypatch)
    monkeypatch.setenv("ANTON_IS_CI", "true")
    captured = _capture_posthog(monkeypatch)

    analytics.send_event(_PosthogSettings(), "turn_completed")

    assert captured == []


def test_turn_completed_respects_the_opt_out(monkeypatch):
    _clear_ci(monkeypatch)

    class _OptedOut(_PosthogSettings):
        analytics_enabled = False

    captured = _capture_posthog(monkeypatch)

    analytics.send_event(_OptedOut(), "turn_completed")

    assert captured == []


def _capture_posthog_request(monkeypatch, status=None) -> list:
    """Record the Request `_fire_posthog` builds.

    `status` raises `HTTPError` the way real urllib does on a 4xx/5xx — it does
    NOT return a response object carrying a status. An earlier version of this
    helper returned a fake response, which made the rejection test pass against
    behaviour urllib does not have.
    """
    seen: list[urllib.request.Request] = []

    def _urlopen(req, timeout=None):
        seen.append(req)
        if status is not None:
            raise urllib.error.HTTPError(
                req.full_url, status, "rejected", hdrs=None, fp=None
            )
        return None

    monkeypatch.setattr(analytics.urllib.request, "urlopen", _urlopen)
    return seen


def test_posthog_request_is_a_post_with_json(monkeypatch):
    """Exercises the real `_fire_posthog` rather than a stub: a GET would land
    the whole payload in gateway and CDN access logs, which is exactly how the
    sibling endpoint ended up logging email addresses."""
    seen = _capture_posthog_request(monkeypatch)

    analytics._fire_posthog("https://ph.example.test/capture/", b'{"a":1}')

    assert len(seen) == 1
    assert seen[0].method == "POST"
    assert seen[0].get_header("Content-type") == "application/json"
    assert "python-urllib" not in seen[0].get_header("User-agent", "").lower()


def test_posthog_rejection_logs_the_status_code(monkeypatch, caplog):
    """A rejection must name its status, and still not raise.

    The stub **raises** `HTTPError`, which is what real urllib does on a 4xx —
    it does not return a response whose `.status` can be read. That distinction
    is the whole point of this test: a previous version asserted against a
    fake that returned, so it passed while the code under it could never run.

    What this can and cannot buy, measured against the live endpoint on
    2026-08-13: a **bogus api_key returns HTTP 200** `{"status":"Ok"}`, so a
    wrong/rotated/revoked token is NOT detectable here at all. The 401 exists
    only for an absent key, which `send_event` short-circuits before building a
    request. Only malformed payloads and transport failures are visible. The
    zero-volume alert stays the only possible detector for a token problem.
    """
    seen = _capture_posthog_request(monkeypatch, status=400)

    with caplog.at_level("DEBUG", logger=analytics.logger.name):
        analytics._fire_posthog("https://ph.example.test/capture/", b'{"bad":1}')

    assert len(seen) == 1  # still attempted; never raises
    # The code, not just a traceback — someone reading a user's logs needs it.
    assert any("400" in r.getMessage() for r in caplog.records), caplog.text


def test_posthog_transport_failure_never_raises(monkeypatch):
    """Analytics must never break a turn — the guarantee the module docstring
    makes. A raising urlopen has to be swallowed."""
    def _boom(req, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr(analytics.urllib.request, "urlopen", _boom)

    analytics._fire_posthog("https://ph.example.test/capture/", b'{"a":1}')  # no raise


def test_posthog_malformed_host_never_raises():
    """`Request()` itself raises ValueError on a malformed URL, and the host is
    a user-supplied setting (`ANTON_POSTHOG_HOST`).

    This runs in a daemon thread, *outside* `send_event`'s guard, so anything
    escaping reaches the user as a traceback mid-session. A previous version of
    `_fire_posthog` built the Request above the `try` and did exactly that.
    """
    analytics._fire_posthog("garbage-not-a-url/capture/", b'{"a":1}')  # no raise


def test_send_event_with_a_bad_posthog_host_kills_no_thread(monkeypatch):
    """End to end through the real path: a bad host must not surface at all."""
    _clear_ci(monkeypatch)

    class _BadHost(_PosthogSettings):
        posthog_host = "garbage-not-a-url"

    escaped: list[str] = []

    class _SyncThread:
        def __init__(self, target=None, args=(), daemon=None):
            self._target, self._args = target, args

        def start(self):
            try:
                self._target(*self._args)
            except BaseException as exc:  # what threading would report and print
                escaped.append(type(exc).__name__)

    monkeypatch.setattr(analytics.threading, "Thread", _SyncThread)

    analytics.send_event(_BadHost(), "turn_completed", tokens_total="1")

    assert escaped == [], f"escaped the daemon thread: {escaped}"


def test_rule_retrieval_goes_to_posthog(monkeypatch):
    """ENG-1390's signal, on the same path (ENG-1495).

    It shipped as `rule_retrieval` — no `anton_` prefix — so the collector
    dropped it on the action filter, exactly as it dropped `turn_completed`.
    anton#332 proposed renaming it to `anton_rule_retrieval` to satisfy that
    filter; this makes the rename unnecessary, and the prefix would have been
    wrong in a project whose other events carry no prefix at all.
    """
    _clear_ci(monkeypatch)
    captured = _capture_posthog(monkeypatch)

    analytics.send_event(
        _PosthogSettings(),
        "rule_retrieval",
        outcome="ok",
        when_rules="12",
        kept_rules="3",
        rules_chars="1840",
        stop_reason="end_turn",
        duration_ms="412",
    )

    assert len(captured) == 1
    _, body = captured[0]
    # The name stays unprefixed: nothing filters on it any more, and the other
    # emitters in project 424726 (first_query, agent_session_started) carry no
    # prefix either.
    assert body["event"] == "rule_retrieval"
    props = body["properties"]
    # `outcome` and `stop_reason` are the two the collector's five-name
    # allowlist would have eaten — they are the whole point of the signal.
    assert props["outcome"] == "ok"
    assert props["stop_reason"] == "end_turn"
    assert props["kept_rules"] == "3"
    assert props["$process_person_profile"] is False
