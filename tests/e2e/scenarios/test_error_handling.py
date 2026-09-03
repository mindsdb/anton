"""Scenario F — Error handling and graceful degradation."""

from __future__ import annotations

import http.server
import threading
import pytest

from tests.e2e.harness import (
    assert_exit_fail, assert_exit_ok, assert_not_output, assert_output,
    base_env, run_anton,
)

# ENG-1361 note for both provider-failure scenarios below.
#
# A provider failure that outlasts the retry budget now FAILS the turn instead
# of being wrapped into assistant prose, so the host can offer recovery — the
# `provider_overloaded` card on cowork-server, and the setup/retry prompt here.
# Previously the count-based path had no such terminal at all: it asked the
# model to explain the outage (a call needing the very provider that was
# failing) and, when that failed too, emitted "An unexpected error occurred:
# <message>. Please try again or rephrase your request." Rephrasing cannot
# reach an unreachable provider.
#
# Consequence for a SCRIPTED run: the CLI now asks a question, and `["hello",
# "exit"]` cannot answer it — "exit" is consumed as an invalid choice, stdin
# runs out, and click aborts with a non-zero exit. That is NOT new behaviour
# introduced here: every provider error that raises into the recovery prompt
# already did this (verified against a 401 on pristine staging, same abort).
# These two scenarios simply joined that class, which is the point of the fix.
# So the assertions below check what "gracefully" actually means — an honest
# diagnosis, an actionable choice, no traceback, and no bogus rephrase advice —
# rather than an exit code that only reflects the scripted stdin.
from tests.e2e.stub_server import StubServer



def test_invalid_provider_fails_fast(cfg, stub, tmp_path):
    env = base_env(stub)
    env["ANTON_PLANNING_PROVIDER"] = "totally-bogus-provider-xyz"
    result = run_anton(["--folder", str(tmp_path)], ["exit"],
                       env=env, timeout=cfg.timeout(15))
    assert_exit_fail(result)
    assert_output(result, "Unknown planning provider")
    assert_output(result, "totally-bogus-provider-xyz")


@pytest.mark.stub_only
def test_http_500_handled_gracefully(cfg, tmp_path):
    class _500Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()
        def log_message(self, *_): pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), _500Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        # StubServer() used only as an env-var template (provider type, api key);
        # ANTON_OPENAI_BASE_URL is overridden below to point at the custom 500 server.
        env = base_env(StubServer())
        env["ANTON_OPENAI_BASE_URL"] = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
        result = run_anton(["--folder", str(tmp_path)], ["hello", "exit"],
                           env=env, timeout=cfg.timeout(30))
    finally:
        httpd.shutdown()

    assert_exit_fail(result)   # scripted stdin cannot answer the prompt — see note above
    assert_not_output(result, "Traceback (most recent call last)")
    # ENG-673: a request-time 5xx is classified as a transient provider error and
    # (since the SDK already retried it) fails fast with the honest typed message
    # rather than the old "Server returned 500" phrasing. ENG-1361 keeps that
    # message when converting to the terminal error — the diagnosis is the part
    # a user or a support thread needs, and only the attempt count is added.
    assert_output(result, "returned 500")
    assert_output(result, "Retried 3 times without success")
    # The turn offers a real remedy instead of blaming the user's wording.
    assert_output(result, "setup/retry")
    assert_not_output(result, "rephrase")


@pytest.mark.stub_only
def test_malformed_json_handled_gracefully(cfg, tmp_path):
    class _BadJSONHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = b"not valid json {"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_): pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), _BadJSONHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        # StubServer() used only as an env-var template; see test_http_500_handled_gracefully.
        env = base_env(StubServer())
        env["ANTON_OPENAI_BASE_URL"] = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
        result = run_anton(["--folder", str(tmp_path)], ["hello", "exit"],
                           env=env, timeout=cfg.timeout(20))
    finally:
        httpd.shutdown()

    assert_exit_fail(result)   # scripted stdin cannot answer the prompt — see note above
    assert_not_output(result, "Traceback (most recent call last)")
    # Same ENG-1361 terminal as the 5xx case: an unusable response body is a
    # provider failure, so it earns a diagnosis and a choice, not prose telling
    # the user to reword a request that was never the problem.
    assert_output(result, "setup/retry")
    assert_not_output(result, "rephrase")


@pytest.mark.stub_only
def test_large_input_no_crash(cfg, stub, tmp_path):
    stub.queue_text("Got your big message.")
    stub.queue_verification_ok()
    result = run_anton(["--folder", str(tmp_path)], ["x" * 100_000, "exit"],
                       env=base_env(stub), timeout=cfg.timeout(60))
    assert_exit_ok(result)
    assert_output(result, "Got your big message.")
    assert_not_output(result, "Traceback (most recent call last)")
