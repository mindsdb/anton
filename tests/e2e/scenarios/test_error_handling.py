"""Scenario F — Error handling and graceful degradation."""

from __future__ import annotations

import http.server
import threading
import pytest

from tests.e2e.harness import (
    assert_exit_fail, assert_exit_ok, assert_not_output, assert_output,
    base_env, run_anton,
)
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

    assert_exit_ok(result)
    assert_not_output(result, "Traceback (most recent call last)")
    # After 3 retries the fallback text surfaces the ConnectionError message (openai.py:364)
    assert_output(result, "Server returned 500")


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

    assert_exit_ok(result)
    assert_not_output(result, "Traceback (most recent call last)")


@pytest.mark.stub_only
def test_large_input_no_crash(cfg, stub, tmp_path):
    stub.queue_text("Got your big message.")
    stub.queue_verification_ok()
    result = run_anton(["--folder", str(tmp_path)], ["x" * 100_000, "exit"],
                       env=base_env(stub), timeout=cfg.timeout(60))
    assert_exit_ok(result)
    assert_not_output(result, "Traceback (most recent call last)")
    assert_output(result, "Got your big message.")
