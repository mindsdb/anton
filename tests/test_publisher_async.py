"""anton.publisher against the async /upload contract (ENG-1580).

The server may answer POST /upload with 202 {job_id, report_id, ...} when the
client sends "async": true; publish() then polls GET /upload/jobs/{job_id} and
returns the job's result as if the upload had been synchronous.
"""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import pytest

from anton import publisher
from anton.publisher import PublishJobFailed, PublishJobTimeout, publish


class _Resp(io.BytesIO):
    def __init__(self, status, body):
        super().__init__(json.dumps(body).encode())
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _metadata(artifact_id: str, artifact_type: str, primary: str) -> str:
    """A metadata.json that passes Artifact.model_validate (32-hex id, all
    required fields). An invalid one silently degrades to a static directory
    publish and the fullstack branch is never exercised."""
    return json.dumps({
        "schemaVersion": 1,
        "id": artifact_id,
        "slug": "app",
        "createdAt": "2026-01-01T00:00:00Z",
        "updatedAt": "2026-01-01T00:00:00Z",
        "name": "App",
        "description": "",
        "type": artifact_type,
        "primary": primary,
    })


def _fullstack(tmp_path: Path) -> Path:
    art = tmp_path / "app"
    (art / "static").mkdir(parents=True)
    (art / "metadata.json").write_text(_metadata("a" * 32, "fullstack-stateless-app", "static/index.html"))
    (art / "backend.py").write_text("print(1)")
    (art / "static" / "index.html").write_text("<html></html>")
    return art


def _static(tmp_path: Path) -> Path:
    f = tmp_path / "report.html"
    f.write_text("<html></html>")
    return f


def _urlopen(responses):
    """Sequence of (status, body) or Exception; records each request."""
    calls = []
    it = iter(responses)

    def fake(req, context=None, timeout=None):
        calls.append(req)
        nxt = next(it)
        if isinstance(nxt, Exception):
            raise nxt
        return _Resp(*nxt)

    return fake, calls


def _body(req):
    return json.loads(req.data.decode())


def test_fullstack_sends_async_flag_and_polls_until_done(tmp_path):
    fake, calls = _urlopen([
        (202, {"job_id": "a" * 32, "report_id": "rid", "status": "queued",
               "status_url": "https://elsewhere/upload/jobs/" + "a" * 32, "poll_after_s": 3}),
        (200, {"job_id": "a" * 32, "status": "running"}),
        (200, {"job_id": "a" * 32, "status": "done", "result": {"report_id": "rid", "view_url": "https://v/a/rid"}}),
    ])
    sleeps = []
    with mock.patch.object(urllib.request, "urlopen", fake), \
            mock.patch.object(publisher, "_collect_datasource_secrets", return_value=({}, [])), \
            mock.patch.object(publisher.time, "sleep", sleeps.append):
        result = publish(_fullstack(tmp_path), api_key="k", publish_url="https://pub.test")
    assert result == {"report_id": "rid", "view_url": "https://v/a/rid"}
    assert calls[0].full_url == "https://pub.test/upload" and calls[0].get_method() == "POST"
    # Guard against the metadata silently failing validation: a static
    # directory publish would carry no artifact_type at all.
    assert _body(calls[0])["artifact_type"] == "fullstack-stateless-app"
    assert _body(calls[0])["async"] is True
    # Polls the publish_url host, not status_url; GET with no body.
    assert [c.full_url for c in calls[1:]] == ["https://pub.test/upload/jobs/" + "a" * 32] * 2
    assert all(c.get_method() == "GET" and c.data is None for c in calls[1:])
    assert sleeps == [3.0, 3.0]


def test_static_does_not_send_async_flag_and_handles_200(tmp_path):
    fake, calls = _urlopen([(200, {"report_id": "r8", "view_url": "https://v/view/u/r8"})])
    with mock.patch.object(urllib.request, "urlopen", fake):
        result = publish(_static(tmp_path), api_key="k", publish_url="https://pub.test")
    assert result["report_id"] == "r8"
    assert "async" not in _body(calls[0])
    assert len(calls) == 1


def test_fullstack_200_from_old_server_is_returned_as_is(tmp_path):
    fake, calls = _urlopen([(200, {"report_id": "rid", "view_url": "u"})])
    with mock.patch.object(urllib.request, "urlopen", fake), \
            mock.patch.object(publisher, "_collect_datasource_secrets", return_value=({}, [])):
        assert publish(_fullstack(tmp_path), api_key="k", publish_url="https://pub.test")["report_id"] == "rid"
    assert len(calls) == 1


def test_failed_job_raises_publish_job_failed_with_server_code_and_text(tmp_path):
    fake, _ = _urlopen([
        (202, {"job_id": "b" * 32, "report_id": "rid", "status": "queued", "poll_after_s": 3}),
        (200, {"job_id": "b" * 32, "status": "failed",
               "error": {"status_code": 400, "message": "pip install failed: no wheel"}}),
    ])
    with mock.patch.object(urllib.request, "urlopen", fake), \
            mock.patch.object(publisher, "_collect_datasource_secrets", return_value=({}, [])), \
            mock.patch.object(publisher.time, "sleep", lambda s: None), \
            pytest.raises(PublishJobFailed) as ei:
        publish(_fullstack(tmp_path), api_key="k", publish_url="https://pub.test")
    exc = ei.value
    assert exc.status_code == 400 and exc.job_id == "b" * 32 and exc.report_id == "rid"
    assert str(exc) == "Publish failed (HTTP 400): pip install failed: no wheel [job " + "b" * 32 + "]"


def test_budget_exhausted_raises_publish_job_timeout(tmp_path):
    running = (200, {"job_id": "c" * 32, "status": "running"})
    fake, calls = _urlopen([(202, {"job_id": "c" * 32, "report_id": "rid", "status": "queued", "poll_after_s": 3})]
                           + [running] * 100)
    clock = iter(range(0, 10_000, 30))  # each monotonic() read advances 30s
    with mock.patch.object(urllib.request, "urlopen", fake), \
            mock.patch.object(publisher, "_collect_datasource_secrets", return_value=({}, [])), \
            mock.patch.object(publisher.time, "sleep", lambda s: None), \
            mock.patch.object(publisher, "_monotonic", lambda: next(clock)), \
            pytest.raises(PublishJobTimeout) as ei:
        publish(_fullstack(tmp_path), api_key="k", publish_url="https://pub.test", job_budget_s=90)
    assert ei.value.job_id == "c" * 32 and ei.value.report_id == "rid"
    assert "job " + "c" * 32 in str(ei.value) and "report rid" in str(ei.value)
    assert len(calls) < 10  # stopped by the clock, not by exhausting responses


def test_transient_poll_error_is_retried_within_budget(tmp_path):
    fake, calls = _urlopen([
        (202, {"job_id": "d" * 32, "report_id": "rid", "status": "queued", "poll_after_s": 3}),
        urllib.error.URLError("timed out"),
        (200, {"job_id": "d" * 32, "status": "done", "result": {"report_id": "rid"}}),
    ])
    with mock.patch.object(urllib.request, "urlopen", fake), \
            mock.patch.object(publisher, "_collect_datasource_secrets", return_value=({}, [])), \
            mock.patch.object(publisher.time, "sleep", lambda s: None):
        assert publish(_fullstack(tmp_path), api_key="k", publish_url="https://pub.test")["report_id"] == "rid"
    assert len(calls) == 3


def test_poll_404_fails_fast_without_waiting_the_budget(tmp_path):
    fake, calls = _urlopen([
        (202, {"job_id": "f" * 32, "report_id": "rid", "status": "queued", "poll_after_s": 3}),
        urllib.error.HTTPError("u", 404, "Not Found", {}, io.BytesIO(b'{"error":"Job not found"}')),
    ])
    with mock.patch.object(urllib.request, "urlopen", fake), \
            mock.patch.object(publisher, "_collect_datasource_secrets", return_value=({}, [])), \
            mock.patch.object(publisher.time, "sleep", lambda s: None), \
            pytest.raises(PublishJobFailed) as ei:
        publish(_fullstack(tmp_path), api_key="k", publish_url="https://pub.test")
    assert ei.value.status_code == 404 and ei.value.job_id == "f" * 32
    assert len(calls) == 2


def test_poll_5xx_is_retried(tmp_path):
    fake, calls = _urlopen([
        (202, {"job_id": "9" * 32, "report_id": "rid", "status": "queued", "poll_after_s": 3}),
        urllib.error.HTTPError("u", 502, "Bad Gateway", {}, io.BytesIO(b"")),
        (200, {"job_id": "9" * 32, "status": "done", "result": {"report_id": "rid"}}),
    ])
    with mock.patch.object(urllib.request, "urlopen", fake), \
            mock.patch.object(publisher, "_collect_datasource_secrets", return_value=({}, [])), \
            mock.patch.object(publisher.time, "sleep", lambda s: None):
        assert publish(_fullstack(tmp_path), api_key="k", publish_url="https://pub.test")["report_id"] == "rid"
    assert len(calls) == 3


def test_poll_interval_is_clamped_and_never_exceeds_budget(tmp_path):
    """A server-suggested poll_after_s far above the 15s ceiling must be
    clamped, and the loop must never sleep past the remaining budget."""
    running = (200, {"job_id": "1" * 32, "status": "running"})
    fake, calls = _urlopen(
        [(202, {"job_id": "1" * 32, "report_id": "rid", "status": "queued", "poll_after_s": 600})]
        + [running] * 1000
    )
    clock = {"t": 0.0}
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        clock["t"] += s

    with mock.patch.object(urllib.request, "urlopen", fake), \
            mock.patch.object(publisher, "_collect_datasource_secrets", return_value=({}, [])), \
            mock.patch.object(publisher.time, "sleep", fake_sleep), \
            mock.patch.object(publisher, "_monotonic", lambda: clock["t"]), \
            pytest.raises(PublishJobTimeout):
        publish(_fullstack(tmp_path), api_key="k", publish_url="https://pub.test", job_budget_s=90)
    assert sleeps  # at least one poll happened
    assert all(s <= 15.0 for s in sleeps)
    assert sum(sleeps) <= 90


@pytest.mark.parametrize("bad_poll_after_s", [-5, "soon"])
def test_negative_or_garbage_poll_after_s_falls_back(tmp_path, bad_poll_after_s):
    fake, calls = _urlopen([
        (202, {"job_id": "2" * 32, "report_id": "rid", "status": "queued", "poll_after_s": bad_poll_after_s}),
        (200, {"job_id": "2" * 32, "status": "done", "result": {"report_id": "rid"}}),
    ])
    sleeps = []
    with mock.patch.object(urllib.request, "urlopen", fake), \
            mock.patch.object(publisher, "_collect_datasource_secrets", return_value=({}, [])), \
            mock.patch.object(publisher.time, "sleep", sleeps.append):
        publish(_fullstack(tmp_path), api_key="k", publish_url="https://pub.test")
    assert sleeps[0] == 3.0


def test_poll_interval_updates_from_status_body(tmp_path):
    fake, calls = _urlopen([
        (202, {"job_id": "3" * 32, "report_id": "rid", "status": "queued", "poll_after_s": 3}),
        (200, {"job_id": "3" * 32, "status": "running", "poll_after_s": 7}),
        (200, {"job_id": "3" * 32, "status": "done", "result": {"report_id": "rid"}}),
    ])
    sleeps = []
    with mock.patch.object(urllib.request, "urlopen", fake), \
            mock.patch.object(publisher, "_collect_datasource_secrets", return_value=({}, [])), \
            mock.patch.object(publisher.time, "sleep", sleeps.append):
        publish(_fullstack(tmp_path), api_key="k", publish_url="https://pub.test")
    assert sleeps == [3.0, 7.0]


def test_timeout_checked_before_sleeping(tmp_path):
    """When the deadline has already passed by the time a poll response
    comes back, the next loop iteration must raise WITHOUT issuing another
    status request."""
    fake, calls = _urlopen([
        (202, {"job_id": "4" * 32, "report_id": "rid", "status": "queued", "poll_after_s": 3}),
        (200, {"job_id": "4" * 32, "status": "running"}),
    ])
    clock = iter([0, 0, 0, 1000])
    with mock.patch.object(urllib.request, "urlopen", fake), \
            mock.patch.object(publisher, "_collect_datasource_secrets", return_value=({}, [])), \
            mock.patch.object(publisher.time, "sleep", lambda s: None), \
            mock.patch.object(publisher, "_monotonic", lambda: next(clock)), \
            pytest.raises(PublishJobTimeout):
        publish(_fullstack(tmp_path), api_key="k", publish_url="https://pub.test", job_budget_s=90)
    assert len(calls) == 2


def test_on_job_accepted_callback_receives_202_body(tmp_path):
    fake, _ = _urlopen([
        (202, {"job_id": "e" * 32, "report_id": "rid", "status": "queued", "poll_after_s": 3}),
        (200, {"job_id": "e" * 32, "status": "done", "result": {"report_id": "rid"}}),
    ])
    seen = []
    with mock.patch.object(urllib.request, "urlopen", fake), \
            mock.patch.object(publisher, "_collect_datasource_secrets", return_value=({}, [])), \
            mock.patch.object(publisher.time, "sleep", lambda s: None):
        publish(_fullstack(tmp_path), api_key="k", publish_url="https://pub.test", on_job_accepted=seen.append)
    assert seen and seen[0]["job_id"] == "e" * 32
