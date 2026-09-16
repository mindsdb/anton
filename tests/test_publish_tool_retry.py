"""handle_publish_or_preview retry semantics (ENG-1580).

Before: any publish error with a known report_id triggered a second publish
without report_id. With async publishes a 3-minute PublishJobTimeout — or a
404 from the status poll after the job record expired — would then create a
duplicate artifact while the first job is still running. Now only an HTTP 404
from POST /upload itself retries.
"""
from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from unittest import mock

import pytest
from rich.console import Console

import anton.tools as tools
from anton.publisher import PublishJobFailed, PublishJobTimeout


def _artifact(tmp_path: Path, *, published: bool = True) -> Path:
    root = tmp_path / "artifacts"
    art = root / "sales"
    art.mkdir(parents=True)
    (art / "metadata.json").write_text(json.dumps({
        "schemaVersion": 1, "id": "b" * 32, "slug": "sales", "name": "Sales", "description": "",
        "createdAt": "2026-01-01T00:00:00Z", "updatedAt": "2026-01-01T00:00:00Z",
        "type": "html-app", "primary": "report.html",
    }))
    (art / "report.html").write_text("<html><title>Sales</title></html>")
    if published:
        (art / ".published.json").write_text(json.dumps({"report.html": {"report_id": "old-id", "view_url": "https://v/x", "mode": "public"}}))
    return art / "report.html"


def _session(tmp_path, console_buf=None):
    s = mock.Mock()
    s._console = Console(file=console_buf if console_buf is not None else io.StringIO())
    ws = mock.Mock()
    ws.base = str(tmp_path)
    s._workspace = ws
    s._settings = None
    return s


def _settings(root, workspace=None):
    # SimpleNamespace, not Mock: an unexpected `settings.<attr>` read must fail
    # loudly instead of returning a truthy Mock that steers the handler into an
    # interactive branch (API-key prompt, access questions).
    from types import SimpleNamespace
    return SimpleNamespace(
        minds_api_key="key",
        publish_url="https://view.test",
        minds_ssl_verify=True,
        artifacts_dir=str(root),
        workspace_path=str(workspace) if workspace is not None else str(root.parent),
        # handle_publish_or_preview calls this on freshly built settings to
        # bind artifacts_dir to the session workspace (anton/tools.py:566).
        # chat._handle_publish does not; a no-op keeps one helper for both.
        resolve_workspace=lambda *a, **k: None,
    )


def _run(tmp_path, side_effects):
    f = _artifact(tmp_path)
    calls = []
    console_buf = io.StringIO()

    def fake_publish(path, **kw):
        calls.append(kw.get("report_id"))
        eff = side_effects[len(calls) - 1]
        if isinstance(eff, Exception):
            raise eff
        return eff

    with mock.patch("anton.publisher.publish", fake_publish), \
            mock.patch("anton.config.settings.AntonSettings", return_value=_settings(f.parent.parent)), \
            mock.patch("webbrowser.open"):
        import asyncio
        session = _session(tmp_path, console_buf=console_buf)
        out = asyncio.run(tools.handle_publish_or_preview(session, {"file_path": str(f), "action": "publish"}))
    return out, calls, console_buf.getvalue()


def test_404_retries_without_report_id(tmp_path):
    err = urllib.error.HTTPError("u", 404, "nf", {}, io.BytesIO(b"{}"))
    out, calls, _ = _run(tmp_path, [err, {"report_id": "new-id", "view_url": "https://v/new"}])
    assert calls == ["old-id", None]
    assert "https://v/new" in out


def test_job_failed_404_does_not_retry(tmp_path):
    """A 404 reported through the job path comes from the status poll (record
    expired, wrong publish_url), not from the upload — the job may well be
    finishing. Re-publishing would create a duplicate."""
    err = PublishJobFailed(404, "job status request rejected: Not Found", job_id="j" * 32, report_id="old-id")
    out, calls, _ = _run(tmp_path, [err])
    assert calls == ["old-id"]
    assert out.startswith("PUBLISH FAILED:") and "j" * 32 in out


def test_job_timeout_does_not_retry(tmp_path):
    err = PublishJobTimeout(job_id="j" * 32, report_id="old-id", waited_s=180)
    out, calls, console_out = _run(tmp_path, [err])
    assert calls == ["old-id"]
    assert out.startswith("PUBLISH FAILED:") and "j" * 32 in out
    # The console.print in handle_publish_or_preview must escape the exception
    # text so Rich markup does not swallow the "[job ..., report ...]" segment.
    assert "j" * 32 in console_out


@pytest.mark.parametrize("err", [
    PublishJobFailed(400, "pip install failed: x", job_id="j" * 32, report_id="old-id"),
    urllib.error.HTTPError("u", 500, "boom", {}, io.BytesIO(b"{}")),
    RuntimeError("local zip error"),
])
def test_other_errors_do_not_retry(tmp_path, err):
    out, calls, _ = _run(tmp_path, [err])
    assert calls == ["old-id"]
    assert out.startswith("PUBLISH FAILED:")


@pytest.mark.asyncio
async def test_chat_publish_prints_job_timeout_as_failure_line(tmp_path):
    import anton.chat as chat
    _artifact(tmp_path, published=False)
    root = tmp_path / "artifacts"
    settings = _settings(root, workspace=tmp_path)
    buf = io.StringIO()
    err = PublishJobTimeout(job_id="j" * 32, report_id="old-id", waited_s=180)

    async def fake_prompt_access(*a, **k):
        return {"mode": "public"}

    with mock.patch("anton.publisher.publish", mock.Mock(side_effect=err)), \
            mock.patch("anton.publish_access.prompt_access", side_effect=fake_prompt_access), \
            mock.patch("webbrowser.open"):
        await chat._handle_publish(Console(file=buf, width=200), settings, file_arg="sales/report.html")
    out = buf.getvalue()
    assert "Publish failed:" in out and "j" * 32 in out
    assert "Traceback" not in out
