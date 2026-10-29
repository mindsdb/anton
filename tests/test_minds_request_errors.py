"""What a failed Minds API request tells its readers.

These run the real `minds_request` against a local HTTP server, so the error
objects are urllib's own: a mocked transport cannot show that the body
survives for every reader that wants it.
"""

import json
import socket
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import pytest
from rich.console import Console

import anton.chat as chat
import anton.tools as tools
from anton.minds_client import (
    describe_minds_request_failure,
    minds_request,
    test_llm as probe_llm,
)
from anton.publisher import ArtifactOwnedByOtherUserError, publish


class _Handler(BaseHTTPRequestHandler):
    routes: dict[str, tuple[int, str, bytes]] = {}

    def _answer(self):
        status, content_type, body = self.routes[self.path]
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = do_DELETE = _answer

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    routes: dict[str, tuple[int, str, bytes]] = {}
    handler = type("Handler", (_Handler,), {"routes": routes})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    # A short poll keeps shutdown() from waiting half a second per test.
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()

    def answer(path: str, status: int, body, content_type: str = "application/json"):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        routes[path] = (status, content_type, raw)

    yield f"http://127.0.0.1:{srv.server_port}", answer
    srv.shutdown()
    srv.server_close()


def _closed_port_url() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def _artifact(tmp_path: Path) -> Path:
    art = tmp_path / "artifacts" / "sales"
    art.mkdir(parents=True)
    (art / "metadata.json").write_text('{"type": "html-report", "primary": "report.html"}')
    f = art / "report.html"
    f.write_text("<html></html>")
    return f


async def _tool_publish(tmp_path: Path, publish_url: str) -> str:
    f = _artifact(tmp_path)
    settings = mock.Mock(
        minds_api_key="key", publish_url=publish_url, minds_ssl_verify=True,
        artifacts_dir=str(f.parent.parent),
    )
    session = mock.Mock(_console=Console(), _settings=None)
    session._workspace.base = str(tmp_path)
    with mock.patch("anton.config.settings.AntonSettings", return_value=settings), \
         mock.patch("webbrowser.open"):
        return await tools.handle_publish_or_preview(
            session, {"file_path": str(f), "action": "publish"},
        )


@pytest.mark.asyncio
async def test_publish_tool_names_the_request_status_and_server_message(tmp_path, server):
    base, answer = server
    answer("/upload", 401, {"error": "Invalid token"})

    out = await _tool_publish(tmp_path, base)

    assert out.startswith(f"PUBLISH FAILED: POST {base}/upload returned HTTP 401: Invalid token.")
    assert "ANTON_MINDS_API_KEY" in out


@pytest.mark.asyncio
async def test_publish_tool_shows_the_status_line_not_an_html_error_page(tmp_path, server):
    base, answer = server
    answer("/upload", 502, b"<html><body>Bad gateway</body></html>", "text/html")

    out = await _tool_publish(tmp_path, base)

    assert f"POST {base}/upload returned HTTP 502 Bad Gateway." in out
    assert "<html" not in out


@pytest.mark.asyncio
async def test_publish_tool_names_the_host_that_did_not_answer(tmp_path):
    base = _closed_port_url()

    out = await _tool_publish(tmp_path, base)

    assert out.startswith(f"PUBLISH FAILED: POST {base}/upload got no response.")


@pytest.mark.asyncio
async def test_server_text_is_not_read_as_console_markup(tmp_path, server):
    base, answer = server
    answer("/upload", 400, {"error": "bad [/bold] value"})

    out = await _tool_publish(tmp_path, base)

    assert "HTTP 400: bad [/bold] value" in out


def test_another_owners_409_is_still_typed(tmp_path, server):
    base, answer = server
    answer("/upload", 409, {"code": "artifact_owned_by_other_user", "error": "owned"})

    with pytest.raises(ArtifactOwnedByOtherUserError):
        publish(_artifact(tmp_path), api_key="key", publish_url=base, report_id="rid")


@pytest.mark.parametrize("status,body", [
    (409, {"code": "something_else", "error": "conflict"}),
    (403, {"code": "permission_denied", "error": "Publishing is not enabled for your plan"}),
    (503, {"code": "auth_unavailable", "error": "auth is down"}),
])
def test_the_error_body_is_left_for_the_callers_reader(tmp_path, server, status, body):
    """cowork-server classifies upload refusals by reading the body off the
    raised error; it has to still be there after anton is done with it."""
    base, answer = server
    answer("/upload", status, body)

    with pytest.raises(urllib.error.HTTPError) as caught:
        publish(_artifact(tmp_path), api_key="key", publish_url=base, report_id="rid")

    assert json.loads(caught.value.read(65_536)) == body


def test_status_line_and_reason_are_unchanged(server):
    base, answer = server
    answer("/list", 401, {"error": "Invalid token"})

    with pytest.raises(urllib.error.HTTPError) as caught:
        minds_request(f"{base}/list", "key")

    assert str(caught.value) == "HTTP Error 401: Unauthorized"
    assert caught.value.reason == "Unauthorized"
    assert caught.value.code == 401


def test_llm_probe_keeps_the_unwrapped_provider_error(server):
    base, answer = server
    answer("/v1/chat/completions", 404, {
        "error": {"code": "model_not_found", "message": "The model `x` does not exist"},
    })

    result = probe_llm(base, "key", model="x")

    assert result.error == "model_not_found: The model `x` does not exist"
    assert result.http_status == 404


def test_long_server_text_keeps_its_end(server):
    base, answer = server
    answer("/upload", 400, {"error": "pip install log " + "x" * 2000 + " ERROR: no matching dist"})

    with pytest.raises(urllib.error.HTTPError) as caught:
        minds_request(f"{base}/upload", "key", method="POST", payload=b"{}")

    line = describe_minds_request_failure(caught.value)
    assert line.endswith("no matching dist. The server refused the request.")
    assert len(line) < 700


def test_a_local_failure_is_not_described_as_a_request():
    assert describe_minds_request_failure(FileNotFoundError("Path not found: x")) is None


@pytest.mark.parametrize("status,body", [
    (500, {"error": "KeyError: 'files'"}),
    (502, b"<html>Bad gateway</html>"),
    (503, b""),
    (504, {"error": "pip install timed out (540s)"}),
])
def test_a_server_error_is_reported_without_a_cause(server, status, body):
    """The agent read "outage" into a bare status once; the line it reads
    must not hand it that story either."""
    base, answer = server
    answer("/upload", status, body, "application/json" if isinstance(body, dict) else "text/html")

    with pytest.raises(urllib.error.HTTPError) as caught:
        minds_request(f"{base}/upload", "key", method="POST", payload=b"{}")

    line = describe_minds_request_failure(caught.value)
    assert line.startswith(f"POST {base}/upload returned HTTP {status}")
    assert "outage" not in line.lower()


def test_no_response_is_reported_without_a_cause():
    base = _closed_port_url()

    with pytest.raises(urllib.error.URLError) as caught:
        minds_request(f"{base}/list", "key")

    line = describe_minds_request_failure(caught.value)
    assert line.startswith(f"GET {base}/list got no response.")
    assert "outage" not in line.lower()


def _chat_settings(tmp_path: Path, publish_url: str):
    return mock.Mock(
        minds_api_key="key", publish_url=publish_url, minds_ssl_verify=True,
        artifacts_dir=str(tmp_path / "artifacts"), workspace_path=str(tmp_path),
    )


@pytest.mark.asyncio
async def test_slash_publish_names_the_request_and_server_message(tmp_path, server):
    base, answer = server
    answer("/upload", 400, {"error": "bad [/bold] value"})
    _artifact(tmp_path)
    console = Console(record=True, width=300)

    async def public(*args, **kwargs):
        return {"mode": "public"}

    with mock.patch("anton.publish_access.prompt_access", side_effect=public), \
         mock.patch("webbrowser.open"):
        await chat._handle_publish(console, _chat_settings(tmp_path, base), file_arg="sales/report.html")

    assert f"Publish failed: POST {base}/upload returned HTTP 400: bad [/bold] value." in console.export_text()


@pytest.mark.asyncio
async def test_slash_unpublish_names_the_failed_listing(tmp_path, server):
    base, answer = server
    answer("/list", 500, {"error": "[/oops] listing broke"})
    console = Console(record=True, width=300)

    await chat._handle_unpublish(console, _chat_settings(tmp_path, base), mock.Mock())

    assert f"Failed to list reports: GET {base}/list returned HTTP 500: [/oops] listing broke." in console.export_text()


@pytest.mark.asyncio
async def test_slash_unpublish_names_the_failed_delete(tmp_path, server):
    base, answer = server
    answer("/list", 200, {"reports": [
        {"title": "sales", "report_id": "247c4983", "md5": "m", "view_url": "u"},
    ]})
    answer("/delete/247c4983", 404, {"error": "Report [/x] not found"})
    console = Console(record=True, width=300)

    with mock.patch("anton.chat.prompt_or_cancel", new=mock.AsyncMock(side_effect=["1", "y"])):
        await chat._handle_unpublish(console, _chat_settings(tmp_path, base), mock.Mock())

    assert f"Failed to remove: DELETE {base}/delete/247c4983 returned HTTP 404: Report [/x] not found." in console.export_text()
