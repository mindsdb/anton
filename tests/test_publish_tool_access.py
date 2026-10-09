"""publish_or_preview tool: access fields + preserve-previous default."""

import io
import json
import urllib.error
from pathlib import Path
from unittest import mock

import pytest
from rich.console import Console

import anton.tools as tools


def _artifact(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    art = root / "sales"
    art.mkdir(parents=True)
    (art / "metadata.json").write_text('{"type": "html-report", "primary": "report.html"}')
    f = art / "report.html"
    f.write_text("<html></html>")
    return f


def _fullstack_artifact(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "artifacts"
    art = root / "server-time-display-2"
    (art / "static").mkdir(parents=True)
    (art / "metadata.json").write_text(
        '{"type": "fullstack-stateless-app", "primary": "static/index.html"}'
    )
    (art / "backend.py").write_text("print(1)")
    (art / "static" / "index.html").write_text("<html></html>")
    return root, art


def _session(tmp_path):
    s = mock.Mock()
    s._console = Console()
    ws = mock.Mock()
    ws.base = str(tmp_path)
    s._workspace = ws
    # A host that passed no settings, so the handler falls back to resolving
    # them itself — which is what the `AntonSettings` patch in each test below
    # stands in for. Without this, Mock would auto-create `_settings` as a Mock
    # that passes the handler's `hasattr(..., "minds_api_key")` check and
    # silently shadow the patch (ENG-1424). The session-carries-settings path
    # is covered in tests/test_publish_session_identity.py.
    s._settings = None
    return s


def _settings(root):
    s = mock.Mock()
    s.minds_api_key = "key"
    s.publish_url = "https://view.test"
    s.minds_ssl_verify = True
    s.artifacts_dir = str(root)
    return s


@pytest.mark.asyncio
async def test_tool_explicit_password(tmp_path):
    f = _artifact(tmp_path)
    fake_publish = mock.Mock(return_value={"view_url": "u", "report_id": "r", "md5": "m", "version": 1})
    with mock.patch("anton.publisher.publish", fake_publish), \
         mock.patch("anton.config.settings.AntonSettings", return_value=_settings(f.parent.parent)), \
         mock.patch("webbrowser.open"):
        out = await tools.handle_publish_or_preview(
            _session(tmp_path),
            {"file_path": str(f), "action": "publish",
             "access_mode": "password", "password": "hunter2"},
        )
    _, kwargs = fake_publish.call_args
    assert kwargs["access"] == {"mode": "password", "password": "hunter2"}
    assert "Published" in out


@pytest.mark.asyncio
async def test_tool_preserves_previous_when_no_fields(tmp_path):
    f = _artifact(tmp_path)
    (f.parent / ".published.json").write_text(json.dumps({
        "report.html": {"report_id": "r", "url": "u", "last_md5": "m",
                          "mode": "password", "requires_password": True,
                          "access_password": "old", "pwd_version": 1},
    }))
    fake_publish = mock.Mock(return_value={"view_url": "u", "report_id": "r", "md5": "m2", "version": 2})
    with mock.patch("anton.publisher.publish", fake_publish), \
         mock.patch("anton.config.settings.AntonSettings", return_value=_settings(f.parent.parent)), \
         mock.patch("webbrowser.open"):
        await tools.handle_publish_or_preview(
            _session(tmp_path), {"file_path": str(f), "action": "publish"},
        )
    _, kwargs = fake_publish.call_args
    assert kwargs["access"] == {"mode": "password", "password": "old"}  # NOT public


@pytest.mark.asyncio
async def test_tool_password_no_value_non_tty_cancels(tmp_path):
    f = _artifact(tmp_path)
    fake_publish = mock.Mock()
    with mock.patch("anton.publisher.publish", fake_publish), \
         mock.patch("anton.config.settings.AntonSettings", return_value=_settings(f.parent.parent)), \
         mock.patch("sys.stdin") as stdin:
        stdin.isatty.return_value = False
        out = await tools.handle_publish_or_preview(
            _session(tmp_path),
            {"file_path": str(f), "action": "publish", "access_mode": "password"},
        )
    assert "CANCELLED" in out
    fake_publish.assert_not_called()


@pytest.mark.asyncio
async def test_tool_fullstack_publishes_folder(tmp_path):
    """Given the artifact folder, publish() receives the folder (fullstack bundle)."""
    root, art = _fullstack_artifact(tmp_path)
    fake_publish = mock.Mock(return_value={"view_url": "u", "report_id": "r", "md5": "m", "version": 1})
    with mock.patch("anton.publisher.publish", fake_publish), \
         mock.patch("anton.config.settings.AntonSettings", return_value=_settings(root)), \
         mock.patch("webbrowser.open"):
        await tools.handle_publish_or_preview(
            _session(tmp_path), {"file_path": str(art), "action": "publish"},
        )
    args, _ = fake_publish.call_args
    assert Path(args[0]) == art  # the folder, not an inner file


@pytest.mark.asyncio
async def test_tool_fullstack_from_inner_file_publishes_folder(tmp_path):
    """Even if the model points at an inner HTML file, publish() gets the folder."""
    root, art = _fullstack_artifact(tmp_path)
    fake_publish = mock.Mock(return_value={"view_url": "u", "report_id": "r", "md5": "m", "version": 1})
    with mock.patch("anton.publisher.publish", fake_publish), \
         mock.patch("anton.config.settings.AntonSettings", return_value=_settings(root)), \
         mock.patch("webbrowser.open"):
        await tools.handle_publish_or_preview(
            _session(tmp_path),
            {"file_path": str(art / "static" / "index.html"), "action": "publish"},
        )
    args, _ = fake_publish.call_args
    assert Path(args[0]) == art  # normalized up to the fullstack folder


@pytest.mark.asyncio
async def test_tool_owner_only_publishes_restricted(tmp_path):
    """An explicit owner_only from the agent must NOT degrade to public."""
    f = _artifact(tmp_path)
    fake_publish = mock.Mock(return_value={"view_url": "u", "report_id": "r", "md5": "m", "version": 1})
    with mock.patch("anton.publisher.publish", fake_publish), \
         mock.patch("anton.config.settings.AntonSettings", return_value=_settings(f.parent.parent)), \
         mock.patch("webbrowser.open"):
        await tools.handle_publish_or_preview(
            _session(tmp_path),
            {"file_path": str(f), "action": "publish",
             "access_mode": "restricted", "emails": [], "owner_only": True},
        )
    _, kwargs = fake_publish.call_args
    assert kwargs["access"] == {"mode": "restricted", "emails": [], "org_allowed": False}
    entry = json.loads((f.parent / ".published.json").read_text())[f.name]
    assert entry["mode"] == "restricted"
    assert entry["owner_only"] is True


@pytest.mark.asyncio
async def test_tool_restricted_without_owner_only_still_degrades(tmp_path):
    """The safety net for careless programmatic callers stays in place."""
    f = _artifact(tmp_path)
    fake_publish = mock.Mock(return_value={"view_url": "u", "report_id": "r", "md5": "m", "version": 1})
    with mock.patch("anton.publisher.publish", fake_publish), \
         mock.patch("anton.config.settings.AntonSettings", return_value=_settings(f.parent.parent)), \
         mock.patch("webbrowser.open"):
        await tools.handle_publish_or_preview(
            _session(tmp_path),
            {"file_path": str(f), "action": "publish", "access_mode": "restricted", "emails": []},
        )
    _, kwargs = fake_publish.call_args
    assert kwargs["access"] == {"mode": "public"}


@pytest.mark.asyncio
async def test_tool_restricted_invalid_email_returns_error(tmp_path):
    """A malformed address must not collapse into an owner-only or public publish."""
    f = _artifact(tmp_path)
    fake_publish = mock.Mock(return_value={"view_url": "u", "report_id": "r", "md5": "m", "version": 1})
    with mock.patch("anton.publisher.publish", fake_publish), \
         mock.patch("anton.config.settings.AntonSettings", return_value=_settings(f.parent.parent)), \
         mock.patch("webbrowser.open"):
        out = await tools.handle_publish_or_preview(
            _session(tmp_path),
            {"file_path": str(f), "action": "publish",
             "access_mode": "restricted", "emails": ["colleague@corp"]},
        )
    assert "colleague@corp" in out
    assert "INVALID" in out
    fake_publish.assert_not_called()


def test_publish_tool_schema_exposes_owner_only():
    props = tools.PUBLISH_TOOL.input_schema["properties"]
    assert props["owner_only"]["type"] == "boolean"


# ---------------------------------------------------------------------------
# Password hash kept owner-side, and the 409 for another owner
# ---------------------------------------------------------------------------


def _recording_request(sent: list):
    """Stands in for the upload service; the real publish() builds the payload."""
    def fake_request(url, api_key, *, method="POST", payload=None, verify=True, timeout=30):
        sent.append(json.loads(payload.decode())["access"])
        return json.dumps({"view_url": "u", "report_id": "r", "md5": f"m{len(sent)}", "version": len(sent)})
    return fake_request


async def _tool_publish(tmp_path, f, tc_input: dict, sent: list) -> str:
    with mock.patch("anton.publisher.minds_request", _recording_request(sent)), \
         mock.patch("anton.config.settings.AntonSettings", return_value=_settings(f.parent.parent)), \
         mock.patch("webbrowser.open"):
        return await tools.handle_publish_or_preview(
            _session(tmp_path), {"file_path": str(f), "action": "publish", **tc_input},
        )


def _entry(f: Path) -> dict:
    return json.loads((f.parent / ".published.json").read_text())[f.name]


@pytest.mark.asyncio
async def test_tool_stores_the_sent_hash_and_sends_it_again(tmp_path):
    f = _artifact(tmp_path)
    sent: list = []

    await _tool_publish(tmp_path, f, {"access_mode": "password", "password": "hunter2"}, sent)
    first = sent[0]["password_hash"]
    assert _entry(f)["password_hash"] == first
    assert _entry(f)["access_password"] == "hunter2"

    # A content-only re-publish (no access fields) keeps the password — and its hash.
    await _tool_publish(tmp_path, f, {}, sent)
    assert sent[1]["password_hash"] == first
    assert _entry(f)["password_hash"] == first


@pytest.mark.asyncio
async def test_tool_new_password_gets_a_new_hash(tmp_path):
    f = _artifact(tmp_path)
    sent: list = []

    await _tool_publish(tmp_path, f, {"access_mode": "password", "password": "hunter2"}, sent)
    await _tool_publish(tmp_path, f, {"access_mode": "password", "password": "correct-horse"}, sent)

    assert sent[1]["password_hash"] != sent[0]["password_hash"]
    assert _entry(f)["password_hash"] == sent[1]["password_hash"]


@pytest.mark.parametrize("detour", [
    {"access_mode": "public"},
    {"access_mode": "restricted", "emails": ["a@x.com"]},
])
@pytest.mark.asyncio
async def test_tool_password_after_a_detour_gets_a_new_hash(tmp_path, detour):
    """password → public/restricted → password with the SAME password: the
    detour drops the stored plaintext and hash, so old grants must not survive."""
    f = _artifact(tmp_path)
    sent: list = []

    await _tool_publish(tmp_path, f, {"access_mode": "password", "password": "hunter2"}, sent)
    await _tool_publish(tmp_path, f, detour, sent)
    assert "password_hash" not in _entry(f)
    await _tool_publish(tmp_path, f, {"access_mode": "password", "password": "hunter2"}, sent)

    assert sent[2]["password_hash"] != sent[0]["password_hash"]


@pytest.mark.asyncio
async def test_tool_does_not_retry_without_report_id_for_another_owner(tmp_path):
    from anton.publisher import ArtifactOwnedByOtherUserError

    root, art = _fullstack_artifact(tmp_path)
    published = art / ".published.json"
    published.write_text(json.dumps({
        "index.html": {"report_id": "rid-owned-elsewhere", "url": "u", "last_md5": "m",
                       "mode": "public", "requires_password": False},
    }))
    before = published.read_text()
    fake_publish = mock.Mock(side_effect=ArtifactOwnedByOtherUserError())
    with mock.patch("anton.publisher.publish", fake_publish), \
         mock.patch("anton.config.settings.AntonSettings", return_value=_settings(root)), \
         mock.patch("webbrowser.open"):
        out = await tools.handle_publish_or_preview(
            _session(tmp_path), {"file_path": str(art), "action": "publish"},
        )

    assert fake_publish.call_count == 1  # no second call without report_id
    assert fake_publish.call_args.kwargs["report_id"] == "rid-owned-elsewhere"
    assert out.startswith("PUBLISH FAILED: This artifact was published by another owner.")
    assert "/publish and choosing 'new'" in out
    assert published.read_text() == before


def _http_error(status: int, reason: str, body: bytes = b"") -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://view.test/upload", status, reason, {}, io.BytesIO(body))


@pytest.mark.parametrize("failure", [
    pytest.param(_http_error(409, "Conflict", b"not json"), id="unrecognised-409"),
    pytest.param(_http_error(503, "Service Unavailable"), id="5xx"),
    pytest.param(urllib.error.URLError(TimeoutError("timed out")), id="timeout"),
])
@pytest.mark.asyncio
async def test_tool_does_not_retry_without_report_id_while_the_report_may_exist(tmp_path, failure):
    """The upload may have landed or the report may belong to someone else:
    a retry without report_id would publish a copy under a new URL."""
    f = _artifact(tmp_path)
    published = f.parent / ".published.json"
    published.write_text(json.dumps({
        "report.html": {"report_id": "rid", "url": "u", "last_md5": "m",
                        "mode": "public", "requires_password": False},
    }))
    before = published.read_text()
    fake_publish = mock.Mock(side_effect=failure)
    with mock.patch("anton.publisher.publish", fake_publish), \
         mock.patch("anton.config.settings.AntonSettings", return_value=_settings(f.parent.parent)), \
         mock.patch("webbrowser.open"):
        out = await tools.handle_publish_or_preview(
            _session(tmp_path), {"file_path": str(f), "action": "publish"},
        )

    assert fake_publish.call_count == 1
    assert fake_publish.call_args.kwargs["report_id"] == "rid"
    assert out.startswith("PUBLISH FAILED:")
    assert published.read_text() == before


@pytest.mark.asyncio
async def test_tool_retries_without_report_id_when_the_report_is_gone(tmp_path):
    f = _artifact(tmp_path)
    (f.parent / ".published.json").write_text(json.dumps({
        "report.html": {"report_id": "gone", "url": "u", "last_md5": "m",
                        "mode": "public", "requires_password": False},
    }))
    fake_publish = mock.Mock(side_effect=[
        _http_error(404, "Not Found"),
        {"view_url": "u2", "report_id": "fresh", "md5": "m2", "version": 1},
    ])
    with mock.patch("anton.publisher.publish", fake_publish), \
         mock.patch("anton.config.settings.AntonSettings", return_value=_settings(f.parent.parent)), \
         mock.patch("webbrowser.open"):
        await tools.handle_publish_or_preview(
            _session(tmp_path), {"file_path": str(f), "action": "publish"},
        )

    assert fake_publish.call_count == 2
    assert "report_id" not in fake_publish.call_args_list[1].kwargs
    assert _entry(f)["report_id"] == "fresh"
