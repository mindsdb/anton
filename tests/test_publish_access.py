"""Tests for the publish access spec (ENG-322): build_access_payload + publish()."""

import base64
import json
from pathlib import Path
from unittest import mock

from anton import publisher
from anton.publisher import build_access_payload, publish


# ---------------------------------------------------------------------------
# build_access_payload
# ---------------------------------------------------------------------------


def test_public_mode():
    assert build_access_payload(None) == {"mode": "public"}
    assert build_access_payload({"mode": "public"}) == {"mode": "public"}


def test_password_mode_hashes_and_drops_plaintext():
    out = build_access_payload({"mode": "password", "password": "hunter2"}, pwd_version=2)
    assert out["mode"] == "password"
    assert out["pwd_version"] == 2
    assert out["password_hash"].startswith("pbkdf2_sha256$")
    assert "password" not in out  # plaintext never leaves


def test_restricted_mode_normalizes_and_excludes_secrets():
    out = build_access_payload(
        {"mode": "restricted", "emails": [" A@X.com ", "a@x.com", "B@Y.com"], "org_allowed": True},
        access_version=4,
    )
    assert out == {
        "mode": "restricted",
        "allowed_emails": ["a@x.com", "b@y.com"],
        "org_allowed": True,
        "access_version": 4,
    }


def test_restricted_mode_defaults():
    out = build_access_payload({"mode": "restricted"}, access_version=1)
    assert out == {"mode": "restricted", "allowed_emails": [], "org_allowed": False, "access_version": 1}


# ---------------------------------------------------------------------------
# publish() wiring
# ---------------------------------------------------------------------------


def _capture_publish(tmp_path: Path, **publish_kwargs) -> dict:
    f = tmp_path / "index.html"
    f.write_text("<html>hi</html>", encoding="utf-8")
    captured: dict = {}

    def fake_request(url, api_key, *, method="POST", payload=None, verify=True, timeout=30):
        captured["payload"] = json.loads(payload.decode())
        return json.dumps(
            {"user_prefix": "u", "report_id": "r", "md5": "m", "view_url": "url", "version": 1, "files": []}
        )

    with mock.patch.object(publisher, "minds_request", fake_request):
        publish(f, api_key="k", **publish_kwargs)
    return captured["payload"]


def test_publish_sends_restricted_access(tmp_path: Path):
    payload = _capture_publish(
        tmp_path,
        access={"mode": "restricted", "emails": ["a@x.com"], "org_allowed": True},
        access_version=2,
    )
    assert payload["access"] == {
        "mode": "restricted",
        "allowed_emails": ["a@x.com"],
        "org_allowed": True,
        "access_version": 2,
    }


def test_publish_restricted_emails_do_not_enter_bundle(tmp_path: Path):
    payload = _capture_publish(
        tmp_path,
        access={"mode": "restricted", "emails": ["secret@x.com"], "org_allowed": False},
        access_version=1,
    )
    zip_bytes = base64.b64decode(payload["file_payload"])
    assert b"secret@x.com" not in zip_bytes


def test_publish_public_by_default(tmp_path: Path):
    payload = _capture_publish(tmp_path)
    assert payload["access"] == {"mode": "public"}


def test_publish_back_compat_password(tmp_path: Path):
    payload = _capture_publish(tmp_path, password="hunter2", pwd_version=3)
    assert payload["access"]["mode"] == "password"
    assert payload["access"]["pwd_version"] == 3
    assert payload["access"]["password_hash"].startswith("pbkdf2_sha256$")
    assert "password" not in payload["access"]
