"""Tests for the publish access spec (ENG-322): build_access_payload + publish()."""

import base64
import hashlib
import http.client
import io
import json
import urllib.error
from pathlib import Path
from unittest import mock

import pytest

from anton import publisher
from anton.publisher import (
    OWNED_BY_OTHER_USER_CODE,
    ArtifactOwnedByOtherUserError,
    build_access_payload,
    hash_access_password,
    publish,
)


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


# ---------------------------------------------------------------------------
# The previous hash is sent again while the password is unchanged
# ---------------------------------------------------------------------------


def _password_entry(password: str, password_hash: str | None) -> dict:
    """An owner-side `.published.json` entry as the publish callers write it."""
    entry = {
        "mode": "password", "requires_password": True,
        "access_password": password, "pwd_version": 1,
        "report_id": "r", "url": "u", "last_md5": "m",
    }
    if password_hash is not None:
        entry["password_hash"] = password_hash
    return entry


def test_a_fresh_hash_differs_on_every_call():
    """Why reuse is needed at all: every call salts anew."""
    first = build_access_payload({"mode": "password", "password": "hunter2"})
    second = build_access_payload({"mode": "password", "password": "hunter2"})
    assert first["password_hash"] != second["password_hash"]


def test_same_password_reuses_the_previous_hash():
    prev_hash = hash_access_password("hunter2")
    out = build_access_payload(
        {"mode": "password", "password": "hunter2"},
        previous=_password_entry("hunter2", prev_hash),
    )
    assert out["password_hash"] == prev_hash


def test_changed_password_gets_a_new_hash():
    prev_hash = hash_access_password("hunter2")
    out = build_access_payload(
        {"mode": "password", "password": "correct-horse"},
        previous=_password_entry("hunter2", prev_hash),
    )
    assert out["password_hash"] != prev_hash
    assert out["password_hash"].startswith("pbkdf2_sha256$")


def test_legacy_entry_without_mode_still_reuses_the_hash():
    prev_hash = hash_access_password("hunter2")
    previous = {"requires_password": True, "access_password": "hunter2", "password_hash": prev_hash}
    out = build_access_payload({"mode": "password", "password": "hunter2"}, previous=previous)
    assert out["password_hash"] == prev_hash


_HUNTER2_HASH = hash_access_password("hunter2")


def _hash_with(password: str, iterations: int, salt: bytes = b"0123456789abcdef") -> str:
    """A well-formed stored hash, with the iteration count chosen by the test."""
    b64 = lambda b: base64.b64encode(b).decode("ascii")
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${b64(salt)}${b64(dk)}"


@pytest.mark.parametrize(
    "previous",
    [
        None,
        "not-a-dict",
        {"mode": "public", "requires_password": False},
        # Left password mode in between, but the plaintext and hash survived.
        {"mode": "restricted", "requires_password": False, "emails": ["a@x.com"],
         "access_password": "hunter2", "password_hash": _HUNTER2_HASH},
        {"mode": "public", "requires_password": False,
         "access_password": "hunter2", "password_hash": _HUNTER2_HASH},
        # Written before the hash was stored: plaintext only.
        _password_entry("hunter2", None),
        _password_entry("hunter2", ""),
        _password_entry("hunter2", "sha1$not-ours"),
        # A well-formed hash of another password: the entry disagrees with itself.
        _password_entry("hunter2", hash_access_password("correct-horse")),
        # Another iteration count, even with a valid digest for that count.
        _password_entry("hunter2", _hash_with("hunter2", 1_000)),
        # The right digest, but the iteration count is not written the way
        # `hash_access_password` writes it.
        _password_entry("hunter2", _hash_with("hunter2", 200_000).replace("$200000$", "$0200000$", 1)),
        # Malformed values.
        _password_entry("hunter2", "pbkdf2_sha256$200000$c2FsdA==$ZGs="),
        _password_entry("hunter2", "pbkdf2_sha256$many$c2FsdA==$ZGs="),
        _password_entry("hunter2", "pbkdf2_sha256$200000$c2FsdA=="),
        _password_entry("hunter2", f"{_HUNTER2_HASH}$extra"),
        _password_entry("hunter2", "pbkdf2_sha256$200000$@@@@$@@@@"),
        _password_entry("hunter2", 12345),
    ],
)
def test_no_reusable_hash_means_a_fresh_one(previous):
    out = build_access_payload({"mode": "password", "password": "hunter2"}, previous=previous)
    assert out["password_hash"].startswith(f"pbkdf2_sha256${publisher._PBKDF2_ITERATIONS}$")
    if isinstance(previous, dict):
        assert out["password_hash"] != previous.get("password_hash")


def test_a_hash_that_verifies_the_password_is_reused():
    """The check is a recomputation, not a prefix test: a verifying hash passes it."""
    stored = _hash_with("hunter2", publisher._PBKDF2_ITERATIONS)
    out = build_access_payload(
        {"mode": "password", "password": "hunter2"}, previous=_password_entry("hunter2", stored),
    )
    assert out["password_hash"] == stored


def _publish_with(tmp_path: Path, fake_request, **publish_kwargs) -> dict:
    f = tmp_path / "index.html"
    f.write_text("<html>hi</html>", encoding="utf-8")
    with mock.patch.object(publisher, "minds_request", fake_request):
        return publish(f, api_key="k", **publish_kwargs)


def _accepting_request(captured: dict):
    def fake_request(url, api_key, *, method="POST", payload=None, verify=True, timeout=30):
        captured["payload"] = json.loads(payload.decode())
        return json.dumps(
            {"user_prefix": "u", "report_id": "r", "md5": "m", "view_url": "url", "version": 1, "files": []}
        )
    return fake_request


def test_publish_sends_the_previous_hash_and_returns_it(tmp_path: Path):
    prev_hash = hash_access_password("hunter2")
    captured: dict = {}
    result = _publish_with(
        tmp_path, _accepting_request(captured),
        access={"mode": "password", "password": "hunter2"},
        previous_access=_password_entry("hunter2", prev_hash),
    )
    assert captured["payload"]["access"]["password_hash"] == prev_hash
    assert result["password_hash"] == prev_hash


def test_publish_returns_the_fresh_hash_it_sent(tmp_path: Path):
    captured: dict = {}
    result = _publish_with(
        tmp_path, _accepting_request(captured), access={"mode": "password", "password": "hunter2"},
    )
    assert result["password_hash"] == captured["payload"]["access"]["password_hash"]


def test_publish_returns_no_hash_outside_password_mode(tmp_path: Path):
    result = _publish_with(
        tmp_path, _accepting_request({}),
        access={"mode": "restricted", "emails": ["a@x.com"], "org_allowed": False},
    )
    assert "password_hash" not in result


# ---------------------------------------------------------------------------
# 409 artifact_owned_by_other_user
# ---------------------------------------------------------------------------


def _http_error(status: int, body) -> urllib.error.HTTPError:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return urllib.error.HTTPError("https://view.test/upload", status, "Rejected", {}, io.BytesIO(raw))


def test_owned_by_other_user_code_is_the_wire_value():
    assert OWNED_BY_OTHER_USER_CODE == "artifact_owned_by_other_user"


def test_publish_raises_a_typed_error_for_another_owner(tmp_path: Path):
    def reject(url, api_key, **kwargs):
        raise _http_error(409, {"error": "Report belongs to another user", "code": OWNED_BY_OTHER_USER_CODE})

    with pytest.raises(ArtifactOwnedByOtherUserError) as err:
        _publish_with(tmp_path, reject, report_id="r")
    assert str(err.value) == "This artifact was published by another owner."
    assert isinstance(err.value, RuntimeError)
    assert isinstance(err.value.__cause__, urllib.error.HTTPError)


class _UnreadableBody(io.BytesIO):
    def read(self, *args):
        raise http.client.IncompleteRead(b"")


def test_a_409_with_an_unreadable_body_propagates_the_original_error(tmp_path: Path):
    """A body that cannot be read is "not recognised", never a different exception."""
    conflict = urllib.error.HTTPError("https://view.test/upload", 409, "Conflict", {}, _UnreadableBody())

    def reject(url, api_key, **kwargs):
        raise conflict

    with pytest.raises(urllib.error.HTTPError) as err:
        _publish_with(tmp_path, reject, report_id="r")
    assert err.value is conflict
    assert err.value.code == 409


@pytest.mark.parametrize(
    "status, body",
    [
        (409, {"error": "Artifact key is owned by another user"}),  # auth's artifact_key conflict: no code
        (409, b"not json"),
        (500, {"error": "boom", "code": OWNED_BY_OTHER_USER_CODE}),
    ],
)
def test_other_http_errors_propagate_unchanged(tmp_path: Path, status, body):
    def reject(url, api_key, **kwargs):
        raise _http_error(status, body)

    with pytest.raises(urllib.error.HTTPError) as err:
        _publish_with(tmp_path, reject, report_id="r")
    assert err.value.code == status
