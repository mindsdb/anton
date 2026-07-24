"""Tests for anton.publish_access — access resolution ported from cowork-server."""

from pathlib import Path

import pytest

from anton.publish_access import (
    access_from_owner_side,
    normalize_emails,
    parse_emails,
    prompt_access,
    resolve_access,
    resolve_publish_target,
)


def _write(p: Path, text: str = "x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_normalize_emails_strips_lowers_dedupes():
    assert normalize_emails([" A@x.com ", "a@x.com", "B@x.com"]) == ["a@x.com", "b@x.com"]
    assert normalize_emails(None) == []


def test_resolve_public_default():
    eff, pwd_v, acc_v, owner = resolve_access(None, None, None)
    assert eff == {"mode": "public"}
    assert owner == {"mode": "public", "requires_password": False}


def test_resolve_legacy_password():
    eff, pwd_v, acc_v, owner = resolve_access("hunter2", None, None)
    assert eff == {"mode": "password", "password": "hunter2"}
    assert pwd_v == 1
    assert owner["mode"] == "password"
    assert owner["requires_password"] is True
    assert owner["access_password"] == "hunter2"
    assert owner["pwd_version"] == 1


def test_resolve_empty_password_degrades_to_public():
    eff, *_ = resolve_access("   ", None, None)
    assert eff == {"mode": "public"}


def test_resolve_password_change_bumps_pwd_version():
    prev = {"mode": "password", "access_password": "old", "pwd_version": 2}
    _, pwd_v, _, owner = resolve_access("new", None, prev)
    assert pwd_v == 3
    assert owner["pwd_version"] == 3


def test_resolve_password_unchanged_keeps_pwd_version():
    prev = {"mode": "password", "access_password": "same", "pwd_version": 2}
    _, pwd_v, _, _ = resolve_access("same", None, prev)
    assert pwd_v == 2


def test_resolve_restricted_normalizes():
    eff, _, acc_v, owner = resolve_access(
        None, {"mode": "restricted", "emails": [" A@x.com ", "a@x.com"], "org_allowed": True}, None
    )
    assert eff == {"mode": "restricted", "emails": ["a@x.com"], "org_allowed": True}
    assert acc_v == 1
    assert owner["mode"] == "restricted"
    assert owner["emails"] == ["a@x.com"]
    assert owner["org_allowed"] is True
    assert owner["access_version"] == 1


def test_resolve_restricted_change_bumps_access_version():
    prev = {"mode": "restricted", "emails": ["a@x.com"], "org_allowed": False, "access_version": 2}
    _, _, acc_v, _ = resolve_access(
        None, {"mode": "restricted", "emails": ["a@x.com", "b@x.com"], "org_allowed": False}, prev
    )
    assert acc_v == 3


def test_resolve_restricted_empty_degrades_to_public():
    eff, *_ = resolve_access(None, {"mode": "restricted", "emails": [], "org_allowed": False}, None)
    assert eff == {"mode": "public"}


def test_parse_emails_splits_and_validates():
    valid, invalid = parse_emails("a@x.com, b@x.com foo@ ; c@y.io")
    assert valid == ["a@x.com", "b@x.com", "c@y.io"]
    assert invalid == ["foo@"]


def test_parse_emails_accepts_list():
    valid, invalid = parse_emails(["A@x.com", "bad"])
    assert valid == ["a@x.com"]
    assert invalid == ["bad"]


def test_access_from_owner_side_password():
    entry = {"mode": "password", "requires_password": True, "access_password": "s3cret"}
    assert access_from_owner_side(entry) == {"mode": "password", "password": "s3cret"}


def test_access_from_owner_side_restricted():
    entry = {"mode": "restricted", "emails": ["a@x.com"], "org_allowed": True}
    assert access_from_owner_side(entry) == {
        "mode": "restricted", "emails": ["a@x.com"], "org_allowed": True,
    }


def test_access_from_owner_side_public_and_legacy():
    assert access_from_owner_side({"mode": "public"}) == {"mode": "public"}
    assert access_from_owner_side({"requires_password": True, "access_password": "p"}) == {
        "mode": "password", "password": "p",
    }
    assert access_from_owner_side({}) == {"mode": "public"}


def test_resolve_target_static_html(tmp_path):
    root = tmp_path / "artifacts"
    art = root / "sales"
    _write(art / "metadata.json", '{"type": "html-report", "primary": "report.html"}')
    _write(art / "report.html", "<html></html>")
    target, pub_dir, key, is_fs = resolve_publish_target(art, [root])
    assert is_fs is False
    assert target == (art / "report.html")
    assert pub_dir == art
    assert key == "report.html"


def test_resolve_target_static_addressed_by_file(tmp_path):
    root = tmp_path / "artifacts"
    art = root / "sales"
    _write(art / "metadata.json", '{"type": "html-report", "primary": "report.html"}')
    f = _write(art / "report.html", "<html></html>")
    target, pub_dir, key, is_fs = resolve_publish_target(f, [root])
    assert is_fs is False
    assert key == "report.html"
    assert pub_dir == art


def test_resolve_target_fullstack(tmp_path):
    root = tmp_path / "artifacts"
    art = root / "app"
    _write(art / "metadata.json", '{"type": "fullstack-stateless-app", "primary": "static/index.html"}')
    _write(art / "backend.py", "print(1)")
    _write(art / "static" / "index.html", "<html></html>")
    target, pub_dir, key, is_fs = resolve_publish_target(art, [root])
    assert is_fs is True
    assert target == art
    assert pub_dir == art
    assert key == "index.html"


class _FakePrompt:
    """Feeds scripted answers to a prompt_or_cancel-compatible callable."""

    def __init__(self, answers):
        self._answers = list(answers)
        self.asked = []

    async def __call__(self, label, *, default="", password=False,
                       choices=None, choices_display="", allow_cancel=True, default_text=""):
        self.asked.append(label)
        return self._answers.pop(0)


@pytest.mark.asyncio
async def test_prompt_access_public():
    fp = _FakePrompt(["public"])
    assert await prompt_access(fp) == {"mode": "public"}


@pytest.mark.asyncio
async def test_prompt_access_password():
    fp = _FakePrompt(["password", "hunter2"])
    assert await prompt_access(fp) == {"mode": "password", "password": "hunter2"}


@pytest.mark.asyncio
async def test_prompt_access_password_empty_reprompts_then_cancel():
    fp = _FakePrompt(["password", "", None])
    assert await prompt_access(fp) is None


@pytest.mark.asyncio
async def test_prompt_access_restricted():
    fp = _FakePrompt(["restricted", "a@x.com, b@x.com", "y"])
    assert await prompt_access(fp) == {
        "mode": "restricted", "emails": ["a@x.com", "b@x.com"], "org_allowed": True,
    }


@pytest.mark.asyncio
async def test_prompt_access_cancel_at_mode():
    fp = _FakePrompt([None])
    assert await prompt_access(fp) is None
