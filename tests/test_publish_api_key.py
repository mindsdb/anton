"""Tests for /publish API key handling — the 401 fix and where the key lands.

Covers:
- Bad key entered on first /publish: nothing is persisted before the publish
  call, and the 401 clears the key so the user is re-prompted (STRC-987)
- Good key: persisted to the GLOBAL vault (~/.anton/.env) after success, not
  to the project vault (ENG-1424)

Every test isolates ``Path.home()``. Before ENG-1424 these tests mocked the
project workspace but not the home directory, so `_handle_publish`'s eager
pre-validation write put ``ANTON_MINDS_API_KEY=goodkey`` into the developer's
real ``~/.anton/.env`` on every run — and passed while doing it.
"""
from __future__ import annotations

import os
import urllib.error
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A PRISTINE home per test, overriding conftest's session-wide one.

    conftest's `_no_real_home` shares one isolated home for the whole session,
    for speed (a fresh one per test cost 230s of e2e subprocess first-runs).
    That sharing is fine for "did anything touch the real home", but not for
    asserting on the vault's CONTENTS: `test_publish_never_writes_the_real_home`
    would otherwise pass on the residue
    `test_successful_publish_persists_key_to_global_vault` left behind — delete
    the persist call and it still goes green. Setting $HOME is enough;
    conftest's patched `Path.home()` defers to any test that names its own.
    """
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    return h


def _global_vault_text(home: Path) -> str:
    f = home / ".anton" / ".env"
    return f.read_text() if f.is_file() else ""


def _make_settings(tmp_path: Path, api_key: str | None = None) -> MagicMock:
    settings = MagicMock()
    settings.minds_api_key = api_key
    settings.workspace_path = str(tmp_path)
    settings.artifacts_dir = str(tmp_path / "artifacts")
    settings.publish_url = "https://4nton.ai"
    settings.minds_ssl_verify = True
    return settings


def _make_console() -> MagicMock:
    console = MagicMock()
    console.print = MagicMock()
    return console


def _make_html_file(tmp_path: Path) -> Path:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    html = artifacts_dir / "report.html"
    html.write_text("<html><title>Test</title></html>")
    return html


def _http_401() -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://4nton.ai/upload", code=401, msg="Unauthorized", hdrs=None, fp=None
    )


@pytest.mark.asyncio
async def test_401_clears_api_key(tmp_path, home):
    """When publish returns 401, the key is cleared so the user can re-enter it."""
    from anton.chat import _handle_publish

    html = _make_html_file(tmp_path)
    settings = _make_settings(tmp_path, api_key=None)
    console = _make_console()

    with (
        patch("anton.chat.prompt_or_cancel", new=AsyncMock(side_effect=["y", "wrongkey", "public"])),
        patch("anton.publisher.publish", side_effect=_http_401()),
    ):
        await _handle_publish(console, settings, file_arg=str(html))

    # Key must be cleared after 401
    assert settings.minds_api_key is None
    # ...in the GLOBAL vault, which is where /publish writes it. Clearing the
    # project vault instead is what let a rejected key survive (ENG-1424).
    assert "wrongkey" not in _global_vault_text(home)
    # User must see a helpful message, not a raw exception
    error_calls = [str(c) for c in console.print.call_args_list]
    assert any("Invalid API key" in c for c in error_calls)


@pytest.mark.asyncio
async def test_key_is_not_on_disk_when_publish_is_called(tmp_path, home):
    """STRC-987: a key that has never authenticated must not reach disk.

    Asserted AT the publish call, not after it. Checking the vault afterwards
    cannot see this bug now that the 401 handler clears the same file the
    writer used — an eager pre-validation write would be silently undone on
    this path and the test would pass while the hazard was back. What actually
    matters is the window: between "user typed a key" and "the server accepted
    it" nothing may be persisted, because not every failure is a 401 that
    triggers the cleanup (see the 500 case below).
    """
    from anton.chat import _handle_publish

    html = _make_html_file(tmp_path)
    settings = _make_settings(tmp_path, api_key=None)
    seen_at_call_time: list[str] = []

    def _capture_then_401(*a, **k):
        seen_at_call_time.append(_global_vault_text(home))
        raise _http_401()

    with (
        patch("anton.chat.prompt_or_cancel", new=AsyncMock(side_effect=["y", "typo-key", "public"])),
        patch("anton.publisher.publish", side_effect=_capture_then_401),
    ):
        await _handle_publish(_make_console(), settings, file_arg=str(html))

    assert seen_at_call_time, "publish was never reached — the test proves nothing"
    assert "typo-key" not in seen_at_call_time[0]
    # And the next session must therefore re-prompt rather than reuse it.
    assert settings.minds_api_key is None


@pytest.mark.asyncio
async def test_bad_key_survives_nothing_when_publish_fails_non_401(tmp_path, home):
    """The hole an eager write leaves: a failure that is not a 401.

    The cleanup at the 401 branch is the only thing that removes a persisted
    key, so anything else — 500, timeout, connection reset — would strand an
    unvalidated key in the global vault forever. Persisting only after success
    is what closes it.
    """
    from anton.chat import _handle_publish

    html = _make_html_file(tmp_path)
    settings = _make_settings(tmp_path, api_key=None)

    http_500 = urllib.error.HTTPError(
        url="https://4nton.ai/upload", code=500, msg="Server Error", hdrs=None, fp=None
    )

    with (
        patch("anton.chat.prompt_or_cancel", new=AsyncMock(side_effect=["y", "typo-key", "public"])),
        patch("anton.publisher.publish", side_effect=http_500),
    ):
        await _handle_publish(_make_console(), settings, file_arg=str(html))

    assert "typo-key" not in _global_vault_text(home)


@pytest.mark.asyncio
async def test_successful_publish_persists_key_to_global_vault(tmp_path, home):
    """When publish succeeds, the key is saved to ~/.anton/.env.

    ENG-1424: it used to go to the PROJECT vault, which `_ensure_workspace`
    promotes into os.environ ahead of every config file — so each project
    folder kept a frozen snapshot that outranked the current key for
    publishing alone once the real key rotated.
    """
    from anton.chat import _handle_publish

    html = _make_html_file(tmp_path)
    settings = _make_settings(tmp_path, api_key=None)
    console = _make_console()

    publish_result = {
        "view_url": "https://4nton.ai/r/abc123",
        "report_id": "abc123",
        "md5": "deadbeef",
        "version": 1,
        "unchanged": False,
    }

    with (
        patch("anton.chat.prompt_or_cancel", new=AsyncMock(side_effect=["y", "goodkey", "public"])),
        patch("anton.publisher.publish", return_value=publish_result),
        patch("webbrowser.open"),
    ):
        await _handle_publish(console, settings, file_arg=str(html))

    assert "ANTON_MINDS_API_KEY=goodkey" in _global_vault_text(home)
    # The project vault cannot be written: `_handle_publish` no longer takes a
    # workspace at all, which is the divergence removed structurally rather
    # than asserted against a mock.


@pytest.mark.asyncio
async def test_401_clears_a_key_that_is_already_on_disk(tmp_path, home):
    """A rejected key must be REMOVED from the global vault, not just from memory.

    This is the test that pins the 401-clear. The other 401 tests cannot: with
    the eager pre-validation write gone, nothing has written the key by the
    time they assert, so "the key is absent" holds even if the clear is deleted
    outright — verified by mutation (delete the clear, whole suite still green).
    Pre-seeding the vault is what makes the assertion load-bearing: it is on
    disk before the call, so only the clear can remove it.
    """
    from anton.chat import _handle_publish
    from anton.workspace import Workspace

    Workspace(home).set_secret("ANTON_MINDS_API_KEY", "stale-bad-key")
    assert "stale-bad-key" in _global_vault_text(home)  # precondition

    html = _make_html_file(tmp_path)
    settings = _make_settings(tmp_path, api_key="stale-bad-key")

    console = _make_console()
    with (
        # /publish now asks for an access mode before publishing; the key is
        # already set so no key prompts fire — just answer the Access prompt.
        patch("anton.chat.prompt_or_cancel", new=AsyncMock(side_effect=["public"])),
        patch("anton.publisher.publish", side_effect=_http_401()),
    ):
        await _handle_publish(console, settings, file_arg=str(html))

    assert settings.minds_api_key is None
    assert "stale-bad-key" not in _global_vault_text(home)
    # REMOVED, not blanked. `ANTON_MINDS_API_KEY=` is still a value: it outranks
    # a real key in a lower-precedence file, and the scratchpad launcher tests
    # membership rather than truthiness when deriving OPENAI_API_KEY, so a child
    # would get an empty credential instead of falling through.
    from anton.workspace import Workspace as _W

    assert _W(home).get_secret("ANTON_MINDS_API_KEY") is None
    assert os.environ.get("ANTON_MINDS_API_KEY") is None
    printed = " ".join(str(c) for c in console.print.call_args_list)
    assert "Invalid API key" in printed


@pytest.mark.asyncio
async def test_a_key_from_config_is_not_copied_into_the_global_vault(tmp_path, home):
    """Only a key typed at this prompt is ours to persist.

    A key that arrived from an existing config file (``~/.cowork/.env``, written
    by the desktop app) already has an owner. Copying it into ``~/.anton/.env``
    creates a second, unmanaged copy that no sign-out path in any repo scrubs —
    cowork's Sign out clears ``~/.cowork/.env`` only — so signing out of the
    desktop would leave a live credential behind and the CLI would keep
    publishing as the account the user just left.
    """
    from anton.chat import _handle_publish

    html = _make_html_file(tmp_path)
    settings = _make_settings(tmp_path, api_key="key-owned-by-the-desktop-app")

    with (
        patch("anton.chat.prompt_or_cancel", new=AsyncMock(side_effect=["public"])),
        patch("anton.publisher.publish", return_value={
            "view_url": "u", "report_id": "r", "md5": "m", "version": 1, "unchanged": False,
        }),
        patch("webbrowser.open"),
    ):
        await _handle_publish(_make_console(), settings, file_arg=str(html))

    assert "key-owned-by-the-desktop-app" not in _global_vault_text(home)


@pytest.mark.asyncio
async def test_publish_never_writes_the_real_home(tmp_path, home):
    """The suite itself must not touch the developer's real ~/.anton/.env.

    This ran for months: the tests mocked the project workspace but not the
    home directory, so the eager write persisted a 7-character all-letters key
    ("goodkey") into the real global vault on every run — indistinguishable
    from a user typo when it later 401'd.

    Asserts ABSENCE at the developer's real path, not presence at the fake one.
    An earlier version checked ``home in (home / ".anton" / ".env").parents``,
    which is true for every possible value of ``home`` and so tested nothing.

    ``REAL_HOME`` is captured at conftest import, before the fixture redirects
    ``$HOME`` — once it is armed there is no other way to name the real home,
    and ``expanduser("~")`` here would resolve to the isolated one and make the
    assertion vacuous a second time.
    """
    from anton.chat import _handle_publish
    from tests.conftest import REAL_HOME

    real_vault = Path(REAL_HOME) / ".anton" / ".env"
    before = real_vault.read_bytes() if real_vault.is_file() else None

    html = _make_html_file(tmp_path)
    settings = _make_settings(tmp_path, api_key=None)

    with (
        patch("anton.chat.prompt_or_cancel", new=AsyncMock(side_effect=["y", "goodkey", "public"])),
        patch("anton.publisher.publish", return_value={
            "view_url": "u", "report_id": "r", "md5": "m", "version": 1, "unchanged": False,
        }),
        patch("webbrowser.open"),
    ):
        await _handle_publish(_make_console(), settings, file_arg=str(html))

    after = real_vault.read_bytes() if real_vault.is_file() else None
    assert after == before, f"the real {real_vault} was modified by the test suite"
    # ...and the write did land, in the isolated home.
    assert "ANTON_MINDS_API_KEY=goodkey" in _global_vault_text(home)


@pytest.mark.asyncio
async def test_401_does_not_clear_a_key_the_global_vault_did_not_supply(tmp_path, home):
    """Review finding: the 401 handler destroyed an unrelated working key.

    The chain has four files and ``~/.cowork/.env`` outranks ``~/.anton/.env``,
    so a session can be running a key the global vault does not hold. Blanking
    it unconditionally deleted a good key AND left the one that really 401'd in
    place — making the lock-out permanent instead of fixing it.
    """
    from anton.chat import _handle_publish
    from anton.workspace import Workspace

    Workspace(home).set_secret("ANTON_MINDS_API_KEY", "GOOD_A")

    html = _make_html_file(tmp_path)
    settings = _make_settings(tmp_path, api_key="EXPIRED_B")  # came from elsewhere
    console = _make_console()

    with (
        patch("anton.chat.prompt_or_cancel", new=AsyncMock(side_effect=["public"])),
        patch("anton.publisher.publish", side_effect=_http_401()),
    ):
        await _handle_publish(console, settings, file_arg=str(html))

    assert Workspace(home).get_secret("ANTON_MINDS_API_KEY") == "GOOD_A"
    # ...and the user is told where to look instead of being told it was cleared.
    printed = " ".join(str(c) for c in console.print.call_args_list)
    assert "cowork" in printed
