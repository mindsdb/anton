"""ENG-1424: one CLI session must resolve ONE identity for every operation.

`handle_publish_or_preview` used to build its own `AntonSettings()`. That is a
second resolve, and by the time it runs `_ensure_workspace` (cli.py) has copied
the project `.anton/.env` into `os.environ` — which pydantic-settings ranks
above every `env_file`. So the publish tool authenticated as a different
account than the LLM client, `/publish` and `/unpublish`, which were all built
from the settings resolved at startup. Reports published under one account were
invisible to `/unpublish` in the very session that created them.

The containment test below is the other half: this handler is reachable only
from the CLI. The desktop harness rebinds PUBLISH_TOOL to its own server-side
handler and the cloud pod registers no host tools at all, so the fix cannot
change behaviour on those surfaces — but only for as long as that stays true.
"""
from __future__ import annotations

import os
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


def _settings(root: Path, api_key: str):
    s = mock.Mock()
    s.minds_api_key = api_key
    s.publish_url = "https://view.test"
    s.minds_ssl_verify = True
    s.artifacts_dir = str(root)
    return s


def _session(tmp_path: Path, settings):
    s = mock.Mock()
    s._console = Console()
    ws = mock.Mock()
    ws.base = str(tmp_path)
    s._workspace = ws
    s._settings = settings
    return s


@pytest.mark.asyncio
async def test_publish_uses_session_settings_not_promoted_env(tmp_path, monkeypatch):
    """The key reaching `publish` is the session's, whatever os.environ holds.

    Mirrors the real failure: the session was built from `~/.cowork/.env`
    (account A) while `apply_env_to_process` had promoted the project vault's
    key (account B) into os.environ.
    """
    f = _artifact(tmp_path)
    root = f.parent.parent

    # What `_ensure_workspace` promotes — a DIFFERENT account than the session.
    monkeypatch.setenv("ANTON_MINDS_API_KEY", "KEY_B_PROMOTED")

    fake_publish = mock.Mock(
        return_value={"view_url": "u", "report_id": "r", "md5": "m", "version": 1}
    )
    session = _session(tmp_path, _settings(root, "KEY_A_SESSION"))

    with mock.patch("anton.publisher.publish", fake_publish), mock.patch("webbrowser.open"):
        await tools.handle_publish_or_preview(
            session, {"file_path": str(f), "action": "publish"}
        )

    assert fake_publish.call_count == 1
    assert fake_publish.call_args.kwargs["api_key"] == "KEY_A_SESSION"


@pytest.mark.asyncio
async def test_publish_falls_back_when_host_passed_no_settings(tmp_path, monkeypatch):
    """A host that passes no settings still gets a working resolve.

    `ChatSessionConfig.settings` is typed `CoreSettings | None`, and
    `CoreSettings` carries none of the publish fields — so the handler must
    fall back rather than assume the session has them.
    """
    f = _artifact(tmp_path)
    root = f.parent.parent
    monkeypatch.delenv("ANTON_MINDS_API_KEY", raising=False)

    fake_publish = mock.Mock(
        return_value={"view_url": "u", "report_id": "r", "md5": "m", "version": 1}
    )
    session = _session(tmp_path, None)

    with (
        mock.patch("anton.publisher.publish", fake_publish),
        mock.patch(
            "anton.config.settings.AntonSettings",
            return_value=_settings(root, "KEY_FROM_FALLBACK"),
        ),
        mock.patch("webbrowser.open"),
    ):
        await tools.handle_publish_or_preview(
            session, {"file_path": str(f), "action": "publish"}
        )

    assert fake_publish.call_args.kwargs["api_key"] == "KEY_FROM_FALLBACK"


@pytest.mark.asyncio
async def test_bare_core_settings_falls_back_too(tmp_path, monkeypatch):
    """A session holding a settings object without the publish fields."""
    f = _artifact(tmp_path)
    root = f.parent.parent
    monkeypatch.delenv("ANTON_MINDS_API_KEY", raising=False)

    class _NoPublishFields:
        """Stands in for a bare CoreSettings — no `minds_api_key` at all."""

    fake_publish = mock.Mock(
        return_value={"view_url": "u", "report_id": "r", "md5": "m", "version": 1}
    )
    session = _session(tmp_path, _NoPublishFields())

    with (
        mock.patch("anton.publisher.publish", fake_publish),
        mock.patch(
            "anton.config.settings.AntonSettings",
            return_value=_settings(root, "KEY_FROM_FALLBACK"),
        ),
        mock.patch("webbrowser.open"),
    ):
        await tools.handle_publish_or_preview(
            session, {"file_path": str(f), "action": "publish"}
        )

    assert fake_publish.call_args.kwargs["api_key"] == "KEY_FROM_FALLBACK"


def test_antons_own_publish_tool_still_routes_to_the_session_aware_handler():
    """Half of the containment story — the half a test in THIS repo can hold.

    `DEFAULT_SESSION_TOOLS` is what the CLI registers, so this pins that the
    CLI's `publish_or_preview` really is the handler the ENG-1424 fix changed
    (and not, say, a wrapper that re-resolves settings on the way in).

    What it deliberately does NOT prove — the earlier docstring claimed it did:
    that no OTHER surface registers anton's handler. Both other surfaces build
    their tool list in a different repo or a different module, so a foreign
    registrant is structurally invisible from here. That containment is
    asserted where it actually lives:
      * desktop — cowork-server's `harnesses/anton_harness/harness.py` rebinds
        PUBLISH_TOOL to `build_cowork_publish_tool()`; its guard belongs in
        cowork-server's suite, not this one;
      * web/pod — covered by the allowlist test below, which CAN see it because
        the pod's tool policy lives in this repo.
    """
    publish_tools = [t for t in tools.DEFAULT_SESSION_TOOLS if t.name == "publish_or_preview"]
    assert len(publish_tools) == 1
    assert publish_tools[0].handler is tools.handle_publish_or_preview


def test_cloud_turn_allowlist_excludes_the_publish_tool():
    """The hosted pod must not expose a host publish tool (ENG-1424 containment).

    Asserted against the real allowlist object, not the source text. An earlier
    version grepped session.py for the literal `tools=[]`, which failed both
    ways: it would false-alarm on a behaviour-preserving refactor (`tools=list()`,
    or the list moved to a constant), and it would pass while the pod actually
    registered the tool, because the string it required is unrelated to the tool
    policy it claimed to check.

    `CLOUD_TOOL_ALLOWLIST` is the real gate — `cloud_turn/session.py` passes it
    as `tool_allowlist=`, and core drops every tool whose name is absent from it.
    """
    from anton.cloud_turn.session import CLOUD_TOOL_ALLOWLIST

    assert "publish_or_preview" not in CLOUD_TOOL_ALLOWLIST
    # The allowlist is only a containment guarantee if it is actually a closed
    # set — an empty or absent one would let everything through.
    assert CLOUD_TOOL_ALLOWLIST, "the cloud allowlist is empty — nothing is contained"
