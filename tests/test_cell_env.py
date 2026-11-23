"""What a real cell started with no sandbox policy finds in its environment.

The parent is the real test environment plus planted server secrets. The cell
reports variable names, and a SHA-256 for each value a test compares, so a
failing assertion never prints a value.
"""

from __future__ import annotations

import ast
import getpass
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import anton
from anton.chat_session import get_runtime_factory
import anton.core.backends.local as local
from anton.core.backends.local import (
    LocalScratchpadRuntime,
    _apply_workspace_overlay,
    _overlay_for_agent_code,
)
from anton.core.backends.sandbox import agent_env

_WINDOWS = sys.platform == "win32"

_PLANTED_SECRETS = {
    "COWORK_AUTH_TOKEN": "planted-cowork-token",
    "DATABASE_URI": "postgresql://cowork:planted-pw@db/cowork",
    "MINDS_API_KEY": "planted-minds-key",
    "ANTON_MINDS_API_KEY": "planted-anton-minds-key",
    "GOOGLE_DRIVE_CLIENT_SECRET": "planted-oauth-secret",
    "DS_INHERITED__PASSWORD": "planted-inherited-ds",
    "UNLISTED_TOOL_TOKEN": "planted-unlisted",
}

_REPORT = """
import getpass, hashlib, json, os, site, tempfile
from pathlib import Path
def h(v):
    return None if v is None else hashlib.sha256(v.encode()).hexdigest()
print(json.dumps({
    "names": sorted(os.environ),
    "hashes": {k: h(v) for k, v in os.environ.items()},
    "home": h(str(Path.home())),
    "tmp": h(os.path.realpath(tempfile.gettempdir())),
    "user": h(getpass.getuser()),
    "user_site": site.ENABLE_USER_SITE,
}))
"""


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def planted(monkeypatch, tmp_path) -> dict[str, str]:
    """The real environment plus server secrets, and home and temp folders no default would pick."""
    home = tmp_path / "planted-home"
    tmp = tmp_path / "planted-tmp"
    home.mkdir()
    tmp.mkdir()
    folders = (
        {"USERPROFILE": str(home), "TEMP": str(tmp), "TMP": str(tmp)}
        if _WINDOWS
        else {"HOME": str(home), "TMPDIR": str(tmp)}
    )
    for key, value in {**_PLANTED_SECRETS, **folders}.items():
        monkeypatch.setenv(key, value)
    return {"home": str(home), "tmp": os.path.realpath(tmp)}


async def _report(tmp_path: Path, code: str = _REPORT, **pad_kwargs) -> dict | list:
    """Run ``code`` in a real pad and parse the JSON on its last output line."""
    coding = {
        "coding_provider": "openai",
        "coding_model": "",
        "coding_api_key": "planted-coding-key",
        "coding_base_url": "",
    }
    pad = LocalScratchpadRuntime(
        "cell-env",
        workspace_path=tmp_path / "project",
        **{**coding, **pad_kwargs},
    )
    (tmp_path / "project").mkdir(exist_ok=True)
    await pad.start()
    try:
        cell = await pad.execute(code)
        assert cell.error is None, cell.error
        return json.loads(cell.stdout.strip().splitlines()[-1])
    finally:
        await pad.close()


async def test_a_cell_holds_no_server_secret_and_keeps_its_own_home_temp_and_key(tmp_path, planted):
    report = await _report(tmp_path, scratchpad_ds_env={"DS_PAD__HOST": "pad-host"})

    assert not set(_PLANTED_SECRETS) & set(report["names"])
    assert report["hashes"]["DS_PAD__HOST"] == _sha("pad-host")
    assert report["hashes"]["OPENAI_API_KEY"] == _sha("planted-coding-key")
    assert report["hashes"]["PATH"] == _sha(os.environ["PATH"])
    assert report["home"] == _sha(planted["home"])
    assert report["tmp"] == _sha(planted["tmp"])
    assert report["user"] == _sha(getpass.getuser())


async def test_a_cell_with_no_connections_of_its_own_gets_no_inherited_ds_value(tmp_path, planted):
    report = await _report(tmp_path, scratchpad_ds_env=None)

    assert "DS_INHERITED__PASSWORD" not in report["names"]


async def test_a_cli_cell_keeps_the_servers_environment(tmp_path, planted):
    # The full report of a server environment can pass the cell's output cap.
    present = await _report(
        tmp_path,
        f"import json, os; print(json.dumps([n for n in {sorted(_PLANTED_SECRETS)!r} if n in os.environ]))",
        scratchpad_ds_env=None,
        inherit_server_env=True,
    )

    assert sorted(present) == sorted(_PLANTED_SECRETS)


async def test_a_pod_shaped_cell_keeps_the_turn_overlay_and_user_site_off(tmp_path, planted, monkeypatch):
    """Regression guard: passes before the change too."""
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")

    report = await _report(tmp_path, workspace_env_overlay={"ANTON_CLOUD_TURN": "1"})

    assert report["hashes"]["ANTON_CLOUD_TURN"] == _sha("1")
    assert report["hashes"]["PYTHONNOUSERSITE"] == _sha("1")
    assert report["user_site"] is False


def _windows_child(parent: dict[str, str], dotenv: dict[str, str]) -> dict[str, str]:
    """The cell's env built the way ``_start_process`` builds it on Windows."""
    env = dict(parent)
    overlay = _overlay_for_agent_code(dotenv, windows=True)
    overlaid = _apply_workspace_overlay(env, overlay)
    return agent_env(env, overlay_keys=overlaid, ds_env={}, host_basics=True)


def test_on_windows_a_lowercase_dotenv_name_cannot_carry_or_shadow_a_server_value():
    parent = {
        "PATH": "C:\\Windows",
        "GOOGLE_DRIVE_CLIENT_SECRET": "server-secret",
        "OPENAI_BASE_URL": "https://server.example/v1",
    }
    dotenv = {
        "google_drive_client_secret": "from-dotenv",
        "pythonpath": "C:\\project\\evil",
        "openai_base_url": "https://dotenv.example/v1",
    }

    child = _windows_child(parent, dotenv)

    upper = {key.upper(): value for key, value in child.items()}
    assert "GOOGLE_DRIVE_CLIENT_SECRET" not in upper
    assert "PYTHONPATH" not in upper
    assert [k for k in child if k.upper() == "OPENAI_BASE_URL"] == ["OPENAI_BASE_URL"]
    assert child["OPENAI_BASE_URL"] == "https://server.example/v1"


async def test_on_windows_a_real_cell_gets_no_lowercase_dotenv_secret(tmp_path, planted, monkeypatch):
    """Runs on every OS: the Windows switch is set for the pad, the cell is real."""
    monkeypatch.setattr(local, "_WINDOWS", True)

    report = await _report(
        tmp_path,
        workspace_env_overlay={"google_drive_client_secret": "from-dotenv", "pythonpath": "evil-path"},
    )

    upper_names = {name.upper() for name in report["names"]}
    assert "GOOGLE_DRIVE_CLIENT_SECRET" not in upper_names
    assert "pythonpath" not in report["names"]
    assert not {_sha("from-dotenv"), _sha("evil-path")} & set(report["hashes"].values())


async def test_a_cell_pointed_at_another_endpoint_never_gets_the_mindshub_key(tmp_path, planted):
    report = await _report(
        tmp_path,
        coding_provider="openai-compatible",
        coding_api_key="",
        coding_base_url="http://127.0.0.1:11434/v1",
    )

    assert _sha(_PLANTED_SECRETS["ANTON_MINDS_API_KEY"]) not in set(report["hashes"].values())


async def test_the_cli_runtime_factory_builds_cells_with_the_servers_environment(tmp_path, planted):
    factory = get_runtime_factory(
        SimpleNamespace(backend="local"), cancel_ends_turn=False, inherit_server_env=True
    )
    (tmp_path / "project").mkdir()
    pad = factory(
        name="cli-env",
        coding_provider="openai",
        coding_model="",
        coding_api_key="",
        coding_base_url="",
        cells=None,
        workspace_path=tmp_path / "project",
    )
    await pad.start()
    try:
        cell = await pad.execute("import os; print('UNLISTED_TOOL_TOKEN' in os.environ)")
        assert cell.stdout.strip() == "True"
    finally:
        await pad.close()


def test_every_cli_call_for_a_runtime_factory_keeps_the_servers_environment():
    """A new CLI entry through get_runtime_factory must pass the opt-out."""
    package = Path(anton.__file__).parent
    calls = []
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            called = (
                node.func.id if isinstance(node.func, ast.Name)
                else node.func.attr if isinstance(node.func, ast.Attribute)
                else None
            )
            if called == "get_runtime_factory":
                keywords = {k.arg: k.value for k in node.keywords}
                opted_out = (
                    isinstance(keywords.get("inherit_server_env"), ast.Constant)
                    and keywords["inherit_server_env"].value is True
                )
                calls.append((path.relative_to(package).as_posix(), node.lineno, opted_out))

    assert len(calls) >= 5
    assert [call for call in calls if not call[2]] == []
