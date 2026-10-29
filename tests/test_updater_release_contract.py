"""The automatic updater must install the selected release's reviewed packages."""

import io
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import urllib.error
import zipfile

from packaging.version import Version
import pytest

import anton
import anton.updater as updater


TAG = "v9.9.9.9.9"
COMMIT = "1" * 40
TAG_OBJECT = "f" * 40
CONTRACT = b"release-dependency==1.0\n"
UV = shutil.which("uv")


def update_fixture(monkeypatch, tmp_path, *, contract=CONTRACT, refs=None, install=None):
    calls = {"install": [], "downloads": [], "git": []}
    monkeypatch.setattr(updater.shutil, "which", lambda _: UV or "/usr/bin/uv")
    monkeypatch.setattr(updater, "_fetch_latest_release_tag", lambda: TAG)
    monkeypatch.setattr(updater, "_SKIP_MARKER", tmp_path / "skip")
    monkeypatch.setattr(updater, "_read_installed_anton_version", lambda: Version(TAG[1:]))
    monkeypatch.setattr(anton, "__version__", "2.0.0")

    def download(url, **kwargs):
        calls["downloads"].append(url)
        if contract is None:
            raise urllib.error.HTTPError(url, 404, "missing contract", {}, None)
        return io.BytesIO(contract)

    def run(args, **kwargs):
        if args[0] == "git":
            calls["git"].append(args)
            return SimpleNamespace(returncode=0, stdout=refs if refs is not None else f"{TAG_OBJECT}\trefs/tags/{TAG}\n{COMMIT}\trefs/tags/{TAG}^{{}}\n")
        calls["install"].append(args)
        if install:
            return install(args, **kwargs)
        if "--constraints" in args:
            path = Path(args[args.index("--constraints") + 1])
            calls["contract"] = path.read_bytes()
            calls["contract_path"] = path
        return SimpleNamespace(returncode=0, stdout=b"")

    monkeypatch.setattr(updater.urllib.request, "urlopen", download)
    monkeypatch.setattr(updater.subprocess, "run", run)
    return calls


def test_update_binds_code_and_contract_to_the_same_peeled_commit(monkeypatch, tmp_path):
    calls = update_fixture(monkeypatch, tmp_path)
    result = {}
    updater._check_and_update(result, settings=None)

    assert calls["downloads"] == [f"https://raw.githubusercontent.com/mindsdb/anton/{COMMIT}/anton/release-requirements.txt"]
    assert calls["install"] == [["uv", "tool", "install", f"git+https://github.com/mindsdb/anton.git@{COMMIT}", "--force", "--constraints", str(calls["contract_path"])]]
    assert calls["contract"] == CONTRACT
    assert not calls["contract_path"].exists()
    assert result["new_version"] == TAG[1:]


@pytest.mark.parametrize(("contract", "refs"), [(None, None), (b"", None), (CONTRACT, "invalid\trefs/tags/" + TAG)])
def test_unavailable_release_contract_never_replaces_the_install(monkeypatch, tmp_path, contract, refs):
    calls = update_fixture(monkeypatch, tmp_path, contract=contract, refs=refs)
    result = {}
    updater._check_and_update(result, settings=None)
    assert calls["install"] == []
    assert "new_version" not in result


def wheel(directory, name, version, *, dependency=None):
    dist = f"{name}-{version}.dist-info"
    path = directory / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        metadata = f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\n"
        if dependency:
            metadata += f"Requires-Dist: {dependency}\n"
            archive.writestr(f"{dist}/entry_points.txt", "[console_scripts]\nrelease-tool = release_tool:main\n")
            archive.writestr("release_tool.py", "def main(): pass\n")
        archive.writestr(f"{dist}/METADATA", metadata)
        archive.writestr(f"{dist}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(f"{dist}/RECORD", "")
    return path


@pytest.mark.skipif(UV is None, reason="requires uv for the controlled-index install")
def test_newer_upload_cannot_change_the_automatic_updates_dependency(monkeypatch, tmp_path):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    wheel(wheels, "release_dependency", "1.0")
    wheel(wheels, "release_dependency", "2.0")
    target = wheel(wheels, "release_tool", "1.0", dependency="release-dependency>=1")
    tools = tmp_path / "tools"
    monkeypatch.setenv("UV_TOOL_DIR", str(tools))
    monkeypatch.setenv("UV_TOOL_BIN_DIR", str(tmp_path / "bin"))
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("UV_CONSTRAINT", "")
    monkeypatch.setenv("UV_OVERRIDE", "")
    real_run = subprocess.run

    def install(args, **kwargs):
        # Substitute only the Git project with a local wheel; retain the updater's
        # real flags and use actual uv resolution against both available versions.
        return real_run([UV, "tool", "install", str(target), *args[4:], "--python", sys.executable, "--no-config", "--no-index", "--find-links", str(wheels)], **kwargs)

    update_fixture(monkeypatch, tmp_path, install=install)
    result = {}
    updater._check_and_update(result, settings=None)
    assert result.get("new_version") == TAG[1:], result
    python = tools / "release-tool" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    installed = real_run([str(python), "-c", "from importlib.metadata import version; print(version('release-dependency'))"], check=True, capture_output=True, text=True)
    assert installed.stdout.strip() == "1.0"
