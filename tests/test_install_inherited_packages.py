"""A package the pad already inherits is not installed a second time.

A pad sees the interpreter's site-packages (the Web image's pinned libraries,
or the Desktop app's), but `uv pip install --python <pad>` does not, so listing
an inherited package used to install a second, unpinned copy that shadowed the
pinned one.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import anton.core.backends.local as local
from anton.core.backends.local import LocalScratchpadRuntime, _inherited_requirements

_DEFAULTS = dict(coding_provider="anthropic", coding_model="", coding_api_key="", coding_base_url="")


def _site_with(tmp_path: Path, *dists: tuple[str, str], pth: tuple[str, ...] = (),
               no_record: tuple[str, ...] = ()) -> Path:
    """A site-packages folder holding `dists`.

    A name in `pth` is laid out like pywin32: its module sits in a subfolder
    reached only through its own top-level `.pth` file. A name in `no_record`
    has no RECORD, so its files can't be listed.
    """
    site_dir = tmp_path / "site-packages"
    for name, version in dists:
        module = name.replace("-", "_")
        info = site_dir / f"{module}-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        if name in pth:
            files = [f"{module}.pth", f"lib/{module}.py"]
        else:
            files = [f"{module}/__init__.py"]
        for f in files:  # importlib.metadata lists only files that exist
            (site_dir / f).parent.mkdir(parents=True, exist_ok=True)
            (site_dir / f).write_text("")
        if name not in no_record:
            record = [*files, f"{info.name}/METADATA", f"{info.name}/RECORD"]
            (info / "RECORD").write_text("".join(f"{f},,\n" for f in record))
    return site_dir


@pytest.fixture
def inherited_site(tmp_path, monkeypatch):
    site_dir = _site_with(tmp_path, ("requests", "2.34.2"), ("python-docx", "1.2.0"))
    monkeypatch.setattr("site.getsitepackages", lambda: [str(site_dir)])
    return site_dir


def test_a_package_that_needs_its_own_pth_is_not_inherited(tmp_path, monkeypatch):
    # pywin32's modules are reached through pywin32.pth, which a pad does not
    # read from a folder it inherits, so it must still be installed in the pad.
    site_dir = _site_with(tmp_path, ("pywin32", "311"), ("requests", "2.34.2"), pth=("pywin32",))
    monkeypatch.setattr("site.getsitepackages", lambda: [str(site_dir)])
    assert _inherited_requirements(["pywin32", "requests"]) == ["requests"]


def test_a_package_without_a_file_list_is_not_inherited(tmp_path, monkeypatch):
    site_dir = _site_with(tmp_path, ("requests", "2.34.2"), no_record=("requests",))
    monkeypatch.setattr("site.getsitepackages", lambda: [str(site_dir)])
    assert _inherited_requirements(["requests"]) == []


def test_bare_names_installed_in_the_interpreter_are_inherited(inherited_site):
    asked = ["requests", "Python_Docx", "requests==2.0", "requests[socks]",
             'requests; python_version > "3"', "pdfplumber"]
    assert _inherited_requirements(asked) == ["requests", "Python_Docx"]


class _Spawns:
    def __init__(self):
        self.argv: list[list[str]] = []

    async def __call__(self, *argv, **kwargs):
        self.argv.append(list(argv))
        raise RuntimeError("stop after recording the install command")


@pytest.fixture
def pad(tmp_path, monkeypatch):
    pad = LocalScratchpadRuntime(name="probe", _venvs_base=tmp_path / "venvs", **_DEFAULTS)

    async def no_venv_work(fn):
        pad._venv_python = str(tmp_path / "pad-python")

    monkeypatch.setattr(pad, "_run_venv_work", no_venv_work)
    monkeypatch.setattr(pad, "_uv", lambda: "/fake/uv")
    return pad


def test_an_inherited_package_is_not_installed_again(pad, inherited_site, monkeypatch):
    spawns = _Spawns()
    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", spawns)

    assert asyncio.run(pad.install_packages(["requests", "python-docx"])) == "All packages already installed."
    assert spawns.argv == []
    # Not recorded as installed in the pad: that record is saved with the pad
    # and trusted when it is reused, maybe under a host without the package.
    assert pad._installed_packages == set()


def test_a_reused_pad_installs_a_package_its_host_no_longer_has(pad, tmp_path, monkeypatch):
    spawns = _Spawns()
    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", spawns)
    # Host A had requests: the listing is skipped.
    monkeypatch.setattr("site.getsitepackages", lambda: [str(_site_with(tmp_path / "a", ("requests", "2.34.2")))])
    assert asyncio.run(pad.install_packages(["requests"])) == "All packages already installed."
    # The same pad under host B, which lacks it: the installer must run.
    monkeypatch.setattr("site.getsitepackages", lambda: [str(_site_with(tmp_path / "b"))])
    with pytest.raises(RuntimeError, match="stop after recording"):
        asyncio.run(pad.install_packages(["requests"]))
    assert spawns.argv and spawns.argv[-1][-1] == "requests"


def test_a_pinned_or_missing_package_still_goes_to_the_installer(pad, inherited_site, monkeypatch):
    spawns = _Spawns()
    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", spawns)

    with pytest.raises(RuntimeError, match="stop after recording"):
        asyncio.run(pad.install_packages(["requests", "requests==2.0", "pdfplumber"]))

    (argv,) = spawns.argv
    assert argv[-2:] == ["requests==2.0", "pdfplumber"]
    assert "requests" not in argv
