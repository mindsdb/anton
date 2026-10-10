"""ScratchpadManager.probe_packages reuses its answer until sys.path changes.

cowork-server builds a manager, and so a probe, per turn, and importlib.metadata
re-reads every installed distribution's METADATA on each call.
"""

from __future__ import annotations

import importlib.metadata
import itertools
import os
import shutil
from pathlib import Path

import pytest

from anton.core.backends import manager as manager_mod
from anton.core.backends.manager import ScratchpadManager


@pytest.fixture(autouse=True)
def _no_earlier_probe(monkeypatch):
    monkeypatch.setattr(manager_mod, "_last_probe", None)


@pytest.fixture()
def site(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "site-packages"
    path.mkdir()
    monkeypatch.syspath_prepend(str(path))
    return path


# Seconds past 2020-09-13, one per call: never a value the kernel stamped.
_fresh_mtimes = itertools.count(1_600_000_000)


def _move_mtime_on(directory: Path) -> None:
    """Give `directory` an mtime it has not had before.

    Linux stamps mtimes from a coarse clock, so two changes a few milliseconds
    apart can leave a directory's mtime where it was, and importlib.metadata's
    own listing cache would miss that too. The test sets the mtime itself
    rather than depend on the clock.
    """
    st = directory.stat()
    os.utime(directory, ns=(st.st_atime_ns, next(_fresh_mtimes) * 1_000_000_000))


def _install(site: Path, name: str) -> Path:
    info = site / f"{name.replace('-', '_')}-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n", encoding="utf-8"
    )
    _move_mtime_on(site)
    return info


def test_unchanged_sys_path_is_probed_once(monkeypatch):
    calls: list[None] = []
    real = importlib.metadata.distributions

    def counting(**kwargs):
        calls.append(None)
        return real(**kwargs)

    monkeypatch.setattr(importlib.metadata, "distributions", counting)

    first = ScratchpadManager.probe_packages()
    second = ScratchpadManager.probe_packages()

    assert first == second
    assert "pyyaml" in {name.lower() for name in first}
    assert len(calls) == 1


def test_install_and_removal_are_seen_on_the_next_probe(site):
    assert "eng-fake-dist" not in ScratchpadManager.probe_packages()

    info = _install(site, "eng-fake-dist")
    assert "eng-fake-dist" in ScratchpadManager.probe_packages()

    shutil.rmtree(info)
    _move_mtime_on(site)
    assert "eng-fake-dist" not in ScratchpadManager.probe_packages()


def test_new_sys_path_entry_is_seen_on_the_next_probe(tmp_path, monkeypatch):
    extra = tmp_path / "extra-site"
    extra.mkdir()
    _install(extra, "eng-fake-extra")
    assert "eng-fake-extra" not in ScratchpadManager.probe_packages()

    monkeypatch.syspath_prepend(str(extra))
    assert "eng-fake-extra" in ScratchpadManager.probe_packages()


def test_each_probe_returns_its_own_list():
    first = ScratchpadManager.probe_packages()
    first.append("not-installed")
    assert "not-installed" not in ScratchpadManager.probe_packages()
