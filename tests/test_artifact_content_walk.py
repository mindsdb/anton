from __future__ import annotations

import contextlib
import errno
import os
from pathlib import Path

import pytest

from anton.core.artifacts.internal_files import is_bytecode_name
from anton.core.artifacts.store import iter_content_files
from anton.core.tools.tool_handlers import FileStamp, _artifact_files
from anton.publish_access import _user_files


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _names(folder: Path) -> set[str]:
    return {p.relative_to(folder).as_posix() for p, _ in iter_content_files(folder)}


class _StatFails:
    """A scan entry whose stat call fails, as for a file removed mid-walk."""

    def __init__(self, entry):
        self._entry = entry

    def __getattr__(self, name):
        return getattr(self._entry, name)

    def stat(self, *, follow_symlinks=True):
        raise OSError(errno.EIO, "boom", self._entry.path)


def _fail_stat_for(monkeypatch, path: Path) -> None:
    real_scandir = os.scandir
    target = os.fspath(path)

    @contextlib.contextmanager
    def scandir(where):
        with real_scandir(where) as it:
            yield [_StatFails(e) if e.path == target else e for e in it]

    monkeypatch.setattr(os, "scandir", scandir)


def test_prunes_non_content_top_level_but_keeps_nested_namesakes(tmp_path):
    _touch(tmp_path / "index.html")
    _touch(tmp_path / "prd.md")
    _touch(tmp_path / "metadata.json")
    _touch(tmp_path / ".revisions" / "entries" / "old.html")
    _touch(tmp_path / "static" / "prd.md")

    assert _names(tmp_path) == {"index.html", "static/prd.md"}


def test_skips_symlinked_file_and_directory(tmp_path):
    outside = tmp_path / "outside"
    _touch(outside / "secret.txt")
    folder = tmp_path / "artifact"
    _touch(folder / "real.txt")
    (folder / "link.txt").symlink_to(outside / "secret.txt")
    (folder / "linkdir").symlink_to(outside, target_is_directory=True)
    _touch(folder / "sub" / "keep.txt")
    (folder / "sub" / "nested_link").symlink_to(outside, target_is_directory=True)

    assert _names(folder) == {"real.txt", "sub/keep.txt"}


def test_yields_lstat_result(tmp_path):
    path = _touch(tmp_path / "a.txt", "hello")

    [(found, st)] = list(iter_content_files(tmp_path))

    assert found == path
    assert st.st_size == 5


def test_file_whose_stat_fails_is_skipped_others_yielded(tmp_path, monkeypatch):
    _touch(tmp_path / "good.txt")
    bad = _touch(tmp_path / "bad.txt")
    _touch(tmp_path / "sub" / "also_good.txt")
    _fail_stat_for(monkeypatch, bad)

    assert _names(tmp_path) == {"good.txt", "sub/also_good.txt"}


def test_unreadable_directory_is_skipped_others_yielded(tmp_path, monkeypatch):
    _touch(tmp_path / "good.txt")
    locked = tmp_path / "locked"
    _touch(locked / "hidden.txt")
    real_scandir = os.scandir

    def flaky_scandir(path):
        if os.fspath(path) == os.fspath(locked):
            raise PermissionError(errno.EACCES, "denied", os.fspath(path))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", flaky_scandir)

    assert _names(tmp_path) == {"good.txt"}


def test_missing_folder_yields_nothing(tmp_path):
    assert list(iter_content_files(tmp_path / "nope")) == []


def test_user_files_survive_one_unreadable_entry(tmp_path, monkeypatch):
    _touch(tmp_path / "index.html")
    _fail_stat_for(monkeypatch, _touch(tmp_path / "bad.txt"))

    assert [p.name for p in _user_files(tmp_path)] == ["index.html"]


def test_unreadable_file_leaves_the_others_in_the_snapshot(tmp_path, monkeypatch):
    good = _touch(tmp_path / "index.html")
    os.utime(good, (1000, 1000))
    _fail_stat_for(monkeypatch, _touch(tmp_path / "gone.html"))

    assert _artifact_files(tmp_path).stamps == {"index.html": FileStamp(mtime_ns=1000 * 10**9, size=1)}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("__pycache__", True),
        ("x.pyc", True),
        ("X.PYC", True),
        ("backend.cpython-313.pyc", True),
        ("pycache", False),
        ("x.py", False),
        ("static", False),
        ("pyc", False),
        ("x.pyc.bak", False),
    ],
)
def test_is_bytecode_name(name, expected):
    assert is_bytecode_name(name) is expected


def test_skips_bytecode_at_any_depth(tmp_path):
    _touch(tmp_path / "backend.py")
    _touch(tmp_path / "__pycache__" / "backend.cpython-313.pyc")
    _touch(tmp_path / "routes" / "x.py")
    _touch(tmp_path / "routes" / "__pycache__" / "x.cpython-313.pyc")
    _touch(tmp_path / "stray.pyc")
    _touch(tmp_path / "static" / "prd.md")

    assert _names(tmp_path) == {"backend.py", "routes/x.py", "static/prd.md"}


def test_bytecode_stays_out_of_the_content_snapshot(tmp_path):
    index = _touch(tmp_path / "index.html")
    os.utime(index, (1000, 1000))
    cache = _touch(tmp_path / "routes" / "__pycache__" / "x.cpython-313.pyc")
    os.utime(cache, (5000, 5000))
    stray = _touch(tmp_path / "stray.pyc")
    os.utime(stray, (6000, 6000))

    assert _artifact_files(tmp_path).stamps == {"index.html": FileStamp(mtime_ns=1000 * 10**9, size=1)}
