import io
import json
import zipfile
from pathlib import Path

import pytest
from anton.publisher import (
    _FULLSTACK_EXCLUDED,
    _read_state_manifest,
    _zip_fullstack,
)


def _make_fullstack_dir(tmp_path: Path) -> Path:
    d = tmp_path / "art"
    d.mkdir()
    (d / "backend.py").write_text("STATE = None\n", encoding="utf-8")
    (d / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    (d / "state_manifest.json").write_text(
        json.dumps({"pk": {"name": "pk"}, "sk": {"name": "sk"}}), encoding="utf-8"
    )
    (d / ".anton_state.db").write_bytes(b"SQLite format 3\x00local-only")
    static = d / "static"
    static.mkdir()
    (static / "index.html").write_text("<html></html>", encoding="utf-8")
    return d


def test_bundle_vendors_anton_state_package(tmp_path):
    zbytes, included = _zip_fullstack(_make_fullstack_dir(tmp_path))
    names = zipfile.ZipFile(io.BytesIO(zbytes)).namelist()
    assert "anton_state/__init__.py" in names
    assert "anton_state/sqlite_driver.py" in names
    assert "anton_state/factory.py" in names


def test_bundle_includes_manifest(tmp_path):
    zbytes, included = _zip_fullstack(_make_fullstack_dir(tmp_path))
    names = zipfile.ZipFile(io.BytesIO(zbytes)).namelist()
    assert "state_manifest.json" in names


def test_local_db_never_bundled_defensive(tmp_path):
    # _zip_fullstack is allowlist-based, so root files like the SQLite db are
    # never bundled regardless. The exclusion set is a defensive contract (in
    # case bundling changes) and covers the WAL side-files (-wal has the freshest data).
    zbytes, _ = _zip_fullstack(_make_fullstack_dir(tmp_path))
    names = zipfile.ZipFile(io.BytesIO(zbytes)).namelist()
    assert ".anton_state.db" not in names
    for name in (".anton_state.db", ".anton_state.db-wal", ".anton_state.db-shm"):
        assert name in _FULLSTACK_EXCLUDED


def test_bundle_omits_manifest_when_absent(tmp_path):
    d = tmp_path / "art"
    d.mkdir()
    (d / "backend.py").write_text("x=1\n", encoding="utf-8")
    zbytes, _ = _zip_fullstack(d)
    names = zipfile.ZipFile(io.BytesIO(zbytes)).namelist()
    assert "state_manifest.json" not in names
    # anton_state is vendored regardless (the backend may use it)
    assert "anton_state/__init__.py" in names


def test_read_state_manifest_parses_json(tmp_path):
    d = tmp_path / "art"
    d.mkdir()
    (d / "state_manifest.json").write_text(
        json.dumps({"pk": {"name": "pk"}, "ttl_attribute": "expires_at"}), encoding="utf-8"
    )
    m = _read_state_manifest(d)
    assert m == {"pk": {"name": "pk"}, "ttl_attribute": "expires_at"}


def test_read_state_manifest_none_when_absent(tmp_path):
    d = tmp_path / "art"
    d.mkdir()
    assert _read_state_manifest(d) is None
