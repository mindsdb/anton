"""Reject drift between the reviewed lock and the artifacts users install."""

import importlib.util
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import warnings
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_contract_under_test", ROOT / "scripts/release_contract.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
CONTRACT = "fixture-dependency==1.0\n"


def distributions(directory, *, wheel="valid", sdist="valid"):
    member = release.CONTRACT.relative_to(release.ROOT).as_posix()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Duplicate name:", category=UserWarning)
        with zipfile.ZipFile(directory / "fixture-1.0-py3-none-any.whl", "w") as archive:
            if wheel != "missing":
                archive.writestr(member, CONTRACT if wheel != "altered" else "fixture-dependency==2.0\n")
            if wheel == "duplicate":
                archive.writestr(member, CONTRACT)
    with tarfile.open(directory / "fixture-1.0.tar.gz", "w:gz") as archive:
        for _ in range(2 if sdist == "duplicate" else 1):
            if sdist == "missing":
                continue
            info = tarfile.TarInfo("fixture-1.0/" + member)
            if sdist == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "outside-contract.txt"
                archive.addfile(info)
            else:
                data = (CONTRACT if sdist != "altered" else "fixture-dependency==2.0\n").encode()
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))


def test_identical_contract_is_accepted_in_both_artifacts(tmp_path):
    distributions(tmp_path)
    release.check_distributions(tmp_path, CONTRACT)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "altered"])
def test_wheel_contract_must_be_present_once_and_unchanged(tmp_path, kind):
    distributions(tmp_path, wheel=kind)
    with pytest.raises(ValueError, match="Wheel runtime contract"):
        release.check_distributions(tmp_path, CONTRACT)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "altered", "symlink"])
def test_sdist_contract_must_be_one_unchanged_regular_file(tmp_path, kind):
    distributions(tmp_path, sdist=kind)
    with pytest.raises(ValueError, match="Sdist"):
        release.check_distributions(tmp_path, CONTRACT)


def test_check_rejects_stale_export_without_rewriting_it(tmp_path, monkeypatch, capsys):
    contract = tmp_path / "requirements.txt"
    original = "fixture-dependency==0.9\n"
    contract.write_text(original)
    monkeypatch.setattr(release, "CONTRACT", contract)
    monkeypatch.setattr(release.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=CONTRACT))
    monkeypatch.setattr(sys, "argv", ["release_contract.py", "--check"])

    assert release.main() == 1
    assert "stale" in capsys.readouterr().err
    assert contract.read_text() == original


@pytest.mark.skipif(shutil.which("uv") is None, reason="requires the CI-pinned uv executable")
def test_export_refuses_changed_project_metadata_without_rewriting_lock(tmp_path, monkeypatch):
    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nname = "contract-fixture"\nversion = "1.0"\nrequires-python = ">=3.11"\ndependencies = []\n')
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("UV_OFFLINE", "1")
    subprocess.run(["uv", "lock", "--offline"], cwd=tmp_path, check=True, capture_output=True)
    lock = tmp_path / "uv.lock"
    original = lock.read_bytes()
    # This can resolve offline without changing any package version. An unlocked
    # export would silently repair the lock instead of rejecting this mismatch.
    project.write_text(project.read_text().replace(">=3.11", ">=3.12"))
    contract = tmp_path / "requirements.txt"
    contract.write_text("")
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(release, "CONTRACT", contract)
    monkeypatch.setattr(sys, "argv", ["release_contract.py", "--check"])

    assert release.main() != 0
    assert lock.read_bytes() == original
    assert contract.read_text() == ""
