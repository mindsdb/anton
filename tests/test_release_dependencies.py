"""Owned release environments must not resolve new runtime packages."""

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_scratchpad_sdk_is_in_the_lock_and_installed_by_sync():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    assert project["dependency-groups"]["scratchpad"]
    names = {package["name"] for package in lock["package"]}
    assert {"boto3", "botocore", "s3transfer"} <= names
    commands = "\n".join(
        line.split("#", 1)[0] for line in (ROOT / "Dockerfile").read_text().splitlines()
    )
    assert "uv sync --locked --no-dev --group scratchpad" in commands
    assert "uv pip install" not in commands


def test_release_export_is_packaged_and_exact():
    contract = ROOT / "anton" / "release-requirements.txt"
    assert contract.is_file(), "release wheels need the reviewed runtime contract"
    requirements = [
        line for line in contract.read_text().splitlines()
        if line and not line.startswith((" ", "#"))
    ]
    assert requirements
    assert all("==" in line for line in requirements)
    assert not any(line.startswith(("anton-agent", "pytest", "boto3")) for line in requirements)
