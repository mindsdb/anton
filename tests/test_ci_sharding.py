"""CI shards must keep every collected test and fail the stable check on any failed job."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def collection_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    # Load the real hooks, including their option registration and validation.
    # Only record pytest's resulting collection; no copy of the shard algorithm.
    conftest = (_ROOT / "tests/conftest.py").read_text()
    conftest += """

_shard_test_deselected = []

def pytest_deselected(items):
    _shard_test_deselected.extend(item.nodeid for item in items)

def pytest_collection_finish(session):
    from pathlib import Path
    Path(session.config.rootpath, "collection.json").write_text(json.dumps({
        "selected": [item.nodeid for item in session.items],
        "deselected": _shard_test_deselected,
    }))
"""
    (project / "conftest.py").write_text(conftest)
    for number in range(16):
        folder = project / "tests" / ("nested" if number % 2 else "unit")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"test_example_{number}.py").write_text("""
import pytest

def test_plain():
    raise AssertionError("collection must not execute tests")

class TestCases:
    @pytest.mark.parametrize("value", ["alpha", "beta"])
    def test_parameter(self, value):
        raise AssertionError("collection must not execute tests")
""")
    return project


def _collect(project: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    environment.pop("PYTEST_ADDOPTS", None)
    environment["PYTHONPATH"] = str(_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
         "--rootdir", str(project), "--confcutdir", str(project), *arguments],
        cwd=project, env=environment, capture_output=True, text=True, timeout=30,
    )


def _collected(project: Path, *arguments: str) -> dict[str, list[str]]:
    result = _collect(project, *arguments)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((project / "collection.json").read_text())


def test_four_shards_cover_every_item_once_and_keep_files_together(
    collection_project: Path, tmp_path: Path,
) -> None:
    baseline = _collected(collection_project)
    expected = {
        f"tests/{'nested' if number % 2 else 'unit'}/test_example_{number}.py::{case}"
        for number in range(16)
        for case in ("test_plain", "TestCases::test_parameter[alpha]", "TestCases::test_parameter[beta]")
    }
    assert set(baseline["selected"]) == expected
    assert len(baseline["selected"]) == len(expected)
    assert baseline["deselected"] == []
    shards = [
        _collected(collection_project, "--shard-index", str(index), "--shard-count", "4")
        for index in range(4)
    ]
    seen: set[str] = set()
    file_shards: dict[str, set[int]] = {}
    for index, shard in enumerate(shards):
        selected = set(shard["selected"])
        assert selected, f"shard {index} selected no fixture files"
        assert len(shard["selected"]) == len(selected)
        assert not seen.intersection(selected)
        seen.update(selected)
        assert set(shard["deselected"]) == expected - selected
        assert len(shard["deselected"]) == len(expected - selected)
        assert shard["selected"] == [item for item in baseline["selected"] if item in selected]
        for item in shard["selected"]:
            file_shards.setdefault(item.split("::", 1)[0], set()).add(index)
    assert seen == expected
    assert all(len(indices) == 1 for indices in file_shards.values())
    # Runner checkout locations must not alter the assignment.
    relocated = tmp_path / "another-checkout"
    shutil.copytree(collection_project, relocated)
    assert _collected(relocated, "--shard-index", "0", "--shard-count", "4") == shards[0]
    assert _collected(collection_project) == baseline


@pytest.mark.parametrize("arguments", [
    ("--shard-index", "0"),
    ("--shard-count", "4"),
    ("--shard-index", "0", "--shard-count", "0"),
    ("--shard-index", "0", "--shard-count", "-1"),
    ("--shard-index", "-1", "--shard-count", "4"),
    ("--shard-index", "4", "--shard-count", "4"),
    ("--shard-index", "wrong", "--shard-count", "4"),
    ("--shard-index", "0", "--shard-count", "wrong"),
])
def test_invalid_shard_options_fail_usage(collection_project: Path, arguments: tuple[str, ...]) -> None:
    result = _collect(collection_project, *arguments)
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, result.stdout + result.stderr
    assert "shard" in result.stderr
    assert "unrecognized arguments" not in result.stderr


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load((_ROOT / ".github/workflows/tests.yml").read_text())


def test_workflow_runs_four_isolated_shards_and_concurrent_e2e(workflow: dict) -> None:
    jobs = workflow["jobs"]
    unit = jobs["unit-tests"]
    assert unit["runs-on"] == "ubuntu-latest"
    assert unit["strategy"]["matrix"]["shard"] == [0, 1, 2, 3]
    assert unit["strategy"]["fail-fast"] is False
    assert not unit.get("needs")
    step = next(step for step in unit["steps"] if "pytest tests/" in step.get("run", ""))
    command = step["run"]
    assert step["env"]["SHARD_INDEX"] == "${{ matrix.shard }}"
    assert '--shard-index "$SHARD_INDEX"' in command
    assert "--shard-count 4" in command
    assert "--ignore=tests/e2e" in command
    e2e = jobs["stub-e2e"]
    assert e2e["runs-on"] == "ubuntu-latest"
    assert not e2e.get("needs")
    assert any("pytest tests/e2e/" in step.get("run", "") for step in e2e["steps"])
    assert set(jobs["run-tests"]["needs"]) == {"unit-tests", "stub-e2e", "windows-installer"}
    condition = str(jobs["run-tests"]["if"]).replace("${{", "").replace("}}", "").strip()
    assert condition == "always()"


@pytest.mark.parametrize(("changed", "outcome"), [
    (None, "success"),
    *((name, outcome) for name in ("UNIT_RESULT", "E2E_RESULT", "WINDOWS_RESULT")
      for outcome in ("failure", "cancelled", "skipped")),
])
def test_aggregate_executes_actual_gate_for_each_job_outcome(
    workflow: dict, changed: str | None, outcome: str,
) -> None:
    steps = [step for step in workflow["jobs"]["run-tests"]["steps"] if "run" in step]
    assert len(steps) == 1
    gate = steps[0]
    assert gate["env"] == {
        "UNIT_RESULT": "${{ needs.unit-tests.result }}",
        "E2E_RESULT": "${{ needs.stub-e2e.result }}",
        "WINDOWS_RESULT": "${{ needs.windows-installer.result }}",
    }
    environment = {name: "success" for name in gate["env"]}
    if changed:
        environment[changed] = outcome
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", gate["run"]],
        env=environment, capture_output=True, text=True, timeout=5,
    )
    assert (result.returncode == 0) is (changed is None), result.stdout + result.stderr
