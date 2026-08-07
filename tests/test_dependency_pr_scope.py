"""The dependency-PR scope guard (ENG-1234).

The regression it exists for is anton#293: a PR titled "chore(deps): bump postcss
from 8.5.15 to 8.5.25 in /docs" that also reverted merged verifier code in
`anton/core/session.py`. The first case below is that PR's real file list, so this
test fails if the guard ever stops catching it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github" / "scripts" / "check_dependency_pr_scope.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("dep_scope", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def guard():
    assert _SCRIPT.exists(), f"guard script missing: {_SCRIPT}"
    return _load()


# The actual files anton#293 changed.
PR_293_FILES = [
    "anton/core/llm/structured.py",
    "anton/core/session.py",
    "docs/package-lock.json",
]


def test_catches_the_pr_293_regression(guard):
    bad = guard.offenders(PR_293_FILES)
    assert bad == ["anton/core/llm/structured.py", "anton/core/session.py"], (
        "the guard must flag the source files #293 smuggled in"
    )
    assert guard.main(["prog", *PR_293_FILES]) == 1, "exit code must be non-zero"


def test_a_clean_bump_passes(guard):
    clean = ["docs/package.json", "docs/package-lock.json"]
    assert guard.offenders(clean) == []
    assert guard.main(["prog", *clean]) == 0


@pytest.mark.parametrize("path", [
    "docs/package.json",
    "docs/package-lock.json",
    "pyproject.toml",
    "uv.lock",
    # github-actions ecosystem bumps pinned versions inside workflows.
    ".github/workflows/tests.yml",
    ".github/dependabot.yml",
    # Nested manifests must work on basename, so a new workspace needs no change.
    "packages/web/package.json",
])
def test_manifests_and_lockfiles_are_allowed(guard, path):
    assert guard.is_allowed(path), f"{path} should be allowed"


@pytest.mark.parametrize("path", [
    "anton/core/session.py",
    "anton/core/llm/structured.py",
    "tests/test_verifier_failsafe.py",
    "docs/src/pages/index.js",
    # A lookalike must not slip through on a substring match.
    "anton/package_json_helper.py",
    # Only real workflow files, not anything merely under .github/.
    ".github/CODEOWNERS",
])
def test_source_files_are_rejected(guard, path):
    assert not guard.is_allowed(path), f"{path} should be rejected"


def test_empty_and_blank_input_is_not_a_failure(guard):
    assert guard.offenders([]) == []
    assert guard.offenders(["", "   ", "\n"]) == []
