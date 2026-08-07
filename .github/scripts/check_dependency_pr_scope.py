#!/usr/bin/env python3
"""Fail a dependency-bump PR that touches anything but manifests and lockfiles.

Why this exists (ENG-1234): anton#293 was titled "bump postcss from 8.5.15 to
8.5.25 in /docs" and also reverted merged verifier code in
`anton/core/session.py` — re-adding the silent fail-safe ENG-1079 had deleted.
The branch came from `main`, the PR was retargeted at `staging`, the resulting
conflict was resolved by keeping both sides, and a squash merge hid all of it
under the postcss title.

No amount of Dependabot configuration prevents that: security updates must be
based on the default branch (GitHub's docs say not to set `target-branch` for
them), so cross-branch merges will keep happening. What is preventable is a
dependency PR silently carrying source changes. This turns that into a red check.

Logic lives here rather than inline in the workflow so it can be tested against
a real diff — see tests/test_dependency_pr_scope.py, which feeds it #293's
actual file list.

Usage:
    git diff --name-only base...HEAD | check_dependency_pr_scope.py
    check_dependency_pr_scope.py path/one path/two
"""

from __future__ import annotations

import sys
from pathlib import PurePosixPath

# Manifests and lockfiles a bump is allowed to rewrite, matched on basename so
# nested projects (docs/, any future workspace) work without new entries.
ALLOWED_BASENAMES = frozenset({
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "npm-shrinkwrap.json",
    "pyproject.toml",
    "uv.lock",
    "poetry.lock",
    "requirements.txt",
    "requirements-dev.txt",
    "Pipfile",
    "Pipfile.lock",
    "go.mod",
    "go.sum",
    "Cargo.toml",
    "Cargo.lock",
})

# The github-actions ecosystem bumps pinned action versions inside workflows,
# and Dependabot may touch its own config.
ALLOWED_PREFIXES = (".github/workflows/",)
ALLOWED_EXACT = frozenset({".github/dependabot.yml", ".github/dependabot.yaml"})


def is_allowed(path: str) -> bool:
    path = path.strip()
    if not path:
        return True
    if path in ALLOWED_EXACT:
        return True
    if path.startswith(ALLOWED_PREFIXES):
        return True
    return PurePosixPath(path).name in ALLOWED_BASENAMES


def offenders(paths: list[str]) -> list[str]:
    return [p.strip() for p in paths if p.strip() and not is_allowed(p)]


def main(argv: list[str]) -> int:
    paths = argv[1:] if len(argv) > 1 else sys.stdin.read().splitlines()
    bad = offenders(paths)
    if not bad:
        print(f"OK — {len([p for p in paths if p.strip()])} changed file(s), all manifests/lockfiles.")
        return 0

    print("Dependency PR is out of scope. Files that are not manifests or lockfiles:")
    for p in bad:
        print(f"  {p}")
    print(
        "\nA dependency bump must not carry source changes. This usually means the "
        "branch was based on a different branch than it targets, and a conflict was "
        "resolved by keeping both sides — see ENG-1234 / anton#293.\n"
        "Fix: rebase the branch onto its target branch and re-resolve, or split the "
        "source changes into their own PR. Do not retarget a Dependabot security PR "
        "at staging — merge it into main and let the weekly sync carry it."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
