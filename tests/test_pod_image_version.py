"""The pod image must report a derived version, never a constant (ENG-1796).

Until 2026-08-26 ``Dockerfile`` set ``SETUPTOOLS_SCM_PRETEND_VERSION=2.0.0``, so
every turn a scratchpad pod ever served reported ``anton_version=2.0.0``. That
was not a fallback firing occasionally — it was the only value cloud reported,
and because ``2.0.0`` is a well-formed release number it read as a legitimate
cohort in any breakdown rather than as a null. At its peak it was 46% of the
measured install population.

Nothing failed when it was wrong, which is why it survived so long: the build
succeeded, the pod ran, and the number looked plausible. These are build
assertions rather than behaviour tests for the same reason — there is no
runtime symptom to assert on, so the contract has to be pinned at the seam
where it can actually be broken.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_DOCKERFILE = _ROOT / "Dockerfile"
_WORKFLOW = _ROOT / ".github/workflows/scratchpad-dev-build.yml"

# The Dockerfile ENV line, whatever it is set to.
_PRETEND = re.compile(r"SETUPTOOLS_SCM_PRETEND_VERSION=(\S+?)\s*\\?$", re.MULTILINE)


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return _DOCKERFILE.read_text()


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(_WORKFLOW.read_text())


def test_the_version_is_not_a_literal(dockerfile: str) -> None:
    """The regression itself: a hardcoded version, in any form."""
    found = _PRETEND.findall(dockerfile)
    assert found, "Dockerfile no longer sets SETUPTOOLS_SCM_PRETEND_VERSION at all"
    for value in found:
        assert value.startswith("${") and value.endswith("}"), (
            f"SETUPTOOLS_SCM_PRETEND_VERSION={value!r} is a literal. Every pod would "
            "report it forever, and a well-formed one is indistinguishable from a "
            "real release downstream (ENG-1796). Pass it in as a build arg."
        )


def test_the_build_arg_is_declared(dockerfile: str) -> None:
    """``ARG`` before use, or docker silently substitutes an empty string."""
    arg_at = dockerfile.find("ARG ANTON_VERSION")
    use_at = dockerfile.find("SETUPTOOLS_SCM_PRETEND_VERSION=${ANTON_VERSION}")
    assert arg_at != -1, "ARG ANTON_VERSION is not declared"
    assert use_at != -1, "SETUPTOOLS_SCM_PRETEND_VERSION does not read ANTON_VERSION"
    assert arg_at < use_at, "ARG must be declared before it is referenced"


def test_an_empty_version_fails_the_build(dockerfile: str) -> None:
    """No silent fallback: an unset arg must stop the build, not invent a version.

    This is the criterion that keeps the fix from decaying back into the bug --
    a build that quietly substitutes something plausible is exactly what 2.0.0
    was.
    """
    assert 'test -n "$SETUPTOOLS_SCM_PRETEND_VERSION"' in dockerfile, (
        "the empty-arg guard is gone; a build with no ANTON_VERSION would fail "
        "deep inside the sync with an error that never names the build arg"
    )
    assert "exit 1" in dockerfile


def _steps(workflow: dict, job: str) -> list[dict]:
    return workflow["jobs"][job]["steps"]


def test_the_version_job_fetches_tags(workflow: dict) -> None:
    """hatch-vcs describes against tags, and the default fetch-depth fetches none.

    The failure mode is silent: with no tags the version resolves to
    ``0.0.0.dev1+g<sha>`` and the build still succeeds. cowork-server shipped
    exactly that for months (ENG-1796).
    """
    checkout = [s for s in _steps(workflow, "version") if "actions/checkout" in str(s.get("uses", ""))]
    assert checkout, "the version job no longer checks out the repo"
    for step in checkout:
        assert (step.get("with") or {}).get("fetch-depth") == 0, (
            "checkout must use fetch-depth: 0 — without tags hatch-vcs cannot "
            "derive a version and the build succeeds anyway with a wrong one"
        )


def test_the_build_waits_for_the_resolved_version(workflow: dict) -> None:
    """Without the dependency the build reads an empty output and the guard fires.

    Worth pinning because the failure is a red build rather than a wrong
    version, so it would be diagnosed as flakiness rather than as this.
    """
    assert "version" in (workflow["jobs"]["build"].get("needs") or []), (
        "the build job must need the version job, or the build arg is empty"
    )


def test_the_resolved_version_reaches_the_build(workflow: dict) -> None:
    """The seam that made this reachable at all: the arg must be passed through.

    Resolving the version on the runner and then not handing it to docker would
    leave the guard above firing on every build.
    """
    passed = [
        (s.get("with") or {}).get("extra-build-args", "")
        for s in _steps(workflow, "build")
        if "build-push-ecr" in str(s.get("uses", ""))
    ]
    assert passed, "the build job no longer calls build-push-ecr"
    assert any("--build-arg ANTON_VERSION=" in str(p) for p in passed), (
        "the resolved version is not passed to the image build"
    )
