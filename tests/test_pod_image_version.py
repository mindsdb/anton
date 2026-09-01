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

import ast
import pathlib
import re
from pathlib import Path

import pytest

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_DOCKERFILE = _ROOT / "Dockerfile"
_WORKFLOW = _ROOT / ".github/workflows/scratchpad-dev-build.yml"

# The Dockerfile ENV line, whatever it is set to.
_PRETEND = re.compile(r"SETUPTOOLS_SCM_PRETEND_VERSION=(\S+?)\s*\\?$", re.MULTILINE)


def _strip_comments(text: str) -> str:
    """Drop comment-only lines before matching. Load-bearing, not cosmetic.

    This file has already been bitten twice by the same thing. The single-line
    capture assertion matched ``tail -n 1`` anywhere in the ``run`` block, and
    the comment explaining the pipeline contained that literal, so deleting the
    pipe left the test green. The cowork-server half of ENG-1796 then repeated
    it exactly: a comment reading ``already checks out with fetch-depth: 0``
    satisfied the assertion guarding that setting, and both mutations passed.

    Prose about a guard must never be able to stand in for the guard, so the
    stripping happens once here rather than at each call site.
    """
    # Truncate each line at its first `#` rather than dropping comment-only
    # lines. A prefix check misses the likelier disabling pattern -- keeping the
    # guard's text as a trailing note beside a no-op, e.g.
    # `true # test ! -e /app/.git`, which reads as intact and is not. Verified:
    # that exact mutation passed until this truncated instead.
    #
    # A `#` inside a quoted string would truncate a real line early, and that is
    # the safe direction: the assertion fails loudly rather than passing on
    # prose. None of the lines asserted on here contain one.
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return _strip_comments(_DOCKERFILE.read_text())


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


def test_the_fallback_version_is_rejected_by_the_build(dockerfile: str) -> None:
    """The hole a non-empty check leaves open, and the nastiest one here.

    With no tags reachable hatch-vcs resolves ``2.0.0.dev1+g<sha>``, and with no
    .git at all it resolves ``2.0.0-dev`` (both measured). Those are well-formed
    versions that a non-empty guard happily accepts -- so losing the tags would
    bake the 2.0.0 family straight back in, which is the entire bug.

    anton is CalVer, so a 2.0.0 version is never legitimate and always means the
    tags did not arrive. In the Dockerfile rather than only the workflow so it
    holds however the image is built.
    """
    assert 'in 2.0.0*)' in dockerfile, (
        "the fallback guard is gone; a tagless resolve would rebuild the exact "
        "2.0.0 constant this change removes (ENG-1796)"
    )


def test_git_is_excluded_from_the_build_context() -> None:
    """anton is SINGLE-STAGE, so .dockerignore is the only thing keeping history out.

    This is the wrong-quietly case, and it is worse here than the version bug it
    accompanies. Un-ignoring ``.git`` produces no signal whatsoever:
    ``SETUPTOOLS_SCM_PRETEND_VERSION`` wins over VCS discovery (measured: it
    overrides a real ``2.26.8.12.1rc6.dev8+g19c6e7515`` with whatever it is set
    to), so the version stays correct, every guard passes, and the build is
    green -- while ``COPY . /app`` puts the repo's entire history into the
    runtime image of every scratchpad pod.

    cowork-server is multi-stage and deletes ``.git`` in the builder, so its
    equivalent mistake is caught by a boundary that does not exist here. The
    exclusion is load-bearing in this repo and was the only unguarded thing in
    it.
    """
    lines = [ln.strip() for ln in (_ROOT / ".dockerignore").read_text().splitlines()]
    assert ".git" in lines, (
        ".git is no longer excluded from the build context. This image is "
        "single-stage, so the full history would ship in every pod — and "
        "nothing at build or run time would report it (ENG-1796)."
    )


def test_the_build_refuses_a_context_containing_git(dockerfile: str) -> None:
    """Belt for the above: catches .git arriving by any route, not just this file.

    A negation pattern, a different context, or a build that bypasses
    .dockerignore would all defeat the assertion above. Mirrors the reasoning
    behind cowork-server's 0.0.0 guard -- the check belongs where the mistake
    happens, not only where one spelling of it is written down.
    """
    assert "test ! -e /app/.git" in dockerfile, (
        "the build no longer refuses a context containing .git"
    )



def _resolve_step(workflow: dict) -> dict:
    for step in workflow["jobs"]["version"]["steps"]:
        if step.get("id") == "resolve":
            return step
    raise AssertionError("no `resolve` step in the version job")


def _resolve_run(workflow: dict) -> str:
    """The resolve step's script, comments removed — see :func:`_strip_comments`."""
    return _strip_comments(_resolve_step(workflow).get("run", ""))


def test_the_resolved_version_is_validated_whole_line(workflow: dict) -> None:
    r"""A prefix match is not enough, because the value reaches a shell unquoted.

    The shared action expands ``extra-build-args`` unquoted into ``docker buildx
    build``, so ``2.26.8 --build-arg EVIL=1`` passes a ``^[0-9]+\.[0-9]`` prefix
    check and becomes extra arguments to that command. ``grep -Eqx`` anchors
    both ends, which is what actually closes it.
    """
    run = _resolve_run(workflow)
    assert "grep -Eqx" in run, (
        "the version is no longer whole-line validated; a prefix match lets "
        "whitespace through into an unquoted shell expansion"
    )
    assert "2.0.0*)" in run, "the workflow no longer rejects the hatch-vcs fallback"


def test_only_one_line_is_captured(workflow: dict) -> None:
    """A second stdout line would be parsed by $GITHUB_OUTPUT as another pair.

    Asserted on the pipeline itself, not on the bare string: the comment above
    it in the same ``run`` block names ``tail -n 1`` too, so a substring check
    stayed green when the pipe was deleted -- caught by mutation, not by review.
    """
    run = _resolve_run(workflow)
    assert "uvx hatch version | tail -n 1" in run, (
        "a multi-line capture would corrupt $GITHUB_OUTPUT rather than fail"
    )


def test_every_dockerfile_read_in_this_file_is_comment_stripped() -> None:
    """Closes the vector rather than the instance, on the fourth attempt.

    A helper named ``_DOCKERFILE_TEXT()`` survived the previous round: dead
    code returning raw text, sitting directly under the assertion it used to
    serve. Nothing was broken by it -- but rerouting one assertion through it
    and neutering the guard with a trailing comment gave ``11 passed``, so it
    was a loaded vector with an inviting name, which is how this class recurred
    twice already.

    Deleting it is not enough on its own: the next person adding a Dockerfile
    assertion would reach for the raw read directly and land in the same place.
    So there is now one stripped accessor -- the ``dockerfile`` fixture -- and
    this asserts nothing bypasses it.

    Walked as an AST rather than scanned line by line, and that was not a
    stylistic choice. The line-scanning version's first act was to flag this
    docstring, because the prose here names the very expression it looks for --
    prose mistaken for code, which is precisely the defect this whole file has
    been chasing. An AST cannot see a docstring's contents, so the tool now
    matches the problem.
    """
    tree = ast.parse(pathlib.Path(__file__).read_text())

    stripped_args = {
        id(arg)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_strip_comments"
        for arg in node.args
    }
    bare = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read_text"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "_DOCKERFILE"
        and id(node) not in stripped_args
    ]
    assert not bare, (
        "the Dockerfile is read without stripping comments at line(s) "
        f"{bare}; a comment there could satisfy the assertion made on it. "
        "Use the `dockerfile` fixture."
    )


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
