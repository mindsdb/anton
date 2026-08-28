"""The scratchpad image build must never run a fork's code on the cluster runner.

``mdb-dev`` is a pod inside the newdev cluster, not a disposable VM. Whatever
runs there holds the runner's Kubernetes service-account token, its IRSA role
into the build account, and a filesystem other repositories' credentialed jobs
write into. A ``pull_request`` run builds the merge ref, which is the fork's
tree, and GitHub reads the workflow from that same tree -- so the fork chooses
the code and we choose only whether to start it.

Nothing fails at runtime when the guard goes: the build succeeds, for someone it
should not have succeeded for. So these are build assertions rather than
behaviour tests, and the two that matter run the gate's own shell rather than
grepping it, because a substring check passes on a script whose comparison has
been inverted.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_WORKFLOW = _ROOT / ".github/workflows/scratchpad-dev-build.yml"

_THIS_REPO = "mindsdb/anton"

# GitHub-hosted runner images all carry one of these prefixes. Anything else is
# one of ours, which is the whole point -- a label added later that nobody here
# has heard of must read as self-hosted, not as safe.
_HOSTED_PREFIXES = ("ubuntu-", "windows-", "macos-")


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(_WORKFLOW.read_text())


def _labels(job: dict) -> list[str]:
    runs_on = job.get("runs-on")
    if isinstance(runs_on, str):
        return [runs_on]
    if isinstance(runs_on, list):
        return [str(label) for label in runs_on]
    if isinstance(runs_on, dict):
        return [str(label) for label in (runs_on.get("labels") or [])] or ["<group>"]
    return ["<missing>"]


def _is_hosted(job: dict) -> bool:
    labels = _labels(job)
    return all(label.startswith(_HOSTED_PREFIXES) for label in labels)


def _decide_step(workflow: dict) -> dict:
    for step in workflow["jobs"]["gate"]["steps"]:
        if step.get("id") == "decide":
            return step
    raise AssertionError("no `decide` step in the gate job")


def _run_the_gate(workflow: dict, tmp_path: Path, head_repo: str) -> tuple[dict, str]:
    """Execute the gate's real shell and hand back what it wrote."""
    script = tmp_path / "decide.sh"
    script.write_text(_decide_step(workflow)["run"])
    output = tmp_path / "github_output"
    summary = tmp_path / "github_step_summary"
    output.touch()
    summary.touch()

    subprocess.run(
        ["bash", str(script)],
        check=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HEAD_REPO": head_repo,
            "THIS_REPO": _THIS_REPO,
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
    )

    written = dict(
        line.split("=", 1) for line in output.read_text().splitlines() if "=" in line
    )
    return written, summary.read_text()


def test_a_fork_pull_request_is_refused(workflow: dict, tmp_path: Path) -> None:
    """The one that matters, run rather than read.

    A pull request from any account outside this repository must not reach the
    build. Executing the block is what catches a comparison someone flipped to
    ``=`` while tidying, which every substring assertion in this file would
    happily pass.
    """
    written, _ = _run_the_gate(workflow, tmp_path, head_repo="stranger/anton")
    assert written.get("run") == "false", (
        "the gate let a fork through: a pull request from any GitHub account "
        "would build its own tree on a pod inside the cluster"
    )


def test_the_refusal_says_why(workflow: dict, tmp_path: Path) -> None:
    """A silently skipped job reads as a broken pipeline to the contributor.

    The reviewer who approved the run and the person who opened it both see the
    run page and nothing else, so the reason has to be on it.
    """
    _, summary = _run_the_gate(workflow, tmp_path, head_repo="stranger/anton")
    assert summary.strip(), "the gate skipped the build without writing a reason"
    assert "fork" in summary.lower(), (
        "the run summary does not name the fork as the reason, so the next "
        "reviewer has to read the workflow to find out why nothing built"
    )


def test_a_branch_in_this_repository_is_allowed(workflow: dict, tmp_path: Path) -> None:
    """The other half: this must not become a guard that skips everything.

    A gate that refuses every pull request would pass the test above and stop
    the scratchpad image being built at all, and no other workflow builds it.
    """
    written, _ = _run_the_gate(workflow, tmp_path, head_repo=_THIS_REPO)
    assert written.get("run") == "true", (
        "the gate refuses a first-party branch, so no pull request builds the "
        "scratchpad image any more"
    )


def test_the_gate_reads_the_head_repository_from_the_event(workflow: dict) -> None:
    """Which value gets compared is the part that is quietly wrong-able.

    ``github.head_ref`` is a branch name the fork chooses, so a gate comparing
    that would pass a fork calling its branch ``staging``. The repository's full
    name on the pull request's head is the only field the fork does not control.
    """
    env = _decide_step(workflow).get("env") or {}
    assert "github.event.pull_request.head.repo.full_name" in str(env.get("HEAD_REPO")), (
        "the gate no longer compares the head repository; a branch name is "
        "chosen by whoever opened the pull request"
    )
    assert "github.repository" in str(env.get("THIS_REPO"))


def test_every_job_off_the_hosted_runners_is_gated(workflow: dict) -> None:
    """The durable one: it covers the job nobody has added yet.

    Guarding ``build`` alone protects today's file. The next job pointed at
    ``mdb-dev`` will be written by someone who has never read this, so the
    assertion is over every job rather than over that one.
    """
    for name, job in workflow["jobs"].items():
        if name == "gate" or _is_hosted(job):
            continue
        condition = str(job.get("if", ""))
        assert "needs.gate.outputs.run" in condition, (
            f"job `{name}` runs on {_labels(job)} and is not gated on `gate`. "
            "A fork pull request would execute its code inside the cluster."
        )
        assert "gate" in (job.get("needs") or []), (
            f"job `{name}` reads the gate's output without needing it, so the "
            "condition sees an empty string and the job is skipped for everyone"
        )


def test_the_gate_stays_on_a_hosted_runner(workflow: dict) -> None:
    """A guard that runs on the thing it is guarding has already lost."""
    assert _is_hosted(workflow["jobs"]["gate"]), (
        f"the gate runs on {_labels(workflow['jobs']['gate'])}, which is a pod "
        "inside the cluster. It has to decide from outside."
    )
