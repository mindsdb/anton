"""The scratchpad image build must never run a fork's code on the cluster runner.

``mdb-dev`` is a pod inside the newdev cluster, not a disposable VM. Whatever
runs there holds the runner's Kubernetes service-account token, its IRSA role
into the build account, and a filesystem other repositories' credentialed jobs
write into. A ``pull_request`` run builds the merge ref, which is the fork's
tree, and GitHub reads the workflow from that same tree -- so the fork chooses
the code and we choose only whether to start it.

Nothing fails at runtime when the guard goes: the build succeeds, for someone it
should not have succeeded for. So these are build assertions rather than
behaviour tests, and none of them may be a substring check on the thing they
guard. A substring passes on an inverted comparison, which is the likeliest way
this decays. The shell is executed instead of read, and the job condition that
starts the runner is compared whole rather than searched.

One boundary is worth stating because no assertion here covers it. A job that
delegates to a reusable workflow in another repository decides its runner label
there, and nothing in this tree can see it. ``_reachable_labels`` returns
``None`` for those and the sweep skips them; today they are the shared
``mindsdb/github-actions`` callers. Pinning that surface is the runner-group
change, not this file.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import NamedTuple

import pytest

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_WORKFLOW_DIR = _ROOT / ".github/workflows"
_WORKFLOW = _WORKFLOW_DIR / "scratchpad-dev-build.yml"

_THIS_REPO = "mindsdb/anton"

# GitHub-hosted runner images all carry one of these prefixes. Anything else is
# one of ours, which is the whole point -- a label added later that nobody here
# has heard of must read as self-hosted, not as safe.
_HOSTED_PREFIXES = ("ubuntu-", "windows-", "macos-")

# The exact condition a job on one of our runners has to carry. Compared whole,
# never searched for: `== 'false'` and `always() || ...` both contain this
# expression, and both hand a fork the runner.
_GATE_CONDITION = "needs.gate.outputs.run == 'true'"

# The events that let an account outside this organisation start a run. A job on
# our runners in a workflow triggered only by `push` is a different question and
# belongs to the runner-group change.
_FORK_REACHABLE_TRIGGERS = ("pull_request", "pull_request_target")


class _GateRun(NamedTuple):
    """What one execution of the gate's shell wrote."""

    outputs: dict[str, str]
    summary: str


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(_WORKFLOW.read_text())


def _workflow_files() -> list[Path]:
    """Both suffixes. GitHub reads `.yaml` too, and a sweep that globs one of
    them is a sweep the next file can be added just outside of."""
    return sorted([*_WORKFLOW_DIR.glob("*.yml"), *_WORKFLOW_DIR.glob("*.yaml")])


@pytest.fixture(scope="module")
def workflows() -> dict[str, dict]:
    """Every workflow in the repo, keyed by filename."""
    return {p.name: yaml.safe_load(p.read_text()) for p in _workflow_files()}


def _labels(job: dict) -> list[str]:
    runs_on = job.get("runs-on")
    if isinstance(runs_on, str):
        return [runs_on]
    if isinstance(runs_on, list):
        return [str(label) for label in runs_on]
    if isinstance(runs_on, dict):
        return [str(label) for label in (runs_on.get("labels") or [])] or ["<group>"]
    return ["<missing>"]


def _needs(job: dict) -> list[str]:
    """``needs`` normalised. It is a string or a list, exactly like ``runs-on``.

    Left as the raw value, ``"gate" in needs`` is a substring test on the string
    form, so a job needing ``propagate`` reads as gated.
    """
    needs = job.get("needs")
    if isinstance(needs, str):
        return [needs]
    return [str(name) for name in (needs or [])]


def _condition(job: dict) -> str:
    """The job's ``if``, with ``${{ }}`` stripped and whitespace collapsed.

    Both spellings mean the same thing to GitHub, so both have to compare equal
    here or the assertion turns into a formatting rule.
    """
    raw = str(job.get("if", ""))
    return " ".join(raw.replace("${{", " ").replace("}}", " ").split())


def _is_hosted(labels: list[str]) -> bool:
    """True only when every label names a GitHub-hosted image.

    ``bool(labels)`` first: ``all()`` over an empty list is ``True``, which would
    read a job with no resolvable label as safe.
    """
    return bool(labels) and all(label.startswith(_HOSTED_PREFIXES) for label in labels)


def _triggers(workflow: dict) -> list[str]:
    # PyYAML resolves the bare `on:` key to the boolean True, so read both.
    on = workflow.get(True, workflow.get("on"))
    if isinstance(on, dict):
        return list(on)
    if isinstance(on, list):
        return [str(event) for event in on]
    return [str(on)] if on else []


def _reachable_labels(job: dict, workflows: dict[str, dict], depth: int = 0) -> list[str] | None:
    """Every runner label this job can land work on, following local ``uses``.

    ``None`` means the answer lives in another repository, so this tree cannot
    tell -- see the module docstring.
    """
    uses = str(job.get("uses", ""))
    if not uses:
        return _labels(job)
    if not uses.startswith("./"):
        return None
    assert depth < 3, f"`uses: {uses}` nests reusable workflows deeper than this resolves"
    callee = workflows.get(Path(uses).name)
    assert callee is not None, f"`uses: {uses}` names a workflow this repo does not have"

    labels: list[str] = []
    for called in (callee.get("jobs") or {}).values():
        resolved = _reachable_labels(called, workflows, depth + 1)
        if resolved is None:
            return None
        labels.extend(resolved)
    return labels


def _decide_step(workflow: dict) -> dict:
    for step in workflow["jobs"]["gate"]["steps"]:
        if step.get("id") == "decide":
            return step
    raise AssertionError("no `decide` step in the gate job")


def _run_the_gate(
    workflow: dict, tmp_path: Path, head_repo: str, is_pr: str = "true"
) -> _GateRun:
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
            "IS_PR": is_pr,
            "HEAD_REPO": head_repo,
            "THIS_REPO": _THIS_REPO,
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
    )

    written = dict(
        line.split("=", 1) for line in output.read_text().splitlines() if "=" in line
    )
    return _GateRun(outputs=written, summary=summary.read_text())


def test_a_fork_pull_request_is_refused(workflow: dict, tmp_path: Path) -> None:
    """The one that matters, run rather than read.

    A pull request from any account outside this repository must not reach the
    build. Executing the block is what catches a comparison someone flipped to
    ``=`` while tidying, which every substring assertion in this file would
    happily pass.
    """
    run = _run_the_gate(workflow, tmp_path, head_repo="stranger/anton")
    assert run.outputs.get("run") == "false", (
        "the gate let a fork through: a pull request from any GitHub account "
        "would build its own tree on a pod inside the cluster"
    )


def test_the_refusal_says_why(workflow: dict, tmp_path: Path) -> None:
    """A silently skipped job reads as a broken pipeline to the contributor.

    The reviewer who approved the run and the person who opened it both see the
    run page and nothing else, so the reason has to be on it.
    """
    run = _run_the_gate(workflow, tmp_path, head_repo="stranger/anton")
    assert run.summary.strip(), "the gate skipped the build without writing a reason"
    assert "fork" in run.summary.lower(), (
        "the run summary does not name the fork as the reason, so the next "
        "reviewer has to read the workflow to find out why nothing built"
    )


def test_a_branch_in_this_repository_is_allowed(workflow: dict, tmp_path: Path) -> None:
    """The other half: this must not become a guard that skips everything.

    A gate that refuses every pull request would pass the test above and stop
    the scratchpad image being built at all, and no other workflow builds it.
    """
    run = _run_the_gate(workflow, tmp_path, head_repo=_THIS_REPO)
    assert run.outputs.get("run") == "true", (
        "the gate refuses a first-party branch, so no pull request builds the "
        "scratchpad image any more"
    )


def test_an_event_with_no_pull_request_is_refused_for_the_right_reason(
    workflow: dict, tmp_path: Path
) -> None:
    """A trigger added later must not be answered with a sentence about forks.

    ``HEAD_REPO`` comes from the pull request payload, so a ``workflow_dispatch``
    or a ``push`` leaves it empty and it compares unequal on its own. Refusing is
    right. Refusing while telling the run page a fork did it sends whoever added
    the trigger looking for a fork that does not exist.
    """
    run = _run_the_gate(workflow, tmp_path, head_repo="", is_pr="false")
    assert run.outputs.get("run") == "false", (
        "an event carrying no pull request reached the cluster build; the gate "
        "cannot tell a first-party branch from a fork without that payload"
    )
    assert "fork" not in run.summary.lower(), (
        "the run summary blames a fork on an event that has no pull request at "
        f"all: {run.summary.strip()!r}"
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
    assert "github.event_name" in str(env.get("IS_PR")), (
        "the gate no longer checks the event, so a trigger added later is "
        "refused with a message naming a fork that does not exist"
    )


def test_every_job_that_can_reach_our_runners_is_gated(workflows: dict[str, dict]) -> None:
    """The durable one: it covers the job, and the file, nobody has added yet.

    Guarding ``build`` alone protects today's file, and reading only that file
    protects nothing against the next workflow. So the sweep walks every
    workflow that a fork can trigger, resolves each job's runner labels through
    local ``uses`` calls, and treats any label outside the hosted families as
    ours.
    """
    for filename, workflow in workflows.items():
        if not any(event in _FORK_REACHABLE_TRIGGERS for event in _triggers(workflow)):
            continue
        for name, job in (workflow.get("jobs") or {}).items():
            if name == "gate" and _is_hosted(_labels(job)):
                continue
            labels = _reachable_labels(job, workflows)
            if labels is None or _is_hosted(labels):
                continue

            assert _condition(job) == _GATE_CONDITION, (
                f"job `{name}` in {filename} runs on {labels} and its condition is "
                f"{_condition(job)!r}, not {_GATE_CONDITION!r}. An inverted or "
                "`always()`-prefixed condition still names the gate and still "
                "runs a fork's code inside the cluster."
            )
            assert "gate" in _needs(job), (
                f"job `{name}` in {filename} reads the gate's output without "
                "needing it, so the condition sees an empty string and the job "
                "is skipped for everyone"
            )
            gate = (workflow.get("jobs") or {}).get("gate")
            assert gate is not None and _is_hosted(_labels(gate)), (
                f"{filename} gates `{name}` on a `gate` job it does not define "
                "on a hosted runner"
            )


def test_the_gate_stays_on_a_hosted_runner(workflow: dict) -> None:
    """A guard that runs on the thing it is guarding has already lost."""
    labels = _labels(workflow["jobs"]["gate"])
    assert _is_hosted(labels), (
        f"the gate runs on {labels}, which is a pod inside the cluster. It has "
        "to decide from outside."
    )
