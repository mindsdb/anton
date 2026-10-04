"""The scratchpad image build must never push a fork's code to our registry.

The build runs on a GitHub-hosted runner and pushes with an AWS role it assumes
through GitHub OIDC. A ``pull_request`` run builds the merge ref, which is the
fork's tree, and GitHub reads the workflow from that same tree, so the fork
chooses the code and we choose only whether to start it. GitHub withholds
``id-token: write`` from a fork's ``pull_request`` run, so a fork cannot assume
the role even without the gate. The gate turns that into a skip that says why,
and it still holds for any trigger added later that does hand a fork a token.

These are build assertions rather than behaviour tests, and none of them may be
a substring check on the thing they guard. A substring passes on an inverted
comparison, which is the likeliest way this decays. The shell is executed
instead of read, and the job condition that starts the build is compared whole
rather than searched.

One boundary is worth stating because no assertion here covers it. A job that
delegates to a reusable workflow in another repository decides its runner label
there, and nothing in this tree can see it. ``_reachable_labels`` returns
``None`` for those and the sweep skips them; today they are the shared
``mindsdb/github-actions`` callers.
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

# The exact condition a job on one of our runners, or a job that can mint an
# OIDC token, has to carry. Compared whole, never searched for: `== 'false'` and
# `always() || ...` both contain this expression, and both hand a fork the job.
_GATE_CONDITION = "needs.gate.outputs.run == 'true'"

# The events only an account with write access to this repository can cause.
# `workflow_call` counts too: a called workflow runs inside its caller's run, and
# the sweeps check the calling job. Both sweeps skip a workflow that only these
# events start.
_WRITE_ACCESS_TRIGGERS = frozenset({"push", "schedule", "workflow_dispatch", "workflow_call"})

_DEV_ROLE = "arn:aws:iam::168681354662:role/gha-anton-ecr-dev"
_PROD_ROLE = "arn:aws:iam::168681354662:role/gha-anton-ecr-prod"


class _GateRun(NamedTuple):
    """What one execution of the gate's shell wrote."""

    outputs: dict[str, str]
    summary: str


class _Target(NamedTuple):
    """Where one way into the workflow builds: the gate's ``target`` outputs."""

    environment: str
    role: str
    repository: str


class _TargetRun(NamedTuple):
    """What one execution of the gate's ``target`` shell did."""

    returncode: int
    outputs: dict[str, str]


# What each way in must resolve to. The empty key is a pull request, which
# carries no workflow_call input. The roles' trust policies expect exactly these
# pairs: the pull_request subject and the staging environment on the dev role,
# and only the prod environment on the prod role.
_TARGETS = {
    "": _Target(environment="", role=_DEV_ROLE, repository="minds-anton-scratchpad-dev"),
    "staging": _Target(
        environment="staging", role=_DEV_ROLE, repository="minds-anton-scratchpad-dev"
    ),
    "production": _Target(
        environment="prod", role=_PROD_ROLE, repository="minds-anton-scratchpad"
    ),
}


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


def _expression(value: object) -> str:
    """``value`` with ``${{ }}`` stripped and whitespace collapsed.

    Both spellings mean the same thing to GitHub, so both have to compare equal
    here or the assertion turns into a formatting rule.
    """
    return " ".join(str(value).replace("${{", " ").replace("}}", " ").split())


def _condition(job: dict) -> str:
    """The job's ``if``, normalised by :func:`_expression`."""
    return _expression(job.get("if", ""))


def _grants_id_token(permissions: object) -> bool:
    """True when a declared ``permissions`` value includes ``id-token: write``."""
    if permissions == "write-all":
        return True
    return isinstance(permissions, dict) and permissions.get("id-token") == "write"


def _step_using(job: dict, action: str) -> dict:
    """The one step in ``job`` whose ``uses`` names ``action``, at any ref."""
    matches = [
        step
        for step in job.get("steps") or []
        if str(step.get("uses", "")).split("@", 1)[0] == action
    ]
    assert len(matches) == 1, f"expected one `{action}` step, found {len(matches)}"
    return matches[0]


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


def _outsider_can_start(workflow: dict) -> bool:
    """True when an account without write access here can start a run of ``workflow``.

    Most events allow that: a fork's pull request, a comment, an issue, or a
    ``workflow_run`` chained from one of those. So the answer comes by
    exclusion, and an event nobody here has heard of reads as reachable, the
    same way an unknown runner label reads as ours.
    """
    return any(event not in _WRITE_ACCESS_TRIGGERS for event in _triggers(workflow))


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


def _gate_step(workflow: dict, step_id: str) -> dict:
    for step in workflow["jobs"]["gate"]["steps"]:
        if step.get("id") == step_id:
            return step
    raise AssertionError(f"no `{step_id}` step in the gate job")


def _outputs_written(output: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1) for line in output.read_text().splitlines() if "=" in line
    )


def _run_the_gate(
    workflow: dict,
    tmp_path: Path,
    head_repo: str,
    is_pr: str = "true",
    called_for: str = "",
) -> _GateRun:
    """Execute the gate's real shell and hand back what it wrote."""
    script = tmp_path / "decide.sh"
    script.write_text(_gate_step(workflow, "decide")["run"])
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
            "CALLED_FOR": called_for,
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
    )

    return _GateRun(outputs=_outputs_written(output), summary=summary.read_text())


def _run_the_target(workflow: dict, tmp_path: Path, called_for: str) -> _TargetRun:
    """Execute the gate's real ``target`` shell and hand back what it did."""
    script = tmp_path / "target.sh"
    script.write_text(_gate_step(workflow, "target")["run"])
    output = tmp_path / "github_output"
    output.touch()

    finished = subprocess.run(
        ["bash", str(script)],
        check=False,
        capture_output=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "CALLED_FOR": called_for,
            "GITHUB_OUTPUT": str(output),
        },
    )

    return _TargetRun(returncode=finished.returncode, outputs=_outputs_written(output))


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
        "would reach the job that assumes our ECR writer role"
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
        "an event carrying no pull request reached the build; the gate cannot "
        "tell a first-party branch from a fork without that payload"
    )
    assert "fork" not in run.summary.lower(), (
        "the run summary blames a fork on an event that has no pull request at "
        f"all: {run.summary.strip()!r}"
    )


def test_a_release_pipeline_call_is_allowed(workflow: dict, tmp_path: Path) -> None:
    """release.yml and publish-staging.yml call this workflow from a push.

    A push carries no pull request, so the case above would refuse it. The
    `environment` input is what says a release pipeline is asking, and those
    only ever run on main and staging, whose trees branch protection admitted.
    """
    run = _run_the_gate(
        workflow, tmp_path, head_repo="", is_pr="false", called_for="production"
    )
    assert run.outputs.get("run") == "true", (
        "the gate refuses a release pipeline's call, so no release builds the "
        "staging or production scratchpad image"
    )


# The release pipelines and the environment each one builds the image for. The
# moving `<env>` tag scratchpad-controller pulls exists only because a caller
# here names that environment (ENG-2129).
_RELEASE_CALLERS = {"release.yml": "production", "publish-staging.yml": "staging"}


def _image_build_caller(workflows: dict[str, dict], filename: str) -> dict:
    """The one job in ``filename`` that calls the image build workflow."""
    callers = [
        job
        for job in (workflows[filename].get("jobs") or {}).values()
        if str(job.get("uses", "")).endswith(_WORKFLOW.name)
    ]
    assert len(callers) == 1, f"{filename} must call {_WORKFLOW.name} exactly once"
    return callers[0]


def test_each_release_pipeline_builds_its_environment_image(
    workflows: dict[str, dict],
) -> None:
    """Dropping the job, or its input, leaves that environment's tag frozen at
    the last image anyone built, and nothing downstream notices."""
    for filename, environment in _RELEASE_CALLERS.items():
        passed = _image_build_caller(workflows, filename).get("with") or {}
        built_for = passed.get("environment")
        assert built_for == environment, (
            f"{filename} builds the scratchpad image for {built_for!r}, not {environment!r}"
        )
        assert "outputs.version" in str(passed.get("version", "")), (
            f"{filename} does not hand the minted version to the image build, so "
            "a re-run on a tagged head would resolve the older tag"
        )


def test_each_release_pipeline_grants_the_token_to_the_image_build_only(
    workflows: dict[str, dict],
) -> None:
    """The build job declares ``id-token: write``, so each caller has to grant it.

    GitHub caps a called workflow at the calling job's grant, and it checks that
    when it loads the file. A caller without the scope fails its whole run before
    any job starts, so the pipeline's own notify job never reports it. The grant
    belongs on the calling job: a workflow-level grant would hand the token to
    every job in the release pipeline.
    """
    for filename in _RELEASE_CALLERS:
        assert not _grants_id_token(workflows[filename].get("permissions")), (
            f"{filename} grants id-token: write to every job; grant it only on "
            f"the job that calls {_WORKFLOW.name}"
        )
        caller = _image_build_caller(workflows, filename)
        assert _grants_id_token(caller.get("permissions")), (
            f"{filename} calls {_WORKFLOW.name} without granting id-token: write, "
            "so every run of it fails before any job starts"
        )


@pytest.mark.parametrize("called_for", sorted(_TARGETS), ids=lambda key: key or "pull_request")
def test_each_way_in_builds_into_its_own_tier(
    workflow: dict, tmp_path: Path, called_for: str
) -> None:
    """The pairs the roles' trust policies expect, executed rather than read.

    A pull request that named an environment would get that environment's OIDC
    subject instead of ``pull_request``, and no role would accept it. A
    production build that pushed to the dev repository would leave
    ``:production`` frozen. Both fail only at run time, and the production row
    runs only after a merge to main.
    """
    run = _run_the_target(workflow, tmp_path, called_for)
    way_in = called_for or "a pull request"
    assert run.returncode == 0, f"the target step fails for {way_in}"
    assert run.outputs == _TARGETS[called_for]._asdict(), (
        f"{way_in} builds into the wrong place"
    )


@pytest.mark.parametrize("called_for", ["development", "prod"])
def test_an_unknown_environment_stops_the_build(
    workflow: dict, tmp_path: Path, called_for: str
) -> None:
    """A caller passing a name nobody mapped must fail, not fall through.

    ``development`` is the pull request tag name and ``prod`` the GitHub
    environment name, so both are easy to pass by mistake. A fall-through to the
    pull request row would build a release into the dev repository, and the
    release's moving tag would quietly stop moving.
    """
    run = _run_the_target(workflow, tmp_path, called_for)
    assert run.returncode != 0, f"the target step accepted {called_for!r}"
    assert not run.outputs, (
        f"the target step wrote outputs for {called_for!r}: {run.outputs}"
    )


def test_the_build_uses_the_target_the_gate_picked(workflow: dict) -> None:
    """The right input has to pick the row, and each value has to reach its step.

    A row picked from anything but the release pipeline's input, or a hardcoded
    environment, role or repository in the build job, ignores the mapping. The
    mistake shows only when that path next runs.
    """
    target_env = _gate_step(workflow, "target").get("env") or {}
    assert _expression(target_env.get("CALLED_FOR")) == "inputs.environment", (
        "the target step picks its row from something other than the release "
        "pipeline's input, so a pull request can land on a release row"
    )
    gate_outputs = workflow["jobs"]["gate"].get("outputs") or {}
    for key in _Target._fields:
        assert _expression(gate_outputs.get(key)) == f"steps.target.outputs.{key}", (
            f"the gate's `{key}` output does not come from its target step"
        )

    build = workflow["jobs"]["build"]
    assert _expression(build.get("environment")) == "needs.gate.outputs.environment", (
        "the build no longer names the environment the gate picked, so its OIDC "
        "subject no longer matches the role it assumes"
    )
    credentials = _step_using(build, "aws-actions/configure-aws-credentials").get("with") or {}
    assert _expression(credentials.get("role-to-assume")) == "needs.gate.outputs.role", (
        "the build assumes a role the gate did not pick"
    )
    push = _step_using(build, "mindsdb/github-actions/build-push-ecr").get("with") or {}
    assert _expression(push.get("module-name")) == "needs.gate.outputs.repository", (
        "the build pushes to a repository the gate did not pick"
    )
    assert push.get("builder") == "local", (
        "the build no longer uses this runner's own builder; the shared action's "
        "default reaches for the in-cluster one"
    )


def test_the_build_assumes_the_role_before_it_pushes(workflow: dict) -> None:
    """build-push-ecr in local mode pushes with whatever AWS credentials the job holds.

    A hosted runner holds none, so the build job has to assume the role before
    the push. It also has to grant itself ``id-token: write``. This workflow's
    own ``permissions`` grant only ``contents: read``, and a job that declares
    none gets that, whatever the caller granted. Either mistake breaks all three
    ways in, and only at run time.
    """
    build = workflow["jobs"]["build"]
    assert _grants_id_token(build.get("permissions")), (
        "the build job does not grant itself id-token: write, so it cannot mint "
        "the token it assumes the role with"
    )
    steps = build.get("steps") or []
    assume = _step_using(build, "aws-actions/configure-aws-credentials")
    push = _step_using(build, "mindsdb/github-actions/build-push-ecr")
    assert steps.index(assume) < steps.index(push), (
        "the build pushes before it assumes the role, so the push finds no AWS "
        "credentials"
    )


def test_the_build_runs_on_a_hosted_runner(workflow: dict) -> None:
    """The build brings its own short-lived role, so it needs nothing of ours.

    GitHub recommends GitHub-hosted runners for public repositories like this
    one. A build moved back to a self-hosted runner would still pass every gate
    assertion in this file, because the gate admits exactly the runs it admitted
    before.
    """
    labels = _labels(workflow["jobs"]["build"])
    assert _is_hosted(labels), f"the build runs on {labels}, which is a pod inside the cluster"


def test_the_gate_reads_the_head_repository_from_the_event(workflow: dict) -> None:
    """Which value gets compared is the part that is quietly wrong-able.

    ``github.head_ref`` is a branch name the fork chooses, so a gate comparing
    that would pass a fork calling its branch ``staging``. The repository's full
    name on the pull request's head is the only field the fork does not control.
    """
    env = _gate_step(workflow, "decide").get("env") or {}
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
    workflow an outside account can start, resolves each job's runner labels
    through local ``uses`` calls, and treats any label outside the hosted
    families as ours.
    """
    for filename, workflow in workflows.items():
        if not _outsider_can_start(workflow):
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


def test_every_job_that_can_mint_an_oidc_token_is_gated(workflows: dict[str, dict]) -> None:
    """``id-token: write`` is a credential, wherever the job runs.

    A job holding it can trade its token for any role that trusts the token's
    subject. In a workflow an outside account can start, that job has to carry
    the gate, exactly like a job on our runners. A comment or a ``workflow_run``
    runs on ``main``, the default branch, so a job there that names the ``prod``
    environment gets a token with exactly the subject and ref the prod role
    trusts. A job with no ``permissions`` anywhere takes the repository default,
    which is a setting this tree cannot see, so it counts as able to mint.
    """
    for filename, workflow in workflows.items():
        if not _outsider_can_start(workflow):
            continue
        for name, job in (workflow.get("jobs") or {}).items():
            permissions = job.get("permissions", workflow.get("permissions"))
            if permissions is not None and not _grants_id_token(permissions):
                continue

            assert _condition(job) == _GATE_CONDITION, (
                f"job `{name}` in {filename} can mint an OIDC token and its "
                f"condition is {_condition(job)!r}, not {_GATE_CONDITION!r}"
            )
            assert "gate" in _needs(job), (
                f"job `{name}` in {filename} reads the gate's output without "
                "needing it, so the condition sees an empty string and the job "
                "is skipped for everyone"
            )


def test_the_gate_stays_on_a_hosted_runner(workflow: dict) -> None:
    """A guard that runs on the thing it is guarding has already lost."""
    labels = _labels(workflow["jobs"]["gate"])
    assert _is_hosted(labels), (
        f"the gate runs on {labels}, which is a pod inside the cluster. It has "
        "to decide from outside."
    )
