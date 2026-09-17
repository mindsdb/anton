"""User-facing labels for the generation FSM's steps (ENG-970).

`GenState.step_started` turns an FSM node name into one of these lines and
pushes it onto the progress channel; `handle_generate_artifact` forwards them
as `ToolProgress` markers. The node names themselves (`make_tech_spec`,
`make_api_spec`) are graph vocabulary and must never reach a user, so the
mapping is explicit rather than derived from the identifier.

A node with no entry here produces no progress line. That silence is a bug
rather than a feature, so `test_artifact_progress.py` walks `orchestrator.py`
with AST and fails when a `step_started(...)` call names a node this table
does not cover.
"""

from __future__ import annotations

# Keyed by the node label the orchestrator already uses for `state.record` and
# the debug trace, so one node has one name across all three channels.
STEP_LABELS: dict[str, str] = {
    "gathering": "Gathering what the artifact needs",
    "draft_brief": "Preparing a short brief for you",
    "redraw_brief": "Updating the brief with your changes",
    "write_prd": "Writing down the agreed requirements",
    "define_required_data": "Working out what data is still missing",
    "is_possible_to_fetch": "Checking whether the missing data can be obtained",
    "fetch_data_sample": "Fetching a data sample",
    "make_tech_spec": "Writing the technical specification",
    "make_api_spec": "Designing the API",
    "generate_backend": "Writing the backend",
    "verify_backend": "Verifying the backend",
    "generate_frontend": "Writing the frontend",
    "verify_frontend": "Verifying the frontend",
    "run_app": "Starting the application",
    "verify_fullstack": "Checking the running application",
}

# An html-app has no backend, so "frontend" has nothing to contrast with and
# reads as jargon; it is simply the page. Same node, same verification rules —
# only the wording changes.
_HTML_APP_LABELS: dict[str, str] = {
    "generate_frontend": "Writing the page",
    "verify_frontend": "Verifying the page",
}


# The steps a run is expected to go through once the artifact type is fixed,
# in pipeline order. `gathering` and the brief steps come before that point
# (the type is settled by `finish_gathering`), so they are announced without a
# count; from `write_prd` on every line carries `N of M`.
_PLAN_HTML_APP: tuple[str, ...] = (
    "write_prd",
    "make_tech_spec",
    "generate_frontend",
    "verify_frontend",
)
_PLAN_FULLSTACK: tuple[str, ...] = (
    "write_prd",
    "make_tech_spec",
    "make_api_spec",
    "generate_backend",
    "verify_backend",
    "generate_frontend",
    "verify_frontend",
    "run_app",
    "verify_fullstack",
)
# The data phase is decided at run time (`orchestrator._needs_data_loop`) and
# has not fired on a live run since the gathering step took over the work.
# Its steps are not in the plan; when one starts, the total grows by it.
_DATA_PHASE_STEPS: frozenset[str] = frozenset(
    {"define_required_data", "is_possible_to_fetch", "fetch_data_sample"}
)
# A resumed run starts further down the plan. Keyed by the checkpoint entry
# names (`discovery.checkpoint.ENTRY_*`, spelled out here to keep this module
# import-free; `test_artifact_progress.py` holds the two in step).
_ENTRY_FIRST_STEP: dict[str, tuple[str, str]] = {
    # entry: (first step for html-app, first step for fullstack)
    "resume_spec": ("make_tech_spec", "make_tech_spec"),
    "resume_generate": ("generate_frontend", "generate_backend"),
}


def plan_steps(*, is_fullstack: bool, entry: str = "full") -> list[str]:
    """The steps this run will announce with a count, in order."""
    plan = list(_PLAN_FULLSTACK if is_fullstack else _PLAN_HTML_APP)
    first = _ENTRY_FIRST_STEP.get(entry)
    if first is not None:
        plan = plan[plan.index(first[1] if is_fullstack else first[0]):]
    return plan


class StepCounter:
    """Turns step starts into `(N, M)` positions.

    `N` is the order in which distinct steps START — not the step's index in
    the plan — so the two generation loops, which run in parallel and whose
    verify steps finish in either order, still count up monotonically. A
    repeated start of the same step (a retry) keeps its number. `M` is the
    plan's length, grown by a data-phase step the moment one appears.
    """

    def __init__(self, plan: list[str]) -> None:
        self.plan = list(plan)
        self._started: dict[str, int] = {}

    def position(self, node: str) -> tuple[int, int] | None:
        if node not in self.plan:
            if node not in _DATA_PHASE_STEPS:
                return None
            self.plan.append(node)
        if node not in self._started:
            self._started[node] = len(self._started) + 1
        return self._started[node], len(self.plan)


def label_for(
    node: str,
    *,
    is_fullstack: bool = False,
    attempt: int = 0,
    position: tuple[int, int] | None = None,
) -> str | None:
    """Return the line to show for `node`, or None if it has no label.

    `attempt` is the generate→verify loop's counter: any value above zero
    means this step is being redone after a failure, which is worth saying —
    it explains why the run is taking longer than the step list suggests.
    `position` is the step's `(N, M)` from a `StepCounter`; both go into one
    bracket: `Writing the backend (step 4 of 9, attempt 2)`.
    """
    if not is_fullstack and node in _HTML_APP_LABELS:
        text: str | None = _HTML_APP_LABELS[node]
    else:
        text = STEP_LABELS.get(node)
    if text is None:
        return None
    notes: list[str] = []
    if position is not None:
        notes.append(f"step {position[0]} of {position[1]}")
    if attempt > 0:
        notes.append(f"attempt {attempt + 1}")
    return f"{text} ({', '.join(notes)})" if notes else text


# Channel-control markers, not user-facing text. They travel on the same
# queue as the progress lines because ordering with respect to those lines is
# the entire point: the pipeline raises its question from inside the task the
# handler is draining, so a marker emitted between OPEN and CLOSED would be
# printed on top of a live prompt.
QUESTION_OPEN = "\x00question-open"
QUESTION_CLOSED = "\x00question-closed"
