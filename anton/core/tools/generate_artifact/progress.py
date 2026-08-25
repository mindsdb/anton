"""User-facing labels for the generation FSM's steps (ENG-970).

`GenState.step_started` turns an FSM node name into one of these lines and
pushes it onto the progress channel; `handle_generate_artifact` forwards them
as `ToolProgress` markers. The node names themselves (`is_data_enough`,
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
    "inspect_scratchpads": "Looking at the data already gathered in this session",
    "is_data_enough": "Working out whether that data is enough",
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


def label_for(node: str, *, is_fullstack: bool = False, attempt: int = 0) -> str | None:
    """Return the line to show for `node`, or None if it has no label.

    `attempt` is the generate→verify loop's counter: any value above zero
    means this step is being redone after a failure, which is worth saying —
    it explains why the run is taking longer than the step list suggests.
    """
    if not is_fullstack and node in _HTML_APP_LABELS:
        text: str | None = _HTML_APP_LABELS[node]
    else:
        text = STEP_LABELS.get(node)
    if text is None:
        return None
    return f"{text} (retry)" if attempt > 0 else text
