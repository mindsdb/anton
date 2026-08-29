"""Sub-tools exposed to generate_prd's phase 1 gathering loop.

Four tools are always offered: `scratchpad`, `web_search`, `web_fetch` (thin
schema copies of the existing top-level tools — dispatch reuses their real
handlers, see engine.py) and `finish_gathering` (the control tool the model
calls once the artifact type is clear and enough data has been gathered).

`ask_user` is offered conditionally (see `tool_schemas`'s `include_ask_user`)
and, unlike every other sub-tool here, is NOT dispatched through the existing
`handle_ask_user` — that handler collapses the `limit` and `unavailable`
`AskAnswer` statuses into an indistinguishable `"error"` string
(`tool_handlers.py:998-1001`), which would make it impossible for the
orchestrator to tell "the user declined" apart from "we ran out of budget".
`dispatch_ask_user` below parses the same tool-call shape but calls
`elicit()` directly, keeping the raw status for the caller.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from anton.core.artifacts.models import ARTIFACT_TYPES

if TYPE_CHECKING:
    from anton.chat_session import ChatSession
    from anton.core.interaction.elicit import AskAnswer, AskRequest


ASK_USER_SCHEMA: dict = {
    "name": "ask_user",
    "description": (
        "Ask the user a clarifying or open question and get their answer "
        "back within this same call. Interactive questions are scarce this "
        "turn — use this only when you genuinely cannot proceed without an "
        "answer. Give 2-10 options with unique `value`s; `allow_custom` "
        "(default true) lets the user type their own answer instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "One short line asking what to clarify.",
            },
            "options": {
                "type": "array",
                "minItems": 2,
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "label": {"type": "string"},
                        "detail": {"type": "string"},
                    },
                    "required": ["value"],
                },
            },
            "select": {"type": "string", "enum": ["one", "many"]},
            "allow_custom": {"type": "boolean"},
        },
        "required": ["question", "options"],
    },
}


FINISH_GATHERING_SCHEMA: dict = {
    "name": "finish_gathering",
    "description": (
        "Call this once the artifact type is unambiguous and you have "
        "enough data (and samples, where relevant) to draft a PRD. Ends "
        "phase 1 — a short brief will be drafted from your `notes` next."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "One-line summary of what was determined.",
            },
            "artifact_type": {
                "type": "string",
                "enum": list(ARTIFACT_TYPES),
                "description": "The confirmed artifact type.",
            },
            "notes": {
                "type": "string",
                "description": (
                    "Everything the PRD-drafting step needs: goal, data "
                    "sources found, sample rows, open assumptions, UI/UX "
                    "hints — this is the only context phase 2 starts from."
                ),
            },
        },
        "required": ["summary", "artifact_type"],
    },
}


def _scratchpad_schema() -> dict:
    # Reuse the exact schema the main agent sees, so the gathering loop
    # drives scratchpads with the same contract. Imported lazily to avoid a
    # tool_defs <-> generate_prd import cycle (mirrors generate_artifact's
    # own sub_tools.py).
    from anton.core.tools.tool_defs import SCRATCHPAD_TOOL

    return {
        "name": SCRATCHPAD_TOOL.name,
        "description": SCRATCHPAD_TOOL.description,
        "input_schema": SCRATCHPAD_TOOL.input_schema,
    }


def _web_search_schema() -> dict:
    from anton.core.tools.web_tools import WEB_SEARCH_FALLBACK_TOOL

    return {
        "name": WEB_SEARCH_FALLBACK_TOOL.name,
        "description": WEB_SEARCH_FALLBACK_TOOL.description,
        "input_schema": WEB_SEARCH_FALLBACK_TOOL.input_schema,
    }


def _web_fetch_schema() -> dict:
    from anton.core.tools.web_tools import WEB_FETCH_FALLBACK_TOOL

    return {
        "name": WEB_FETCH_FALLBACK_TOOL.name,
        "description": WEB_FETCH_FALLBACK_TOOL.description,
        "input_schema": WEB_FETCH_FALLBACK_TOOL.input_schema,
    }


async def signal_thinking(session: "ChatSession") -> None:
    """Restart the host's spinner before a direct LLM call.

    Every direct `session._llm.plan/code/generate_object` call in phase 1
    (engine.py) and phase 2 (orchestrator.py) happens outside the outer
    agent loop's own tool-round machinery, so it never sees that loop's
    `StreamTaskProgress(phase="reasoning_start")` emission. `elicit()`, in
    turn, stops the spinner for interactive input
    (`phase="interactive"`) and never restarts it. Without this call in
    between, the gap between the user's answer and the next model reply
    renders as a silent, spinner-less pause.
    """
    from anton.core.llm.provider import StreamTaskProgress

    await session.emit(StreamTaskProgress(phase="reasoning_start", message="Thinking..."))


async def ask_via_elicit(session: "ChatSession", request: "AskRequest") -> "AskAnswer":
    """Call `elicit()` directly, never raising, with the same telemetry
    `handle_ask_user` fires around its own call.

    `elicit()` re-raises whatever `elicitor.ask()` raises instead of
    returning an `"error"` status (there is no such status — see
    `AskAnswer.status`'s docstring). Both generate_prd call sites — this
    one (via `dispatch_ask_user`, phase 1's `ask_user` sub-tool) and
    `orchestrator.show_and_confirm` (phase 2) — bypass `handle_ask_user`
    entirely to keep the raw status, which also means neither gets its
    `ask_user_asked` / `ask_user_<status>` telemetry for free. Firing it
    here once, rather than in each call site, means both phases get it
    without duplicating (or forgetting) the two `_send_ask_user_event`
    calls.
    """
    from anton.core.interaction.elicit import AskAnswer, elicit
    from anton.core.tools.tool_handlers import _send_ask_user_event

    props = {"select": request.select, "options": str(len(request.options))}
    _send_ask_user_event(session, "ask_user_asked", props)
    question_id = f"ask:{uuid.uuid4().hex}"
    try:
        answer = await elicit(session, question_id, request)
    except Exception:
        _send_ask_user_event(session, "ask_user_error", props)
        return AskAnswer(status="error")
    _send_ask_user_event(session, f"ask_user_{answer.status}", props)
    return answer


async def dispatch_ask_user(session: "ChatSession", tc_input: dict) -> dict:
    """Handle one `ask_user` sub-tool call from phase 1's gathering loop.

    Mirrors `tool_handlers.handle_ask_user`'s parsing, but keeps the raw
    `AskAnswer.status` in the return value instead of collapsing
    `limit`/`unavailable` into `"error"` — the gathering loop needs to tell
    "budget exhausted" apart from "user declined" so it can decide whether
    to keep asking or fall back to an assumption. Telemetry is fired inside
    `ask_via_elicit`, not duplicated here.
    """
    from anton.core.tools.tool_handlers import build_ask_request

    elicitor = getattr(session, "elicitor", None)
    request = build_ask_request(tc_input, timeout_s=getattr(elicitor, "timeout_s", None))
    if request is None:
        question = str(tc_input.get("question") or "")
        return {
            "tool_result": (
                "Error: the question was malformed (needs 2-10 options with "
                "unique values) — proceed on a stated assumption."
            ),
            "status": "unavailable",
            "question": question,
            "answer_summary": "(malformed question, not asked)",
        }

    answer = await ask_via_elicit(session, request)

    if answer.status == "limit":
        return {
            "tool_result": (
                "Question limit reached for this turn; proceed on a stated "
                "assumption instead of asking again."
            ),
            "status": "limit",
            "question": request.prompt,
            "answer_summary": "(question budget exhausted, not answered)",
        }
    if answer.status in ("unavailable", "error"):
        return {
            "tool_result": (
                "Interactive questions are unavailable right now; proceed "
                "on a stated assumption."
            ),
            "status": answer.status,
            "question": request.prompt,
            "answer_summary": f"({answer.status}, not answered)",
        }
    if answer.status in ("cancelled", "timeout"):
        return {
            "tool_result": (
                f"The user did not answer ({answer.status}); proceed on a "
                "stated assumption instead of re-asking."
            ),
            "status": answer.status,
            "question": request.prompt,
            "answer_summary": f"(user {answer.status})",
        }

    parts = []
    if answer.values:
        parts.append(", ".join(answer.values))
    if answer.text:
        parts.append(answer.text)
    summary = " / ".join(parts) if parts else "(answered, no content)"
    return {
        "tool_result": (
            f'{{"status": "answered", "values": {list(answer.values)!r}, '
            f'"text": {answer.text!r}}}'
        ),
        "status": "answered",
        "question": request.prompt,
        "answer_summary": summary,
    }


# ── Steps of the shared-prefix region (phases A-D) ──────────────────────────
STEP_GATHERING = "gathering"
STEP_DRAFT_BRIEF = "draft_brief"
STEP_REDRAW_BRIEF = "redraw_brief"
STEP_WRITE_PRD = "write_prd"
STEP_TECH_SPEC = "make_tech_spec"
STEP_API_SPEC = "make_api_spec"

_DATA_TOOLS = frozenset({"scratchpad", "web_search", "web_fetch"})

# Which tools may actually RUN on each step. The schema array itself never
# changes (see `pipeline_tool_schemas`), because system prompt + tools +
# messages form the cached prefix and editing any of them at a phase switch
# costs a full cache miss on the largest context in the pipeline.
#
# `redraw_brief` is a separate row from `draft_brief` for one reason:
# `finish_gathering`. On the hot path it is denied, because a model reaches
# for it whenever a prompt says "you have finished gathering" — that is why
# the old code removed it from the array outright. On the redraw step it is
# required: it is how the artifact type and the declared data sources get
# re-stated after a user correction.
ALLOWED_TOOLS_BY_STEP: dict[str, frozenset[str]] = {
    STEP_GATHERING: _DATA_TOOLS | {"ask_user", "finish_gathering"},
    STEP_DRAFT_BRIEF: _DATA_TOOLS,
    STEP_REDRAW_BRIEF: _DATA_TOOLS | {"finish_gathering"},
    STEP_WRITE_PRD: _DATA_TOOLS,
    STEP_TECH_SPEC: frozenset(),
    STEP_API_SPEC: frozenset(),
}

_STEP_TASK = {
    STEP_GATHERING: "gathering data and settling the artifact type",
    STEP_DRAFT_BRIEF: "drafting the brief — reply with text, no tool calls",
    STEP_REDRAW_BRIEF: "redrawing the brief after the user's correction",
    STEP_WRITE_PRD: "writing the full PRD — reply with markdown, no tool calls",
    STEP_TECH_SPEC: "writing the technical specification — reply with markdown, no tool calls",
    STEP_API_SPEC: "writing the API specification — reply with JSON, no tool calls",
}


def pipeline_tool_schemas() -> list[dict]:
    """The ONE tool array used by every call in phases A-D.

    Fixed contents, fixed order, built the same way every time: it is part of
    the cached prefix, and a reordered array is a changed prefix. Availability
    per step is decided by `rejection_for`, never by dropping entries here.
    """
    return [
        _scratchpad_schema(),
        _web_search_schema(),
        _web_fetch_schema(),
        ASK_USER_SCHEMA,
        FINISH_GATHERING_SCHEMA,
    ]


def rejection_for(step: str, name: str, *, questions_left: int) -> str | None:
    """The refusal text for a disallowed call, or None when it may run.

    The refusal names the current step and the action expected instead: a
    bare "not available" leaves the model to guess, and a guessing model
    spends another round — which on a spec step costs a full re-send of the
    shared history, the most expensive round in the pipeline.
    """
    allowed = ALLOWED_TOOLS_BY_STEP.get(step)
    if allowed is None:
        return (
            f"`{name}` is not available: step `{step}` is unknown to this "
            "pipeline. Reply with text and no tool calls."
        )
    task = _STEP_TASK.get(step, step)
    if name not in allowed:
        return f"`{name}` is not available at this step. You are {task}."
    if name == "ask_user" and questions_left <= 0:
        return (
            "`ask_user` is unavailable: the question budget for this turn is "
            "exhausted. Proceed on a stated assumption instead of asking again."
        )
    return None
