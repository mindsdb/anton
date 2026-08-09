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


def tool_schemas(*, include_ask_user: bool) -> list[dict]:
    """The tool schemas offered to phase 1's gathering loop this round.

    `include_ask_user` is computed once per `run_gathering_loop` call from
    the remaining question budget (see state.gathering_question_budget) —
    when it is False, the model is not even shown the tool, so it spends no
    rounds asking questions guaranteed to come back `limit`/`unavailable`.
    """
    schemas = [
        _scratchpad_schema(),
        _web_search_schema(),
        _web_fetch_schema(),
        FINISH_GATHERING_SCHEMA,
    ]
    if include_ask_user:
        schemas.append(ASK_USER_SCHEMA)
    return schemas


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
