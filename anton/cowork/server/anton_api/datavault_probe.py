"""Headless anton probe — runs as a server-side worker, not a chat turn.

The Form Handler (`datavault_agent.process_submission_stream`) calls
`run_probe(...)` to find out whether a set of credentials actually
works. The probe spins up a FRESH ChatSession with:

  - empty history
  - no history_store → nothing persists to disk
  - no session_id   → ditto
  - a tiny toolbelt: set_status, report_success, report_failure,
    request_extra_field

The session runs `turn_stream(prompt)` once and ends. The only thing
that survives is the events it yielded back to the caller — which the
Form Handler translates into UI updates (form patches, chat lines,
right-rail scratchpad cells).

The probe is intentionally invisible to the user-facing conversation:
no system probe prompt leaks into chat history, no separate user/assistant
turn pollutes `_history.json`, the cached anton session for the
conversation isn't touched.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger(__name__)


# ── Event vocabulary the probe yields back to the Form Handler ───────
#
# Tuples of (kind, payload) so the handler can route without isinstance
# checks. Kept dead simple — V1 only needs four kinds.

# kind = 'text'           payload = str        (anton's prose, into chat)
# kind = 'scratchpad'     payload = dict       (right rail, action=start|end|result)
# kind = 'status'         payload = str        (form's live status row)
# kind = 'field_status'   payload = dict       ({name, status, method_id?})
# kind = 'remove_field'   payload = dict       ({name, method_id?})
# kind = 'extra_field'    payload = dict       ({fields, method_id?})  — added during probe
# kind = 'switch_method'  payload = dict       ({method_id, reason?})
# kind = 'verdict'        payload = ProbeOutcome  (terminal: success/fail/needs_input)


@dataclass
class ProbeOutcome:
    """Final state of a probe run. Set exactly once via the report_*
    or request_extra_field tools (or by the runner if the LLM exits
    without calling any verdict tool — the catch-all is 'failure').
    """
    status: str = "unresolved"  # success | failure | needs_input | unresolved
    summary: str = ""           # one-liner for the chat
    error: str = ""             # the actual problem (failure path)
    extra_fields: list[dict] = field(default_factory=list)  # needs_input
    follow_up: str = ""         # short advice tied to the verdict
    method_id: str | None = None  # multi-method: which method the verdict applies to


def _write_credentials_env(credentials: dict) -> tuple[str, list[str]]:
    """Persist credentials to a tempfile in `.env` format. Returns the
    path + the list of variable names so the prompt can tell anton
    exactly what's available without ever printing the values.

    Caller deletes the file when the probe completes (success or fail).
    """
    var_names: list[str] = []
    lines: list[str] = []
    for key, value in (credentials or {}).items():
        if not key:
            continue
        var = f"DS_{str(key).upper()}"
        var_names.append(var)
        # Backslash + double-quote escaping; literal `\n` for embedded
        # newlines so each var stays single-row. python-dotenv handles
        # this convention natively.
        escaped = (
            str(value)
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
        )
        lines.append(f'{var}="{escaped}"')

    fd, path = tempfile.mkstemp(prefix="anton-vault-", suffix=".env")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        try: os.unlink(path)
        except Exception: pass
        raise
    return path, var_names


def _summarize_field_list(fields: list, filled_names: set[str], skipped_set: set[str], indent: str = "  ") -> str:
    """Render a field list as a compact bullet list. Used both for
    single-method forms (top-level fields[]) and per-method blocks
    on multi-method forms.
    """
    if not fields:
        return f"{indent}(no fields)"
    lines: list[str] = []
    for f in fields:
        if not isinstance(f, dict):
            continue
        name = f.get("name") or ""
        if not name:
            continue
        ftype = f.get("type") or "text"
        label = f.get("label") or name
        if name in skipped_set:
            state = "skipped"
        elif name in filled_names:
            state = "filled"
        else:
            state = "empty"
        lines.append(f"{indent}• `{name}` ({ftype}, {state}) — {label}")
    return "\n".join(lines) if lines else f"{indent}(no fields)"


def _summarize_form(form_spec: dict, filled_names: set[str], skipped: list[str]) -> str:
    """Render the form's structure for the prompt. Branches on shape:

      • multi-method form → enumerate each method (id, label, recommended,
        selected/yes-no, fields list)
      • single-method form → top-level fields list (legacy shape)
    """
    skipped_set = set(skipped or [])
    methods = (form_spec or {}).get("methods") or []
    if methods:
        selected = (form_spec or {}).get("selected_method")
        lines: list[str] = []
        lines.append("This is a MULTI-METHOD form. The user picks ONE method "
                     "before submitting. Each method has its own field list.")
        if selected:
            lines.append(f"Currently selected method: `{selected}`")
        else:
            lines.append("No method selected yet — the user is on the picker.")
        lines.append("")
        for m in methods:
            if not isinstance(m, dict):
                continue
            mid = m.get("id") or ""
            mlabel = m.get("label") or mid
            recommended = " (recommended)" if m.get("recommended") else ""
            sel_marker = " ← selected" if mid == selected else ""
            lines.append(f"Method `{mid}` — {mlabel}{recommended}{sel_marker}")
            if m.get("description"):
                lines.append(f"  {m['description']}")
            lines.append(_summarize_field_list(
                m.get("fields") or [], filled_names, skipped_set, indent="    ",
            ))
            lines.append("")
        return "\n".join(lines).rstrip()

    # Single-method (legacy) — just the top-level fields.
    fields = (form_spec or {}).get("fields") or []
    return _summarize_field_list(fields, filled_names, skipped_set)


def _build_probe_prompt(
    engine: str,
    env_path: str,
    var_names: list[str],
    form_spec: dict,
    skipped: list[str],
) -> str:
    """The single message we pass to the probe session. The system
    prompt (built into ChatSessionConfig.system_prompt_context) is the
    standard anton prompt — we steer behaviour entirely via this user
    message + the toolbelt.
    """
    filled_names = {v.replace("DS_", "", 1).lower() for v in var_names}
    # var_names are uppercased+prefixed env-var names; the field names
    # in form_spec are the original lowercase keys. Normalize both
    # sides for matching the "filled" state in the roster.
    field_names_lower = set()
    for f in (form_spec or {}).get("fields") or []:
        if isinstance(f, dict) and f.get("name"):
            field_names_lower.add(str(f["name"]).lower())
    filled_names = {n for n in filled_names if n in field_names_lower}
    # Map back to original-case names for the roster.
    filled_original = {
        f["name"] for f in ((form_spec or {}).get("fields") or [])
        if isinstance(f, dict) and f.get("name")
        and str(f["name"]).lower() in filled_names
    }
    roster = _summarize_form(form_spec, filled_original, skipped)
    selected_method = (form_spec or {}).get("selected_method") or (form_spec or {}).get("auth_method")
    method_hint = (
        f"\nThe user picked method `{selected_method}` — focus your "
        f"probe on whatever auth flow that implies (e.g. app_password "
        f"→ IMAP/SMTP; service_account → impersonation; oauth_paste → "
        f"refresh-token exchange). If you decide a different method "
        f"would clearly work better, you can call `switch_method` with "
        f"a one-line reason.\n"
    ) if selected_method else (
        "\nNo method has been picked. Probe what the user submitted; "
        "if there's no usable info, request_extra_field with the "
        "minimum needed.\n"
    )
    return (
        f"You are a connection prober for `{engine}`. Your only job is "
        f"to determine if the credentials we just collected actually "
        f"work, and report back via your tools.\n\n"
        f"The user-submitted credentials are in a temporary `.env` file:\n"
        f"  Path: `{env_path}`\n"
        f"  Variable names: {', '.join(var_names) or '(none)'}\n"
        f"{method_hint}\n"
        f"——— CURRENT FORM ROSTER ———\n"
        f"These are the fields ALREADY in the form. Do NOT call "
        f"`request_extra_field` for any of these — they're already "
        f"there (even if empty or skipped). Use exact names when you "
        f"reference them via `set_field_status` / `remove_field`. For "
        f"multi-method forms, ALL field-edit tools (set_field_status, "
        f"remove_field, request_extra_field) take a `method_id` "
        f"parameter — pass the method whose fields you're touching.\n\n"
        f"{roster}\n\n"
        f"——— STEPS (follow in order) ———\n"
        f"1. Call `set_status` with a short message like \"Loading credentials…\".\n"
        f"2. In the scratchpad, parse the .env file (e.g. `dotenv_values('{env_path}')`). "
        f"NEVER print the values. NEVER echo them back in any tool input.\n"
        f"3. Call `set_status` with \"Installing <pkg>…\" if you need a client library, "
        f"then install it via the scratchpad's `packages` array.\n"
        f"4. Call `set_status` with \"Probing {engine}…\" and run a tiny test query "
        f"(e.g. `SELECT 1` for a database, `/me` for an API, list-buckets for storage).\n"
        f"5. **TRY HARD before reporting failure.** A successful auth that "
        f"can't access the resource the user actually needs IS NOT a success — but "
        f"it's also not a reason to immediately give up. Before calling "
        f"`report_failure` or `request_extra_field`:\n"
        f"   a) Try the engine's discovery endpoints first. If the auth handshake "
        f"works but you don't know which project / workspace / org / database to "
        f"target, list them via the API itself (e.g. PostHog: `GET /api/projects/`; "
        f"GitHub: `GET /user/repos`; Snowflake: `SHOW DATABASES`). Pick one "
        f"automatically — usually the only result, the most-recently-used, or the "
        f"one whose name best matches the user's context.\n"
        f"   b) Try multiple fallbacks. A 401 means broken auth; a 403 / 404 / "
        f"\"scope\" / \"project required\" error means the credential works but "
        f"is narrowly scoped — list what IS accessible and pick from there.\n"
        f"   c) Probe the actual resource the user cares about. Use the discovered "
        f"project / workspace / org id to call a real data endpoint (e.g. PostHog: "
        f"`GET /api/projects/<id>/insights/?limit=1`; not just `/api/users/@me/`).\n"
        f"   d) Only after exhausting (a)–(c) without finding a working path, "
        f"call `request_extra_field` for the missing piece. Frame it as a last "
        f"resort, not a first move.\n"
        f"6. Call EXACTLY ONE of:\n"
        f"   • `report_success(summary=...)` — connection works AND a real data "
        f"endpoint returned data. Mention what you confirmed (e.g. \"5 projects "
        f"visible, queried events on project 12345\").\n"
        f"   • `report_failure(error=..., follow_up=...)` — definitively broken "
        f"(bad credential, network unreachable, account suspended, etc.). "
        f"`error` should be the underlying issue in plain language; `follow_up` "
        f"is a one-line hint about what the user should fix.\n"
        f"   • `request_extra_field(fields=[{{name, label, type, help}}, ...])` — "
        f"the credentials we have aren't enough AND you've already tried "
        f"step (5) above to discover the missing piece automatically. Include a "
        f"`reason` saying what you tried and why it didn't work.\n\n"
        f"——— STATUS + FORM EDIT TOOLS ———\n"
        f"• `set_status(text)` — form-WIDE status (the bar at the top of the panel). "
        f"Use for overall phase: \"Loading\", \"Probing\", etc.\n"
        f"• `set_field_status(name, status)` — PER-FIELD status (a small line under "
        f"one specific field). Use when you want to show that you're testing or "
        f"validating a particular value, e.g. set_field_status(name='api_key', "
        f"status='Validating…') then later set_field_status(name='api_key', "
        f"status='OK') or set_field_status(name='api_key', status=null) to clear. "
        f"`name` MUST match an existing field from the roster above.\n"
        f"• `remove_field(name, method_id?)` — delete a field from the form. Use when "
        f"a field is no longer relevant (e.g. user picked OAuth so the password field "
        f"is obsolete). For multi-method forms pass the method_id.\n"
        f"• `switch_method(method_id, reason)` — flip the multi-method form to a "
        f"different method. Use only when the current method is clearly wrong (e.g. "
        f"the user's API key looks like a service-account key but they're on the "
        f"app_password method). Always include a one-line reason.\n\n"
        f"——— DON'T DUPLICATE ———\n"
        f"Before calling `request_extra_field`, scan the roster above and confirm "
        f"the field isn't already there under any name (including close variants — "
        f"`api_token` vs `api_key`, `account_id` vs `project_id`, etc). If a field "
        f"already exists but is empty/skipped, surface that to the user via "
        f"`set_field_status(name, 'Required for this engine')` instead of adding a "
        f"new one with a similar name.\n\n"
        f"——— RULES ———\n"
        f"• Keep prose to one sentence at most. The form panel shows your live "
        f"status; the user can see scratchpad cells in the right rail.\n"
        f"• NEVER print credential values. NEVER include them in tool inputs.\n"
        f"• You MUST call exactly one verdict tool before stopping. If you don't, "
        f"the run is treated as a failure.\n"
        f"• Don't ask follow-up questions in prose — use `request_extra_field` "
        f"if you need more from the user.\n"
    )


async def run_probe(
    *,
    engine: str,
    credentials: dict,
    base_session,
    form_spec: dict | None = None,
    skipped: list[str] | None = None,
    timeout_seconds: float = 90.0,
) -> AsyncIterator[tuple[str, Any]]:
    """Run one probe attempt against `engine` using `credentials`.

    `base_session` is the conversation's anton ChatSession — we crib
    its llm_client, settings, workspace, etc so the probe inherits the
    same model + configuration without re-doing the build dance. The
    probe itself uses a FRESH ChatSession instance with empty history
    and no persistence, so nothing the LLM does here pollutes the user's
    conversation.

    Yields (kind, payload) tuples; the final yield is always
    `('verdict', ProbeOutcome)` — even on timeout / runner exception
    so the Form Handler can rely on a terminal event.
    """
    # Local imports — anton may not be installed in dev environments,
    # and we don't want this module to crash on import if so.
    from anton.core.session import ChatSession, ChatSessionConfig, SystemPromptContext
    from anton.core.llm.provider import (
        StreamTextDelta, StreamToolResult,
        StreamToolUseStart, StreamToolUseEnd, StreamToolUseDelta, StreamComplete,
    )
    from anton.core.tools.tool_defs import ToolDef
    # ChatSession auto-registers SCRATCHPAD_TOOL inside _build_core_tools(),
    # so we don't have to add it to `tools` ourselves.

    env_path, var_names = _write_credentials_env(credentials)

    outcome = ProbeOutcome()
    # Tools push events into this buffer; the runner drains it between
    # each upstream StreamEvent so updates land in roughly real-time.
    pending: list[tuple[str, Any]] = []

    async def _set_status(_session, tc_input):
        text = (tc_input.get("text") or "").strip()
        if text:
            pending.append(("status", text))
        return "ok"

    async def _set_field_status(_session, tc_input):
        name = (tc_input.get("name") or "").strip()
        if not name:
            return "ignored: missing field name"
        status = tc_input.get("status")
        if isinstance(status, str):
            status = status.strip()
        method_id = (tc_input.get("method_id") or "").strip() or None
        pending.append(("field_status", {
            "name": name, "status": status, "method_id": method_id,
        }))
        return "ok"

    async def _remove_field(_session, tc_input):
        name = (tc_input.get("name") or "").strip()
        if not name:
            return "ignored: missing field name"
        method_id = (tc_input.get("method_id") or "").strip() or None
        pending.append(("remove_field", {"name": name, "method_id": method_id}))
        return "ok"

    async def _switch_method(_session, tc_input):
        method_id = (tc_input.get("method_id") or "").strip()
        if not method_id:
            return "ignored: missing method_id"
        reason = (tc_input.get("reason") or "").strip()
        pending.append(("switch_method", {"method_id": method_id, "reason": reason}))
        return "ok"

    async def _report_success(_session, tc_input):
        outcome.status = "success"
        outcome.summary = (tc_input.get("summary") or "").strip()
        return "ok"

    async def _report_failure(_session, tc_input):
        outcome.status = "failure"
        outcome.error = (tc_input.get("error") or "").strip() or "Connection failed."
        outcome.follow_up = (tc_input.get("follow_up") or "").strip()
        return "ok"

    async def _request_extra_field(_session, tc_input):
        outcome.status = "needs_input"
        fields = tc_input.get("fields") or []
        if isinstance(fields, list):
            outcome.extra_fields = [f for f in fields if isinstance(f, dict) and f.get("name")]
        outcome.follow_up = (tc_input.get("reason") or "").strip()
        # Track which method (if any) the extra fields belong to so
        # the agent can build a method-scoped patch instead of
        # appending to the top-level fields[] of a multi-method form.
        outcome.method_id = (tc_input.get("method_id") or "").strip() or None
        return "ok"

    SET_FIELD_STATUS_TOOL = ToolDef(
        name="set_field_status",
        description=(
            "Update the small status line under a SPECIFIC field in "
            "the form (e.g. \"Validating…\" under the api_key field). "
            "Pass `status=null` to clear. For multi-method forms, "
            "include `method_id` so the patch lands on the right "
            "method's field. Never echo credential values."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Field name (must match an existing "
                                   "field in the form/method, e.g. 'api_key').",
                },
                "status": {
                    "type": ["string", "null"],
                    "description": "Short status line (e.g. 'Validating…', "
                                   "'OK'). Pass null to clear.",
                },
                "method_id": {
                    "type": "string",
                    "description": "OPTIONAL — for multi-method forms, the "
                                   "id of the method the field belongs to "
                                   "(e.g. 'app_password'). Omit for "
                                   "single-method forms.",
                },
            },
            "required": ["name"],
        },
        handler=_set_field_status,
    )

    REMOVE_FIELD_TOOL = ToolDef(
        name="remove_field",
        description=(
            "Permanently delete a field from the form. Use when a "
            "field is obsolete (e.g. user picked OAuth and the password "
            "is no longer needed) or was a wrong ask. For multi-method "
            "forms, pass `method_id` to scope the deletion to one "
            "method's field list. The `name` MUST match an existing "
            "field from the roster."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the field to delete.",
                },
                "method_id": {
                    "type": "string",
                    "description": "OPTIONAL — for multi-method forms, the "
                                   "method whose field list this deletion "
                                   "applies to.",
                },
            },
            "required": ["name"],
        },
        handler=_remove_field,
    )

    SWITCH_METHOD_TOOL = ToolDef(
        name="switch_method",
        description=(
            "Flip the form to a different method (multi-method forms "
            "only). Use when the current method's probe failed and a "
            "different one would clearly work better — e.g. PostHog "
            "rejected the personal API key, switch to project_api_key. "
            "The user sees the method picker re-open with the new "
            "method already selected and can change it again if they "
            "disagree. Provide a one-line `reason` so the user "
            "understands the suggestion."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "method_id": {
                    "type": "string",
                    "description": "Target method id (must exist in the "
                                   "form's methods[] roster).",
                },
                "reason": {
                    "type": "string",
                    "description": "One-line explanation shown as a "
                                   "status update.",
                },
            },
            "required": ["method_id"],
        },
        handler=_switch_method,
    )

    SET_STATUS_TOOL = ToolDef(
        name="set_status",
        description=(
            "Update the form's live status line. Call before every "
            "scratchpad step so the user sees the probe progressing. "
            "Use 3-6 word phrases (e.g. 'Loading credentials', "
            "'Installing posthog', 'Probing /api/me'). Never include "
            "credential values."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Short status line."},
            },
            "required": ["text"],
        },
        handler=_set_status,
    )

    REPORT_SUCCESS_TOOL = ToolDef(
        name="report_success",
        description=(
            "Verdict: the connection works. The form panel flips to a "
            "success state and the credentials are saved to the vault. "
            "Call AT MOST ONCE."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "One-line summary for the chat (e.g. "
                                   "'PostHog reachable, found 3 projects').",
                },
            },
            "required": ["summary"],
        },
        handler=_report_success,
    )

    REPORT_FAILURE_TOOL = ToolDef(
        name="report_failure",
        description=(
            "Verdict: the connection does not work. The form panel "
            "shows the error and lets the user edit + resubmit. The "
            "credentials are NOT saved to the vault. Call AT MOST ONCE."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "error": {
                    "type": "string",
                    "description": "Underlying issue in plain language "
                                   "(e.g. 'API key rejected — token "
                                   "expired or malformed').",
                },
                "follow_up": {
                    "type": "string",
                    "description": "One-line hint for the user "
                                   "(e.g. 'Generate a new personal API "
                                   "key from posthog.com/settings').",
                },
            },
            "required": ["error"],
        },
        handler=_report_failure,
    )

    REQUEST_EXTRA_FIELD_TOOL = ToolDef(
        name="request_extra_field",
        description=(
            "Verdict: the credentials we collected aren't enough — we "
            "need more fields from the user. The form panel re-opens "
            "with the new fields appended. Use this when the engine "
            "needs something the original form didn't ask for "
            "(e.g. PostHog needs `project_id`). For multi-method "
            "forms, pass `method_id` so the new fields land on the "
            "right method's field list."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "fields": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "label": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": ["text", "password", "url", "select", "textarea"],
                            },
                            "help": {"type": "string"},
                            "placeholder": {"type": "string"},
                            "required": {"type": "boolean"},
                        },
                        "required": ["name"],
                    },
                },
                "reason": {
                    "type": "string",
                    "description": "Short reason shown above the fields "
                                   "(e.g. 'PostHog also needs your project ID').",
                },
                "method_id": {
                    "type": "string",
                    "description": "OPTIONAL — for multi-method forms, the "
                                   "method to attach the new fields to.",
                },
            },
            "required": ["fields"],
        },
        handler=_request_extra_field,
    )

    # Build a probe-only session. Lift the heavy bits (llm client,
    # workspace, settings, scratchpad runtime factory) off the
    # conversation's anton session so we don't redo the build dance.
    # Persistence-related fields are intentionally omitted/None —
    # nothing about this run survives.
    # Probe-only session: lift llm_client + workspace from the
    # conversation's anton session so we get the same model + working
    # directory; everything else is intentionally default. Persistence
    # fields stay None so this run leaves no trace on disk.
    config = ChatSessionConfig(
        llm_client=base_session._llm,
        system_prompt_context=SystemPromptContext(
            runtime_context="",
            suffix=(
                "You are a connection prober. You are NOT in a user-facing "
                "chat — your job is to call your tools to verify a "
                "credential set, then exit. Don't narrate. Don't ask "
                "the user questions in prose."
            ),
            output_context="",
        ),
        workspace=base_session._workspace,
        tools=[
            SET_STATUS_TOOL,
            SET_FIELD_STATUS_TOOL,
            REMOVE_FIELD_TOOL,
            SWITCH_METHOD_TOOL,
            REPORT_SUCCESS_TOOL,
            REPORT_FAILURE_TOOL,
            REQUEST_EXTRA_FIELD_TOOL,
        ],
    )

    try:
        probe_session = ChatSession(config)
    except Exception as exc:
        logger.exception("Could not build probe session")
        outcome.status = "failure"
        outcome.error = f"Could not start probe: {exc}"
        try: os.unlink(env_path)
        except Exception: pass
        yield ("verdict", outcome)
        return

    prompt = _build_probe_prompt(engine, env_path, var_names, form_spec or {}, skipped or [])

    # Track scratchpad lifecycle so the right rail can render cells.
    # turn_stream emits StreamToolUseStart/End around tool calls and
    # StreamToolResult after — for the scratchpad tool, those mean
    # "cell starting" / "cell finished, here's what it printed".
    current_tool_name: dict[str, str] = {}     # id → name
    current_tool_input_json: dict[str, str] = {}  # id → assembled json

    try:
        # Wrap the iteration in a timeout so a hung probe doesn't lock
        # up the SSE stream forever.
        async def _drive():
            async for event in probe_session.turn_stream(prompt):
                # Drain status updates (and any other tool-pushed events)
                # before yielding the upstream event so they appear in
                # roughly the order anton intended.
                while pending:
                    yield pending.pop(0)

                if isinstance(event, StreamTextDelta):
                    if event.text:
                        yield ("text", event.text)
                elif isinstance(event, StreamToolUseStart):
                    current_tool_name[event.id] = event.name
                    current_tool_input_json[event.id] = ""
                    if event.name == "scratchpad":
                        yield ("scratchpad", {"action": "start"})
                elif isinstance(event, StreamToolUseDelta):
                    current_tool_input_json[event.id] = (
                        current_tool_input_json.get(event.id, "") + (event.json_delta or "")
                    )
                elif isinstance(event, StreamToolUseEnd):
                    name = current_tool_name.pop(event.id, "")
                    raw = current_tool_input_json.pop(event.id, "")
                    if name == "scratchpad":
                        # Parse the assembled tool input so the right
                        # rail can show the code + description.
                        import json as _json
                        try:
                            parsed = _json.loads(raw or "{}")
                        except Exception:
                            parsed = {}
                        if (parsed.get("action") or "") == "exec":
                            yield ("scratchpad", {
                                "action": "end",
                                "name": parsed.get("name", ""),
                                "code": parsed.get("code", ""),
                                "one_line_description": parsed.get("one_line_description", ""),
                            })
                elif isinstance(event, StreamToolResult):
                    if event.name == "scratchpad":
                        # `content` is the rendered result (stdout +
                        # stderr + error). Forward verbatim — the rail
                        # already knows how to display this shape.
                        yield ("scratchpad", {
                            "action": "result",
                            "content": event.content or "",
                        })
                elif isinstance(event, StreamComplete):
                    pass  # turn_stream's outer iteration handles termination

            # Drain any final tool-pushed events that arrived after the
            # last upstream event (e.g. report_success fired in the
            # final assistant message).
            while pending:
                yield pending.pop(0)

        # Manual timeout — wraps each `__anext__` so per-iteration
        # progress resets the clock would be nice but we keep it
        # simple here. Total probe budget is `timeout_seconds`.
        gen = _drive().__aiter__()
        deadline = asyncio.get_event_loop().time() + timeout_seconds
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                outcome.status = "failure"
                outcome.error = f"Probe timed out after {int(timeout_seconds)}s."
                outcome.follow_up = "Try again, or check that the service is reachable."
                break
            try:
                evt = await asyncio.wait_for(gen.__anext__(), timeout=remaining)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                outcome.status = "failure"
                outcome.error = f"Probe timed out after {int(timeout_seconds)}s."
                outcome.follow_up = "Try again, or check that the service is reachable."
                break
            yield evt

    except Exception as exc:
        logger.exception("Probe session crashed")
        outcome.status = "failure"
        outcome.error = f"Probe crashed: {exc}"
    finally:
        try:
            os.unlink(env_path)
        except Exception:
            logger.debug("Could not delete temp env file %s", env_path, exc_info=True)

    # Default verdict if anton stopped without calling any of the
    # report_* tools — treat as failure rather than silently succeeding.
    if outcome.status == "unresolved":
        outcome.status = "failure"
        outcome.error = "Probe ended without a verdict."
        outcome.follow_up = "Try resubmitting; if it persists, check that the engine name matches what you intend to connect to."

    yield ("verdict", outcome)
