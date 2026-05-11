"""Cowork-side tool wrappers.

Anton's stock tools (PUBLISH_TOOL, CONNECT_DATASOURCE_TOOL, …) are
written for the CLI: they assume a Rich Console attached to a TTY,
they pop the system browser, and they hold the user's gaze with
animated spinners. None of that works inside the FastAPI process the
desktop app spawns.

We build cowork-flavoured wrappers that share the LLM-facing schema
(name / description / input_schema) so the model uses them
identically, but whose handlers do the actual work in a way that
makes sense for a server process: no console.print, no Live spinner,
no webbrowser.open. Status flows back to the desktop UI through the
normal SSE event stream and the response string the LLM renders.

Right now we only override PUBLISH_TOOL — the only one users have hit.
Add more here as needed (CONNECT_DATASOURCE_TOOL is the next likely
candidate; same pattern).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


async def _cowork_publish_or_preview(session: Any, tc_input: dict) -> str:
    """Server-side equivalent of anton.tools.handle_publish_or_preview.

    Mirrors the same `action` semantics:
      - 'ask' / 'preview' → return a string pointing the user at the
        Live Artifacts panel; the desktop UI already exposes preview
        and publish buttons there. We don't open a browser here.
      - 'publish' → call anton.publisher.publish directly, persist the
        result in `<output_dir>/.published.json`, return the view URL.
    """
    raw_path = tc_input.get("file_path", "")
    title = tc_input.get("title", "Dashboard")
    action = (tc_input.get("action") or "ask").lower()

    if not raw_path:
        return "publish_or_preview: missing file_path"

    file_path = Path(raw_path).expanduser()
    if not file_path.is_absolute():
        # Anton's session carries the active workspace base.
        workspace = getattr(session, "_workspace", None)
        if workspace is not None:
            base = getattr(workspace, "base", None)
            if base:
                file_path = Path(base) / raw_path
    file_path = file_path.resolve()

    if not file_path.exists():
        return f"File not found: {file_path}"

    # 'ask' and 'preview' are no-ops in the desktop chat — the artifact
    # is already visible in the Live Artifacts panel and inline in the
    # assistant turn. Returning a clear string lets the LLM continue
    # without ever invoking a CLI prompt.
    if action in ("ask", "preview"):
        return (
            f"Created {title} at {file_path}. The user can preview, publish, "
            f"or copy a public URL from the Live Artifacts panel — they don't "
            f"need a /publish command in the desktop app."
        )

    if action != "publish":
        return f"publish_or_preview: unknown action '{action}'"

    # ── action == 'publish' ───────────────────────────────────────────
    # Read the API key the same way the cowork HTTP endpoint does so
    # both code paths agree on what's "configured".
    try:
        from .settings import _get_env
    except Exception as exc:
        logger.exception("Cowork publish tool could not import settings helper")
        return f"PUBLISH FAILED: settings module unavailable ({exc})"

    api_key = _get_env("ANTON_MINDS_API_KEY")
    if not api_key:
        return (
            "STOP: No Minds API key configured. Tell the user to set their "
            "Minds API key in Settings (or in their .env) before publishing. "
            "Do NOT call this tool again until they confirm the key is set."
        )

    publish_url = _get_env("ANTON_PUBLISH_URL", "https://4nton.ai")
    ssl_verify = _get_env("ANTON_MINDS_SSL_VERIFY", "true").lower() == "true"

    try:
        from anton.publisher import publish
    except Exception as exc:
        logger.exception("Cowork publish tool could not import anton.publisher")
        return f"PUBLISH FAILED: anton.publisher unavailable ({exc})"

    output_dir = file_path.parent
    published_json_path = output_dir / ".published.json"
    published_map: dict[str, Any] = {}
    if published_json_path.is_file():
        try:
            published_map = json.loads(published_json_path.read_text(encoding="utf-8"))
        except Exception:
            published_map = {}

    file_key = file_path.name
    prev = published_map.get(file_key)
    report_id = prev.get("report_id") if isinstance(prev, dict) else None

    def _do_publish(rid: str | None):
        return publish(
            file_path,
            api_key=api_key,
            report_id=rid,
            publish_url=publish_url,
            ssl_verify=ssl_verify,
        )

    try:
        result = _do_publish(report_id)
    except Exception as exc:
        # If we tried to update an existing report and the upstream
        # rejected it (e.g. report was deleted), retry as a fresh one
        # — same recovery path anton's CLI tool uses.
        if report_id:
            try:
                result = _do_publish(None)
            except Exception as retry_exc:
                logger.exception("Cowork publish retry failed")
                return f"PUBLISH FAILED: {retry_exc}"
        else:
            logger.exception("Cowork publish failed")
            return f"PUBLISH FAILED: {exc}"

    view_url = result.get("view_url", "") if isinstance(result, dict) else ""
    returned_report_id = result.get("report_id", "") if isinstance(result, dict) else ""

    if returned_report_id:
        published_map[file_key] = {
            "report_id": returned_report_id,
            "url": view_url,
            "last_md5": result.get("md5", "") if isinstance(result, dict) else "",
        }
        try:
            published_json_path.write_text(
                json.dumps(published_map, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            logger.debug("Could not persist .published.json", exc_info=True)

    if not view_url:
        return "Published, but no view URL was returned."
    return f"Published successfully! View URL: {view_url}"


def build_cowork_publish_tool():
    """Construct a ToolDef matching anton's PUBLISH_TOOL schema, but
    with the cowork-aware handler. Lazy-imports anton.tools so callers
    can build the session config without paying the import cost twice.
    """
    from anton.tools import PUBLISH_TOOL
    from anton.core.tools.tool_defs import ToolDef

    return ToolDef(
        name=PUBLISH_TOOL.name,
        description=PUBLISH_TOOL.description,
        input_schema=PUBLISH_TOOL.input_schema,
        handler=_cowork_publish_or_preview,
        prompt=PUBLISH_TOOL.prompt,
    )


# ── Data-vault form tools ──────────────────────────────────────────────
#
# Two cowork-only tools that drive the agentic credential flow
# documented in docs/datavault.md:
#
#   request_credentials(spec)       — render a `data-vault-form`
#                                     markdown block for the user.
#                                     Returns the block to the LLM
#                                     so it can include it verbatim
#                                     in its response.
#   fetch_submission(submission_id) — pull staged credential values
#                                     after the user submits. Anton
#                                     uses these to test / save the
#                                     connection, then either
#                                     presents a new form (with
#                                     errors) or moves on.

import uuid


def _ensure_form_id(spec: dict) -> dict:
    """Normalize a form spec — generate a form_id if missing, fall back
    to a default title. Mutates a copy and returns it.
    """
    out = dict(spec)
    if not out.get("form_id"):
        out["form_id"] = "fm_" + uuid.uuid4().hex[:10]
    if not out.get("title"):
        out["title"] = "Connect"
    has_methods = isinstance(out.get("methods"), list) and out.get("methods")
    if not has_methods and (
        "fields" not in out or not isinstance(out.get("fields"), list)
    ):
        out["fields"] = []
    return out


async def _cowork_request_credentials(session: Any, tc_input: dict) -> str:
    """Tool handler for `request_credentials`.

    The LLM hands us a form spec; we wrap it in a `data-vault-form`
    markdown block (the renderer's MarkdownCode picks this up and
    publishes the spec into the per-conversation form store, which
    the right-rail DataVaultFormPanel mounts). The returned string
    instructs the LLM to relay the block verbatim.
    """
    spec = tc_input.get("spec") if isinstance(tc_input.get("spec"), dict) else tc_input
    if not isinstance(spec, dict):
        return "request_credentials: invalid spec — must be a JSON object with `title` and `fields`"

    spec = _ensure_form_id(spec)
    block = "```data-vault-form\n" + json.dumps(spec, indent=2) + "\n```"
    return (
        "Form ready. Include the following markdown block VERBATIM in your "
        "next message so it renders for the user in the side panel — do not "
        "summarize or paraphrase the JSON.\n\n"
        "FORMATTING (critical): the opening ``` and the closing ``` must "
        "each be on their own line, with a blank line BEFORE the opening "
        "fence and AFTER the closing fence. Do not concatenate the fence "
        "onto the end of a sentence — markdown parsers won't recognise it "
        "as a code block if it isn't at the start of a line.\n\n"
        "After the user submits, you'll receive a continuation message "
        "with `submission_id` (and any skipped field names). Call "
        "`fetch_submission(submission_id)` to retrieve the staged values "
        "when you need them.\n\n"
        f"{block}"
    )


_REQUEST_CREDENTIALS_SCHEMA = {
    "type": "object",
    "properties": {
        "form_id": {
            "type": "string",
            "description": "Stable identifier for this form. Generate a new one for a new question; reuse the same one when re-asking the same form (so the user's typed values persist).",
        },
        "engine": {
            "type": "string",
            "description": "REQUIRED. A short slug for the connector (e.g. 'postgres', 'mysql', 'snowflake', 'github', 'posthog', 'salesforce', 'gmail', 'google_calendar'). Use the closest convention; ANY value is accepted — engines not in anton's built-in registry are saved as 'custom' connections with whatever fields you list here. Don't gate on whether it's a known engine.",
        },
        "_connector_id": {
            "type": "string",
            "description": "OPTIONAL. The slug of the canonical connector spec (from `lookup_connector`) this form was built from. When set, submissions go to POST /v1/connectors/{id}/save (which bypasses anton-core's built-in datasource registry — required for OAuth-shaped saves). ALWAYS set this when the form spec came from `lookup_connector`; leave unset when handcrafting a one-off spec.",
        },
        "how_to": {
            "type": "string",
            "description": "OPTIONAL. Markdown-formatted setup instructions for SINGLE-method forms (use the per-method `how_to` for multi-method specs). The form panel shows a 'How to?' link in the actions row that opens a centered modal with this content.",
        },
        "help_url": {
            "type": "string",
            "description": "OPTIONAL. External help URL for SINGLE-method forms (use the per-method `help_url` for multi-method specs). Used as a fallback when no `how_to` markdown is provided.",
        },
        "logo": {
            "type": "string",
            "description": "Optional icon name from the app's palette — use one of: 'database', 'globe', 'cube', 'doc', 'code', 'image', 'folder', 'brain', 'sparkle', 'wifi', 'key', 'link', 'mindsdb'. URLs are NOT supported; pick the closest semantic match for the connector. Defaults to 'database' when omitted.",
        },
        "logo_color": {
            "type": "string",
            "description": "Optional CSS color for the icon (e.g. '#3b82f6', 'var(--accent)').",
        },
        "title": {
            "type": "string",
            "description": "Short headline (e.g. 'Connect to Postgres').",
        },
        "subtitle": {
            "type": "string",
            "description": "Optional one-liner under the title (e.g. 'Anton needs read-only access — credentials never leave your machine.').",
        },
        "form_warning": {
            "type": "string",
            "description": "Optional amber banner above the fields. Use for cautionary notes ('Last attempt timed out…').",
        },
        "form_error": {
            "type": "string",
            "description": "Optional red banner above the fields. Use when a previous attempt failed at the form level (e.g. wrong engine selected).",
        },
        "fields": {
            "type": "array",
            "description": "Field specs the user fills in (single-method form). Order matters — render top to bottom. For services with MULTIPLE auth options (Gmail = OAuth + app-password + service-account, etc.) prefer `methods` instead.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Field id (env-var-like)."},
                    "label": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["text", "password", "url", "select", "textarea", "boolean"],
                    },
                    "required": {"type": "boolean"},
                    "placeholder": {"type": "string"},
                    "default": {},
                    "value": {"description": "Pre-fill on re-render (e.g. preserve what the user typed last attempt)."},
                    "options": {
                        "type": "array",
                        "description": "For type=select.",
                        "items": {
                            "type": "object",
                            "properties": {"value": {"type": "string"}, "label": {"type": "string"}},
                        },
                    },
                    "error": {"type": "string", "description": "Per-field red text under the input. Set on a retry to call out which field needs attention."},
                    "warning": {"type": "string", "description": "Per-field amber text under the input."},
                    "help": {"type": "string", "description": "Muted helper text under the input."},
                    "skipable": {"type": "boolean", "description": "Defaults to true. Pass false ONLY for absolute requirements where skipping makes no sense."},
                },
                "required": ["name", "label", "type"],
            },
        },
        "methods": {
            "type": "array",
            "description": (
                "Use INSTEAD of `fields` when the engine supports multiple auth methods "
                "(e.g. Gmail can be reached via OAuth, App Password, or Service Account). "
                "The user picks one method first, then fills in just that method's fields. "
                "Each method should have a clear label, a 1-2 sentence description, and "
                "its own fields. Mark the easiest one with `recommended: true`. If the "
                "user has already signalled a preference (\"I have an app password\"), "
                "set `selected_method` at the form's top level so we skip the picker."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Stable id for this method (e.g. 'oauth', 'app-password', 'service-account'). Anton's probe uses this id to decide which auth flow to test, and the vault stamps it onto the saved record as `_method`."},
                    "label": {"type": "string", "description": "Card title (e.g. 'App Password')."},
                    "description": {"type": "string", "description": "1-2 sentence description shown on the picker card. Mention prereqs and difficulty when relevant."},
                    "recommended": {"type": "boolean", "description": "Mark the easiest/most common method. Renders a 'Recommended' pill on the card."},
                    "how_to": {"type": "string", "description": "Optional markdown-formatted setup instructions for THIS method. Picker cards show a 'How to?' affordance that opens a centered modal with this content; the same link travels into the form-fill stage. Prefer this over `help_url` when you can write a good walkthrough."},
                    "help_url": {"type": "string", "description": "Optional external help URL for THIS method. Used when no `how_to` markdown is provided."},
                    "submit_action": {
                        "type": "string",
                        "enum": ["oauth_launch"],
                        "description": "Optional. When set to `oauth_launch`, the panel runs a PKCE OAuth flow on the user's machine when they click Submit (spawns a loopback HTTP server, opens the browser to the consent screen, exchanges the code for tokens, and saves the refresh_token to the vault). Required for any OAuth method. Pair with the `oauth` block.",
                    },
                    "oauth": {
                        "type": "object",
                        "description": "Required when `submit_action` is `oauth_launch`. Provides everything the desktop's PKCE helper needs to run the flow without the LLM in the loop. Anton spawns a loopback server, opens the browser, exchanges the code, and stores the refresh_token in the vault under this connector.",
                        "properties": {
                            "auth_url": {"type": "string", "description": "OAuth 2.0 authorization endpoint (e.g. https://accounts.google.com/o/oauth2/v2/auth)."},
                            "token_url": {"type": "string", "description": "OAuth 2.0 token endpoint (e.g. https://oauth2.googleapis.com/token)."},
                            "scopes": {
                                "type": "array",
                                "description": "Scopes to request. Provider-specific.",
                                "items": {"type": "string"},
                            },
                            "extra_auth_params": {
                                "type": "object",
                                "description": "Extra query params on the auth URL (e.g. {access_type: 'offline', prompt: 'consent'} for Google to force a refresh_token).",
                                "additionalProperties": {"type": "string"},
                            },
                        },
                        "required": ["auth_url", "token_url", "scopes"],
                    },
                    "fields": {
                        "type": "array",
                        "description": "Fields specific to this method. Same shape as the top-level `fields` items. For `oauth_launch` methods, fields are typically `client_id` + `client_secret` (Pattern B — bring-your-own-OAuth-client) and may be empty when the connector ships a hosted client.",
                        "items": {"type": "object"},
                    },
                    "actions": {
                        "type": "array",
                        "description": "OPTIONAL — per-method action buttons. Falls back to the form's top-level actions, then to a default Submit + Cancel pair.",
                        "items": {"type": "object"},
                    },
                },
                "required": ["id", "label", "fields"],
            },
        },
        "selected_method": {
            "type": "string",
            "description": "Pre-pick a method id from `methods[]` (skips the picker, jumps straight to that method's fields). Set when the user has clearly indicated a preference. The user can still hit 'change' to re-open the picker.",
        },
        "actions": {
            "type": "array",
            "description": "Optional. Defaults to a single primary 'Submit' action plus 'Cancel'. Use to surface custom actions like 'Try OAuth' or per-field skip shortcuts.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "kind": {"type": "string", "enum": ["primary", "skip", "cancel"]},
                    "field": {"type": "string", "description": "Only for kind='skip' — the field name to mark skipped."},
                },
                "required": ["id", "label"],
            },
        },
    },
    "required": ["engine", "title"],
}


_REQUEST_CREDENTIALS_PROMPT = (
    "DATA VAULT WORKFLOW — when the user asks to connect to a service or database:\n"
    "1. ALWAYS call `lookup_connector` FIRST with the user's wording (e.g. 'gmail', "
    "'google mail', 'postgres'). When it returns a `form` spec, that is the SAME "
    "spec the in-app Connector Picker uses — pass it to `request_credentials` "
    "VERBATIM (only tweak `selected_method` if the user has clearly signalled a "
    "preference, or `subtitle` to match the conversation context). Registry specs "
    "ship with `methods[]`, OAuth blocks (`submit_action: 'oauth_launch'` + "
    "`oauth: {...}`), `how_to` markdown, and `help_url` already filled in — do "
    "not strip those, do not paraphrase them. ALSO copy `_connector_id` (the "
    "lookup tool returns it) onto the form spec so submissions go to the "
    "registry-aware save endpoint.\n"
    "2. ONLY when `lookup_connector` returns no match, handcraft the spec. The "
    "schema documents OAuth, `how_to`, and `help_url` fields — use them when you "
    "know the auth shape (e.g. an OAuth provider). Better to write a good "
    "registry-grade spec than a bare username/password prompt.\n"
    "3. Call `request_credentials` with the (registry or handcrafted) spec the "
    "FIRST time. Include the returned markdown block VERBATIM (with blank lines "
    "around the fence) so the form renders in the side panel.\n"
    "4. Wait for the user's submission. The follow-up message has a `submission_id` "
    "and the names of any skipped fields. For OAuth methods, the desktop runs the "
    "browser flow itself — you'll just receive the submission with the refresh "
    "token already in the vault.\n"
    "5. Call `fetch_submission(submission_id)` to retrieve the staged values. Test "
    "the connection (`connect_datasource` or a scratchpad probe).\n"
    "6. ON FAILURE — DO NOT re-emit the full form. Use `update_form` (which "
    "returns a `data-vault-form-patch` block) to attach an `error` to the failing "
    "field and tweak `subtitle` / `form_warning` if useful. The user's existing "
    "inputs stay in the panel; only the changed bits update. NEVER include `value` "
    "fields in a patch or full re-emit — that would echo credentials into chat "
    "history. The user re-types what they want to fix.\n"
    "7. On success, summarize what you connected and stop. Do NOT call "
    "`request_credentials` again unless the user asks for another connection.\n\n"
    "MULTI-METHOD SHAPE — for engines with several auth options (Gmail can be "
    "reached via OAuth, App Password, or Service Account; Postgres might support "
    "password auth + IAM, etc.) emit a `methods[]` array INSTEAD of `fields[]`. "
    "Each method has its own id, label, description, fields, and (when relevant) "
    "`how_to` markdown / `help_url` / OAuth block. Mark the simplest one with "
    "`recommended: true`. The form panel renders a picker first; the user "
    "chooses, then types only the fields for the chosen method. The submission "
    "carries `auth_method` so the probe knows which one to test. If the user has "
    "CLEARLY signalled a preference (\"I already have an app password\"), "
    "pre-set `selected_method` to skip the picker.\n\n"
    "STRICT RULES:\n"
    "- Field VALUES never appear in chat. Don't echo them, don't include them in "
    "any form spec, don't paraphrase them. The fetch tool is the only read path.\n"
    "- Use `update_form` for any retry / error / status change after the initial "
    "form is up. Reserve `request_credentials` for first emission and for fully "
    "switching to a different connector.\n"
    "- The chat-emitted form and the picker-emitted form must FEEL identical to "
    "the user — same methods, same OAuth flow, same how-to docs. The registry "
    "lookup is what guarantees that parity; use it."
)


# ── lookup_connector ──────────────────────────────────────────────────
# The chat agent's "find the canonical spec for this service" path —
# same source of truth the in-app Connector Picker uses. Without this
# tool the LLM would hand-craft each form from training-data memory,
# producing slightly different shapes on each call (sometimes missing
# OAuth, sometimes bare username/password where the service has a
# proper OAuth path, etc.). Routing through the registry guarantees
# the chat-emitted form is byte-identical to what the picker emits.
#
# Mirrors POST /v1/connectors/match (server/routes/connectors.py):
# stage 1 exact match (id or alias, normalized) → stage 2 token-overlap.
# We don't reuse the route helpers directly to avoid a circular import
# (routes already pulls from anton_api), but the scoring stays in step.


def _lookup_normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _lookup_score(query: str, c: dict) -> float:
    q_tokens = set(_lookup_normalize(query).split())
    if not q_tokens:
        return 0.0
    label_tokens = set(_lookup_normalize(c.get("label", "")).split())
    alias_tokens: set[str] = set()
    for alias in c.get("aliases", []):
        alias_tokens.update(_lookup_normalize(alias).split())
    keyword_tokens = set(_lookup_normalize(" ".join(c.get("keywords", []))).split())
    desc_tokens = set(_lookup_normalize(c.get("description", "")).split())
    score = 0.0
    score += 3.0 * len(q_tokens & label_tokens)
    score += 2.5 * len(q_tokens & alias_tokens)
    score += 1.0 * len(q_tokens & keyword_tokens)
    score += 0.4 * len(q_tokens & desc_tokens)
    return score


async def _cowork_lookup_connector(session: Any, tc_input: dict) -> str:
    """Tool handler for `lookup_connector`.

    Pulls a registry-backed connector spec by id (slug) or by
    natural-language query. The returned `form` blob is what the LLM
    should pass to `request_credentials` verbatim — methods, OAuth
    blocks, how_to markdown, and help_url all already filled in. We
    also stamp `_connector_id` onto the form so submissions land via
    the registry-aware save path (POST /v1/connectors/{id}/save).
    """
    try:
        from . import connectors_registry as registry
    except Exception as exc:
        logger.exception("Cowork lookup_connector could not import registry")
        return f"lookup_connector: registry unavailable ({exc})"

    cid = (tc_input.get("id") or "").strip()
    query = (tc_input.get("query") or "").strip()

    if not cid and not query:
        return "lookup_connector: provide either `id` (a connector slug) or `query` (a natural-language name)."

    def _present(connector: dict, confidence: float, stage: str) -> str:
        # Stamp `_connector_id` into the form blob so the LLM doesn't
        # have to remember to copy it across. The vault save endpoint
        # uses this to route through POST /v1/connectors/{id}/save
        # (which bypasses anton-core's built-in datasource registry).
        form_blob = dict(connector.get("form") or {})
        form_blob["_connector_id"] = connector.get("id")
        # Also copy the connector's logo onto the form if the form
        # itself doesn't carry one — the picker already does this for
        # the in-app picker path; mirroring it here keeps the chat-
        # emitted form visually identical.
        if "logo" not in form_blob and connector.get("logo"):
            form_blob["logo"] = connector["logo"]
        if "logo_color" not in form_blob and connector.get("logo_color"):
            form_blob["logo_color"] = connector["logo_color"]
        return json.dumps({
            "id": connector.get("id"),
            "label": connector.get("label"),
            "description": connector.get("description"),
            "category": connector.get("category"),
            "confidence": confidence,
            "stage": stage,
            "form": form_blob,
            "next_step": (
                "Pass this `form` blob to `request_credentials` verbatim "
                "(only tweak `selected_method` or `subtitle` if needed). "
                "It already includes `_connector_id`, methods[], OAuth "
                "blocks, how_to markdown, and help_url where applicable."
            ),
        })

    # Stage 0 — explicit id always wins.
    if cid:
        c = registry.get_connector(cid)
        if c:
            return _present(c, 1.0, "id")
        # Fall through to query-style lookup if the id didn't match.
        if not query:
            return json.dumps({
                "id": None,
                "match": "none",
                "message": (
                    f"No connector with id `{cid}` in the registry. "
                    f"Either retry with a `query` (natural-language) or "
                    f"handcraft the form spec — see the request_credentials "
                    f"schema for the OAuth/how_to/help_url fields."
                ),
                "available_ids": sorted(registry.all_connectors().keys()),
            })
        # If both were given and id missed, treat the id as part of the query.
        query = f"{cid} {query}".strip()

    # Stage 1 — exact match (id or alias) on the query.
    nq = _lookup_normalize(query)
    for c in registry.all_connectors().values():
        if _lookup_normalize(c.get("id", "")) == nq:
            return _present(c, 1.0, "exact-id")
        for alias in c.get("aliases", []):
            if _lookup_normalize(alias) == nq:
                return _present(c, 1.0, "exact-alias")

    # Stage 2 — token-overlap. Return up to 3 candidates with confidence.
    scored: list[tuple[float, dict]] = []
    for c in registry.all_connectors().values():
        s = _lookup_score(query, c)
        if s > 0:
            scored.append((s, c))
    scored.sort(reverse=True, key=lambda x: x[0])

    if not scored:
        return json.dumps({
            "id": None,
            "match": "none",
            "message": (
                "No connector matched the query. Either ask the user to "
                "clarify, or handcraft the form spec — see the "
                "request_credentials schema for the OAuth/how_to/help_url "
                "fields you should fill in when you know the auth shape."
            ),
            "available_ids": sorted(registry.all_connectors().keys()),
        })

    top_score, top_c = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0

    # Top is dominant — single confident pick.
    if runner_up == 0.0 or top_score >= runner_up * 2:
        return _present(top_c, 0.85, "scored-single")

    # Otherwise return the top three with normalized confidence so the
    # LLM can either (a) pick one based on chat context or (b) ask the
    # user to clarify. We do NOT inline `form` for the runners-up to
    # keep the response small — the LLM can re-call lookup_connector
    # with an `id` once it picks.
    candidates = [
        {
            "id": c.get("id"),
            "label": c.get("label"),
            "description": c.get("description"),
            "confidence": round(s / top_score, 3),
        }
        for s, c in scored[:3]
    ]
    return json.dumps({
        "match": "ambiguous",
        "candidates": candidates,
        "message": (
            "Multiple connectors matched. Either ask the user to clarify, "
            "or call `lookup_connector` again with the chosen `id` to fetch "
            "the full form spec."
        ),
    })


_LOOKUP_CONNECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {
            "type": "string",
            "description": "Connector slug (e.g. 'gmail', 'postgres', 'slack'). Use this when you already know the canonical id; matches exactly. Returns the full form spec including methods[], OAuth, how_to, help_url, and a stamped `_connector_id`.",
        },
        "query": {
            "type": "string",
            "description": "Natural-language query (e.g. 'google mail', 'my postgres database', 'send slack messages'). Runs id-or-alias exact match, then token-overlap scoring, returning a single confident hit or up to 3 candidates with confidence scores.",
        },
    },
}


_LOOKUP_CONNECTOR_PROMPT = (
    "Use `lookup_connector` to fetch the canonical form spec for any "
    "service the user names. The returned `form` blob is the SAME "
    "spec the in-app Connector Picker uses — pass it to "
    "`request_credentials` verbatim. The registry already encodes "
    "OAuth flows, how_to markdown, help URLs, and method-picker UI "
    "for services like Gmail, so handcrafting from training-data "
    "memory will produce a strictly worse form. Always lookup first; "
    "only handcraft when the registry returns no match."
)


def build_cowork_lookup_connector_tool():
    from anton.core.tools.tool_defs import ToolDef
    return ToolDef(
        name="lookup_connector",
        description=(
            "Look up the canonical connector spec for a service by id or "
            "natural-language query. Returns the same form blob the "
            "in-app Connector Picker uses — pass it straight to "
            "`request_credentials`."
        ),
        input_schema=_LOOKUP_CONNECTOR_SCHEMA,
        handler=_cowork_lookup_connector,
        prompt=_LOOKUP_CONNECTOR_PROMPT,
    )


async def _cowork_fetch_submission(session: Any, tc_input: dict) -> str:
    """Tool handler for `fetch_submission` — return staged values for
    a previously-submitted form, by submission_id.
    """
    sid = tc_input.get("submission_id") or tc_input.get("id")
    if not sid:
        return "fetch_submission: missing submission_id"
    try:
        from . import datavault_submissions
    except Exception as exc:
        logger.exception("Cowork fetch_submission could not import store")
        return f"fetch_submission: store unavailable ({exc})"
    entry = datavault_submissions.get_submission(sid)
    if not entry:
        return (
            f"fetch_submission: submission `{sid}` not found or expired. "
            f"Submissions TTL after 24h. Ask the user to resubmit the form."
        )
    return json.dumps({
        "submission_id": entry.get("submission_id"),
        "form_id": entry.get("form_id"),
        "values": entry.get("values", {}),
        "skipped": entry.get("skipped", []),
    })


_FETCH_SUBMISSION_SCHEMA = {
    "type": "object",
    "properties": {
        "submission_id": {
            "type": "string",
            "description": "The id from the user's continuation message after they submitted the form.",
        },
    },
    "required": ["submission_id"],
}


def build_cowork_request_credentials_tool():
    from anton.core.tools.tool_defs import ToolDef
    return ToolDef(
        name="request_credentials",
        description=(
            "Request credentials / configuration from the user via an interactive "
            "form rendered in the side panel. Returns a markdown block you must "
            "include verbatim in your next assistant message so the form appears."
        ),
        input_schema=_REQUEST_CREDENTIALS_SCHEMA,
        handler=_cowork_request_credentials,
        prompt=_REQUEST_CREDENTIALS_PROMPT,
    )


def build_cowork_fetch_submission_tool():
    from anton.core.tools.tool_defs import ToolDef
    return ToolDef(
        name="fetch_submission",
        description=(
            "Retrieve the staged values from a `data-vault-form` submission. "
            "Returns JSON with `values`, `skipped`, and `form_id`. Field values "
            "never appear in chat history — this tool is the only way to read them."
        ),
        input_schema=_FETCH_SUBMISSION_SCHEMA,
        handler=_cowork_fetch_submission,
        prompt=None,
    )


# ── update_form ───────────────────────────────────────────────────────
# Patch dialect for in-place form updates. Anton uses this on retry
# loops and any time the form needs a field-level error / warning /
# label change without re-emitting the whole spec. The patch never
# carries `value` fields — the user's existing inputs are preserved
# client-side by the form panel.

async def _cowork_update_form(session: Any, tc_input: dict) -> str:
    """Tool handler for `update_form` — emit a patch dialect block
    that the renderer merges into the active form for this
    conversation.
    """
    patch = tc_input.get("patch") if isinstance(tc_input.get("patch"), dict) else tc_input
    if not isinstance(patch, dict):
        return "update_form: invalid patch — must be a JSON object with `form_id`"
    if not patch.get("form_id"):
        return "update_form: `form_id` is required (must match the form you previously emitted via request_credentials)"

    # Strip any `value` keys that snuck in — patches must NEVER carry
    # credential material. We log + drop rather than fail the call so
    # an over-eager LLM doesn't get stuck retrying.
    fields_obj = patch.get("fields")
    if isinstance(fields_obj, dict):
        sanitized_fields = {}
        for name, fp in fields_obj.items():
            if not isinstance(fp, dict):
                continue
            cleaned = {k: v for k, v in fp.items() if k != "value"}
            if "value" in fp:
                logger.info(
                    "update_form: stripped `value` from field %r — patches must not carry credentials",
                    name,
                )
            sanitized_fields[name] = cleaned
        patch = {**patch, "fields": sanitized_fields}

    block = "```data-vault-form-patch\n" + json.dumps(patch, indent=2) + "\n```"
    return (
        "Patch ready. Include the following markdown block VERBATIM in your "
        "next message (with blank lines around the fence). The form panel "
        "will merge it into the existing form — the user's typed values are "
        "preserved.\n\n"
        f"{block}"
    )


_UPDATE_FORM_SCHEMA = {
    "type": "object",
    "properties": {
        "form_id": {
            "type": "string",
            "description": "Must match the `form_id` of the form currently shown in the side panel.",
        },
        "title": {"type": "string", "description": "Optional. Replace the form title."},
        "subtitle": {"type": "string", "description": "Optional. Replace the subtitle."},
        "form_warning": {"type": "string", "description": "Optional. Set the amber form-level banner. Pass null to clear."},
        "form_error": {"type": "string", "description": "Optional. Set the red form-level banner. Pass null to clear."},
        "fields": {
            "type": "object",
            "description": (
                "Map of field NAME → partial field spec. Only the keys you "
                "include override the existing field's properties. Pass `null` "
                "for a key to clear that property (e.g. `error: null` to dismiss "
                "an error). Add a brand-new field by including its full spec "
                "under a name not already in the form. NEVER include `value` — "
                "the user's input is preserved client-side."
            ),
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "error": {"type": ["string", "null"], "description": "Per-field red text. Set on retry."},
                    "warning": {"type": ["string", "null"], "description": "Per-field amber text."},
                    "help": {"type": ["string", "null"]},
                    "placeholder": {"type": ["string", "null"]},
                    "required": {"type": "boolean"},
                    "skipable": {"type": "boolean"},
                },
            },
        },
        "actions": {
            "type": "array",
            "description": "Optional. Replace the actions list.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "kind": {"type": "string", "enum": ["primary", "skip", "cancel"]},
                    "field": {"type": "string"},
                },
                "required": ["id", "label"],
            },
        },
    },
    "required": ["form_id"],
}


_UPDATE_FORM_PROMPT = (
    "Use `update_form` for ANY change to a form already shown by "
    "`request_credentials`. Common cases:\n"
    "  • Connection failed → set `fields: { <name>: { error: 'message' } }` "
    "and `subtitle` to explain.\n"
    "  • Need an extra field → add it under a new key in `fields`.\n"
    "  • Need to clear a previous error → `fields: { <name>: { error: null } }`.\n"
    "Never include `value` — the user's typed input is preserved. The patch "
    "is far cheaper than a full re-emit and avoids leaking credentials into "
    "chat history."
)


def build_cowork_update_form_tool():
    from anton.core.tools.tool_defs import ToolDef
    return ToolDef(
        name="update_form",
        description=(
            "Patch the active data-vault-form for this conversation in place. "
            "Use this for retry loops, error messages, and any field-level "
            "tweak — the user's typed values are preserved client-side."
        ),
        input_schema=_UPDATE_FORM_SCHEMA,
        handler=_cowork_update_form,
        prompt=_UPDATE_FORM_PROMPT,
    )
