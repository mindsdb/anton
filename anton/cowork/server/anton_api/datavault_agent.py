"""Form Handler — owns the credential-test workflow end to end.

Architecture (the model the user asked for):

  POST /v1/datavault/submissions  (single SSE response)
        │
        ▼
  process_submission_stream  ◄─── this module's job
        │
        ├── stage values (datavault_submissions)
        ├── shape-validate (custom engines pass through)
        ├── spin up a HEADLESS probe via datavault_probe.run_probe
        │     • fresh ChatSession, no history, no persistence
        │     • toolbelt: set_status, report_success, report_failure,
        │       request_extra_field
        │     • prompt: "creds at <path>, connect to <engine>, report back"
        ├── translate probe events into UI:
        │     status        → form-patch with status_text
        │     scratchpad    → response.in_progress with thought_role.*
        │                     (right rail picks these up)
        │     text          → text delta (anton's prose into chat)
        │     verdict       → final form patch + chat one-liner +
        │                     vault save (success only)
        └── persist the synthesized turn (so reload reproduces it)

Vault save is deferred to `report_success` — the .env file is the
staging area, the vault is the commit. Failure paths leave the vault
clean.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import AsyncIterator

from . import conversation_manager, datavault_submissions, datavault_probe

logger = logging.getLogger(__name__)


def _sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _new_response_id() -> str:
    return "resp-" + uuid.uuid4().hex[:12]


def _new_message_id() -> str:
    return "msg-" + uuid.uuid4().hex[:12]


def _format_patch_block(patch: dict) -> str:
    """Wrap a patch object as a `data-vault-form-patch` markdown
    block. Surrounding blank lines are critical — without them the
    fence sticks to text and markdown won't recognise it.
    """
    body = json.dumps(patch, indent=2)
    return f"\n\n```data-vault-form-patch\n{body}\n```\n\n"


def _field_patch(form_id: str, method_id: str | None, name: str, value):
    """Build a patch that targets a single field, scoped either to
    the form's top-level fields[] (single-method) or to a specific
    method's fields[] (multi-method). `value` is the per-field patch
    object OR None (the latter signals deletion of the whole field).
    """
    if method_id:
        return {
            "form_id": form_id,
            "methods": {method_id: {"fields": {name: value}}},
        }
    return {
        "form_id": form_id,
        "fields": {name: value},
    }


def _validate_shape(engine: str, credentials: dict, auth_method: str | None):
    """Look the engine up in anton's datasource registry.

    Returns (engine_def, missing_fields). engine_def is None for
    custom engines (anything not in the registry) — the caller then
    skips the strict missing-field gate.
    """
    try:
        from ..routes.utilities import _validate_datasource_payload  # type: ignore
    except Exception:
        logger.debug("Could not import shape validator", exc_info=True)
        return None, []
    try:
        engine_def, _, _, _, missing = _validate_datasource_payload(
            engine, credentials, auth_method,
        )
        return engine_def, missing
    except Exception:
        # Unknown engine — fine, treat as custom.
        return None, []


def _save_connection(engine: str, name: str, credentials: dict, auth_method: str | None):
    """Persist the connection to anton's data vault. Returns the saved
    slug. Custom engines (no engine_def) are saved verbatim; registered
    ones get field-whitelisted.

    Honours the modify-flow sentinel merge: any incoming value equal
    to `ANTON_VAULT_KEEP` is resolved against the existing vault
    record so the chat-driven submission path picks up untouched
    secrets the renderer pre-filled with sentinels. The same
    `resolve_modify_merge` runs on every save — it's a no-op on
    create paths where no prior record exists.
    """
    from anton.core.datasources.data_vault import LocalDataVault
    # `resolve_modify_merge` ships in newer anton-core; fall back to
    # a passthrough when the installed anton predates it so the save
    # still lands (without sentinel resolution). Same defensive shim
    # for the `secure_keys` save kwarg — older vaults silently drop
    # it via Python's positional/kwarg semantics, but to keep one
    # code path we route everything through a thin local helper.
    try:
        from anton.core.datasources.data_vault import resolve_modify_merge as _merge
    except ImportError:
        _merge = None

    engine_def, _ = _validate_shape(engine, credentials, auth_method)

    spec_secret_names: list[str] = []
    if engine_def is None:
        cleaned = {str(k).strip(): str(v) for k, v in credentials.items() if str(k).strip()}
    else:
        from ..routes.utilities import _validate_datasource_payload  # type: ignore
        _, _, _, fields, _ = _validate_datasource_payload(engine, credentials, auth_method)
        known = {f.name for f in fields}
        cleaned = {k: v for k, v in credentials.items() if k in known}
        spec_secret_names = [
            getattr(f, "name", "") for f in fields
            if getattr(f, "secret", False) and getattr(f, "name", "")
        ]

    vault = LocalDataVault()
    save_name = (name or "").strip() or f"{engine}-{vault.next_connection_number(engine)}"
    if _merge is not None:
        merged, secure_keys = _merge(
            vault, engine, save_name, cleaned,
            spec_secret_keys=spec_secret_names,
        )
        try:
            vault.save(engine, save_name, merged, secure_keys=secure_keys)
        except TypeError:
            # Older LocalDataVault.save signature lacks the keyword.
            vault.save(engine, save_name, merged)
    else:
        # Legacy fallback: no sentinel resolution, no secure_keys
        # persistence. The user is on a stale anton-core and needs
        # to reinstall to get the modify-flow merge — log a hint so
        # the trail in the chat is informative.
        logger.warning(
            "anton-core has no `resolve_modify_merge` — saving without "
            "the modify-flow merge. Reinstall anton (e.g. "
            "`uv tool install --reinstall --force-reinstall anton`) "
            "to enable sentinel-based credential preservation."
        )
        vault.save(engine, save_name, cleaned)
    return save_name


async def process_submission_stream(
    *,
    submission_id: str,
    form_spec: dict,
    conversation_id: str | None,
) -> AsyncIterator[str]:
    """Synthesize a chat turn that processes a form submission.

    Yields SSE-formatted strings in the same shape anton's chat stream
    emits (response.created → text deltas + in_progress events →
    response.completed). Single SSE envelope from start to finish —
    the probe's events are translated and forwarded INSIDE this turn.
    """
    started_at_ms = int(time.time() * 1000)
    response_id = _new_response_id()
    message_id = _new_message_id()
    seq = 0
    body_parts: list[str] = []
    recorded_events: list[dict] = []

    def _push(event_type: str, data: dict):
        nonlocal seq
        seq += 1
        payload = {**data, "sequence_number": seq}
        recorded_events.append(payload)
        return _sse(event_type, payload)

    def _delta(text: str) -> str:
        body_parts.append(text)
        return _push("response.output_text.delta", {
            "type": "response.output_text.delta",
            "item_id": message_id,
            "delta": text,
        })

    def _patch_delta(patch: dict) -> str:
        """Emit a form-patch as a text delta. The markdown extension
        on the client extracts these blocks and applies them to the
        side panel; the chat bubble just sees the fenced block (which
        the renderer strips visually).
        """
        return _delta(_format_patch_block(patch))

    form_id = (form_spec or {}).get("form_id")

    # response.created — opens the SSE turn.
    yield _push("response.created", {
        "type": "response.created",
        "response": {"id": response_id, "model": "datavault-agent", "status": "created"},
        "conversation_id": conversation_id,
    })

    # ── Stage check ─────────────────────────────────────────────────
    submission = datavault_submissions.get_submission(submission_id)
    if not submission:
        yield _delta(
            "The form submission expired before I could process it. "
            "Please re-submit the form."
        )
        yield _push("response.completed", {
            "type": "response.completed",
            "response": {"id": response_id, "status": "failed"},
        })
        _persist_turn(conversation_id, body_parts, recorded_events, started_at_ms)
        return

    engine = (form_spec or {}).get("engine") or "unknown"
    auth_method = (form_spec or {}).get("auth_method")
    values = submission.get("values", {}) or {}
    skipped = submission.get("skipped", []) or []
    name_hint = (form_spec or {}).get("name") or (form_spec or {}).get("connection_name") or ""

    credentials = {
        k: v for k, v in values.items()
        if k not in skipped and (v is not None and v != "")
    }

    # No registry-shape gate here — the chat-driven flow uses the
    # connector JSON's own field names (`username`, `connection_uri`,
    # …) which often don't match anton's CLI datasource registry
    # (`user`, `connection_string`, …). Validating against the
    # registry produced "missing required field: user" messages even
    # when the user had filled out the equivalent `username`. The
    # form already enforces `required: true` client-side; if a
    # credential is genuinely insufficient, the probe will surface
    # the real authentication error from the engine itself.

    # ── Probe setup ─────────────────────────────────────────────────
    # Need the conversation's anton session to lift llm_client +
    # workspace from. The probe runs in a fresh session that inherits
    # those but persists nothing. Look up the conversation's project
    # explicitly so a non-active-project conversation still resolves
    # to the right .anton dir (instead of the active project's).
    #
    # Recovery strategy — historically this throws `[Errno 2] No such
    # file or directory` when the conversation's cached session in
    # `conversation_manager._live[cid]` holds a stale path (project
    # renamed / moved / deleted on disk while the in-memory session
    # still points at the old location). A server restart heals it
    # because the cache is in-process. Mimic that here automatically:
    #
    #   1. First attempt — use the conversation's recorded project.
    #   2. On any exception → evict the cached session for this cid,
    #      then retry with `project=None` so resolve falls through to
    #      the active project (which self-heals to default if its own
    #      directory is missing).
    #   3. If that still fails → surface a clean error. The retry
    #      almost always succeeds, but the second guard means we
    #      never silently swallow a deeper bug.
    base_session = None
    last_error: Exception | None = None

    async def _try_resolve(project_name):
        return await conversation_manager._resolve_session(
            conversation_id, project_name, None,
        )

    if conversation_id:
        # Attempt 1 — discovered project.
        try:
            located = conversation_manager._find_conversation_dir(conversation_id)
            project_name = located[0] if located else None
            base_session = await _try_resolve(project_name)
        except Exception as exc:
            logger.warning(
                "Probe base-session resolve failed for %s (project=%r): %r — evicting cache and retrying.",
                conversation_id, project_name if 'project_name' in locals() else None, exc,
            )
            last_error = exc
            base_session = None

        # Attempt 2 — evict any stale cached session, retry against
        # the active project. `_resolve_session` rebuilds from disk
        # when the cid isn't in `_live`.
        if base_session is None:
            try:
                conversation_manager._live.pop(conversation_id, None)
                base_session = await _try_resolve(None)
                logger.info(
                    "Probe base-session recovered for %s after cache eviction.",
                    conversation_id,
                )
                last_error = None
            except Exception as exc:
                logger.exception("Probe base-session retry also failed")
                last_error = exc
                base_session = None

    if last_error is not None and base_session is None:
        # Surface the underlying filename when present so the user
        # has something actionable. `[Errno 2] No such file or
        # directory` alone tells them nothing.
        path_hint = ""
        try:
            if isinstance(last_error, OSError) and last_error.filename:
                path_hint = f" (missing path: {last_error.filename})"
        except Exception:
            pass
        yield _delta(
            f"Could not start the connection probe — `{last_error}`{path_hint}. "
            f"Try again, or restart the app if it persists.\n\n"
        )
        yield _patch_delta({
            "form_id": form_id,
            "form_error": f"Probe setup failed: {last_error}",
        })
        yield _push("response.completed", {
            "type": "response.completed",
            "response": {"id": response_id, "status": "failed"},
        })
        _persist_turn(conversation_id, body_parts, recorded_events, started_at_ms)
        return

    if base_session is None:
        # No conversation to attach to — degrade gracefully: save
        # without a probe verdict, tell the user.
        try:
            slug = _save_connection(engine, name_hint, credentials, auth_method)
        except Exception as exc:
            yield _delta(f"Could not save: `{exc}`.")
            yield _push("response.completed", {
                "type": "response.completed",
                "response": {"id": response_id, "status": "failed"},
            })
            _persist_turn(conversation_id, body_parts, recorded_events, started_at_ms)
            return
        yield _delta(f"Saved as `{slug}` (no live probe — no conversation context).\n\n")
        yield _patch_delta({
            "form_id": form_id,
            "title": f"Saved — {slug}",
            "subtitle": "Stored in the vault. No live verification was performed.",
            "_is_success": True,
            "actions": [{"id": "dismiss", "label": "Close", "kind": "cancel"}],
        })
        yield _push("response.completed", {
            "type": "response.completed",
            "response": {"id": response_id, "status": "success"},
        })
        _persist_turn(conversation_id, body_parts, recorded_events, started_at_ms)
        return

    # Brief intro sentence — tells the user something started; the
    # form panel takes over from here with live status.
    yield _delta(f"Trying to connect to **{engine}**…\n\n")

    # Initial probing patch — collapses the form fields, shows the
    # spinner with a generic status until anton fires its first
    # set_status call.
    yield _patch_delta({
        "form_id": form_id,
        "_is_probing": True,
        "status_text": "Starting probe…",
        "form_error": None,
    })

    # ── Run the probe ───────────────────────────────────────────────
    # Each event from the probe is translated into the existing SSE
    # vocabulary the client already speaks:
    #   text       → text delta (chat bubble)
    #   status     → form-patch text delta (status_text update)
    #   scratchpad → response.in_progress with thought_role (right rail)
    #   verdict    → final form patch + chat verdict + (success) vault save
    final_outcome: datavault_probe.ProbeOutcome | None = None

    # Track scratchpad cell metadata across start/end/result so the
    # rail event has the code+description on the result event.
    pending_cell: dict = {}

    try:
        async for kind, payload in datavault_probe.run_probe(
            engine=engine,
            credentials=credentials,
            base_session=base_session,
            form_spec=form_spec,
            skipped=skipped,
        ):
            if kind == "text":
                # Anton's prose lands directly in the chat bubble.
                yield _delta(payload)
            elif kind == "status":
                # Form panel's live status row.
                yield _patch_delta({
                    "form_id": form_id,
                    "_is_probing": True,
                    "status_text": payload,
                })
            elif kind == "field_status":
                # Granular per-field status — translates into the
                # smallest possible patch: just `fields[name].status`.
                # The form-store merges this into the existing field
                # by name, leaving every other property (label, type,
                # value, error, etc) untouched. For multi-method forms
                # the patch is nested under methods[method_id].
                name = (payload or {}).get("name")
                if name:
                    status_val = (payload or {}).get("status")
                    method_id = (payload or {}).get("method_id")
                    yield _patch_delta(_field_patch(form_id, method_id, name, {"status": status_val}))
            elif kind == "remove_field":
                # Field deletion — `fields[name] = null` at the patch
                # level is the form-store's "remove this whole field"
                # signal (distinct from `fields[name].prop = null`
                # which only clears one property). For multi-method
                # forms the deletion nests inside the method's fields.
                name = (payload or {}).get("name")
                if name:
                    method_id = (payload or {}).get("method_id")
                    yield _patch_delta(_field_patch(form_id, method_id, name, None))
            elif kind == "switch_method":
                # Flip selected_method at the form level. Reason (if
                # provided) becomes the form's status_text so the user
                # sees a one-line explanation in the toast.
                method_id = (payload or {}).get("method_id")
                reason = (payload or {}).get("reason") or ""
                if method_id:
                    patch = {
                        "form_id": form_id,
                        "selected_method": method_id,
                    }
                    if reason:
                        patch["status_text"] = reason
                    yield _patch_delta(patch)
            elif kind == "scratchpad":
                action = payload.get("action")
                if action == "start":
                    yield _push("response.in_progress", {
                        "type": "response.in_progress",
                        "thought_role": "thought.scratchpad.start",
                        "tool_name": "scratchpad",
                    })
                elif action == "end":
                    pending_cell.update(payload)
                    yield _push("response.in_progress", {
                        "type": "response.in_progress",
                        "thought_role": "thought.scratchpad.end",
                        "content": json.dumps({
                            "name": payload.get("name", ""),
                            "one_line_description": payload.get("one_line_description", ""),
                            "code": payload.get("code", ""),
                        }),
                    })
                elif action == "result":
                    yield _push("response.in_progress", {
                        "type": "response.in_progress",
                        "thought_role": "thought.scratchpad.result",
                        "content": json.dumps({
                            "code": pending_cell.get("code", ""),
                            "stdout": payload.get("content", ""),
                            "stderr": "",
                        }),
                    })
                    pending_cell = {}
            elif kind == "verdict":
                final_outcome = payload
                break
    except Exception as exc:
        logger.exception("Probe iteration failed")
        final_outcome = datavault_probe.ProbeOutcome(
            status="failure",
            error=f"Probe runner crashed: {exc}",
            follow_up="Try resubmitting; if it persists, restart the app.",
        )

    # ── Apply the verdict ───────────────────────────────────────────
    if final_outcome is None:
        final_outcome = datavault_probe.ProbeOutcome(
            status="failure",
            error="Probe ended without a verdict.",
        )

    saved_slug: str | None = None
    if final_outcome.status == "success":
        try:
            saved_slug = _save_connection(engine, name_hint, credentials, auth_method)
        except Exception as exc:
            logger.exception("Vault save failed despite probe success")
            final_outcome.status = "failure"
            final_outcome.error = f"Probe succeeded but save failed: {exc}"

    if final_outcome.status == "success":
        # Brief verdict line for the chat (form panel shows the rest).
        summary = final_outcome.summary or "Connection works."
        yield _delta(f"\n\n{summary}\n")
        # Don't include `actions` — the form widget renders a
        # default pair on success: "Close" + "View connectors →"
        # (the latter wired by the panel host to navigate to the
        # Connect Apps and Data page).
        yield _patch_delta({
            "form_id": form_id,
            "title": f"Connected — {saved_slug}",
            "subtitle": summary,
            "status_text": None,
            "form_error": None,
            "_is_probing": False,
            "_is_success": True,
        })
        yield _push("response.completed", {
            "type": "response.completed",
            "response": {"id": response_id, "status": "success"},
        })
    elif final_outcome.status == "needs_input":
        # Reopen the form with the new fields appended. The user
        # resubmits and the whole flow runs again with the merged
        # credential set.
        reason = final_outcome.follow_up or "We need a few more details before we can connect."
        yield _delta(f"\n\nI need a bit more info before I can finish: {reason}\n")
        extra = {}
        for f in final_outcome.extra_fields:
            extra[f.get("name")] = {
                "label": f.get("label") or f.get("name"),
                "type": f.get("type") or "text",
                "help": f.get("help") or "",
                "placeholder": f.get("placeholder") or "",
                "required": bool(f.get("required", True)),
            }
        # Multi-method needs_input scopes the new fields under the
        # method id anton tagged on the verdict; otherwise they go
        # to the form's top-level fields[].
        target_method = getattr(final_outcome, "method_id", None)
        if target_method:
            yield _patch_delta({
                "form_id": form_id,
                "subtitle": reason,
                "status_text": None,
                "_is_probing": False,
                "form_error": None,
                "methods": {target_method: {"fields": extra}},
            })
        else:
            yield _patch_delta({
                "form_id": form_id,
                "subtitle": reason,
                "status_text": None,
                "_is_probing": False,
                "form_error": None,
                "fields": extra,
            })
        yield _push("response.completed", {
            "type": "response.completed",
            "response": {"id": response_id, "status": "needs_input"},
        })
    else:
        err = final_outcome.error or "Connection failed."
        hint = final_outcome.follow_up or "Update the form and try again."
        yield _delta(f"\n\n{err} {hint}\n")
        yield _patch_delta({
            "form_id": form_id,
            "subtitle": hint,
            "status_text": None,
            "_is_probing": False,
            "form_error": err,
            "_is_success": False,
        })
        yield _push("response.completed", {
            "type": "response.completed",
            "response": {"id": response_id, "status": "retry"},
        })

    _persist_turn(conversation_id, body_parts, recorded_events, started_at_ms)


def _persist_turn(
    conversation_id: str | None,
    body_parts: list[str],
    events: list[dict],
    started_at_ms: int,
) -> None:
    """Append the synthesized turn to the conversation's history file +
    write its events into the per-turn sidecar so reload reconstructs
    the chat (form patches, scratchpad steps, verdict text).
    """
    if not conversation_id:
        return
    text = "".join(body_parts).strip()
    if not text:
        return

    located = conversation_manager._find_conversation_dir(conversation_id)
    if not located:
        logger.debug("datavault_agent: conversation %s not found, skipping persist", conversation_id)
        return
    project, _ = located

    try:
        history = conversation_manager._load_history(project, conversation_id) or []
        if not isinstance(history, list):
            history = []
        history.append({"role": "assistant", "content": text})
        history_path = conversation_manager._history_path(project, conversation_id)
        conversation_manager._write_history(history_path, history)
    except Exception:
        logger.debug("Could not persist datavault turn to history", exc_info=True)
        return

    try:
        conversation_manager.record_turn_events(conversation_id, started_at_ms, events)
    except Exception:
        logger.debug("Could not persist datavault turn events", exc_info=True)

    try:
        conversation_manager._live.pop(conversation_id, None)
    except Exception:
        pass
