from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from anton.core.backends.base import Cell
from anton.core.tools.registry import ToolOutcome
from anton.core.tools.side_effect import SideEffectResult, now_iso
from anton.core.utils.scratchpad import (
    prepare_scratchpad_exec,
    format_cell_result,
    observe_scratchpad_cell,
    cell_failure_reason,
    install_call_installed_something,
    reject_invalid_packages,
    send_package_install_event,
)

if TYPE_CHECKING:
    from anton.chat_session import ChatSession


_log = logging.getLogger(__name__)


async def _fire_pre_execute(session: "ChatSession", cell: Cell) -> None:
    """Notify pre-execute observers (e.g. cerebellum) before a cell runs.

    Best-effort: a buggy observer never kills a cell. The list of
    observers is owned by the session — typically populated in
    ChatSession.__init__. Empty list (or attribute missing) means no
    observers and this is a no-op.
    """
    observers = getattr(session, "_scratchpad_observers", None) or []
    for obs in observers:
        on_pre = getattr(obs, "on_pre_execute", None)
        if on_pre is None:
            continue
        try:
            await on_pre(cell)
        except Exception as exc:
            _log.warning(
                "scratchpad pre-execute observer %s failed: %s",
                type(obs).__name__,
                exc,
            )


async def _fire_post_execute(session: "ChatSession", cell: Cell) -> None:
    """Notify post-execute observers (e.g. cerebellum) after a cell finishes.

    Same best-effort contract as `_fire_pre_execute`.
    """
    observers = getattr(session, "_scratchpad_observers", None) or []
    for obs in observers:
        on_post = getattr(obs, "on_post_execute", None)
        if on_post is None:
            continue
        try:
            await on_post(cell)
        except Exception as exc:
            _log.warning(
                "scratchpad post-execute observer %s failed: %s",
                type(obs).__name__,
                exc,
            )


def _artifact_store(session: "ChatSession"):
    """Return the artifact store rooted at the session's workspace.

    Returns None when the session has no workspace (e.g. CLI calls
    that don't go through `resolve_workspace`). Tool handlers fall
    back to a clear error string in that case rather than raising.
    """
    workspace = getattr(session, "_workspace", None)
    if workspace is None:
        return None
    from anton.core.artifacts import ArtifactStore
    return ArtifactStore(workspace.artifacts_dir)


def _track_artifact(session: "ChatSession", store, slug: str, *, summary: str = "") -> None:
    """Record that THIS turn created or opened `slug`.

    Two records, for two different readers:

    * `session._artifacts_touched` — in-memory, per-turn. The host reads it
      after the turn to build the turn's artifact cards. This is what makes
      attribution correct without diffing the artifacts directory: several
      turns can share one project-wide artifacts folder and each still knows
      exactly what it touched.
    * `provenance` in the artifact's own `metadata.json`, via
      `record_turn()` — durable, and the only record that survives the
      process. Answers "which conversations have ever worked on this?" for
      any later reader.

    Best-effort: attribution is bookkeeping, so a failure here must never
    fail the tool call the agent actually made.
    """
    try:
        session._artifacts_touched.add(slug)
    except AttributeError:
        # A session predating this field (or a test double) still gets
        # durable provenance below.
        pass
    conversation_id = str(getattr(session, "_session_id", "") or "")
    if not conversation_id:
        # No host conversation to attribute to (a bare CLI session). The
        # in-memory set above is still useful; provenance would be a row
        # keyed by nothing.
        return
    try:
        store.record_turn(
            slug,
            conversation_id=conversation_id,
            conversation_title=None,
            turn_index=getattr(session, "_turn_count", 0) + 1,
            summary=summary,
            files_touched=[],
        )
    except Exception:
        _log.warning("could not record turn provenance for artifact %s", slug, exc_info=True)


def _artifact_content_mtime(folder: Path) -> float:
    """Max mtime across an artifact folder's user-content files.

    Excludes the store's own housekeeping files, mirroring cowork-server's
    `content_mtime` gate so both sides agree on what "changed" means.
    """
    from anton.core.artifacts.store import (
        METADATA_FILENAME,
        PUBLISHED_FILENAME,
        README_FILENAME,
    )

    housekeeping = {METADATA_FILENAME, README_FILENAME, PUBLISHED_FILENAME}
    try:
        return max(
            (
                p.stat().st_mtime
                for p in folder.rglob("*")
                if p.is_file()
                and not p.is_symlink()
                and str(p.relative_to(folder)) not in housekeeping
            ),
            default=0.0,
        )
    except OSError:
        return 0.0


def _snapshot_existing_artifact_mtimes(store) -> dict[str, float]:
    """slug -> content mtime, for every artifact folder that exists right now."""
    root = store.root
    if not root.is_dir():
        return {}
    mtimes: dict[str, float] = {}
    for child in root.iterdir():
        if child.is_dir() and (child / "metadata.json").is_file():
            mtimes[child.name] = _artifact_content_mtime(child)
    return mtimes


def _track_edits_since(session: "ChatSession", store, before: dict[str, float]) -> None:
    """Catch an artifact edit the agent made without calling `open_artifact`.

    `open_artifact` is how attribution is SUPPOSED to work (see
    `_track_artifact` above), but nothing forces the agent to call it again
    once it already has an artifact's path from earlier in the conversation —
    it can (and in practice does) just write straight into a remembered
    folder via the scratchpad, skipping the tool call entirely. Without this,
    that edit's `_artifacts_touched` stays empty and the host can never card
    it, even though the file genuinely changed this turn (ENG-1933 follow-up).

    Scoped to THIS cell's own execution window — the snapshot taken right
    before `pad.execute` vs. right after — rather than the whole turn, to
    keep the diff-based race this reintroduces (a concurrent sibling
    conversation happening to bump some other artifact's mtime) as narrow as
    possible: seconds, not minutes. Narrower than the pre-ENG-1933 exposure,
    not zero — the same trade-off `index_turn_artifacts` already accepts for
    Hermes's edits.
    """
    already_touched = getattr(session, "_artifacts_touched", None) or ()
    after = _snapshot_existing_artifact_mtimes(store)
    for slug, prev_mtime in before.items():
        current = after.get(slug)
        if current is not None and current > prev_mtime and slug not in already_touched:
            _track_artifact(session, store, slug, summary="Edited via scratchpad")


async def handle_create_artifact(session: "ChatSession", tc_input: dict) -> ToolOutcome:
    """Create a fresh artifact folder + metadata.json + README.md.

    Returns a `SideEffectResult` whose `message` carries the artifact path the
    agent writes output files under (`<path>/...`) after this call returns.
    """
    store = _artifact_store(session)
    if store is None:
        return SideEffectResult.failed(
            "Artifact store unavailable (no workspace bound to this session).",
            reason="store_unavailable",
        )

    name = (tc_input.get("name") or "").strip()
    description = (tc_input.get("description") or "").strip()
    artifact_type = (tc_input.get("type") or "").strip()
    primary = tc_input.get("primary")
    if not name:
        return SideEffectResult.failed("Error: `name` is required.", reason="missing_name")
    if not description:
        return SideEffectResult.failed(
            "Error: `description` is required.", reason="missing_description"
        )

    from anton.core.artifacts.models import ARTIFACT_TYPES

    if artifact_type not in ARTIFACT_TYPES:
        return SideEffectResult.failed(
            f"Error: `type` must be one of {ARTIFACT_TYPES}. Got: {artifact_type!r}.",
            reason="invalid_type",
        )

    artifact = store.create(  # type: ignore[arg-type]
        name=name,
        description=description,
        type=artifact_type,
        primary=primary if isinstance(primary, str) else None,
    )
    folder = store.folder_for(artifact.slug)
    _track_artifact(session, store, artifact.slug, summary=f"Created artifact: {name}")
    return SideEffectResult(
        success=True,
        message=(
            f"Created artifact `{artifact.slug}` ({artifact.type}). "
            f"Write output files under: {folder}"
            + (f" (primary: {artifact.primary})" if artifact.primary else "")
        ),
        resource_id=artifact.slug,
        idempotency_key=artifact.slug,
        committed_at=now_iso(),
        details={
            "slug": artifact.slug,
            "path": str(folder),
            "name": artifact.name,
            "type": artifact.type,
            "primary": artifact.primary,
        },
    ).to_outcome()


async def handle_update_artifact_metadata(session: "ChatSession", tc_input: dict) -> ToolOutcome:
    """Update mutable metadata fields on an existing artifact.

    Only fields present in the input are modified. Supports:
    - `primary`: entry-point file path (empty string to clear)
    - `port`: backend port number (fullstack apps only)
    - `datasources`: list of vault-connection slugs the backend reads from.
      `engine`, `name`, and `env_prefix` are derived from the vault.
    """
    store = _artifact_store(session)
    if store is None:
        return SideEffectResult.failed(
            "Artifact store unavailable (no workspace bound to this session).",
            reason="store_unavailable",
        )

    slug = (tc_input.get("slug") or "").strip()
    if not slug:
        return SideEffectResult.failed("Error: `slug` is required.", reason="missing_slug")

    kwargs: dict = {}
    if "primary" in tc_input:
        kwargs["primary"] = tc_input["primary"]
    if "port" in tc_input:
        try:
            kwargs["port"] = int(tc_input["port"]) if tc_input["port"] is not None else None
        except (TypeError, ValueError):
            return SideEffectResult.failed("Error: `port` must be a number.", reason="invalid_port")

    if "datasources" in tc_input:
        from anton.core.artifacts.models import DatasourceRef
        from anton.core.datasources.data_vault import LocalDataVault

        raw_list = tc_input.get("datasources") or []
        if not isinstance(raw_list, list):
            return SideEffectResult.failed(
                "Error: `datasources` must be a list of slug strings.",
                reason="invalid_datasources",
            )

        vault = session._data_vault or LocalDataVault()
        known = {f"{c['engine']}-{c['name']}": (c["engine"], c["name"])
                 for c in vault.list_connections()}

        refs: list[DatasourceRef] = []
        unknown: list[str] = []
        for item in raw_list:
            if not isinstance(item, str):
                return SideEffectResult.failed(
                    "Error: each entry in `datasources` must be a slug string.",
                    reason="invalid_datasources",
                )
            ref_slug = item.strip()
            if not ref_slug:
                continue
            if ref_slug not in known:
                unknown.append(ref_slug)
                continue
            engine, name = known[ref_slug]
            refs.append(DatasourceRef(engine=engine, name=name))
        if unknown:
            return SideEffectResult.failed(
                f"Error: unknown datasource slug(s): {', '.join(unknown)}. "
                f"Each slug must match an existing vault connection "
                f"(format: `<engine>-<name>`).",
                reason="unknown_datasource",
            )
        kwargs["datasources"] = refs

    artifact = store.update(slug, **kwargs)
    if artifact is None:
        return SideEffectResult.failed(
            f"Error: no artifact found for slug `{slug}`.", reason="artifact_not_found"
        )
    datasources = [d.slug for d in artifact.datasources]
    _track_artifact(session, store, artifact.slug, summary=f"Updated artifact metadata: {artifact.name}")
    return SideEffectResult(
        success=True,
        message=(
            f"Updated artifact `{artifact.slug}` "
            f"(primary={artifact.primary}, port={artifact.port}, "
            f"datasources={datasources})."
        ),
        resource_id=artifact.slug,
        idempotency_key=artifact.slug,
        committed_at=now_iso(),
        details={
            "slug": artifact.slug,
            "primary": artifact.primary,
            "port": artifact.port,
            "datasources": datasources,
        },
    ).to_outcome()


async def handle_launch_backend(session: "ChatSession", tc_input: dict) -> ToolOutcome:
    """Launch the artifact's backend script as a standalone subprocess.

    Thin wrapper over `launch_artifact_backend`: validates tool-call shape,
    resolves the artifact folder via the session's ArtifactStore, hands
    the session's scratchpad pool + tracked-backends dict to the helper,
    then persists the discovered port into metadata.json.

    The actual subprocess lifecycle (free-port discovery, dependency
    install, health probe, idempotent reap) lives in
    `anton.core.artifacts.backend_launcher.launch_artifact_backend` so
    other entry points (e.g. cowork's auto-relaunch) can reuse it.
    """
    from anton.core.artifacts.backend_launcher import launch_artifact_backend

    store = _artifact_store(session)
    if store is None:
        return SideEffectResult.failed(
            "Artifact store unavailable (no workspace bound to this session).",
            reason="store_unavailable",
        )

    slug = (tc_input.get("slug") or "").strip()
    if not slug:
        return SideEffectResult.failed("Error: `slug` is required.", reason="missing_slug")
    artifact = store.open(slug)
    if artifact is None:
        return SideEffectResult.failed(
            f"Error: no artifact found for slug `{slug}`.", reason="artifact_not_found"
        )

    rel_path = (tc_input.get("path") or "backend.py").strip()
    extra_args = tc_input.get("extra_args") or []
    health_path = tc_input.get("health_path") or "/"
    try:
        health_timeout = float(tc_input.get("health_timeout", 10))
    except (TypeError, ValueError):
        return SideEffectResult.failed(
            "Error: `health_timeout` must be a number.", reason="invalid_health_timeout"
        )

    tracked = getattr(session, "_tracked_backends", None)
    if tracked is None:
        tracked = {}
        session._tracked_backends = tracked

    # A subprocess, so its credentials go in its own env — and only for the
    # datasources the artifact declared.
    vault = getattr(session, "_data_vault", None)
    ds_env: dict[str, str] = {}
    for ref in artifact.datasources:
        if vault is None:
            _log.warning("Artifact %s declares datasources but the session has no vault", slug)
            break
        # Per-ref so one unreadable connection cannot deny the others, and
        # env_for is the resolver a pad's own DS_* are built from.
        try:
            env = vault.env_for(ref.engine, ref.name)
        except Exception:
            _log.warning(
                "Could not resolve %s/%s for backend %s", ref.engine, ref.name, slug,
                exc_info=True,
            )
            continue
        if env is None:
            # Declared in metadata but gone from the vault: the backend would
            # fail on its first query with nothing saying why.
            _log.warning(
                "Artifact %s declares %s/%s, which is not in the vault",
                slug, ref.engine, ref.name,
            )
            continue
        # Enforced here, not left to the vault: TurnKeyDataVault.env_for does
        # not drop `_`-prefixed bookkeeping though its contract says it does.
        field_prefix = f"{ref.env_prefix}__"
        for key, value in env.items():
            field = key[len(field_prefix):] if key.startswith(field_prefix) else key
            if field.startswith("_"):
                continue
            ds_env[key] = value

    # Only-if-unset, like the scratchpad, so a project .env cannot override
    # PATH or a key this process already has.
    overlay = getattr(session, "_workspace_env_overlay", None) or {}
    extra_env = {k: v for k, v in overlay.items() if k not in os.environ}

    result = await launch_artifact_backend(
        slug=slug,
        artifact_folder=store.folder_for(slug),
        scratchpad_pool=session._scratchpads,
        tracked_backends=tracked,
        path=rel_path,
        extra_args=extra_args,
        extra_env=extra_env,
        ds_env=ds_env,
        health_path=health_path,
        health_timeout=health_timeout,
    )
    # The launcher rolls back on failure (kills the process, never tracks it),
    # so a string result means nothing committed.
    if isinstance(result, str):
        return SideEffectResult.failed(result, reason="launch_failed")

    store.update(slug, port=result["port"])
    url = result.get("url", "")
    return SideEffectResult(
        success=True,
        message=(
            f"Backend for `{slug}` is running at {url} "
            f"(pid {result.get('pid')}, port {result.get('port')}, "
            f"log {result.get('log_path')})."
        ),
        resource_id=slug,
        external_url=url or None,
        idempotency_key=slug,
        committed_at=now_iso(),
        details={
            "slug": slug,
            "port": result.get("port"),
            "pid": result.get("pid"),
            "log_path": result.get("log_path"),
        },
    ).to_outcome()


async def handle_list_artifacts(
    session: "ChatSession", tc_input: dict
) -> "str | ToolOutcome":
    """List every artifact in the workspace, newest first.

    Output is a JSON array of summaries — slug, name, type,
    description, file count, last-update timestamp. The agent uses
    this to decide whether to create a new artifact or modify an
    existing one.
    """
    import json

    store = _artifact_store(session)
    if store is None:
        # Tier 2 (ENG-2248): the tool cannot operate at all, so a retry
        # cannot help and repetition is thrash. Reuses the existing
        # `store_unavailable` sentinel key (external_wall/service_unavailable).
        return ToolOutcome(
            content="Artifact store unavailable (no workspace bound to this session).",
            ok=False, reason="store_unavailable",
        )

    artifacts = store.list()
    summaries = [
        {
            "slug": a.slug,
            "name": a.name,
            "type": a.type,
            "description": a.description,
            "file_count": len(a.files),
            "updatedAt": a.updatedAt,
        }
        for a in artifacts
    ]
    # Tier 1: a listing was produced. An EMPTY list is still a success —
    # "there are no artifacts" is the correct answer, not a failure.
    return ToolOutcome(content=json.dumps(summaries, indent=2), ok=True)


async def handle_open_artifact(
    session: "ChatSession", tc_input: dict
) -> "str | ToolOutcome":
    """Load an existing artifact's metadata + folder path.

    Returns the same shape as `create_artifact` plus the file list
    so the agent can decide what to edit. 404-shaped error when the
    slug is unknown.
    """
    import json

    store = _artifact_store(session)
    if store is None:
        # Tier 2 (ENG-2248): the tool cannot operate at all, so a retry
        # cannot help and repetition is thrash. Reuses the existing
        # `store_unavailable` sentinel key (external_wall/service_unavailable).
        return ToolOutcome(
            content="Artifact store unavailable (no workspace bound to this session).",
            ok=False, reason="store_unavailable",
        )

    slug = (tc_input.get("slug") or "").strip()
    if not slug:
        # Tier 2: a malformed call; retrying it unchanged cannot work.
        return ToolOutcome(
            content="Error: `slug` is required.",
            ok=False, reason="missing_slug",
        )
    artifact = store.open(slug)
    if artifact is None:
        # Tier 3 (ENG-2248): deliberately left ok=None. Same shape as
        # `recall_skill`'s NO MATCH — the store worked and the artifact simply
        # does not exist. The model can list artifacts and pick a real slug, so
        # this is arguably its own error; but it is also how a model discovers
        # what exists, and ok=False would feed that exploration to the breaker.
        # Needs its own decision, not a side effect of this pass.
        return f"Error: no artifact found for slug `{slug}`."
    folder = store.folder_for(artifact.slug)
    # Opening is how the agent gets an artifact's path in order to write to
    # it, so this is the turn's declaration of intent to modify. Tracked here
    # rather than at write time because the writes themselves happen in
    # scratchpad cells the tool layer never sees.
    _track_artifact(session, store, artifact.slug, summary=f"Opened artifact: {artifact.name}")
    # Tier 1: the artifact was opened and its descriptor returned.
    return ToolOutcome(content=json.dumps({
        "id": artifact.id,
        "slug": artifact.slug,
        "name": artifact.name,
        "type": artifact.type,
        "description": artifact.description,
        "path": str(folder),
        "files": [{"path": f.path, "bytes": f.bytes} for f in artifact.files],
    }, indent=2), ok=True)


async def handle_recall(session: ChatSession, tc_input: dict) -> str:
    """Process a recall tool call — search episodic memory."""
    if session._episodic is None or not session._episodic.enabled:
        return "Episodic memory is not available."

    query = tc_input.get("query", "")
    if not query:
        return "No query provided."

    kwargs: dict = {}
    if "max_results" in tc_input:
        kwargs["max_results"] = int(tc_input["max_results"])
    if "days_back" in tc_input:
        kwargs["days_back"] = int(tc_input["days_back"])

    return session._episodic.recall_formatted(query, **kwargs)


async def handle_memorize(
    session: ChatSession, tc_input: dict
) -> "str | ToolOutcome":
    """Process a memorize tool call and return a result string.

    Encoding is fire-and-forget so it never blocks scratchpad execution.
    """
    import asyncio

    if session._cortex is None:
        # Tier 2 (ENG-2248): no memory system wired, so no entry can ever be
        # stored. Retrying cannot help.
        return ToolOutcome(
            content="Memory system not available.",
            ok=False, reason="store_unavailable",
        )

    if session._cortex.mode == "off":
        # Tier 3 (ENG-2248): deliberately ok=None. This is a CONFIGURED state,
        # not a failure — the user turned memory off, and the tool reported that
        # correctly. Marking it ok=False would nudge and then break the tool for
        # every user who has memory disabled on purpose.
        return "Memory encoding is disabled. Change memory mode via /setup to enable."

    from anton.core.memory.base import Engram

    raw_entries = tc_input.get("entries", [])
    if not raw_entries:
        # Tier 2: a malformed call.
        return ToolOutcome(
            content="No entries provided.",
            ok=False, reason="missing_name",
        )

    engrams: list[Engram] = []
    for entry in raw_entries:
        if not isinstance(entry, dict) or "text" not in entry:
            continue

        kind = entry.get("kind", "lesson")
        if kind not in ("always", "never", "when", "lesson", "profile"):
            kind = "lesson"

        scope = entry.get("scope", "project")
        if scope not in ("global", "project"):
            scope = "project"

        # User-sourced memories (via explicit tool call) get high confidence
        engrams.append(
            Engram(
                text=entry["text"],
                kind=kind,
                scope=scope,
                confidence="high",
                topic=entry.get("topic", ""),
                source="user",
            )
        )

    if not engrams:
        # Tier 2: every entry was rejected by the shape checks above, so the
        # call carried nothing usable.
        return ToolOutcome(
            content="No valid entries provided.",
            ok=False, reason="invalid_type",
        )

    # Always encode immediately via fire-and-forget — the LLM explicitly
    # chose to memorize these, so we never interrupt the user mid-turn
    # with confirmation prompts.  Confirmations are reserved for the
    # post-turn consolidator (lessons extracted from scratchpad sessions).
    async def _encode_bg(cortex, entries):
        try:
            await cortex.encode(entries)
        except Exception:
            pass  # Best-effort; don't disrupt the conversation

    # Tracked so a host that tears down at end of turn can await it.
    session._track_memory_write(asyncio.create_task(_encode_bg(session._cortex, engrams)))

    descriptions = [f"Encoded {e.kind}: {e.text}" for e in engrams]
    # Tier 1: at least one entry was stored.
    return ToolOutcome(
        content="Memory updated: " + "; ".join(descriptions), ok=True
    )


async def handle_scratchpad(
    session: ChatSession, tc_input: dict
) -> str | ToolOutcome:
    """Dispatch a scratchpad tool call by action.

    The exec path returns a ``ToolOutcome`` carrying the runtime's own
    failure verdict, so the error streak doesn't re-classify the result by
    reading it (ENG-1276). The other actions still return plain strings
    (legacy substring classification).
    """
    action = tc_input.get("action", "")
    name = tc_input.get("name", "")

    if not name:
        # Explicit failure: this text has no legacy marker phrase, so the
        # substring fallback used to RESET the streak on it (ENG-1276).
        return ToolOutcome(
            content="Scratchpad name is required.",
            ok=False,
            reason="scratchpad_missing_name",
        )

    # ACC emit helper: use the session's safe wrapper if it exists,
    # otherwise no-op. Defined as a local closure so each emit site
    # stays a single line.
    def _acc_observe(kind: str, detail: dict, *, severity: int = 1) -> None:
        fn = getattr(session, "_acc_observe", None)
        if fn is not None:
            fn(kind, detail, severity=severity)

    if action == "exec":
        # The single-scratchpad guard and the pre-execute ACC events
        # (scratchpad_empty_code / scratchpad_call) live in
        # prepare_scratchpad_exec — the SHARED entry point that the streaming
        # path (ChatSession.turn_stream) also calls — so they fire on both
        # paths. A str return is a message the call should not run past
        # (empty code, single-scratchpad challenge, or install failure).
        result = await prepare_scratchpad_exec(session, tc_input)
        if isinstance(result, ToolOutcome):
            return result
        pad, code, description, estimated_time, estimated_seconds = result

        # Notify pre-execute observers (e.g. cerebellum). The runtime
        # never sees these — observation is an orchestration concern,
        # so it lives at the dispatcher layer where the data is most
        # natural and where local/remote runtimes stay interchangeable.
        prelim_cell = Cell(
            code=code,
            stdout="",
            stderr="",
            error=None,
            description=description,
            estimated_time=estimated_time or str(estimated_seconds),
        )
        await _fire_pre_execute(session, prelim_cell)

        # Snapshot existing artifacts' content mtimes before the cell runs, so
        # an edit the cell makes without a prior `open_artifact` call this
        # turn still gets attributed below (see `_track_edits_since`).
        artifact_store = _artifact_store(session)
        before_artifact_mtimes = (
            _snapshot_existing_artifact_mtimes(artifact_store)
            if artifact_store is not None
            else {}
        )

        cell = await pad.execute(
            code,
            description=description,
            estimated_time=estimated_time,
            estimated_seconds=estimated_seconds,
        )
        if cell is not None:
            session._record_cell_explainability(
                pad_name=name, description=description, cell=cell,
            )
            await _fire_post_execute(session, cell)
            # Post-execute ACC event (killed vs result) via the shared helper —
            # the streaming path emits the same.
            observe_scratchpad_cell(session, name, cell)
            if artifact_store is not None:
                _track_edits_since(session, artifact_store, before_artifact_mtimes)
        # The runtime's verdict: a raised error/timeout/kill is a failure;
        # stderr-only output (warnings) is not, and stdout containing words
        # like "failed" is not either — the streak reads this flag, never the
        # text (ENG-1276). The reason is the traceback's LAST line (the cause),
        # the machine-comparable key ENG-1286's thrash breaker will consume.
        error = (cell.error or "").strip() if cell is not None else ""
        return ToolOutcome(
            content=format_cell_result(cell),
            ok=not error,
            reason=cell_failure_reason(error),
        )

    elif action == "view":
        # get_or_create: new ChatSession has empty _pads but replayed cells on the
        # manager — same hydration path as exec so view works on the first tool call.
        pad = await session._scratchpads.get_or_create(name)
        # ok=True: viewing succeeded even when the notebook being viewed
        # contains old "[error]" cells — the substring fallback used to count
        # a successful view of a failed cell as a fresh tool failure
        # (ENG-1276 false positive).
        return ToolOutcome(content=pad.view(), ok=True)

    elif action == "reset":
        pad = session._scratchpads.pads.get(name)
        if pad is None:
            return f"No scratchpad named '{name}'."
        await pad.reset()
        _acc_observe(
            "scratchpad_reset",
            {"name": name, "reason": "manual"},
            severity=5,
        )
        return f"Scratchpad '{name}' reset. All state cleared."

    elif action == "remove":
        return await session._scratchpads.remove(name)

    elif action == "dump":
        # get_or_create: dump must materialize the runtime from replayed cells when this
        # is the first scratchpad call in a new session (pads.get would miss every time).
        pad = await session._scratchpads.get_or_create(name)
        # ok=True for the same reason as view: rendering a notebook whose
        # cells include past "[error]" output is a success, not a failure.
        return ToolOutcome(content=pad.render_notebook(), ok=True)

    elif action == "install":
        packages = tc_input.get("packages", [])
        if not packages:
            return "No packages specified."
        refused = reject_invalid_packages(packages)
        if refused:
            return ToolOutcome(
                content=refused, ok=False, reason="package_install_rejected"
            )
        pad = await session._scratchpads.get_or_create(name)
        result = await pad.install_packages(packages)
        if install_call_installed_something(result):
            send_package_install_event(session, packages)
        return result

    else:
        return f"Unknown scratchpad action: {action}"


async def handle_read_image(
    session: "ChatSession", tc_input: dict
) -> "ToolOutcome":
    """Read an image file from disk and return it as an image content block.

    Returns a list of content blocks (image + text) on success so the model
    sees the picture on its next turn. Returns a plain error string on
    failure (missing file, non-image extension, oversized image, etc.).
    """
    import base64
    from pathlib import Path

    from anton.clipboard import is_image_path
    from anton.utils.clipboard import (
        MAX_IMAGE_BYTES,
        _media_type_for,
        human_size,
    )

    file_path = (tc_input.get("file_path") or "").strip()
    if not file_path:
        # Tier 2 (ENG-2248): a malformed call; a retry cannot fix it.
        return ToolOutcome(
            content="Error: file_path is required.",
            ok=False, reason="missing_name",
        )

    try:
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            # Resolve relative paths against the project workspace (where
            # artifacts/decks live), not the process cwd. Under the desktop
            # app the agent's cwd is NOT the project, so a project-relative
            # image path would otherwise land in the wrong directory and
            # come back "file not found". Mirrors publish_or_preview. Falls
            # back to cwd for the CLI, where cwd already is the project.
            base = getattr(getattr(session, "_workspace", None), "base", None)
            root = Path(base) if base else Path.cwd()
            path = (root / path).resolve()
    except OSError as exc:
        # Tier 2: the model supplied a path that will not parse.
        return ToolOutcome(
            content=f"Error: invalid path '{file_path}': {exc}",
            ok=False, reason="invalid_type",
        )

    if not path.is_file():
        # Tier 2: the file is genuinely absent. Unlike `recall_skill`'s NO
        # MATCH, there is no listing the model can consult to self-correct and
        # nothing here tells it to proceed regardless, so repeating the same
        # path IS thrash and belongs in the streak.
        return ToolOutcome(
            content=f"Error: file not found: {path}",
            ok=False, reason="path_not_found",
        )

    if not is_image_path(path.name):
        # Tier 2: the model pointed the image tool at a non-image. Its own
        # argument, and the message names the accepted extensions.
        return ToolOutcome(
            content=(
                f"Error: '{path.name}' is not a supported image format "
                "(expected .png/.jpg/.jpeg/.gif/.webp/.bmp)."
            ),
            ok=False, reason="not_an_image",
        )

    try:
        raw = path.read_bytes()
    except OSError as exc:
        # Tier 2: the read itself failed. `ok=False` is certain — the model got
        # no image. The CAUSE is not: this wraps a bare `except Exception`, so
        # `read_failed` is mapped TIER_UNCLASSIFIED rather than guessing at a
        # permissions wall.
        return ToolOutcome(
            content=f"Error: cannot read '{path}': {exc}",
            ok=False, reason="read_failed",
        )

    suffix = path.suffix.lstrip(".").lower()
    if suffix == "bmp":
        try:
            import io
            from PIL import Image as _PILImage

            buf = io.BytesIO()
            _PILImage.open(io.BytesIO(raw)).save(buf, format="PNG")
            raw = buf.getvalue()
            suffix = "png"
        except Exception as exc:
            # Tier 2 for the verdict, unclassified for the cause: a bare
            # `except Exception` around a PIL call covers a missing Pillow (a
            # wall) and a corrupt BMP (self-inflicted) with one sentinel.
            return ToolOutcome(
                content=f"Error: failed to convert BMP to PNG: {exc}",
                ok=False, reason="bmp_convert_failed",
            )

    if len(raw) * 4 // 3 > MAX_IMAGE_BYTES:
        # Tier 2: over the API's hard limit. The model chose the file and the
        # message tells it what to do instead.
        return ToolOutcome(
            content=(
                f"Error: image is too large ({human_size(len(raw))}); "
                "the API limit is ~3.7 MB raw / 5 MB base64. "
                "Resize the image and try again."
            ),
            ok=False, reason="image_too_large",
        )

    b64 = base64.standard_b64encode(raw).decode("ascii")
    media_type = _media_type_for(suffix)

    summary = f"Loaded {path.name} ({human_size(len(raw))})."
    try:
        from PIL import Image as _PILImage

        with _PILImage.open(path) as im:
            summary = f"Loaded {path.name} ({im.width}x{im.height}, {human_size(len(raw))})."
    except Exception:
        pass

    # Tier 1, and the ONLY multimodal verdict in the tree. `ok=True` is free
    # here: the list arms of both tool loops already treat a list result as a
    # success unless a handler says otherwise, so this changes the emitted
    # `ok` from "unknown" to `true` and nothing else.
    #
    # It must stay `ok=True`. A list carrying `ok=False` would reach two
    # documented gaps in the list arms — the nudge/breaker text is never
    # appended there, and `_record_root_cause` is never called — so the model
    # would be silently retried past the breaker. `_tool_failure_cause`'s
    # docstring in session.py describes that shape as unreachable; it stays
    # unreachable because every failure above returns a plain string.
    # Pinned by `test_read_image_never_pairs_a_list_with_a_failure_verdict`.
    return ToolOutcome(
        content=[
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            },
            {"type": "text", "text": summary},
        ],
        ok=True,
    )


# ---------------------------------------------------------------------------
# select_path — interactive file/folder disambiguation
# ---------------------------------------------------------------------------

# Hard caps keep the picker fast and the prompt readable: never offer more
# than _SELECTION_MAX_CANDIDATES options, and stop walking a glob after
# _SELECTION_SCAN_LIMIT entries so a `**` pattern on a huge tree can't stall.
_SELECTION_MAX_CANDIDATES = 50
_SELECTION_SCAN_LIMIT = 5000


def _selection_root(session: "ChatSession") -> "Path":
    """The directory candidates are confined to: the project root (or cwd)."""
    workspace = getattr(session, "_workspace", None)
    return (workspace.base if workspace is not None else Path.cwd()).resolve()


def _is_within(root: "Path", candidate: "Path") -> bool:
    """True when *candidate* is inside *root* — the path-traversal guard."""
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _collect_selection_candidates(tc_input: dict, root: "Path", kind: str) -> "list[Path]":
    """Resolve candidate paths: confined to *root*, deduped, kind-filtered, capped.

    Prefers the model's explicit ``candidates`` list; otherwise globs
    ``pattern`` (under an optional ``base_dir``). The private ``.anton``
    workspace is never exposed.
    """
    seen: set[Path] = set()
    found: list[Path] = []

    def consider(path: Path) -> bool:
        """Add *path* if it qualifies. Returns False once the cap is hit."""
        if len(found) >= _SELECTION_MAX_CANDIDATES:
            return False
        resolved = path.resolve()
        if resolved in seen or not _is_within(root, resolved) or not resolved.exists():
            return True
        if ".anton" in resolved.relative_to(root).parts:
            return True
        if (kind == "file" and not resolved.is_file()) or (kind == "folder" and not resolved.is_dir()):
            return True
        seen.add(resolved)
        found.append(resolved)
        return True

    explicit = tc_input.get("candidates")
    if isinstance(explicit, list) and explicit:
        for raw in explicit:
            if not isinstance(raw, str) or not raw.strip():
                continue
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = root / candidate
            if not consider(candidate):
                break
    else:
        pattern = (tc_input.get("pattern") or "").strip()
        if pattern:
            base_dir = (tc_input.get("base_dir") or "").strip()
            search_root = root
            if base_dir:
                rel = Path(base_dir).expanduser()
                search_root = (rel if rel.is_absolute() else root / rel).resolve()
            if _is_within(root, search_root):
                for scanned, match in enumerate(search_root.glob(pattern)):
                    if scanned >= _SELECTION_SCAN_LIMIT or not consider(match):
                        break

    found.sort(key=lambda p: str(p).lower())
    return found


def _selection_option(path: "Path", root: "Path"):
    """One picker entry for *path*, labelled relative to the project root."""
    from anton.core.interaction.elicit import AskOption

    try:
        label = str(path.relative_to(root))
    except ValueError:
        label = str(path)
    return AskOption(
        value=str(path),
        label=label,
        kind="folder" if path.is_dir() else "file",
    )


def _chosen_path(answer) -> "str | None":
    """The single path an ``AskAnswer`` carries, or None if there is none."""
    if answer.status != "answered":
        return None
    if answer.values:
        return answer.values[0]
    return (answer.text or "").strip() or None


def _path_answer_failure(answer) -> "str | None":
    """Map a non-``answered`` outcome onto ``select_path``'s own status, or None.

    Collapsing every failure into ``cancelled`` would tell the model "the user
    dismissed the picker" when the user was never asked — the per-turn question
    budget was spent, the question timed out, or no host could render it. The
    model's documented reaction to ``cancelled`` is to ask how to proceed, so a
    false ``cancelled`` costs a turn on a question nobody ever saw.
    """
    if answer.status == "answered":
        return None
    if answer.status == "cancelled":
        return _status(
            "cancelled",
            "The user dismissed the picker without choosing. Ask how they would like to proceed.",
        )
    if answer.status == "unavailable":
        return _status(
            "picker_unavailable",
            "An interactive picker is unavailable here; ask the user for the path in plain text.",
        )
    if answer.status == "limit":
        return _status(
            "error",
            "Too many questions this turn; choose a path yourself and state which you picked, "
            "or ask in plain text.",
        )
    return _status("error", f"The picker did not return a selection ({answer.status}).")


def _browse_start_dir(tc_input: dict, root: "Path") -> "Path":
    """Resolve the browse-mode starting directory (defaults to the project root)."""
    raw = (tc_input.get("start_dir") or "").strip()
    if not raw:
        return root
    start = Path(raw).expanduser()
    if not start.is_absolute():
        start = root / start
    start = start.resolve()
    return start if start.is_dir() else root


def _status(status: str, message: str = "", **extra) -> str:
    """Serialize a select_path tool result, omitting an empty message."""
    return json.dumps({"status": status, **({"message": message} if message else {}), **extra})


def _needs_confirmation(candidate: "Path", label: str) -> str:
    """The model-supplied candidate cannot be silently accepted and no
    confirmation card can render on this host: hand the decision back."""
    return _status(
        "needs_confirmation",
        f"You supplied '{label}' as the only candidate — that is your guess, not "
        "the user's choice, so it was not accepted. Ask the user to confirm it "
        "before using it, and do not present it as chosen or connected until "
        "they do. If they asked for a file or folder outside the project, no "
        "path inside the project is an answer: tell them plainly that this host "
        "cannot reach paths outside the project, and ask them to attach the "
        "relevant files to the conversation.",
        candidates=[str(candidate)],
    )


async def _confirm_single_candidate(
    session: "ChatSession", candidate: "Path", root: "Path", timeout_s: "int | None"
) -> str:
    """Confirm a model-supplied single candidate with the user (ENG-1852).

    A lone entry in ``candidates`` is the model's own guess echoed back, not a
    discovery: in a fresh project exactly one directory exists (``skills``), so
    silently resolving it reported folders nobody chose as connected — to the
    first-run prompt "give you access to a folder on my computer", no less.
    Confirmation is enforced here rather than in the prompt, because the model
    already narrates such guesses as decisions ("I'll use it as the folder").

    Renders a yes/no choice card (supported on every cowork host, which has no
    path picker). Where no card can render either, returns
    ``needs_confirmation`` so the model must ask in plain text.
    """
    from anton.core.interaction.elicit import AskOption, AskRequest, elicit

    try:
        label = str(candidate.relative_to(root))
    except ValueError:
        label = str(candidate)
    noun = "folder" if candidate.is_dir() else "file"
    request = AskRequest(
        prompt=f"Only one candidate was found — use this {noun}?",
        kind="choice",
        timeout_s=timeout_s,
        options=(
            AskOption(value="yes", label=f"Yes — use {label}", kind=noun, style="primary"),
            AskOption(value="no", label="No — that's not it"),
        ),
    )
    try:
        answer = await elicit(session, f"path:{uuid.uuid4().hex}", request)
    except Exception as exc:  # noqa: BLE001 — the card is host code
        _log.warning("select_path confirmation elicitor failed: %s", exc, exc_info=True)
        return _status("error", f"Selection failed: {exc}")
    if answer.status == "unavailable":
        return _needs_confirmation(candidate, label)
    if answer.status == "cancelled":
        return _status(
            "cancelled",
            "The user dismissed the confirmation without choosing. Ask how they "
            "would like to proceed.",
        )
    if answer.status == "limit":
        return _status(
            "error",
            "Too many questions this turn; do not use the candidate without the "
            "user's confirmation — ask in plain text.",
        )
    if answer.status != "answered":
        return _status("error", f"The confirmation did not return an answer ({answer.status}).")
    typed = (answer.text or "").strip()
    if "yes" in answer.values:
        # Deliberately NOT auto_resolved: the user confirmed this path.
        return _status(
            "resolved",
            f'The user added: "{typed}"' if typed else "",
            path=str(candidate),
        )
    if typed and "no" not in answer.values:
        # A free-typed reply is not a decline the user made — it may even be
        # the word "yes" — so assert nothing; hand it over verbatim. Parsing
        # typed affirmations here would be a worse failure than one re-ask.
        return _status(
            "cancelled",
            f'The user typed a reply instead of choosing an option: "{typed}". '
            f"'{label}' is not confirmed — do not use it or present it as "
            "connected; act on their reply. If it reads as agreement, call "
            "select_path again for an explicit confirmation. If they want a "
            "file or folder outside the project, tell them plainly that this "
            "host cannot reach it and ask them to attach the relevant files "
            "to the conversation.",
        )
    return _status(
        "cancelled",
        f"The user declined '{label}'. Do not use this path and do not present "
        "it as connected. Ask what they actually meant — and if it is a file or "
        "folder outside the project, tell them plainly that this host cannot "
        "reach paths outside the project, and ask them to attach the relevant "
        "files to the conversation."
        + (f' The user said: "{typed}"' if typed else ""),
    )


def _finalize_browse_choice(chosen: "str | None", kind: str, root: "Path") -> str:
    """Validate a browse-mode pick: any existing path of the requested kind."""
    if chosen is None:
        return _status("cancelled", "The user dismissed the picker without choosing. Ask how they would like to proceed.")
    path = Path(chosen).expanduser()
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        return _status("invalid", "The selected path no longer exists.")
    if kind == "file" and not path.is_file():
        return _status("invalid", "A folder was selected but a file was expected.")
    if kind == "folder" and not path.is_dir():
        return _status("invalid", "A file was selected but a folder was expected.")
    return _status("resolved", path=str(path.resolve()))


async def handle_select_path(session: "ChatSession", tc_input: dict) -> str:
    """Have the user choose a file/folder; return the chosen path as JSON.

    Two modes, chosen automatically from the inputs:

    * **browse** — no ``candidates``/``pattern`` given: the location is unknown,
      so the user navigates a picker to locate it. Use this instead of asking
      the user to type or paste a path. **Host-gated:** only reachable where an
      elicitor supports ``kind="path"``. Elsewhere — every cowork session today
      — it returns ``picker_unavailable`` pointing at attachment, and the model
      is not offered browse at all, because ``session.py`` registers
      ``SELECT_PATH_TOOL_PICK_ONLY`` there (ENG-1357).
    * **pick** — ``candidates`` or ``pattern`` given: disambiguate concrete
      matches within the project. A single *pattern* match auto-resolves and
      zero matches reports "no matches"; a single model-supplied candidate is
      confirmed with the user first (ENG-1852) — via the path picker where one
      renders, a yes/no choice card elsewhere, or ``needs_confirmation`` when
      neither can.

    The result is fed back as the tool result, so the agent continues without a
    separate user message.
    """
    from anton.core.interaction.elicit import AskRequest, elicit

    prompt = (tc_input.get("prompt") or "Select a file or folder.").strip()
    kind = (tc_input.get("kind") or "any").strip().lower()
    if kind not in ("file", "folder", "any"):
        kind = "any"

    root = _selection_root(session)
    elicitor = getattr(session, "elicitor", None)
    # Early-out only: elicit() re-checks kind support itself and answers
    # "unavailable" (mapped to picker_unavailable below) if this check were
    # ever removed or bypassed, so this is not the sole guard.
    can_pick = elicitor is not None and "path" in getattr(elicitor, "supported_kinds", ())
    timeout_s = getattr(elicitor, "timeout_s", None)
    has_candidates = isinstance(tc_input.get("candidates"), list) and bool(tc_input.get("candidates"))
    has_pattern = bool((tc_input.get("pattern") or "").strip())

    # ── browse — locate an unspecified path ──────────────────────────────
    if not has_candidates and not has_pattern:
        if not can_pick:
            # NOT "ask for the path in plain text" (ENG-1357). Browse means the
            # file's location is unknown, so it is very likely outside the
            # project — and every host that lands here forbids reading files
            # that are neither in the project nor attached, so a typed path is
            # unusable even when the user supplies it. Naming the one route
            # that works matters: with no legitimate exit offered, the model
            # fabricated the user's data instead of asking.
            return _status(
                "picker_unavailable",
                "This host cannot render a file browser. Ask the user to attach the "
                "file to the conversation — that is how they grant you access to a "
                "file outside the project. Do not ask them to type or paste a path, "
                "and do not proceed with invented or example data in its place.",
            )
        request = AskRequest(
            prompt=prompt,
            kind="path",
            timeout_s=timeout_s,
            path_kind=kind,
            path_mode="browse",
            root=str(_browse_start_dir(tc_input, root)),
        )
        try:
            answer = await elicit(session, f"path:{uuid.uuid4().hex}", request)
        except Exception as exc:  # noqa: BLE001 — the picker is host code
            _log.warning("select_path elicitor failed: %s", exc, exc_info=True)
            return _status("error", f"Selection failed: {exc}")
        failure = _path_answer_failure(answer)
        if failure is not None:
            return failure
        browse_root = Path(request.root) if request.root else root
        return _finalize_browse_choice(_chosen_path(answer), kind, browse_root)

    # ── pick — disambiguate concrete candidates within the project ───────
    candidates = _collect_selection_candidates(tc_input, root, kind)
    if not candidates:
        # The browse suggestion is only honest where browse can actually run —
        # on a host without a path elicitor it leads straight to
        # picker_unavailable, and "ask in plain text" is unusable for a file
        # outside the project (ENG-1357).
        return _status(
            "no_matches",
            (
                "No match found. Refine the pattern, omit candidates/pattern to let "
                "the user browse, or ask in plain text."
            )
            if can_pick
            else (
                "No match found in the project. Refine the pattern, or — if the file "
                "is not in the project at all — ask the user to attach it to the "
                "conversation. This host has no file browser, and you cannot read a "
                "path the user types."
            ),
        )
    if len(candidates) == 1:
        if not has_candidates:
            # A lone *pattern* match is a genuine discovery — the glob searched
            # the project and found exactly one thing — so resolving it without
            # a prompt is safe. A lone *explicit* candidate is not (ENG-1852):
            # it is the model's own guess echoed back, so it must be confirmed
            # by the user — via the confirmation card below, or by picking it
            # in the path picker where one renders.
            return _status("resolved", auto_resolved=True, path=str(candidates[0]))
        if not can_pick:
            return await _confirm_single_candidate(session, candidates[0], root, timeout_s)
    if not can_pick:
        return _status(
            "picker_unavailable",
            "An interactive picker is unavailable here; ask the user which of these paths they meant.",
            candidates=[str(p) for p in candidates],
        )

    options = tuple(_selection_option(p, root) for p in candidates)
    request = AskRequest(
        prompt=prompt,
        kind="path",
        timeout_s=timeout_s,
        options=options,
        path_kind=kind,
    )
    try:
        answer = await elicit(session, f"path:{uuid.uuid4().hex}", request)
    except Exception as exc:  # noqa: BLE001
        _log.warning("select_path elicitor failed: %s", exc, exc_info=True)
        return _status("error", f"Selection failed: {exc}")
    failure = _path_answer_failure(answer)
    if failure is not None:
        return failure
    chosen = _chosen_path(answer)
    if chosen not in {option.value for option in options}:
        return _status("invalid", "The returned selection was not one of the offered options.")
    return _status("resolved", path=chosen)


# ---------------------------------------------------------------------------
# ask_user — a multiple-choice question answered inside the current turn
# ---------------------------------------------------------------------------


def build_ask_request(tc_input: dict, timeout_s: "int | None"):
    """Parse the tool input into an ``AskRequest``, or None if it is junk.

    Parsing only — the well-formedness rules (option count, uniqueness) live
    in ``validate_request`` so the orchestrator, which builds requests in
    Python and never comes through here, gets them too.
    """
    from anton.core.interaction.elicit import AskOption, AskRequest

    question = (tc_input.get("question") or "").strip()
    if not question:
        return None

    raw_options = tc_input.get("options")
    if not isinstance(raw_options, list) or not raw_options:
        return None
    options = []
    for raw in raw_options:
        if not isinstance(raw, dict):
            return None
        value = (raw.get("value") or "").strip()
        if not value:
            return None
        options.append(
            AskOption(
                value=value,
                label=(raw.get("label") or value).strip(),
                detail=(raw.get("detail") or "").strip(),
            )
        )

    select = (tc_input.get("select") or "one").strip().lower()
    if select not in ("one", "many"):
        return None

    allow_custom = tc_input.get("allow_custom")
    return AskRequest(
        prompt=question,
        kind="choice",
        timeout_s=timeout_s,
        options=tuple(options),
        select=select,
        allow_custom=True if allow_custom is None else bool(allow_custom),
    )


_ASK_USER_UNAVAILABLE = (
    "Interactive questions are unavailable here, or the question was malformed "
    "(it needs 2-10 options with unique values). Ask in plain text instead and "
    "end your turn."
)
_ASK_USER_LIMIT = (
    "Question limit reached for this turn; proceed on a stated assumption "
    "instead of asking again."
)


def _ask_user_telemetry_settings(session: "ChatSession"):
    """Best-effort settings object for `send_event`, mirroring the pattern
    already used at every other call site (see `anton/tools.py`)."""
    settings = getattr(session, "_settings", None)
    if settings is not None:
        return settings
    try:
        from anton.config.settings import AntonSettings

        return AntonSettings()
    except Exception:
        return None


def _send_ask_user_event(session: "ChatSession", action: str, props: dict) -> None:
    """Fire one telemetry event. Never raises — analytics must not break a turn."""
    try:
        settings = _ask_user_telemetry_settings(session)
        if not settings:
            return
        from anton.analytics import send_event

        send_event(settings, action, **props)
    except Exception:
        pass


async def handle_ask_user(session: "ChatSession", tc_input: dict) -> str:
    """Ask the user to choose, and return the answer as the tool result."""
    from anton.core.interaction.elicit import elicit

    elicitor = getattr(session, "elicitor", None)
    request = build_ask_request(
        tc_input, timeout_s=getattr(elicitor, "timeout_s", None)
    )
    if request is None:
        return _status("error", _ASK_USER_UNAVAILABLE)

    # send_event only accepts string extras (see every existing call site).
    props = {"select": request.select, "options": str(len(request.options))}
    _send_ask_user_event(session, "ask_user_asked", props)

    # The question id doubles as the correlation key the host echoes back
    # with the answer. The originating tool_use.id is not visible here
    # (dispatch_tool passes only name + input), so mint one.
    question_id = f"ask:{uuid.uuid4().hex}"
    answer = await elicit(session, question_id, request)
    _send_ask_user_event(session, f"ask_user_{answer.status}", props)

    if answer.status == "limit":
        return _status("error", _ASK_USER_LIMIT)
    if answer.status == "unavailable":
        return _status("error", _ASK_USER_UNAVAILABLE)
    if answer.status in ("cancelled", "timeout"):
        return _status(answer.status)

    if answer.status == "answered":
        extra: dict = {}
        if answer.values:
            extra["values"] = list(answer.values)
        if answer.text:
            extra["text"] = answer.text
        return _status("answered", **extra)

    # Explicit rather than a fall-through to "answered": `Elicitor` is a
    # structural Protocol implemented out of tree, so an unlisted status (a
    # host-side typo, a future status) is reachable without touching this
    # repo — and telling the LLM the user answered and chose nothing is the
    # worst failure shape a decision tool has. Same shape as
    # `_path_answer_failure`.
    return _status("error", f"The question did not return an answer ({answer.status}).")
