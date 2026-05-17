from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from anton.core.backends.base import Cell
from anton.core.utils.scratchpad import prepare_scratchpad_exec, format_cell_result

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


async def handle_create_artifact(session: "ChatSession", tc_input: dict) -> str:
    """Create a fresh artifact folder + metadata.json + README.md.

    Returns a JSON-shaped string the LLM can parse into the artifact
    path. The agent is expected to write its output files under
    `<path>/...` after this call returns.
    """
    import json

    store = _artifact_store(session)
    if store is None:
        return "Artifact store unavailable (no workspace bound to this session)."

    name = (tc_input.get("name") or "").strip()
    description = (tc_input.get("description") or "").strip()
    artifact_type = (tc_input.get("type") or "").strip()
    primary = tc_input.get("primary")
    if not name:
        return "Error: `name` is required."
    if not description:
        return "Error: `description` is required."

    from anton.core.artifacts.models import ARTIFACT_TYPES

    if artifact_type not in ARTIFACT_TYPES:
        return (
            f"Error: `type` must be one of {ARTIFACT_TYPES}. "
            f"Got: {artifact_type!r}."
        )

    artifact = store.create(  # type: ignore[arg-type]
        name=name,
        description=description,
        type=artifact_type,
        primary=primary if isinstance(primary, str) else None,
    )
    folder = store.folder_for(artifact.slug)
    return json.dumps({
        "id": artifact.id,
        "slug": artifact.slug,
        "name": artifact.name,
        "type": artifact.type,
        "primary": artifact.primary,
        "path": str(folder),
    }, indent=2)


async def handle_set_artifact_primary(session: "ChatSession", tc_input: dict) -> str:
    """Update or clear the primary-file pointer on an existing artifact.

    The agent calls this when it created an artifact without a
    primary and now knows what it should be, or when the primary
    file's name changed. Pass `primary: null` to clear and revert
    the renderer to its heuristic.
    """
    import json

    store = _artifact_store(session)
    if store is None:
        return "Artifact store unavailable (no workspace bound to this session)."

    slug = (tc_input.get("slug") or "").strip()
    if not slug:
        return "Error: `slug` is required."
    raw = tc_input.get("primary")
    primary = raw if isinstance(raw, str) else None
    artifact = store.set_primary(slug, primary)
    if artifact is None:
        return f"Error: no artifact found for slug `{slug}`."
    return json.dumps({
        "slug": artifact.slug,
        "primary": artifact.primary,
    }, indent=2)


async def handle_list_artifacts(session: "ChatSession", tc_input: dict) -> str:
    """List every artifact in the workspace, newest first.

    Output is a JSON array of summaries — slug, name, type,
    description, file count, last-update timestamp. The agent uses
    this to decide whether to create a new artifact or modify an
    existing one.
    """
    import json

    store = _artifact_store(session)
    if store is None:
        return "Artifact store unavailable (no workspace bound to this session)."

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
    return json.dumps(summaries, indent=2)


async def handle_open_artifact(session: "ChatSession", tc_input: dict) -> str:
    """Load an existing artifact's metadata + folder path.

    Returns the same shape as `create_artifact` plus the file list
    so the agent can decide what to edit. 404-shaped error when the
    slug is unknown.
    """
    import json

    store = _artifact_store(session)
    if store is None:
        return "Artifact store unavailable (no workspace bound to this session)."

    slug = (tc_input.get("slug") or "").strip()
    if not slug:
        return "Error: `slug` is required."
    artifact = store.open(slug)
    if artifact is None:
        return f"Error: no artifact found for slug `{slug}`."
    folder = store.folder_for(artifact.slug)
    return json.dumps({
        "id": artifact.id,
        "slug": artifact.slug,
        "name": artifact.name,
        "type": artifact.type,
        "description": artifact.description,
        "path": str(folder),
        "files": [{"path": f.path, "bytes": f.bytes} for f in artifact.files],
    }, indent=2)


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


async def handle_memorize(session: ChatSession, tc_input: dict) -> str:
    """Process a memorize tool call and return a result string.

    Encoding is fire-and-forget so it never blocks scratchpad execution.
    """
    import asyncio

    if session._cortex is None:
        return "Memory system not available."

    if session._cortex.mode == "off":
        return "Memory encoding is disabled. Change memory mode via /setup to enable."

    from anton.core.memory.base import Engram

    raw_entries = tc_input.get("entries", [])
    if not raw_entries:
        return "No entries provided."

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
        return "No valid entries provided."

    # Always encode immediately via fire-and-forget — the LLM explicitly
    # chose to memorize these, so we never interrupt the user mid-turn
    # with confirmation prompts.  Confirmations are reserved for the
    # post-turn consolidator (lessons extracted from scratchpad sessions).
    async def _encode_bg(cortex, entries):
        try:
            await cortex.encode(entries)
        except Exception:
            pass  # Best-effort; don't disrupt the conversation

    asyncio.create_task(_encode_bg(session._cortex, engrams))

    descriptions = [f"Encoded {e.kind}: {e.text}" for e in engrams]
    return "Memory updated: " + "; ".join(descriptions)


async def handle_scratchpad(session: ChatSession, tc_input: dict) -> str:
    """Dispatch a scratchpad tool call by action."""
    action = tc_input.get("action", "")
    name = tc_input.get("name", "")

    if not name:
        return "Scratchpad name is required."

    # ACC emit helper: use the session's safe wrapper if it exists,
    # otherwise no-op. Defined as a local closure so each emit site
    # stays a single line.
    def _acc_observe(kind: str, detail: dict, *, severity: int = 1) -> None:
        fn = getattr(session, "_acc_observe", None)
        if fn is not None:
            fn(kind, detail, severity=severity)

    if action == "exec":
        result = await prepare_scratchpad_exec(session, tc_input)
        if isinstance(result, str):
            # Empty / malformed code parameter — the dispatcher rejected
            # it before reaching the runtime. This is exactly the
            # "silent code-clip" failure mode the ACC's
            # detect_oversized_cell watches for.
            _acc_observe("scratchpad_empty_code", {"name": name}, severity=7)
            return result
        pad, code, description, estimated_time, estimated_seconds = result

        _acc_observe(
            "scratchpad_call",
            {
                "name": name,
                "code_len": len(code or ""),
                "one_line_description": description or "",
            },
        )

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
            # ACC: distinguish "killed" (timeout/cancel/OOM) from a
            # plain runtime error. The local backend sets cell.error
            # to a string starting with "Cancelled" or matching the
            # "Cell timed out"/"Cell killed" prefixes from the
            # asyncio.TimeoutError path. Everything else (NameError,
            # ImportError, …) is a regular result with success=False.
            err = (cell.error or "").strip()
            if err.startswith(("Cancelled", "Cell timed out", "Cell killed")):
                _acc_observe(
                    "scratchpad_killed",
                    {"name": name, "reason": err[:120]},
                    severity=6,
                )
            else:
                success = not err and not (cell.stderr or "").strip()
                _acc_observe(
                    "scratchpad_result",
                    {
                        "name": name,
                        "success": success,
                        "stdout_len": len(cell.stdout or ""),
                        "error": err[:300] if err else "",
                    },
                    severity=5 if not success else 1,
                )
        return format_cell_result(cell)

    elif action == "view":
        # get_or_create: new ChatSession has empty _pads but replayed cells on the
        # manager — same hydration path as exec so view works on the first tool call.
        pad = await session._scratchpads.get_or_create(name)
        return pad.view()

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
        return pad.render_notebook()

    elif action == "install":
        packages = tc_input.get("packages", [])
        if not packages:
            return "No packages specified."
        pad = await session._scratchpads.get_or_create(name)
        return await pad.install_packages(packages)

    else:
        return f"Unknown scratchpad action: {action}"
