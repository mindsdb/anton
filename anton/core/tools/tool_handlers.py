from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from anton.core.backends.base import Cell
from anton.core.utils.scratchpad import (
    prepare_scratchpad_exec,
    format_cell_result,
    observe_scratchpad_cell,
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


async def handle_update_artifact_metadata(session: "ChatSession", tc_input: dict) -> str:
    """Update mutable metadata fields on an existing artifact.

    Only fields present in the input are modified. Supports:
    - `primary`: entry-point file path (empty string to clear)
    - `port`: backend port number (fullstack apps only)
    - `datasources`: list of vault-connection slugs the backend reads from.
      `engine`, `name`, and `env_prefix` are derived from the vault.
    """
    import json

    store = _artifact_store(session)
    if store is None:
        return "Artifact store unavailable (no workspace bound to this session)."

    slug = (tc_input.get("slug") or "").strip()
    if not slug:
        return "Error: `slug` is required."

    kwargs: dict = {}
    if "primary" in tc_input:
        kwargs["primary"] = tc_input["primary"]
    if "port" in tc_input:
        try:
            kwargs["port"] = int(tc_input["port"]) if tc_input["port"] is not None else None
        except (TypeError, ValueError):
            return "Error: `port` must be a number."

    if "datasources" in tc_input:
        from anton.core.artifacts.models import DatasourceRef
        from anton.core.datasources.data_vault import LocalDataVault

        raw_list = tc_input.get("datasources") or []
        if not isinstance(raw_list, list):
            return "Error: `datasources` must be a list of slug strings."

        vault = session._data_vault or LocalDataVault()
        known = {f"{c['engine']}-{c['name']}": (c["engine"], c["name"])
                 for c in vault.list_connections()}

        refs: list[DatasourceRef] = []
        unknown: list[str] = []
        for item in raw_list:
            if not isinstance(item, str):
                return "Error: each entry in `datasources` must be a slug string."
            ref_slug = item.strip()
            if not ref_slug:
                continue
            if ref_slug not in known:
                unknown.append(ref_slug)
                continue
            engine, name = known[ref_slug]
            refs.append(DatasourceRef(engine=engine, name=name))
        if unknown:
            return (
                f"Error: unknown datasource slug(s): {', '.join(unknown)}. "
                f"Each slug must match an existing vault connection "
                f"(format: `<engine>-<name>`)."
            )
        kwargs["datasources"] = refs

    artifact = store.update(slug, **kwargs)
    if artifact is None:
        return f"Error: no artifact found for slug `{slug}`."
    return json.dumps({
        "slug": artifact.slug,
        "primary": artifact.primary,
        "port": artifact.port,
        "datasources": [d.slug for d in artifact.datasources],
    }, indent=2)


async def handle_launch_backend(session: "ChatSession", tc_input: dict) -> str:
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
    import json

    from anton.core.artifacts.backend_launcher import launch_artifact_backend

    store = _artifact_store(session)
    if store is None:
        return "Artifact store unavailable (no workspace bound to this session)."

    slug = (tc_input.get("slug") or "").strip()
    if not slug:
        return "Error: `slug` is required."
    artifact = store.open(slug)
    if artifact is None:
        return f"Error: no artifact found for slug `{slug}`."

    rel_path = (tc_input.get("path") or "backend.py").strip()
    extra_args = tc_input.get("extra_args") or []
    health_path = tc_input.get("health_path") or "/"
    try:
        health_timeout = float(tc_input.get("health_timeout", 10))
    except (TypeError, ValueError):
        return "Error: `health_timeout` must be a number."

    tracked = getattr(session, "_tracked_backends", None)
    if tracked is None:
        tracked = {}
        session._tracked_backends = tracked

    result = await launch_artifact_backend(
        slug=slug,
        artifact_folder=store.folder_for(slug),
        scratchpad_pool=session._scratchpads,
        tracked_backends=tracked,
        path=rel_path,
        extra_args=extra_args,
        health_path=health_path,
        health_timeout=health_timeout,
    )
    if isinstance(result, str):
        return result

    store.update(slug, port=result["port"])
    return json.dumps(
        {k: v for k, v in result.items() if k != "proc"},
        indent=2,
    )


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
        # The single-scratchpad guard and the pre-execute ACC events
        # (scratchpad_empty_code / scratchpad_call) live in
        # prepare_scratchpad_exec — the SHARED entry point that the streaming
        # path (ChatSession.turn_stream) also calls — so they fire on both
        # paths. A str return is a message the call should not run past
        # (empty code, single-scratchpad challenge, or install failure).
        result = await prepare_scratchpad_exec(session, tc_input)
        if isinstance(result, str):
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


async def handle_read_image(
    session: "ChatSession", tc_input: dict
) -> str | list[dict]:
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
        return "Error: file_path is required."

    try:
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
    except OSError as exc:
        return f"Error: invalid path '{file_path}': {exc}"

    if not path.is_file():
        return f"Error: file not found: {path}"

    if not is_image_path(path.name):
        return (
            f"Error: '{path.name}' is not a supported image format "
            "(expected .png/.jpg/.jpeg/.gif/.webp/.bmp)."
        )

    try:
        raw = path.read_bytes()
    except OSError as exc:
        return f"Error: cannot read '{path}': {exc}"

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
            return f"Error: failed to convert BMP to PNG: {exc}"

    if len(raw) * 4 // 3 > MAX_IMAGE_BYTES:
        return (
            f"Error: image is too large ({human_size(len(raw))}); "
            "the API limit is ~3.7 MB raw / 5 MB base64. "
            "Resize the image and try again."
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

    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64,
            },
        },
        {"type": "text", "text": summary},
    ]


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
    from pathlib import Path

    workspace = getattr(session, "_workspace", None)
    return (workspace.base if workspace is not None else Path.cwd()).resolve()


def _is_within(root: "Path", candidate: "Path") -> bool:
    """True when *candidate* is inside *root* — the path-traversal guard."""
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _collect_selection_candidates(
    session: "ChatSession", tc_input: dict, root: "Path", kind: str
) -> "list[Path]":
    """Resolve candidate paths: confined to *root*, deduped, kind-filtered, capped.

    Prefers the model's explicit ``candidates`` list; otherwise globs
    ``pattern`` (under an optional ``base_dir``). The private ``.anton``
    workspace is never exposed.
    """
    from pathlib import Path

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
    """Build a display option for *path* (label = path relative to the root)."""
    from anton.core.interaction.selection import SelectionOption

    try:
        label = str(path.relative_to(root))
    except ValueError:
        label = str(path)
    return SelectionOption(
        value=str(path),
        label=label,
        kind="folder" if path.is_dir() else "file",
    )


def _resolve_selection_elicitor(session: "ChatSession"):
    """The host-injected elicitor, falling back to a terminal picker on the CLI."""
    elicitor = getattr(session, "selection_elicitor", None)
    if elicitor is not None:
        return elicitor
    console = getattr(session, "_console", None)
    if console is None:
        return None
    from anton.core.interaction.cli import CLISelectionElicitor

    return CLISelectionElicitor(console)


def _browse_start_dir(tc_input: dict, root: "Path") -> "Path":
    """Resolve the browse-mode starting directory (defaults to the project root)."""
    from pathlib import Path

    raw = (tc_input.get("start_dir") or "").strip()
    if not raw:
        return root
    start = Path(raw).expanduser()
    if not start.is_absolute():
        start = root / start
    start = start.resolve()
    return start if start.is_dir() else root


async def _run_elicitor(elicitor, request):
    """Run the elicitor, returning (chosen, error_json). error_json is None on success."""
    import json

    try:
        return await elicitor.elicit(request), None
    except Exception as exc:
        _log.warning("select_path elicitor failed: %s", exc, exc_info=True)
        return None, json.dumps({"status": "error", "message": f"Selection failed: {exc}"})


def _finalize_browse_choice(chosen: "str | None", kind: str) -> str:
    """Validate a browse-mode pick: any existing path of the requested kind."""
    import json
    from pathlib import Path

    if chosen is None:
        return json.dumps({
            "status": "cancelled",
            "message": "The user dismissed the picker without choosing. Ask how they would like to proceed.",
        })
    path = Path(chosen).expanduser()
    if not path.exists():
        return json.dumps({"status": "invalid", "message": "The selected path no longer exists."})
    if kind == "file" and not path.is_file():
        return json.dumps({"status": "invalid", "message": "A folder was selected but a file was expected."})
    if kind == "folder" and not path.is_dir():
        return json.dumps({"status": "invalid", "message": "A file was selected but a folder was expected."})
    return json.dumps({"status": "resolved", "path": str(path.resolve())})


async def handle_select_path(session: "ChatSession", tc_input: dict) -> str:
    """Have the user choose a file/folder; return the chosen path as JSON.

    Two modes, chosen automatically from the inputs:

    * **browse** — no ``candidates``/``pattern`` given: the location is unknown,
      so the user navigates a picker to locate it. Use this instead of asking
      the user to type or paste a path.
    * **pick** — ``candidates`` or ``pattern`` given: disambiguate concrete
      matches within the project. Auto-resolves a single match and reports
      "no matches" for none, so the picker appears only for a genuine (≥2)
      ambiguity.

    The result is fed back as the tool result, so the agent continues without a
    separate user message.
    """
    import json

    from anton.core.interaction.selection import SelectionRequest

    prompt = (tc_input.get("prompt") or "Select a file or folder.").strip()
    kind = (tc_input.get("kind") or "any").strip().lower()
    if kind not in ("file", "folder", "any"):
        kind = "any"

    root = _selection_root(session)
    elicitor = _resolve_selection_elicitor(session)
    has_candidates = isinstance(tc_input.get("candidates"), list) and bool(tc_input.get("candidates"))
    has_pattern = bool((tc_input.get("pattern") or "").strip())

    # ── browse — locate an unspecified path ──────────────────────────────
    if not has_candidates and not has_pattern:
        if elicitor is None:
            return json.dumps({
                "status": "picker_unavailable",
                "message": "An interactive picker is unavailable here; ask the user for the path in plain text.",
            })
        request = SelectionRequest(
            prompt=prompt, kind=kind, mode="browse", root=str(_browse_start_dir(tc_input, root))
        )
        chosen, error = await _run_elicitor(elicitor, request)
        return error or _finalize_browse_choice(chosen, kind)

    # ── pick — disambiguate concrete candidates within the project ───────
    candidates = _collect_selection_candidates(session, tc_input, root, kind)
    if not candidates:
        return json.dumps({
            "status": "no_matches",
            "message": "No match found. Refine the pattern, omit candidates/pattern to let the user browse, or ask in plain text.",
        })
    if len(candidates) == 1:
        return json.dumps({"status": "resolved", "auto_resolved": True, "path": str(candidates[0])})
    if elicitor is None:
        return json.dumps({
            "status": "picker_unavailable",
            "candidates": [str(p) for p in candidates],
            "message": "An interactive picker is unavailable here; ask the user which of these paths they meant.",
        })

    options = tuple(_selection_option(p, root) for p in candidates)
    chosen, error = await _run_elicitor(elicitor, SelectionRequest(prompt=prompt, options=options, kind=kind))
    if error:
        return error
    if chosen is None:
        return json.dumps({
            "status": "cancelled",
            "message": "The user dismissed the picker without choosing. Ask how they would like to proceed.",
        })
    if chosen not in {option.value for option in options}:
        return json.dumps({"status": "invalid", "message": "The returned selection was not one of the offered options."})
    return json.dumps({"status": "resolved", "path": chosen})
