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


async def handle_update_artifact_metadata(session: "ChatSession", tc_input: dict) -> str:
    """Update mutable metadata fields on an existing artifact.

    Only fields present in the input are modified. Supports:
    - `primary`: entry-point file path (empty string to clear)
    - `port`: backend port number (fullstack-stateful-app only)
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
        kwargs["port"] = tc_input["port"]

    artifact = store.update(slug, **kwargs)
    if artifact is None:
        return f"Error: no artifact found for slug `{slug}`."
    return json.dumps({
        "slug": artifact.slug,
        "primary": artifact.primary,
        "port": artifact.port,
    }, indent=2)


async def handle_launch_backend(session: "ChatSession", tc_input: dict) -> str:
    """Launch the artifact's backend script as a standalone subprocess.

    Picks a free TCP port, runs
    `<scratchpad venv python> <backend script> --port <port> [...extra_args]`
    with the artifact folder as cwd, waits for the server to become
    reachable, persists the port in metadata.json, and tracks the
    process on the session so it can be reaped on close.

    Idempotent: a fresh call for the same slug terminates any
    previously-tracked backend before launching the new one.
    """
    import asyncio
    import json
    import os
    import signal
    import socket
    import sys
    import urllib.error
    import urllib.request

    store = _artifact_store(session)
    if store is None:
        return "Artifact store unavailable (no workspace bound to this session)."

    slug = (tc_input.get("slug") or "").strip()
    if not slug:
        return "Error: `slug` is required."
    artifact = store.open(slug)
    if artifact is None:
        return f"Error: no artifact found for slug `{slug}`."

    folder = store.folder_for(slug)
    rel_path = (tc_input.get("path") or "backend.py").strip()
    script = (folder / rel_path).resolve()
    try:
        script.relative_to(folder.resolve())
    except ValueError:
        return f"Error: `path` must stay within the artifact folder ({folder})."
    if not script.is_file():
        return f"Error: backend script not found at {script}."

    extra_args = tc_input.get("extra_args") or []
    if not isinstance(extra_args, list) or not all(isinstance(x, str) for x in extra_args):
        return "Error: `extra_args` must be a list of strings."
    health_path = tc_input.get("health_path") or "/"
    if not health_path.startswith("/"):
        health_path = "/" + health_path
    try:
        health_timeout = float(tc_input.get("health_timeout", 10))
    except (TypeError, ValueError):
        return "Error: `health_timeout` must be a number."

    venv_python = await session._scratchpads.venv_python(slug)
    if not venv_python:
        return (
            "Error: scratchpad venv Python is not available. "
            "This usually means the runtime is remote, or no scratchpad cell "
            "has run yet to provision the venv."
        )

    req_path = folder / "requirements.txt"
    if req_path.is_file():
        packages: list[str] = []
        for raw_line in req_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            packages.append(line)
        if packages:
            from datetime import datetime, timezone

            pad = await session._scratchpads.get_or_create(slug)
            install_result = await pad.install_packages(packages)
            banner = (
                f"\n=== requirements.txt install "
                f"({datetime.now(timezone.utc).isoformat(timespec='seconds')}) ===\n"
            )
            with open(folder / "backend.log", "ab", buffering=0) as install_log:
                install_log.write(banner.encode("utf-8"))
                install_log.write(install_result.encode("utf-8"))
                install_log.write(b"\n")
            if install_result.startswith("Install failed") or install_result.startswith(
                "Install timed out"
            ):
                return (
                    "Error: dependency install failed for `requirements.txt`.\n"
                    + install_result
                )

    tracked = getattr(session, "_tracked_backends", None)
    if tracked is None:
        tracked = {}
        session._tracked_backends = tracked

    # Reap any previously-tracked backend for this slug before launching
    # the new one — keeps the call idempotent across hot reloads.
    prev = tracked.pop(slug, None)
    if prev is not None:
        prev_proc = prev.get("proc")
        if prev_proc is not None and prev_proc.returncode is None:
            try:
                prev_proc.terminate()
                try:
                    await asyncio.wait_for(prev_proc.wait(), timeout=3)
                except asyncio.TimeoutError:
                    prev_proc.kill()
                    await prev_proc.wait()
            except ProcessLookupError:
                pass

    # Bind-and-close to discover a free port. There is a TOCTOU window
    # before the backend picks it up — acceptable in single-user dev.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    cmd = [venv_python, str(script), "--port", str(port), *extra_args]
    log_path = folder / "backend.log"
    log_fd = open(log_path, "ab", buffering=0)

    # PR_SET_PDEATHSIG so the backend dies with Anton on Linux. macOS
    # has no equivalent; we rely on close() to reap there.
    preexec_fn = None
    if sys.platform.startswith("linux"):
        def _set_pdeathsig() -> None:
            try:
                import ctypes

                libc = ctypes.CDLL("libc.so.6", use_errno=True)
                PR_SET_PDEATHSIG = 1
                libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
            except Exception:
                pass

        preexec_fn = _set_pdeathsig

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(folder),
            stdout=log_fd,
            stderr=log_fd,
            stdin=asyncio.subprocess.DEVNULL,
            preexec_fn=preexec_fn,
            env={**os.environ},
        )
    except OSError as exc:
        log_fd.close()
        return f"Error: failed to spawn backend: {exc}"
    finally:
        # The subprocess holds its own dup of the fd; we can close ours.
        try:
            log_fd.close()
        except OSError:
            pass

    # Readiness — try HTTP first, fall back to TCP-connect. HTTP 4xx
    # still counts as "process is alive and answering" → ready.
    loop = asyncio.get_event_loop()
    deadline = loop.time() + health_timeout
    ready = False
    last_err: str | None = None
    while loop.time() < deadline:
        if proc.returncode is not None:
            tail = ""
            try:
                tail = log_path.read_text(errors="replace")[-2000:]
            except OSError:
                pass
            return (
                f"Error: backend exited early (rc={proc.returncode}) before "
                f"binding to :{port}.\nLog tail:\n{tail}"
            )
        url = f"http://127.0.0.1:{port}{health_path}"
        try:
            await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: urllib.request.urlopen(url, timeout=1).close()
                ),
                timeout=1.5,
            )
            ready = True
            break
        except urllib.error.HTTPError:
            # 4xx/5xx → process is alive and listening
            ready = True
            break
        except Exception as exc:
            last_err = str(exc)
            # Fallback: bare TCP connect
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    ready = True
                    break
            except OSError:
                await asyncio.sleep(0.2)

    if not ready:
        try:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        except ProcessLookupError:
            pass
        tail = ""
        try:
            tail = log_path.read_text(errors="replace")[-2000:]
        except OSError:
            pass
        return (
            f"Error: backend did not become ready on :{port} within "
            f"{health_timeout}s (last error: {last_err}).\nLog tail:\n{tail}"
        )

    tracked[slug] = {"proc": proc, "port": port, "pid": proc.pid, "log_path": str(log_path)}
    store.update(slug, port=port)

    return json.dumps(
        {
            "slug": slug,
            "port": port,
            "pid": proc.pid,
            "url": f"http://127.0.0.1:{port}",
            "log_path": str(log_path),
        },
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

    if action == "exec":
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
