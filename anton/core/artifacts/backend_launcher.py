"""Launch a fullstack artifact's backend script as a standalone subprocess.

Extracted from `anton/core/tools/tool_handlers.handle_launch_backend` so it
can be invoked outside of a ChatSession — notably from cowork, which
auto-relaunches backends when the user opens a preview after the Anton
session that created them has ended.

The helper owns: requirements.txt install into the scratchpad venv, free
port discovery, subprocess spawn with PR_SET_PDEATHSIG on Linux, HTTP/TCP
readiness probe, and idempotent reaping of any previously-tracked process
for the same slug. It does NOT own: artifact metadata writes (caller
updates `metadata.json.port` if appropriate), `--port`-flag protocol on
the backend script (assumed; callers wanting a different protocol should
build their own launcher).
"""
from __future__ import annotations

import asyncio
import os
import signal
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol


class ScratchpadPoolLike(Protocol):
    """Minimal surface the launcher needs from a scratchpad pool.

    Both `anton.core.backends.ScratchpadManager` and cowork's module-level
    pool wrapper satisfy this — the launcher stays decoupled from either
    concrete implementation.
    """

    async def venv_python(self, name: str) -> str | None: ...

    async def get_or_create(self, name: str) -> Any: ...
    # The returned object must expose:
    #     async install_packages(packages: list[str]) -> str


async def launch_artifact_backend(
    *,
    slug: str,
    artifact_folder: Path,
    scratchpad_pool: ScratchpadPoolLike,
    tracked_backends: dict[str, dict],
    path: str = "backend.py",
    extra_args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    health_path: str = "/",
    health_timeout: float = 10.0,
) -> dict | str:
    """Launch the artifact's backend script in its scratchpad venv.

    Returns a dict `{slug, port, pid, url, log_path, proc}` on success
    (caller is responsible for persisting `port` to artifact metadata if
    needed). Returns an error string on failure — the prefix tells the
    caller whether the failure is in script resolution, dependency
    install, or runtime readiness.

    `tracked_backends` is a dict the caller owns; the launcher stores the
    spawned `asyncio.subprocess.Process` under `slug` and reaps any
    previously-tracked process for the same slug before spawning. The
    caller is responsible for cleaning the dict on shutdown.

    `extra_env` is merged over the inherited `os.environ` for the spawned
    process only (e.g. datasource `DS_*` secrets) — it never mutates the
    parent's environment, keeping secrets scoped to the backend subprocess.
    """
    extra_args = list(extra_args or [])
    folder = artifact_folder

    script = (folder / path).resolve()
    try:
        script.relative_to(folder.resolve())
    except ValueError:
        return f"Error: `path` must stay within the artifact folder ({folder})."
    if not script.is_file():
        return f"Error: backend script not found at {script}."

    if not isinstance(extra_args, list) or not all(isinstance(x, str) for x in extra_args):
        return "Error: `extra_args` must be a list of strings."
    if not health_path.startswith("/"):
        health_path = "/" + health_path

    venv_python = await scratchpad_pool.venv_python(slug)
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

            pad = await scratchpad_pool.get_or_create(slug)
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

    # Reap any previously-tracked backend for this slug before launching
    # the new one — keeps the call idempotent across hot reloads.
    prev = tracked_backends.pop(slug, None)
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

    # PR_SET_PDEATHSIG so the backend dies with the parent on Linux. macOS
    # has no equivalent; we rely on caller-side reap there.
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
            env={**os.environ, **(extra_env or {})},
        )
    except OSError as exc:
        log_fd.close()
        return f"Error: failed to spawn backend: {exc}"
    finally:
        try:
            log_fd.close()
        except OSError:
            pass

    # Readiness — try HTTP first, fall back to TCP-connect. HTTP 4xx
    # still counts as "process is alive and answering" → ready.
    loop = asyncio.get_running_loop()
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
            ready = True
            break
        except Exception as exc:
            last_err = str(exc)
            try:
                await loop.run_in_executor(
                    None,
                    lambda: socket.create_connection(
                        ("127.0.0.1", port), timeout=0.5
                    ).close(),
                )
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

    tracked_backends[slug] = {
        "proc": proc,
        "port": port,
        "pid": proc.pid,
        "log_path": str(log_path),
    }

    return {
        "slug": slug,
        "port": port,
        "pid": proc.pid,
        "url": f"http://127.0.0.1:{port}",
        "log_path": str(log_path),
        "proc": proc,
    }
