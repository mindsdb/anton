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
import re
import shutil
import signal
import socket
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol


def _anton_state_pythonpath_dir() -> str:
    """A private directory exposing ONLY the `anton_state` package, for PYTHONPATH.

    Pointing PYTHONPATH at the package's real parent would leak the whole anton
    repo (or its venv's site-packages) onto the backend's sys.path and shadow
    the scratchpad venv's own deps (fastapi/pydantic) — a local != published
    hazard. Instead we expose anton_state alone via a symlink (copy fallback).
    In the cloud the package is vendored into the bundle instead (see publisher).
    """
    import anton_state

    pkg = Path(anton_state.__file__).resolve().parent
    root = Path(tempfile.gettempdir()) / "anton_state_pp"
    link = root / "anton_state"
    root.mkdir(parents=True, exist_ok=True)
    # Always reset the entry, then (re)link — keeps it fresh and simple.
    if link.is_symlink() or link.exists():
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link)
        else:
            link.unlink()
    try:
        link.symlink_to(pkg, target_is_directory=True)
    except OSError:
        shutil.copytree(pkg, link)
    return str(root)


def _build_backend_env(
    extra_env: dict[str, str] | None,
    ds_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Subprocess env: inherited environ + extra_env, with anton_state on PYTHONPATH.

    A non-None `ds_env` replaces the inherited DS_* entirely, so the backend
    sees only the datasources it declared.
    """
    env = {**os.environ}
    # Before the strip, so a caller's DS_* survive only when ds_env is None;
    # a project .env cannot add one to a backend that declared its own.
    env.update(extra_env or {})
    if ds_env is not None:
        for key in [k for k in env if k.startswith("DS_")]:
            del env[key]
        env.update(ds_env)
    isolated = _anton_state_pythonpath_dir()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = isolated + (os.pathsep + existing if existing else "")
    return env


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
    ds_env: dict[str, str] | None = None,
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

    `ds_env`, when given, is the backend's complete `DS_*` set: the inherited
    ones are dropped first, so a connection the artifact did not declare (or
    one a concurrent turn injected) cannot reach it. Callers that still route
    `DS_*` through `extra_env` keep the old merge-only behaviour.
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
            # `anton_state` is the internal STATE SDK — it is provided to the
            # backend at runtime via PYTHONPATH (see _anton_state_pythonpath_dir),
            # not published to any package registry. Drop it if the model listed
            # it in requirements.txt, otherwise the install step fails with
            # "anton-state was not found in the package registry".
            pkg_name = re.split(r"[<>=!~ \[]", line, maxsplit=1)[0].strip()
            if pkg_name.replace("-", "_").lower() == "anton_state":
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
            if install_result.startswith(
                ("Install failed", "Install timed out", "Install refused")
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
            env=_build_backend_env(extra_env, ds_env),
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
