import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import venv

from .base import (
    Cell,
    ScratchpadRuntime,
    _BOOT_SCRIPT_PATH,
    _CELL_DELIM,
    _CELL_TIMEOUT_DEFAULT,
    _CELL_INACTIVITY_TIMEOUT,
    _CELL_INACTIVITY_AFTER_PROGRESS,
    _INSTALL_TIMEOUT,
    _PROGRESS_MARKER,
    _RESULT_START,
    _RESULT_END,
)

@dataclass
class LocalScratchpadRuntime(ScratchpadRuntime):
    # Runtime settings
    # The runtime will be launched in a subprocess.
    _boot_path: str = str(_BOOT_SCRIPT_PATH)
    _proc: asyncio.subprocess.Process | None = None

    # Virtual environment settings
    # Code will be execueted in isolated virtual environments.
    _venvs_base: Path | None = None
    _venv_dir: str | None = None
    _venv_python: str | None = None
    _MAX_VENV_RETRIES = 3

    def __post_init__(self) -> None:
        self._venvs_base = self._workspace_path / "scratchpad-venvs"

    async def start(self) -> None:
        """Write the boot script to a temp file and launch the subprocess."""
        self._ensure_venv()

        # Create a temp file for the boot script.
        # This is done to avoid collisions between multiple scratchpad processes.
        boot_code = self._boot_path.read_text()
        fd, path = tempfile.mkstemp(suffix=".py", prefix="anton_scratchpad_")
        os.write(fd, boot_code.encode())
        os.close(fd)
        self._boot_path = path

        env = os.environ.copy()
        if self._coding_model:
            env["ANTON_SCRATCHPAD_MODEL"] = self._coding_model
        if self._coding_provider:
            env["ANTON_SCRATCHPAD_PROVIDER"] = self._coding_provider
        # Ensure the SDKs can find API keys under their expected names.
        # Anton stores them as ANTON_*_API_KEY; the SDKs expect *_API_KEY.
        if "ANTHROPIC_API_KEY" not in env and "ANTON_ANTHROPIC_API_KEY" in env:
            env["ANTHROPIC_API_KEY"] = env["ANTON_ANTHROPIC_API_KEY"]
        if "OPENAI_API_KEY" not in env and "ANTON_OPENAI_API_KEY" in env:
            env["OPENAI_API_KEY"] = env["ANTON_OPENAI_API_KEY"]
        # If settings provided an explicit API key (e.g. from ~/.anton/.env or
        # Pydantic settings), inject it so the subprocess SDK can authenticate.
        if self._coding_api_key:
            sdk_key = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
            }.get(self._coding_provider, "")
            if sdk_key and sdk_key not in env:
                env[sdk_key] = self._coding_api_key
        # Pass uv path so the boot script can use it for auto-installing
        # missing modules (same installer that created the venv).
        uv = self._find_uv()
        if uv:
            env["ANTON_UV_PATH"] = uv

        # Ensure the anton package is importable in the subprocess (needed for
        # get_llm and skill loading). The boot script runs from a temp file, so
        # the project root isn't on sys.path by default.
        _anton_root = str(Path(__file__).resolve().parent.parent)
        python_path = env.get("PYTHONPATH", "")
        if _anton_root not in python_path:
            env["PYTHONPATH"] = _anton_root + (os.pathsep + python_path if python_path else "")

        try:
            self._proc = await asyncio.create_subprocess_exec(
                self._venv_python, path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                # Own session so os.killpg() kills the whole process tree
                # (grandchildren spawned by user code, pip installs, etc.)
                start_new_session=(sys.platform != "win32"),
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            # Python binary is missing or broken — nuke venv and raise
            self._nuke_venv()
            raise RuntimeError(
                f"Failed to start scratchpad: {exc}. "
                f"The Python venv has been deleted and will be recreated on next attempt."
            ) from exc

    def _ensure_venv(self) -> None:
        """Create a lightweight per-scratchpad venv (idempotent).

        Uses system_site_packages=True so the real system packages are visible.
        If we're running inside a parent venv, we also drop a .pth file so the
        parent venv's site-packages are visible in the child.

        If a persistent venv already exists on disk it is recycled when healthy.
        If the venv is broken (stale symlinks, missing Python binary, version
        mismatch), it is deleted and recreated from scratch. Gives up after
        _MAX_VENV_RETRIES.
        """
        if self._venv_dir is not None and self._verify_venv_python():
            return

        # Try to recycle a persistent venv from a previous session.
        venv_path = self._venvs_base / self.name
        if venv_path.is_dir() and self._try_recycle_venv(venv_path):
            return

        # Recycling failed or no prior venv — nuke leftovers and create fresh.
        if venv_path.is_dir():
            self._nuke_venv()

        last_error: Exception | None = None
        for _ in range(1, self._MAX_VENV_RETRIES + 1):
            try:
                self._create_venv()
                if self._verify_venv_python():
                    self._setup_parent_site_packages()
                    self._save_python_version()
                    return
                # Python binary exists but doesn't run — nuke and retry
                raise RuntimeError(f"venv Python binary at {self._venv_python} is not functional")
            except Exception as exc:
                last_error = exc
                # Clean up the broken venv before retrying
                self._nuke_venv()

        raise RuntimeError(
            f"Failed to create a working Python venv after {self._MAX_VENV_RETRIES} attempts. "
            f"Last error: {last_error}. "
            f"Try running: python3 -c 'print(\"ok\")' to verify your Python installation."
        )

    def _create_venv(self) -> None:
        """Allocate a venv directory and create the virtual environment.

        Prefers ``uv venv`` when available — it is faster, more reliable on
        macOS (doesn't break when Homebrew upgrades Python), and doesn't depend
        on the ``venv`` stdlib module being functional.  Falls back to
        ``venv.create()`` when ``uv`` isn't found.

        The venv is persisted at ``{_venvs_base}/{name}`` on all platforms so
        installed packages survive across sessions.
        """
        import subprocess as _sp

        self._venv_dir = str(self._venvs_base / self.name)
        os.makedirs(self._venv_dir, exist_ok=True)

        uv = self._find_uv()
        if uv:
            _sp.run(
                [uv, "venv", self._venv_dir,
                 "--python", sys.executable,
                 "--system-site-packages", "--seed", "--quiet"],
                check=True,
                capture_output=True,
                timeout=30,
            )
        else:
            venv.create(self._venv_dir, system_site_packages=True, with_pip=False, clear=True)

        if sys.platform == "win32":
            bin_dir = os.path.join(self._venv_dir, "Scripts")
            self._venv_python = os.path.join(bin_dir, "python.exe")
            self._add_windows_firewall_rule()
        else:
            bin_dir = os.path.join(self._venv_dir, "bin")
            self._venv_python = os.path.join(bin_dir, "python")

    @staticmethod
    def _find_uv() -> str | None:
        """Return the path to the ``uv`` binary, or *None* if unavailable."""
        # Fast path: already on PATH
        uv = shutil.which("uv")
        if uv:
            return uv
        # Common install locations
        if sys.platform == "win32":
            candidates = (
                os.path.expanduser("~/.local/bin/uv.exe"),
                os.path.expanduser("~/.cargo/bin/uv.exe"),
            )
        else:
            candidates = (
                os.path.expanduser("~/.local/bin/uv"),
                os.path.expanduser("~/.cargo/bin/uv"),
            )
        for candidate in candidates:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        return None

    def _verify_venv_python(self) -> bool:
        """Check that the venv Python binary exists and can execute."""
        if self._venv_python is None:
            return False
        if not os.path.exists(self._venv_python):
            return False
        # Quick smoke test — run python with a trivial command
        try:
            import subprocess
            result = subprocess.run(
                [self._venv_python, "-c", "print('ok')"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0 and "ok" in result.stdout.decode()
        except Exception:
            return False

    def _nuke_venv(self) -> None:
        """Delete the venv directory entirely so it can be recreated."""
        if self._venv_dir is not None:
            try:
                shutil.rmtree(self._venv_dir)
            except OSError:
                pass
        self._venv_dir = None
        self._venv_python = None

    def _add_windows_firewall_rule(self) -> None:
        """Add a Windows Firewall outbound-allow rule for this venv's python.exe.

        Windows Firewall blocks new executables by default.  Without a rule,
        scratchpad HTTP calls (httpx, requests, etc.) silently time out.
        Runs silently — failures are ignored (user can add rules manually).
        """
        if self._venv_python is None or not os.path.isfile(self._venv_python):
            return
        import subprocess as _sp
        rule_name = f"Anton Scratchpad - {self.name}"
        try:
            _sp.run(
                [
                    "netsh", "advfirewall", "firewall", "add", "rule",
                    f"name={rule_name}", "dir=out", "action=allow",
                    f"program={self._venv_python}",
                ],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass
        self._installed_packages.clear()

    def _setup_parent_site_packages(self) -> None:
        """Make parent venv's packages visible in the child venv."""
        if sys.prefix != sys.base_prefix:
            import site as _site
            parent_site = _site.getsitepackages()
            child_site = None
            for dirpath, dirnames, _ in os.walk(self._venv_dir):
                if "site-packages" in dirnames:
                    child_site = os.path.join(dirpath, "site-packages")
                    break
            if child_site and parent_site:
                pth_path = os.path.join(child_site, "_parent_venv.pth")
                with open(pth_path, "w") as f:
                    for sp in parent_site:
                        f.write(sp + "\n")

    def _try_recycle_venv(self, venv_path: Path) -> bool:
        """Validate and reuse a persistent venv from a previous session.

        Sets internal paths, verifies the Python binary is functional, checks
        the Python version matches, loads saved requirements, and refreshes
        parent-site-packages links.  Returns False on any failure (caller
        should nuke the directory and create a fresh venv).
        """
        try:
            self._venv_dir = str(venv_path)
            if sys.platform == "win32":
                self._venv_python = os.path.join(self._venv_dir, "Scripts", "python.exe")
            else:
                self._venv_python = os.path.join(self._venv_dir, "bin", "python")

            if not self._verify_venv_python():
                return False
            if not self._check_python_version():
                return False
            self._load_requirements()
            self._setup_parent_site_packages()
            return True
        except Exception:
            return False

    def _save_requirements(self) -> None:
        """Write installed package names to requirements.txt (best-effort)."""
        if not self._venv_dir or not self._installed_packages:
            return
        try:
            req_path = os.path.join(self._venv_dir, "requirements.txt")
            with open(req_path, "w") as f:
                for pkg in sorted(self._installed_packages):
                    f.write(pkg + "\n")
        except OSError:
            pass

    def _load_requirements(self) -> None:
        """Read requirements.txt into _installed_packages."""
        if not self._venv_dir:
            return
        req_path = os.path.join(self._venv_dir, "requirements.txt")
        try:
            with open(req_path) as f:
                for line in f:
                    pkg = line.strip()
                    if pkg:
                        self._installed_packages.add(pkg)
        except FileNotFoundError:
            pass

    def _save_python_version(self) -> None:
        """Write the current Python major.minor to .python_version."""
        if not self._venv_dir:
            return
        try:
            ver_path = os.path.join(self._venv_dir, ".python_version")
            with open(ver_path, "w") as f:
                f.write(f"{sys.version_info.major}.{sys.version_info.minor}\n")
        except OSError:
            pass

    def _check_python_version(self) -> bool:
        """Return True if .python_version matches the current Python."""
        if not self._venv_dir:
            return False
        ver_path = os.path.join(self._venv_dir, ".python_version")
        try:
            with open(ver_path) as f:
                saved = f.read().strip()
            expected = f"{sys.version_info.major}.{sys.version_info.minor}"
            return saved == expected
        except FileNotFoundError:
            # No version file — treat as mismatch so it gets recreated with one.
            return False

    async def execute_streaming(
        self,
        code: str,
        *,
        description: str = "",
        estimated_time: str = "",
        estimated_seconds: int = 0,
    ):
        """Async generator that sends code and yields progress strings and a final Cell.

        Yields:
            str — progress messages from progress() calls in the cell code
            Cell — the final execution result (always the last item)
        """
        if self._proc is None or self._proc.returncode is not None:
            yield Cell(
                code=code,
                stdout="",
                stderr="",
                error="Scratchpad process is not running. Use reset to restart.",
                description=description,
                estimated_time=estimated_time,
            )
            return

        payload = code + "\n" + _CELL_DELIM + "\n"
        self._proc.stdin.write(payload.encode())  # type: ignore[union-attr]
        await self._proc.stdin.drain()  # type: ignore[union-attr]

        total_timeout, inactivity_timeout = self._compute_timeouts(estimated_seconds)

        try:
            result_data: dict | None = None
            async for item in self._read_result(
                total_timeout=total_timeout,
                inactivity_timeout=inactivity_timeout,
            ):
                if isinstance(item, str):
                    yield item  # progress message
                else:
                    result_data = item
        except asyncio.TimeoutError as exc:
            self._kill_tree()
            await self._proc.wait()
            cell = Cell(
                code=code,
                stdout="",
                stderr="",
                error=f"{exc}. Process killed — state lost. Use reset to restart.",
                description=description,
                estimated_time=estimated_time,
            )
            self.cells.append(cell)
            yield cell
            return

        if result_data is None:
            result_data = {"stdout": "", "stderr": "", "error": "Process exited unexpectedly."}

        # Track packages that the subprocess auto-installed on ModuleNotFoundError
        for pkg in result_data.get("auto_installed") or []:
            self._installed_packages.add(pkg.lower())

        cell = Cell(
            code=code,
            stdout=result_data.get("stdout", ""),
            stderr=result_data.get("stderr", ""),
            error=result_data.get("error"),
            description=description,
            estimated_time=estimated_time,
            logs=result_data.get("logs", ""),
        )
        self.cells.append(cell)
        yield cell

    async def _read_result(
        self,
        *,
        total_timeout: float = _CELL_TIMEOUT_DEFAULT,
        inactivity_timeout: float = _CELL_INACTIVITY_TIMEOUT,
    ):
        """Async generator that reads lines from stdout until result delimiters.

        Yields:
            str — progress messages (lines starting with _PROGRESS_MARKER)
            dict — the final JSON result (always the last item)

        Raises asyncio.TimeoutError with a descriptive message.

        After a progress() call is received, the inactivity window is extended
        to _CELL_INACTIVITY_AFTER_PROGRESS (60s) so that long-running work
        that signals liveness isn't killed prematurely.
        """
        import time as _time

        lines: list[str] = []
        in_result = False
        start = _time.monotonic()
        current_inactivity = inactivity_timeout

        while True:
            elapsed = _time.monotonic() - start
            remaining_total = total_timeout - elapsed
            if remaining_total <= 0:
                raise asyncio.TimeoutError(
                    f"Cell timed out after {total_timeout:.0f}s total"
                )

            line_timeout = min(current_inactivity, remaining_total)
            try:
                raw = await asyncio.wait_for(
                    self._proc.stdout.readline(),  # type: ignore[union-attr]
                    timeout=line_timeout,
                )
            except asyncio.TimeoutError:
                # Determine which timeout was hit
                elapsed_now = _time.monotonic() - start
                if elapsed_now >= total_timeout - 0.5:
                    raise asyncio.TimeoutError(
                        f"Cell timed out after {total_timeout:.0f}s total"
                    ) from None
                raise asyncio.TimeoutError(
                    f"Cell killed after {current_inactivity:.0f}s of inactivity "
                    f"(no output or progress() calls)"
                ) from None

            if not raw:
                yield {"stdout": "", "stderr": "", "error": "Process exited unexpectedly."}
                return

            line = raw.decode().rstrip("\r\n")

            # Progress marker — yield to caller, don't store.
            # Extend inactivity window: the cell is actively working.
            if line.startswith(_PROGRESS_MARKER):
                current_inactivity = max(
                    current_inactivity, _CELL_INACTIVITY_AFTER_PROGRESS,
                )
                message = line[len(_PROGRESS_MARKER):].strip()
                yield message
                continue

            if line == _RESULT_START:
                in_result = True
                continue
            if line == _RESULT_END:
                break
            if in_result:
                lines.append(line)

        yield json.loads("\n".join(lines))

    async def cancel(self) -> None:
        """Kill the current execution and restart the subprocess.

        Called when the user cancels (ESC / Ctrl-C) during a running cell.
        Kills the entire process tree, records a cancelled cell, then restarts
        so the scratchpad is ready for the next use.
        """
        if self._proc is None or self._proc.returncode is not None:
            return
        self._kill_tree()
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
        # Record the cancelled execution
        self.cells.append(Cell(
            code="# (cancelled by user)",
            stdout="",
            stderr="",
            error="Cancelled by user.",
            description="Cancelled",
        ))
        # Restart so the pad is usable again
        self._proc = None
        await self.start()

    async def _stop_process(self) -> None:
        """Kill the subprocess and delete the boot script, but keep the venv."""
        if self._proc is not None and self._proc.returncode is None:
            try:
                self._kill_tree()
                await self._proc.wait()
            except ProcessLookupError:
                pass
        # Close transport pipes to prevent "Event loop is closed" noise
        # from __del__ during Python shutdown.
        if self._proc is not None:
            for attr in ("stdin", "stdout", "stderr"):
                pipe = getattr(self._proc, attr, None)
                if pipe and not pipe.is_closing() if hasattr(pipe, "is_closing") else False:
                    pipe.close()
        self._proc = None
        if self._boot_path is not None:
            try:
                os.unlink(self._boot_path)
            except OSError:
                pass
            self._boot_path = None

    def _kill_tree(self) -> None:
        """Kill the subprocess and all its children via process group."""
        if self._proc is None or self._proc.returncode is not None:
            return
        pid = self._proc.pid
        if sys.platform != "win32":
            # Kill the entire process group (subprocess + grandchildren)
            import signal
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                # Fallback: kill just the direct child
                try:
                    self._proc.kill()
                except ProcessLookupError:
                    pass
        else:
            self._proc.kill()

    async def reset(self) -> None:
        """Kill the process, clear cells, restart.

        If the venv is healthy, it's reused (installed packages survive).
        If the venv is broken, it's deleted and recreated from scratch.
        """
        await self._stop_process()
        self.cells.clear()
        # If the venv Python is broken, nuke it so _ensure_venv recreates it
        if not self._verify_venv_python():
            self._nuke_venv()
        await self.start()

    async def close(self, cleanup: bool = True) -> None:
        """Kill the process and clean up the boot script temp file.

        If cleanup is True, the venv is deleted and recreated from scratch.
        Otherwise, the venv is preserved on disk so installed packages survive across
        sessions. A ``requirements.txt`` is saved to record what was installed.
        """
        await self._stop_process()
        if self._venv_dir is not None:
            if cleanup:
                self._nuke_venv()
            else:
                self._save_requirements()
            self._venv_dir = None
            self._venv_python = None

    async def install_packages(self, packages: list[str]) -> str:
        """Install packages into the scratchpad's venv via pip (or uv pip)."""
        if not packages:
            return "No packages specified."
        # Skip packages we've already installed in this scratchpad
        needed = [p for p in packages if p.lower() not in self._installed_packages]
        if not needed:
            return "All packages already installed."
        self._ensure_venv()

        uv = self._find_uv()
        if uv:
            cmd = [uv, "pip", "install", "--python", self._venv_python, *needed]
        else:
            cmd = [self._venv_python, "-m", "pip", "install", "--no-input", *needed]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_INSTALL_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"Install timed out after {_INSTALL_TIMEOUT}s."
        output = stdout.decode()
        if proc.returncode != 0:
            return f"Install failed (exit {proc.returncode}):\n{output}"
        # Track successfully installed packages
        for p in needed:
            self._installed_packages.add(p.lower())
        return output