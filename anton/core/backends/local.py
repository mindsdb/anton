"""LocalScratchpadRuntime — venv-based scratchpad for the CLI."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import venv
from pathlib import Path

from anton.core.backends.base import Cell, ScratchpadRuntime
from anton.core.backends.wire import (
    CELL_DELIM,
    PROGRESS_MARKER,
    RESULT_END,
    RESULT_START,
)
from anton.core.settings import CoreSettings

_BOOT_SCRIPT_PATH = Path(__file__).parent / "scratchpad_boot.py"
_MAX_OUTPUT = 10_000


def _compute_timeouts(estimated_seconds: int) -> tuple[float, float]:
    """Compute (total_timeout, inactivity_timeout) from an estimated run time.

    Reads defaults from CoreSettings so they're tunable via env vars.
    """
    s = CoreSettings()
    if estimated_seconds <= 0:
        return float(s.cell_timeout_default), float(s.cell_inactivity_timeout)
    total = max(estimated_seconds * 2, estimated_seconds + 30)
    inactivity = max(estimated_seconds * 0.5, 30)
    return float(total), float(inactivity)


class LocalScratchpadRuntime(ScratchpadRuntime):
    """Runs scratchpad cells in a persistent per-named venv subprocess."""

    _MAX_VENV_RETRIES = 3

    def __init__(
        self,
        name: str,
        *,
        cells: list[Cell] | None = None,
        coding_provider: str = "anthropic",
        coding_model: str = "",
        coding_api_key: str = "",
        coding_base_url: str = "",
        workspace_path: Path | None = None,
        _venvs_base: Path | None = None,
    ) -> None:
        super().__init__(
            name,
            cells=cells,
            coding_provider=coding_provider,
            coding_model=coding_model,
            coding_api_key=coding_api_key,
            workspace_path=workspace_path,
        )
        self._coding_base_url = coding_base_url
        self._proc: asyncio.subprocess.Process | None = None
        self._boot_path: str | None = None
        self._venv_dir: str | None = None
        self._venv_python: str | None = None
        if _venvs_base is not None:
            self._venvs_base = _venvs_base
        elif workspace_path is not None:
            self._venvs_base = workspace_path / ".anton" / "scratchpad-venvs"
        else:
            self._venvs_base = Path("~/.anton/scratchpad-venvs").expanduser()

    def _ensure_venv(self) -> None:
        if self._venv_dir is not None and self._verify_venv_python():
            return

        venv_path = self._venvs_base / self.name
        if venv_path.is_dir() and self._try_recycle_venv(venv_path):
            return

        if venv_path.is_dir():
            self._nuke_venv()

        last_error: Exception | None = None
        for attempt in range(1, self._MAX_VENV_RETRIES + 1):
            try:
                self._create_venv()
                if self._verify_venv_python():
                    self._setup_parent_site_packages()
                    self._save_python_version()
                    return
                raise RuntimeError(
                    f"venv Python binary at {self._venv_python} is not functional"
                )
            except Exception as exc:
                last_error = exc
                self._nuke_venv()

        raise RuntimeError(
            f"Failed to create a working Python venv after {self._MAX_VENV_RETRIES} "
            f"attempts. Last error: {last_error}. "
            f"Try running: python3 -c 'print(\"ok\")' to verify your Python installation."
        )

    @staticmethod
    def _find_uv() -> str | None:
        uv = shutil.which("uv")
        if uv:
            return uv
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

    def _create_venv(self) -> None:
        import subprocess as _sp

        self._venv_dir = str(self._venvs_base / self.name)
        os.makedirs(self._venv_dir, exist_ok=True)

        uv = self._find_uv()
        if uv:
            _sp.run(
                [
                    uv,
                    "venv",
                    self._venv_dir,
                    "--python",
                    sys.executable,
                    "--system-site-packages",
                    "--seed",
                    "--quiet",
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
        else:
            venv.create(
                self._venv_dir,
                system_site_packages=True,
                with_pip=False,
                clear=True,
            )

        if sys.platform == "win32":
            bin_dir = os.path.join(self._venv_dir, "Scripts")
            self._venv_python = os.path.join(bin_dir, "python.exe")
            self._add_windows_firewall_rule()
        else:
            bin_dir = os.path.join(self._venv_dir, "bin")
            self._venv_python = os.path.join(bin_dir, "python")

    def _verify_venv_python(self) -> bool:
        if self._venv_python is None:
            return False
        if not os.path.exists(self._venv_python):
            return False
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
        if self._venv_dir is not None:
            try:
                shutil.rmtree(self._venv_dir)
            except OSError:
                pass
        self._venv_dir = None
        self._venv_python = None

    def _add_windows_firewall_rule(self) -> None:
        if self._venv_python is None or not os.path.isfile(self._venv_python):
            return
        import subprocess as _sp

        rule_name = f"Anton Scratchpad - {self.name}"
        try:
            _sp.run(
                [
                    "netsh",
                    "advfirewall",
                    "firewall",
                    "add",
                    "rule",
                    f"name={rule_name}",
                    "dir=out",
                    "action=allow",
                    f"program={self._venv_python}",
                ],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass
        self._installed_packages.clear()

    def _setup_parent_site_packages(self) -> None:
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
        try:
            self._venv_dir = str(venv_path)
            if sys.platform == "win32":
                self._venv_python = os.path.join(
                    self._venv_dir, "Scripts", "python.exe"
                )
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
        if not self._venv_dir:
            return
        try:
            ver_path = os.path.join(self._venv_dir, ".python_version")
            with open(ver_path, "w") as f:
                f.write(f"{sys.version_info.major}.{sys.version_info.minor}\n")
        except OSError:
            pass

    def _check_python_version(self) -> bool:
        if not self._venv_dir:
            return False
        ver_path = os.path.join(self._venv_dir, ".python_version")
        try:
            with open(ver_path) as f:
                saved = f.read().strip()
            expected = f"{sys.version_info.major}.{sys.version_info.minor}"
            return saved == expected
        except FileNotFoundError:
            return False

    async def start(self) -> None:
        """Write the boot script to a temp file and launch the subprocess."""
        self._ensure_venv()

        boot_code = _BOOT_SCRIPT_PATH.read_text()
        fd, path = tempfile.mkstemp(suffix=".py", prefix="anton_scratchpad_")
        os.write(fd, boot_code.encode())
        os.close(fd)
        self._boot_path = path

        env = os.environ.copy()
        if self._coding_model:
            env["ANTON_SCRATCHPAD_MODEL"] = self._coding_model
        if self._coding_provider:
            env["ANTON_SCRATCHPAD_PROVIDER"] = self._coding_provider
        if "ANTHROPIC_API_KEY" not in env and "ANTON_ANTHROPIC_API_KEY" in env:
            env["ANTHROPIC_API_KEY"] = env["ANTON_ANTHROPIC_API_KEY"]
        if "OPENAI_API_KEY" not in env and "ANTON_OPENAI_API_KEY" in env:
            env["OPENAI_API_KEY"] = env["ANTON_OPENAI_API_KEY"]
        if "OPENAI_BASE_URL" not in env and "ANTON_OPENAI_BASE_URL" in env:
            env["OPENAI_BASE_URL"] = env["ANTON_OPENAI_BASE_URL"]
        if (
            "OPENAI_API_KEY" not in env
            and "ANTON_MINDS_API_KEY" in env
            and self._coding_provider == "openai-compatible"
        ):
            env["OPENAI_API_KEY"] = env["ANTON_MINDS_API_KEY"]
        if (
            "OPENAI_BASE_URL" not in env
            and "ANTON_MINDS_URL" in env
            and self._coding_provider == "openai-compatible"
        ):
            env["OPENAI_BASE_URL"] = f"{env['ANTON_MINDS_URL'].rstrip('/')}/api/v1"
        if self._coding_api_key:
            sdk_key = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "openai-compatible": "OPENAI_API_KEY",
            }.get(self._coding_provider, "")
            if sdk_key:
                env[sdk_key] = self._coding_api_key
        if self._coding_provider in ("openai", "openai-compatible"):
            base_url = (
                self._coding_base_url
                or env.get("ANTON_OPENAI_BASE_URL")
                or env.get("OPENAI_BASE_URL")
                or ""
            )
            if base_url:
                env["OPENAI_BASE_URL"] = base_url
                env["ANTON_OPENAI_BASE_URL"] = base_url
        uv = self._find_uv()
        if uv:
            env["ANTON_UV_PATH"] = uv

        _anton_root = str(Path(__file__).resolve().parent.parent.parent.parent)
        python_path = env.get("PYTHONPATH", "")
        if _anton_root not in python_path:
            env["PYTHONPATH"] = _anton_root + (
                os.pathsep + python_path if python_path else ""
            )

        try:
            self._proc = await asyncio.create_subprocess_exec(
                self._venv_python,
                path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                start_new_session=(sys.platform != "win32"),
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            self._nuke_venv()
            raise RuntimeError(
                f"Failed to start scratchpad: {exc}. "
                "The Python venv has been deleted and will be recreated on next attempt."
            ) from exc

    async def reset(self) -> None:
        """Kill the process, clear cells, and restart."""
        await self._stop_process()
        self.cells.clear()
        if not self._verify_venv_python():
            self._nuke_venv()
        await self.start()

    async def close(self) -> None:
        """Kill the process and save requirements; preserve the venv."""
        await self._stop_process()
        if self._venv_dir is not None:
            self._save_requirements()
            self._venv_dir = None
            self._venv_python = None

    async def cancel(self) -> None:
        """Kill the current cell and restart the runtime."""
        if self._proc is None or self._proc.returncode is not None:
            return
        self._kill_tree()
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
        self.cells.append(
            Cell(
                code="# (cancelled by user)",
                stdout="",
                stderr="",
                error="Cancelled by user.",
                description="Cancelled",
            )
        )
        self._proc = None
        await self.start()

    async def cleanup(self) -> None:
        """Kill process and delete the venv entirely."""
        await self._stop_process()
        self._nuke_venv()

    async def execute_streaming(
        self,
        code: str,
        *,
        description: str = "",
        estimated_time: str = "",
        estimated_seconds: int = 0,
    ):
        """Async generator: yields progress strings then a final Cell."""
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

        payload = code + "\n" + CELL_DELIM + "\n"
        self._proc.stdin.write(payload.encode())  # type: ignore[union-attr]
        await self._proc.stdin.drain()  # type: ignore[union-attr]

        total_timeout, inactivity_timeout = _compute_timeouts(estimated_seconds)

        try:
            result_data: dict | None = None
            async for item in self._read_result(
                total_timeout=total_timeout,
                inactivity_timeout=inactivity_timeout,
            ):
                if isinstance(item, str):
                    yield item
                else:
                    result_data = item
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            self._kill_tree()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
            error_msg = (
                f"{exc}. Process killed — state lost. Use reset to restart.\n\n"
                "If a database query was running, it may still be executing server-side.\n"
                "To check and cancel: run SHOW PROCESSLIST (MySQL) or\n"
                "SELECT * FROM information_schema.processlist WHERE status='running' "
                "and cancel with KILL <id>.\n"
                "For Snowflake: use SHOW RUNNING QUERIES and "
                "SELECT SYSTEM$CANCEL_ALL_QUERIES(<session_id>)."
            )
            cell = Cell(
                code=code,
                stdout="",
                stderr="",
                error=error_msg,
                description=description,
                estimated_time=estimated_time,
            )
            self.cells.append(cell)
            yield cell
            return
        except Exception as exc:
            cell = Cell(
                code=code,
                stdout="",
                stderr="",
                error=(
                    f"Scratchpad result could not be read: {exc}. "
                    "The scratchpad is still running — you can retry."
                ),
                description=description,
                estimated_time=estimated_time,
            )
            self.cells.append(cell)
            yield cell
            return

        if result_data is None:
            result_data = {
                "stdout": "",
                "stderr": "",
                "error": "Process exited unexpectedly.",
            }

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
        total_timeout: float | None = None,
        inactivity_timeout: float | None = None,
    ):
        """Read stdout until result delimiters; yield progress strings then dict."""
        import time as _time

        s = CoreSettings()
        if total_timeout is None:
            total_timeout = float(s.cell_timeout_default)
        if inactivity_timeout is None:
            inactivity_timeout = float(s.cell_inactivity_timeout)

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
                elapsed_now = _time.monotonic() - start
                if elapsed_now >= total_timeout - 0.5:
                    raise asyncio.TimeoutError(
                        f"Cell timed out after {total_timeout:.0f}s total"
                    ) from None
                raise asyncio.TimeoutError(
                    f"Cell killed after {current_inactivity:.0f}s of inactivity "
                    "(no output or progress() calls)"
                ) from None

            if not raw:
                yield {
                    "stdout": "",
                    "stderr": "",
                    "error": "Process exited unexpectedly.",
                }
                return

            line = raw.decode().rstrip("\r\n")

            if line.startswith(PROGRESS_MARKER):
                current_inactivity = max(
                    current_inactivity, float(s.cell_inactivity_after_progress)
                )
                message = line[len(PROGRESS_MARKER) :].strip()
                yield message
                continue

            if line == RESULT_START:
                in_result = True
                continue
            if line == RESULT_END:
                break
            if in_result:
                lines.append(line)

        raw_text = "\n".join(lines)
        try:
            yield json.loads(raw_text)
        except json.JSONDecodeError:
            try:
                start_idx = raw_text.index("{")
                end_idx = raw_text.rindex("}") + 1
                yield json.loads(raw_text[start_idx:end_idx])
            except (ValueError, json.JSONDecodeError):
                yield {
                    "stdout": raw_text,
                    "stderr": "",
                    "logs": "",
                    "error": "Scratchpad result was malformed (JSON parse failed). "
                    "Output above may be partial.",
                }

    async def install_packages(self, packages: list[str]) -> str:
        if not packages:
            return "No packages specified."
        needed = [p for p in packages if p.lower() not in self._installed_packages]
        if not needed:
            return "All packages already installed."
        self._ensure_venv()

        uv = self._find_uv()
        if uv:
            cmd = [uv, "pip", "install", "--python", self._venv_python, *needed]
        else:
            cmd = [self._venv_python, "-m", "pip", "install", "--no-input", *needed]

        _install_timeout = CoreSettings().cell_install_timeout
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=_install_timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"Install timed out after {_install_timeout}s."
        output = stdout.decode()
        if proc.returncode != 0:
            return f"Install failed (exit {proc.returncode}):\n{output}"
        for p in needed:
            self._installed_packages.add(p.lower())
        return output

    async def _stop_process(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            try:
                self._kill_tree()
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except (ProcessLookupError, asyncio.TimeoutError):
                pass
        if self._proc is not None:
            pipe = self._proc.stdin
            if pipe is not None:
                if hasattr(pipe, "is_closing"):
                    if not pipe.is_closing():
                        pipe.close()
                else:
                    pipe.close()
        self._proc = None
        if self._boot_path is not None:
            try:
                os.unlink(self._boot_path)
            except OSError:
                pass
            self._boot_path = None

    def _kill_tree(self) -> None:
        if self._proc is None or self._proc.returncode is not None:
            return
        pid = self._proc.pid
        if sys.platform != "win32":
            import signal

            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    self._proc.kill()
                except ProcessLookupError:
                    pass
        else:
            self._proc.kill()
