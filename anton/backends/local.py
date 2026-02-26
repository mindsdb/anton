from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import venv
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from anton.backends.base import ExecutionEvent, ExecutionResult, ScratchpadBackend, ScratchpadRuntime
from anton.backends.protocol import boot_script_source, encode_cell, read_execution_events


def _anton_project_root() -> str:
    # Ensure `import anton.*` works when boot script runs from a temp file.
    # local.py: .../anton/anton/backends/local.py -> project root is parents[3]
    return str(Path(__file__).resolve().parents[3])


@dataclass(slots=True)
class LocalScratchpadRuntime(ScratchpadRuntime):
    """Local runtime: per-scratchpad venv + persistent Python subprocess."""

    name: str
    venvs_base: Path
    env_overrides: Mapping[str, str] = field(default_factory=dict)
    coding_provider: str = "anthropic"
    coding_model: str = ""
    coding_api_key: str = ""

    _proc: asyncio.subprocess.Process | None = field(default=None, init=False, repr=False)
    _boot_path: str | None = field(default=None, init=False, repr=False)

    _venv_dir: str | None = field(default=None, init=False, repr=False)
    _venv_python: str | None = field(default=None, init=False, repr=False)
    _installed_packages: set[str] = field(default_factory=set, init=False, repr=False)

    _MAX_VENV_RETRIES: int = 3

    # ---------------------------------------------------------------------
    # Venv management (moved from old Scratchpad implementation)
    # ---------------------------------------------------------------------

    def _ensure_venv(self) -> None:
        if self._venv_dir is not None and self._verify_venv_python():
            return

        venv_path = self.venvs_base / self.name
        if venv_path.is_dir() and self._try_recycle_venv(venv_path):
            return

        if venv_path.is_dir():
            self._nuke_venv()

        last_error: Exception | None = None
        for _attempt in range(1, self._MAX_VENV_RETRIES + 1):
            try:
                self._create_venv()
                if self._verify_venv_python():
                    self._setup_parent_site_packages()
                    self._save_python_version()
                    return
                raise RuntimeError(f"venv Python binary at {self._venv_python} is not functional")
            except Exception as exc:  # pragma: no cover (platform dependent)
                last_error = exc
                self._nuke_venv()

        raise RuntimeError(
            f"Failed to create a working Python venv after {self._MAX_VENV_RETRIES} attempts. "
            f"Last error: {last_error}. "
            f"Try running: python3 -c 'print(\"ok\")' to verify your Python installation."
        )

    @staticmethod
    def _find_uv() -> str | None:
        uv_bin = shutil.which("uv")
        if uv_bin:
            return uv_bin
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

        self._venv_dir = str(self.venvs_base / self.name)
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
            venv.create(self._venv_dir, system_site_packages=True, with_pip=False, clear=True)

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
        if sys.prefix != sys.base_prefix and self._venv_dir is not None:
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

    # ---------------------------------------------------------------------
    # Runner lifecycle + execution
    # ---------------------------------------------------------------------

    async def start(self) -> None:
        self._ensure_venv()

        boot_code = boot_script_source()
        fd, path = tempfile.mkstemp(suffix=".py", prefix="anton_scratchpad_")
        os.write(fd, boot_code.encode())
        os.close(fd)
        self._boot_path = path

        env = os.environ.copy()
        env.update({k: str(v) for k, v in self.env_overrides.items()})

        if self.coding_model:
            env["ANTON_SCRATCHPAD_MODEL"] = self.coding_model
        if self.coding_provider:
            env["ANTON_SCRATCHPAD_PROVIDER"] = self.coding_provider

        if "ANTHROPIC_API_KEY" not in env and "ANTON_ANTHROPIC_API_KEY" in env:
            env["ANTHROPIC_API_KEY"] = env["ANTON_ANTHROPIC_API_KEY"]
        if "OPENAI_API_KEY" not in env and "ANTON_OPENAI_API_KEY" in env:
            env["OPENAI_API_KEY"] = env["ANTON_OPENAI_API_KEY"]

        if self.coding_api_key:
            sdk_key = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
            }.get(self.coding_provider, "")
            if sdk_key and sdk_key not in env:
                env[sdk_key] = self.coding_api_key

        uv = self._find_uv()
        if uv:
            env["ANTON_UV_PATH"] = uv

        anton_root = _anton_project_root()
        python_path = env.get("PYTHONPATH", "")
        if anton_root not in python_path:
            env["PYTHONPATH"] = anton_root + (os.pathsep + python_path if python_path else "")

        try:
            self._proc = await asyncio.create_subprocess_exec(
                self._venv_python,  # type: ignore[arg-type]
                path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            self._nuke_venv()
            raise RuntimeError(
                f"Failed to start scratchpad: {exc}. "
                "The Python venv has been deleted and will be recreated on next attempt."
            ) from exc

    async def execute_streaming(
        self,
        code: str,
        *,
        estimated_seconds: int = 0,
    ) -> AsyncIterator[ExecutionEvent]:
        if self._proc is None or self._proc.returncode is not None:
            yield ExecutionResult(
                stdout="",
                stderr="",
                logs="",
                error="Scratchpad process is not running. Use reset to restart.",
            )
            return

        self._proc.stdin.write(encode_cell(code))  # type: ignore[union-attr]
        await self._proc.stdin.drain()  # type: ignore[union-attr]

        import anton.scratchpad as scratchpad_module

        total_timeout, inactivity_timeout = scratchpad_module._compute_timeouts(estimated_seconds)

        try:
            last: ExecutionResult | None = None
            async for event in read_execution_events(
                self._proc.stdout.readline,  # type: ignore[union-attr]
                total_timeout=total_timeout,
                inactivity_timeout=inactivity_timeout,
            ):
                if isinstance(event, str):
                    yield event
                else:
                    last = event
        except asyncio.TimeoutError as exc:
            self._proc.kill()
            await self._proc.wait()
            yield ExecutionResult(
                stdout="",
                stderr="",
                logs="",
                error=f"{exc}. Process killed — state lost. Use reset to restart.",
            )
            return

        yield last or ExecutionResult(
            stdout="",
            stderr="",
            logs="",
            error="No result produced.",
        )

    async def _stop_process(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            try:
                self._proc.kill()
                await self._proc.wait()
            except ProcessLookupError:
                pass
        self._proc = None
        if self._boot_path is not None:
            try:
                os.unlink(self._boot_path)
            except OSError:
                pass
            self._boot_path = None

    async def reset(self) -> None:
        await self._stop_process()
        if not self._verify_venv_python():
            self._nuke_venv()
        await self.start()

    async def close(self) -> None:
        await self._stop_process()
        if self._venv_dir is not None:
            self._save_requirements()
            self._venv_dir = None
            self._venv_python = None

    async def remove(self) -> None:
        await self._stop_process()
        self._nuke_venv()

    async def install_packages(self, packages: list[str]) -> str:
        import anton.scratchpad as scratchpad_module

        if not packages:
            return "No packages specified."
        needed = [p for p in packages if p.lower() not in self._installed_packages]
        if not needed:
            return "All packages already installed."

        self._ensure_venv()

        uv = self._find_uv()
        if uv:
            cmd = [uv, "pip", "install", "--python", self._venv_python, *needed]  # type: ignore[list-item]
        else:
            cmd = [self._venv_python, "-m", "pip", "install", "--no-input", *needed]  # type: ignore[list-item]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=getattr(scratchpad_module, "_INSTALL_TIMEOUT", 120))
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"Install timed out after {getattr(scratchpad_module, '_INSTALL_TIMEOUT', 120)}s."

        output = stdout.decode()
        if proc.returncode != 0:
            return f"Install failed (exit {proc.returncode}):\n{output}"
        for p in needed:
            self._installed_packages.add(p.lower())
        return output


class LocalScratchpadBackend(ScratchpadBackend):
    def __init__(self, *, venvs_base: Path | None = None) -> None:
        self._venvs_base = venvs_base

    async def create_runtime(
        self,
        name: str,
        *,
        workspace_path: Path | None,
        env: Mapping[str, str],
        coding_provider: str,
        coding_model: str,
        coding_api_key: str,
    ) -> ScratchpadRuntime:
        venvs_base = self._venvs_base or (
            (workspace_path / ".anton" / "scratchpad-venvs")
            if workspace_path is not None
            else Path("~/.anton/scratchpad-venvs").expanduser()
        )
        return LocalScratchpadRuntime(
            name=name,
            venvs_base=venvs_base,
            env_overrides=env,
            coding_provider=coding_provider,
            coding_model=coding_model,
            coding_api_key=coding_api_key,
        )

    def probe_packages(self) -> list[str]:
        from importlib.metadata import distributions

        return sorted({d.metadata["Name"] for d in distributions()})

