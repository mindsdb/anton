"""Install-mode-aware dependency check for the optional [server] extra.

Anton is typically installed via `uv tool install`, which gives it an
isolated environment that `uv pip install` cannot reach. This helper
detects the active install mode and routes the install command correctly:

- uv tool install  → `uv tool install --with fastapi --with uvicorn --upgrade anton`
- uv pip / venv    → `uv pip install --python <sys.executable> fastapi uvicorn`
- plain pip        → fallback printed instructions

Mirrors the UX of cli._ensure_dependencies: prompt, install, re-exec.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm

_REQUIRED = ("fastapi", "uvicorn")
_INSTALL_SPECS = ("fastapi>=0.100", "uvicorn>=0.20")


def _missing() -> list[str]:
    """Return the install specs for any missing server packages."""
    import importlib

    out: list[str] = []
    for mod, spec in zip(_REQUIRED, _INSTALL_SPECS):
        try:
            importlib.import_module(mod)
        except ImportError:
            out.append(spec)
    return out


def _find_uv() -> str | None:
    """Find the uv binary on PATH or in conventional locations."""
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
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _is_uv_tool_install() -> bool:
    """True if anton is running from a `uv tool install` environment.

    `uv tool` puts each tool under {data_home}/uv/tools/{tool}/, so
    sys.executable will be inside that directory.
    """
    data_home = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local/share")
    uv_tool_root = Path(data_home) / "uv" / "tools"
    try:
        Path(sys.executable).resolve().relative_to(uv_tool_root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _reexec() -> None:
    """Re-execute the anton binary so the new packages are picked up."""
    binary = shutil.which("anton") or sys.argv[0]
    os.execv(binary, [binary] + sys.argv[1:])


def ensure_server_deps(console: Console) -> None:
    """Verify [server] extras are installed; offer to install if not.

    On confirmed install, re-execs the process. On decline (or no uv),
    prints the right command for the detected install mode and exits.
    """
    missing = _missing()
    if not missing:
        return

    console.print()
    console.print("[anton.warning]Server packages are required to run `anton serve`:[/]")
    for pkg in missing:
        console.print(f"  [bold]- {pkg}[/]")
    console.print()

    uv = _find_uv()
    is_tool = _is_uv_tool_install()

    if uv and is_tool:
        cmd = [uv, "tool", "install", "--with", "fastapi", "--with", "uvicorn", "--upgrade", "anton"]
        manual = "uv tool install --with fastapi --with uvicorn --upgrade anton"
    elif uv:
        cmd = [uv, "pip", "install", "--python", sys.executable, *missing]
        manual = f"uv pip install {' '.join(missing)}"
    else:
        cmd = None
        manual = "pip install 'anton[server]'"

    if cmd is None:
        console.print("To install, run:")
        console.print(f"  [bold]{manual}[/]")
        console.print()
        sys.exit(1)

    if not Confirm.ask("Install with uv?", default=True, console=console):
        console.print()
        console.print("To install manually, run:")
        console.print(f"  [bold]{manual}[/]")
        console.print()
        sys.exit(1)

    console.print(f"[anton.muted]  Running: {' '.join(cmd)}[/]")
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        console.print("[anton.error]  Install failed:[/]")
        console.print(result.stderr.decode() if result.stderr else result.stdout.decode())
        console.print()
        console.print("To install manually, run:")
        console.print(f"  [bold]{manual}[/]")
        sys.exit(1)

    console.print("[anton.success]  Server packages installed.[/]")
    _reexec()
