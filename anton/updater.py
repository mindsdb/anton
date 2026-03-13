"""Auto-update check for Anton."""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time


_TOTAL_TIMEOUT = 10  # Hard ceiling — update check never blocks startup longer than this


def check_and_update(console, settings) -> bool:
    """Check for a newer version of Anton and self-update if available.

    Runs in a thread with a hard timeout so it never blocks startup,
    even if DNS resolution or network calls hang on Windows.
    """
    if settings.disable_autoupdates:
        return False

    result: dict = {}
    deadline = time.monotonic() + _TOTAL_TIMEOUT

    def _worker():
        try:
            _check_and_update(result, settings, deadline=deadline)
        except Exception:
            pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=_TOTAL_TIMEOUT)

    # Print messages collected by the worker (if it finished)
    for msg in result.get("messages", []):
        console.print(msg)

    # Update in-memory version so the banner shows the new version
    if "new_version" in result:
        import anton
        anton.__version__ = result["new_version"]

    return "new_version" in result


def _remaining_time(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _check_and_update(result: dict, settings, *, deadline: float | None = None) -> None:
    messages: list[str] = []
    result["messages"] = messages

    if shutil.which("uv") is None:
        return

    # Fetch remote __init__.py to get __version__
    import urllib.request

    url = "https://raw.githubusercontent.com/mindsdb/anton/main/anton/__init__.py"
    try:
        req = urllib.request.Request(url)
        timeout = min(2.0, _remaining_time(deadline)) if deadline is not None else 2.0
        if timeout <= 0:
            return
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read().decode("utf-8")
    except Exception:
        return

    # Parse remote version
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        return
    remote_version_str = match.group(1)

    # Compare versions
    from packaging.version import InvalidVersion, Version

    import anton

    try:
        local_ver = Version(anton.__version__)
        remote_ver = Version(remote_version_str)
    except InvalidVersion:
        return

    if remote_ver <= local_ver:
        return

    # Newer version available — upgrade
    messages.append(f"  Updating anton {local_ver} \u2192 {remote_ver}...")

    timeout = _remaining_time(deadline) if deadline is not None else 15.0
    if timeout <= 0:
        return

    try:
        proc = subprocess.run(
            ["uv", "tool", "upgrade", "anton"],
            capture_output=True,
            timeout=timeout,
        )
    except Exception:
        messages.append("  [dim]Update failed, continuing...[/]")
        return

    if proc.returncode != 0:
        messages.append("  [dim]Update failed, continuing...[/]")
        return

    messages.append("  \u2713 Updated!")
    result["new_version"] = remote_version_str
