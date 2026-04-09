"""Auto-update check for Anton."""

from __future__ import annotations

import shutil
import subprocess
import threading


_TOTAL_TIMEOUT = 10  # Hard ceiling — update check never blocks startup longer than this


def check_and_update(console, settings) -> bool:
    """Check for a newer version of Anton and self-update if available.

    Runs in a thread with a hard timeout so it never blocks startup,
    even if DNS resolution or network calls hang on Windows.

    Returns True if an update was applied and the process should restart.
    """
    if settings.disable_autoupdates:
        return False

    result: dict = {}

    def _worker():
        try:
            _check_and_update(result, settings)
        except Exception:
            pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=_TOTAL_TIMEOUT)

    if t.is_alive():
        # Deadline exceeded — the upgrade may still be running in the background.
        # Discard the result so we never restart with partially-replaced files on disk.
        return False

    # Print messages collected by the worker (if it finished)
    for msg in result.get("messages", []):
        console.print(msg)

    return "new_version" in result


def _check_and_update(result: dict, settings) -> None:
    messages: list[str] = []
    result["messages"] = messages

    if shutil.which("uv") is None:
        return

    # Fetch latest release tag from GitHub API
    import json
    import urllib.request

    url = "https://api.github.com/repos/mindsdb/anton/releases/latest"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return

    latest_tag = data.get("tag_name", "")
    if not latest_tag:
        return

    # Strip leading 'v' for version comparison (e.g. "v0.3.1" -> "0.3.1")
    remote_version_str = latest_tag.lstrip("v")

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

    # Newer version available — reinstall from the specific release tag
    messages.append(f"  Updating anton {local_ver} \u2192 {remote_ver}...")

    try:
        proc = subprocess.run(
            ["uv", "tool", "install", f"git+https://github.com/mindsdb/anton.git@{latest_tag}", "--force"],
            capture_output=True,
            timeout=_TOTAL_TIMEOUT,
        )
    except Exception:
        messages.append("  [dim]Update failed, continuing...[/]")
        return

    if proc.returncode != 0:
        messages.append("  [dim]Update failed, continuing...[/]")
        return

    # Verify the upgrade actually installed the newer version by checking
    # what uv reports. If PyPI hasn't published the version yet, uv will
    # reinstall the old one and we'd loop forever on restart.
    try:
        verify = subprocess.run(
            ["uv", "tool", "list"],
            capture_output=True,
            timeout=5,
        )
        tool_list = verify.stdout.decode()
        # Look for "anton X.Y.Z" in the output
        installed_match = re.search(r"anton\s+(\S+)", tool_list)
        if installed_match:
            installed_ver = Version(installed_match.group(1))
            if installed_ver < remote_ver:
                # Upgrade didn't actually install the new version — skip restart
                return
    except Exception:
        pass

    messages.append("  \u2713 Updated!")
    result["new_version"] = remote_version_str
