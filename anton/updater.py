"""Auto-update check for Anton."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import urllib.request
from pathlib import Path


_TOTAL_TIMEOUT = 10  # Hard ceiling — update check never blocks startup longer than this

# Records the last release tag whose force-reinstall did NOT take (the installed
# version never matched the tag). A fresh terminal would otherwise re-run
# `uv tool install --force` for that same doomed tag on every launch — which can
# corrupt the running tool env (partial rich/typer, or the package vanishing),
# especially on Windows (ENG-655). We suppress re-attempting the SAME tag until a
# newer one appears.
_SKIP_MARKER = Path("~/.anton/.update_skip_tag").expanduser()

_RELEASES_LATEST_URL = "https://api.github.com/repos/mindsdb/anton/releases/latest"
_GITHUB_API_HEADERS = {"Accept": "application/vnd.github+json"}


def check_and_update(console, settings) -> bool:
    """Check for a newer version of Anton and self-update if available.

    Runs in a thread with a hard timeout so it never blocks startup,
    even if DNS resolution or network calls hang on Windows.

    Returns True if an update was applied and the process should restart.
    """
    import os

    if settings.disable_autoupdates:
        return False

    # Guard against infinite restart loops.  _reexec() sets this before
    # replacing the process; the new process inherits it and skips the check.
    if os.environ.get("_ANTON_UPDATED"):
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

    latest_tag = _fetch_latest_release_tag()
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

    # Anti-thrash: if a previous run already force-reinstalled this exact tag and
    # it didn't take (installed version never matched — e.g. a tag/metadata
    # mismatch), don't reinstall it again. Re-running `uv tool install --force`
    # of the running tool every launch is what corrupts the env (ENG-655); wait
    # for a genuinely newer tag instead.
    try:
        if _SKIP_MARKER.is_file() and _SKIP_MARKER.read_text().strip() == latest_tag:
            return
    except Exception:
        pass

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

    # Verify the upgrade actually installed the exact release version.
    # If the release tag and the installed package version diverge, do not
    # restart into a possibly confusing mixed state.
    installed_ver = _read_installed_anton_version()
    if installed_ver is None:
        messages.append("  [dim]Update could not be verified, continuing...[/]")
        return
    if installed_ver != remote_ver:
        # Remember this tag so we don't force-reinstall it again next launch
        # (see _SKIP_MARKER) \u2014 that repeated reinstall is what bricks installs.
        try:
            _SKIP_MARKER.parent.mkdir(parents=True, exist_ok=True)
            _SKIP_MARKER.write_text(latest_tag)
        except Exception:
            pass
        messages.append(
            "  [dim]Update skipped: installed Anton version does not match the latest release tag.[/]"
        )
        return

    # Success \u2014 clear any stale skip marker so a future re-install isn't blocked.
    try:
        _SKIP_MARKER.unlink(missing_ok=True)
    except Exception:
        pass
    messages.append("  \u2713 Updated!")
    result["new_version"] = remote_version_str


def _fetch_latest_release_tag() -> str:
    """Return the latest GitHub release tag, or an empty string on failure."""
    try:
        req = urllib.request.Request(_RELEASES_LATEST_URL, headers=_GITHUB_API_HEADERS)
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return ""
    tag = data.get("tag_name", "")
    return str(tag).strip()


def _read_installed_anton_version():
    """Return Anton's installed uv tool version, or None if unreadable."""
    from packaging.version import InvalidVersion, Version

    try:
        verify = subprocess.run(
            ["uv", "tool", "list"],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        return None

    if verify.returncode != 0:
        return None

    tool_list = verify.stdout.decode()
    # Match the `anton-agent` tool specifically (line-anchored). A bare
    # `anton\s+` also matches a leftover legacy `anton` tool (from before the
    # package was renamed anton -> anton-agent) — which is listed first, so the
    # verification read the WRONG tool's version and "Update skipped" fired
    # forever even after a correct install (ENG-655, confirmed on a real
    # machine that had both `anton` and `anton-agent` uv tools).
    installed_match = re.search(r"^anton-agent\s+v?(\S+)", tool_list, re.MULTILINE)
    if not installed_match:
        return None

    try:
        return Version(installed_match.group(1))
    except InvalidVersion:
        return None
