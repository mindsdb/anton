"""Auto-update check for Anton.

Design — the GitHub release tag is the single source of truth.

* The currently-installed Anton version is whatever ``anton.__version__``
  reports.  That constant ships inside the wheel that ``uv tool install`` lays
  down from a specific git tag, so the wheel and the tag cannot drift unless
  the release pipeline itself is broken.  We never parse ``uv tool list``.

* The remote source of truth is GitHub's "latest release" tag.  The decision
  reduces to ``Version(remote_tag) > Version(anton.__version__)``.

* ``uv tool install ... --force`` is treated as atomic — uv stages then
  renames.  Either it returns 0 and we trust the new install on disk, or it
  returns non-zero and we leave the previous install in place.  We never
  inspect partial state.

* Restart-loop safety relies on the ``_ANTON_UPDATED`` env var (defense in
  depth) plus the fact that the re-exec'd process re-reads ``__version__``
  and the comparison short-circuits.

* GitHub is hit at most once per ``_CHECK_TTL``.  An exclusive O_EXCL lock
  file prevents two concurrent ``anton`` processes from running
  ``uv tool install --force`` on the same tool dir.
"""
from __future__ import annotations

import errno
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from packaging.version import InvalidVersion, Version

import anton

_RELEASES_LATEST_URL = "https://api.github.com/repos/mindsdb/anton/releases/latest"
_GITHUB_API_HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "anton-updater",
}

_API_DEADLINE = 5.0           # GitHub call ceiling — protects against Windows DNS hangs
_INSTALL_TIMEOUT = 60.0       # uv tool install budget
_CHECK_TTL = 6 * 60 * 60      # minimum gap between GitHub checks
_LOCK_STALE_AFTER = 10 * 60   # abandoned lock files older than this get reclaimed

_STATE_DIR = Path("~/.anton").expanduser()
_STATE_FILE = _STATE_DIR / "updater_state.json"
_LOCK_FILE = _STATE_DIR / "updater.lock"
_LOG_FILE = _STATE_DIR / "updater.log"


def check_and_update(console, settings) -> bool:
    """Check for a newer Anton release and self-update if available.

    Returns True iff a new version was successfully installed and the caller
    should re-exec the process.
    """
    if settings.disable_autoupdates:
        return False
    if os.environ.get("_ANTON_UPDATED"):
        return False
    if _is_editable_dev_install():
        return False

    uv = _resolve_uv()
    if uv is None:
        return False

    state = _load_state()
    cached_tag = state.get("last_known_tag")
    last_checked = float(state.get("last_check_at", 0) or 0)

    if time.time() - last_checked < _CHECK_TTL:
        latest_tag = cached_tag
    else:
        fetched = _fetch_latest_tag(deadline=_API_DEADLINE)
        if fetched:
            _save_state({"last_check_at": time.time(), "last_known_tag": fetched})
            latest_tag = fetched
        else:
            latest_tag = cached_tag

    if not latest_tag:
        return False

    try:
        local_ver = Version(anton.__version__)
        remote_ver = Version(latest_tag.lstrip("v"))
    except InvalidVersion:
        return False

    if remote_ver.is_prerelease:
        return False
    if remote_ver <= local_ver:
        return False

    with _install_lock() as got_lock:
        if not got_lock:
            return False
        return _install_release(uv, console, latest_tag, local_ver, remote_ver)


def _resolve_uv() -> str | None:
    """Return path to the uv binary, mirroring ``cli._find_uv``."""
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


def _is_editable_dev_install() -> bool:
    """True if anton is being run from a source checkout, not a wheel install.

    A sibling ``.git`` dir or ``pyproject.toml`` next to the package directory
    means we're running from a developer's clone — auto-update would clobber it.
    """
    try:
        pkg_root = Path(anton.__file__).resolve().parent.parent
    except (TypeError, AttributeError):
        return False
    return (pkg_root / ".git").exists() or (pkg_root / "pyproject.toml").exists()


def _load_state() -> dict:
    try:
        with _STATE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(state: dict) -> None:
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _STATE_FILE.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, _STATE_FILE)
    except OSError:
        pass


def _fetch_latest_tag(deadline: float) -> str:
    """Return the latest release tag, or empty string on any failure.

    Runs the HTTP call in a daemon thread so a hung DNS lookup (a real risk
    on Windows) cannot block startup beyond ``deadline`` seconds.
    """
    out: dict[str, str] = {}

    def worker() -> None:
        try:
            req = urllib.request.Request(_RELEASES_LATEST_URL, headers=_GITHUB_API_HEADERS)
            with urllib.request.urlopen(req, timeout=deadline) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            tag = payload.get("tag_name") or ""
            out["tag"] = str(tag).strip()
        except Exception:
            pass

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=deadline)
    return out.get("tag", "")


@contextmanager
def _install_lock():
    """Atomic O_EXCL file lock; yields True if acquired, False otherwise.

    Stale locks (older than ``_LOCK_STALE_AFTER`` seconds) are reclaimed once.
    """
    acquired = False
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            try:
                fd = os.open(str(_LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                with os.fdopen(fd, "w") as f:
                    f.write(f"{os.getpid()} {int(time.time())}\n")
                acquired = True
                break
            except OSError as e:
                if e.errno != errno.EEXIST:
                    break
                try:
                    age = time.time() - _LOCK_FILE.stat().st_mtime
                except OSError:
                    age = 0
                if age > _LOCK_STALE_AFTER and attempt == 0:
                    try:
                        _LOCK_FILE.unlink()
                    except OSError:
                        break
                    continue
                break
        yield acquired
    finally:
        if acquired:
            try:
                _LOCK_FILE.unlink()
            except OSError:
                pass


def _install_release(uv: str, console, tag: str, local_ver: Version, remote_ver: Version) -> bool:
    """Run ``uv tool install`` for the given tag.  Trust uv's exit code."""
    console.print(f"  Updating anton {local_ver} → {remote_ver}...")
    try:
        proc = subprocess.run(
            [uv, "tool", "install", f"git+https://github.com/mindsdb/anton.git@{tag}", "--force"],
            capture_output=True,
            timeout=_INSTALL_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        console.print("  [dim]Update timed out, continuing...[/]")
        return False
    except Exception:
        console.print("  [dim]Update failed, continuing...[/]")
        return False

    if proc.returncode != 0:
        _log_failure(proc)
        console.print("  [dim]Update failed, continuing...[/]")
        return False

    console.print("  ✓ Updated!")
    return True


def _log_failure(proc: subprocess.CompletedProcess) -> None:
    """Persist install failure output for later debugging."""
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        with _LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"--- {time.strftime('%Y-%m-%dT%H:%M:%S')} (rc={proc.returncode}) ---\n")
            if proc.stdout:
                f.write(proc.stdout.decode("utf-8", "replace"))
            if proc.stderr:
                f.write(proc.stderr.decode("utf-8", "replace"))
            f.write("\n")
    except OSError:
        pass
