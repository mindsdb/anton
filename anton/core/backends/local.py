"""LocalScratchpadRuntime — venv-based scratchpad for the CLI."""

from __future__ import annotations

import asyncio
import json
import os
import hashlib
import re
import shutil
import sys
import tempfile
import venv
from pathlib import Path

from anton.core.backends.base import Cell, ScratchpadRuntime
from anton.core.backends.wire import (
    CELL_DELIM,
    HEARTBEAT_MARKER,
    INSTALL_END_MARKER,
    INSTALL_START_MARKER,
    PROGRESS_MARKER,
    RESULT_END,
    RESULT_START,
    STDOUT_CHUNK_MARKER,
)
from anton.core.settings import CoreSettings
from anton.core.backends.utils import compute_timeouts

_BOOT_SCRIPT_PATH = Path(__file__).parent / "scratchpad_boot.py"

# Bound on accumulated salvage chunks per cell — mirrors the boot script's
# _MAX_OUTPUT so a killed cell can never report more stdout than a successful
# one would have.
_SALVAGE_MAX = 10_000

# The liveness heartbeat (ENG-578) proves a quiet cell is alive but not that
# it's progressing, so a wedged cell stays silent — for as long as
# cell_total_max — unless we say something. First notice no earlier than
# this many seconds of silence, then no more often than this (ENG-1324).
_QUIET_NOTICE_AFTER = 60.0
_QUIET_NOTICE_EVERY = 60.0

# Extra headroom on top of cell_install_timeout while an in-cell auto-install
# runs: the worker enforces the budget itself and reports a named install
# error, so the parent's windows must outlast the worker's timer for that
# error to win the race against a generic kill (ENG-1275).
_INSTALL_GRACE = 30.0


def _read_boot_script() -> str:
    """Read the boot script as UTF-8 explicitly.

    It contains non-ASCII (…, —), so a host-locale-default read (e.g. GBK on
    Chinese Windows) crashes with a codec error before the scratchpad can start
    (ENG-824). Kept as a helper so the explicit encoding is pinned by a test.
    """
    return _BOOT_SCRIPT_PATH.read_text(encoding="utf-8")


def _encode_cell_payload(payload: str) -> bytes:
    """Encode the cell payload sent to the scratchpad subprocess.

    ``errors="surrogateescape"`` rather than a strict UTF-8 encode: model-written
    cell code can embed a filesystem path that ``os.fsdecode`` surrogate-escaped
    when it couldn't decode the path bytes on a non-UTF-8 host (e.g. a pt-BR
    ``Área de Trabalho`` or an emoji path on Windows → lone surrogates in
    U+DC80..U+DCFF). A strict encode raises ``UnicodeEncodeError: surrogates not
    allowed`` here and takes down the whole session before the code reaches the
    subprocess — the encode-side sibling of ENG-824's decode crash (ENG-940).

    ``surrogateescape`` (not ``surrogatepass``) is deliberate: it's the inverse
    of the ``os.fsdecode`` that created these surrogates, so it restores the
    original path bytes, and it matches how the subprocess — which always runs
    in UTF-8 mode (``_utf8_env``) and so decodes stdin with ``surrogateescape``
    — reads them back. ``surrogatepass`` would emit the 3-byte CESU form that the
    subprocess's ``surrogateescape`` decode then re-mangles, so the path would
    not survive intact.
    """
    return payload.encode("utf-8", errors="surrogateescape")


def _utf8_env(base: "os._Environ[str] | dict[str, str]") -> dict[str, str]:
    """A copy of ``base`` with Python UTF-8 mode forced for the scratchpad
    subprocess.

    The parent may run under a non-UTF-8 host locale (e.g. GBK/cp936 on
    Chinese Windows). Without this, the child interpreter inherits that code
    page and every ``open()``/``print()``/stdio defaults to it — so reading the
    boot script or emitting non-ASCII output crashes (ENG-824). ``setdefault``
    so an explicit operator override still wins.

    Only ``PYTHONUTF8`` is set: UTF-8 mode already makes open()/filesystem/stdio
    UTF-8 with the lenient ``surrogateescape`` stdio handler. We deliberately do
    NOT also set ``PYTHONIOENCODING`` — a bare value downgrades the stdio error
    handler back to ``strict`` (verified), which would re-introduce a crash on
    exotic output rather than round-tripping it.
    """
    env = dict(base)
    env.setdefault("PYTHONUTF8", "1")
    return env


_MAX_OUTPUT = 10_000


# ── Namespace-snapshot layout (ENG-1124) ────────────────────────────────────
# One derivation, shared by the writer (the runtime) and the single-scratchpad
# guard (which needs to know which pads a conversation has already used, across
# turns). Keeping it in one place is the point: if these two disagreed, the guard
# would silently stop firing again.

def default_venvs_base(workspace_path: Path | None) -> Path:
    """Where a workspace's scratchpad venvs live."""
    if workspace_path is not None:
        return workspace_path / ".anton" / "scratchpad-venvs"
    return Path("~/.anton/scratchpad-venvs").expanduser()


def _safe_segment(value: str, fallback: str) -> str:
    """A path-safe version of a model-chosen string.

    Deliberately NOT case-folded or separator-collapsed — that would change which
    pads share state, and belongs with the venv-path normalisation in ENG-1133.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", value or "").strip("._") or fallback


def snapshot_dir(venvs_base: Path, session_id: str | None) -> Path | None:
    """Namespace-snapshot directory for one conversation, or None if it must not persist.

    Requires a session id, and requires that id to be path-safe as supplied. There is
    deliberately NO shared fallback bucket:

    * A shared bucket is a confidentiality boundary, not a convenience. Cowork's
      transient `CredentialProbe` builds a `ChatSession` with **no** session id and
      parses `DS_*` datasource credentials in the scratchpad — and
      `ANTON_SCRATCHPAD_PERSIST_SESSION` is process-global, so a probe inherits it once
      any normal chat has switched it on. With a shared bucket those credentials land on
      disk under a predictable path and a later probe reusing the pad name reloads them.
    * `_safe_segment` is not injective, so `tenant/a` and `tenant_a` would resolve to one
      directory. Rather than transform the id (which would break the path cowork-server
      computes when it prunes), refuse anything that is not already path-safe. A UUID —
      what every real host passes — is unchanged, so this enforces the cross-repo
      invariant instead of merely documenting it.

    The cost is that bare CLI use gets no cross-process persistence. That is the right
    trade: the CLI is one long-lived process, so its namespace lives in memory anyway.
    """
    if not session_id:
        return None
    if _safe_segment(session_id, "") != session_id:
        return None
    return venvs_base.parent / "scratchpad-sessions" / session_id


def _pad_filename(pad_name: str) -> str:
    """An injective, length-bounded filename for a pad.

    `_safe_segment` alone is NOT injective — it maps every unsafe character to `_`, so
    `'my pad'`, `'my_pad'` and `'my/pad'` all collapse to `my_pad` and would share one
    snapshot, meaning one pad loads another's namespace. Appending a digest of the
    ORIGINAL name keeps distinct pads distinct. Also truncated, because a pad name is
    model-chosen and most filesystems cap a path component at 255 bytes — an
    over-long name would fail the write instead of saving state.
    """
    stem = _safe_segment(pad_name, "scratchpad")[:80]
    digest = hashlib.sha1((pad_name or "").encode("utf-8")).hexdigest()[:8]
    return f"{stem}-{digest}.pkl"


def snapshot_file(venvs_base: Path, session_id: str | None, pad_name: str) -> Path | None:
    """Snapshot path for one pad, or None if it must not or cannot be persisted."""
    base = snapshot_dir(venvs_base, session_id)
    if base is None:
        return None
    path = base / _pad_filename(pad_name)
    # Belt for the sanitiser: never hand back a path outside the snapshot root.
    try:
        path.resolve().relative_to(base.resolve())
    except (ValueError, OSError):
        return None
    return path


class LocalScratchpadRuntime(ScratchpadRuntime):
    """Runs scratchpad cells in a persistent per-named venv subprocess."""

    _MAX_VENV_RETRIES = 3
    # ENG-1273: how many times in a row execute_streaming() will silently
    # resume() a dead process before concluding resume() itself isn't the
    # fix (most likely the snapshot it keeps reloading) and falling back to
    # a full reset() instead of retrying resume() forever.
    _MAX_CONSECUTIVE_AUTO_RESUMES = 2

    def __init__(
        self,
        name: str,
        *,
        coding_provider: str,
        coding_model: str,
        coding_api_key: str,
        coding_base_url: str,
        cells: list[Cell] | None = None,
        workspace_path: Path | None = None,
        session_id: str | None = None,
        _venvs_base: Path | None = None,
    ) -> None:
        super().__init__(
            name,
            coding_provider=coding_provider,
            coding_model=coding_model,
            coding_api_key=coding_api_key,
            coding_base_url=coding_base_url,
            cells=cells,
            workspace_path=workspace_path,
        )
        # `_workspace_path` on the base class falls back to
        # `~/.anton` when no path was passed. We need the explicit
        # arg (None when omitted) to decide whether to pin the
        # subprocess cwd to a real project root vs. inheriting the
        # parent's cwd. Don't conflate the two — the base attribute
        # is a "where to put scratchpad venvs" hint; the explicit
        # arg is "the agent's project, when known".
        self._explicit_workspace_path: Path | None = workspace_path
        # Conversation id when the host supplies one (cowork-server passes the
        # conversation's UUID as `session_id`). Only used to scope the namespace
        # snapshot — see `_session_snapshot_path`.
        self._session_id: str | None = session_id
        self._proc: asyncio.subprocess.Process | None = None
        self._boot_path: str | None = None
        self._venv_dir: str | None = None
        self._venv_python: str | None = None
        # recovery bookkeeping for a process that died on its own
        # (watchdog kill, crash) — see resume() / _auto_resume(). The death
        # counter is zeroed by reset(): whatever was wrong before, a reset is
        # a clean slate. The recovery note and error are preserved separately
        # for Task 2 to consume and report to the UI.
        self._consecutive_deaths: int = 0
        self._pending_recovery_note: str | None = None
        self._last_resume_error: str | None = None
        self._last_verify_error: str | None = None
        self._venvs_base = (
            _venvs_base if _venvs_base is not None else default_venvs_base(workspace_path)
        )

    def _session_snapshot_path(self, *, create: bool = False) -> Path | None:
        """Where this pad's namespace snapshot lives, or None if it can't be written.

        Scoped per conversation *and* per pad. Both matter: without the pad segment two
        named scratchpads would overwrite each other, and without the conversation
        segment two conversations in the same workspace that happen to use the same pad
        name would read each other's variables — a correctness bug and a confidentiality
        one.

        Returns None when there is nowhere safe to write: no session id, or one that is
        not already path-safe. There is deliberately no shared fallback bucket — see
        `snapshot_dir` for why (the unscoped `CredentialProbe` would leave `DS_*`
        credentials in it). Bare CLI use and tests therefore get no snapshot at all,
        keeping their namespace in memory as before.

        Also returns None when the directory cannot be created; the caller then leaves
        ANTON_SCRATCHPAD_SESSION_PATH unset so the failure is reported rather than silent.
        """
        path = snapshot_file(self._venvs_base, self._session_id, self.name)
        if path is None or not create:
            return path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        return path

    def _snapshot_configured(self) -> bool:
        """Whether this pad's namespace snapshot will actually be read or
        written for THIS run — both conditions must hold: a session id
        that resolves to a writable path (see `_session_snapshot_path`),
        and persistence turned on for the child, read the same way
        `scratchpad_boot.py` reads `ANTON_SCRATCHPAD_PERSIST_SESSION`.

        Used to keep the kill message honest (ENG-1273): claiming a
        restoration that cannot happen — e.g. bare CLI use with no session
        id — is the same false contract this ticket exists to fix, just
        pointed the other way.
        """
        if self._session_snapshot_path() is None:
            return False
        return os.environ.get(
            "ANTON_SCRATCHPAD_PERSIST_SESSION", "false"
        ).lower() in {"1", "true", "yes", "on"}

    def _discard_session_snapshot(self) -> None:
        """Delete this pad's namespace snapshot, if any. Best-effort."""
        path = self._session_snapshot_path()
        if path is None:
            return
        candidates = [path]
        # The writer suffixes its temp file with a pid, so match any of them.
        try:
            candidates += sorted(path.parent.glob(f"{path.name}.*.tmp"))
        except OSError:
            pass
        for candidate in candidates:
            try:
                candidate.unlink()
            except OSError:
                pass

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
                detail = f" ({self._last_verify_error})" if self._last_verify_error else ""
                raise RuntimeError(
                    f"venv Python binary at {self._venv_python} is not functional{detail}"
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
            local_app_data = os.environ.get("LOCALAPPDATA", "")
            candidates = (
                os.path.expanduser("~/.local/bin/uv.exe"),
                os.path.expanduser("~/.cargo/bin/uv.exe"),
                os.path.expanduser("~/scoop/shims/uv.exe"),
                os.path.join(local_app_data, "Microsoft", "WinGet", "Links", "uv.exe"),
            )
        else:
            # Package-manager locations a GUI-launched parent's PATH may miss.
            # Keep in sync with cowork's uv-paths.ts.
            candidates = (
                os.path.expanduser("~/.local/bin/uv"),
                os.path.expanduser("~/.cargo/bin/uv"),
                "/opt/homebrew/bin/uv",
                "/usr/local/bin/uv",
                "/opt/local/bin/uv",
                "/home/linuxbrew/.linuxbrew/bin/uv",
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
            try:
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
            except (_sp.CalledProcessError, _sp.TimeoutExpired) as exc:
                # CalledProcessError/TimeoutExpired.__str__ omits the captured
                # stderr, so uv's actual reason (bad --python, disk full) was
                # never seen — only "returned non-zero exit status N".
                stderr = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"uv venv failed: {stderr}" if stderr else str(exc)) from exc
        else:
            # symlinks=False is venv.create()'s own default on every platform
            # (only the `python -m venv` CLI defaults it per-OS); a copied
            # macOS Python binary loses its @rpath and crashes on launch.
            venv.create(
                self._venv_dir,
                system_site_packages=True,
                with_pip=False,
                clear=True,
                symlinks=sys.platform != "win32",
            )

        if sys.platform == "win32":
            bin_dir = os.path.join(self._venv_dir, "Scripts")
            self._venv_python = os.path.join(bin_dir, "python.exe")
            self._add_windows_firewall_rule()
        else:
            bin_dir = os.path.join(self._venv_dir, "bin")
            self._venv_python = os.path.join(bin_dir, "python")

    def venv_python(self) -> str | None:
        """Public accessor for the scratchpad's Python interpreter path.

        Returns None when the venv has not been provisioned yet (i.e.
        no exec has run). Auxiliary tools that want to share installed
        packages call this to discover the interpreter.
        """
        if self._venv_python and os.path.isfile(self._venv_python):
            return self._venv_python
        return None

    def ensure_venv(self) -> str | None:
        """Provision the venv on disk (recycle if present, create if not) and
        return its python interpreter path.

        Public counterpart to the internal `_ensure_venv` used by `start()`
        and `install_packages`. Exposed for callers that need only the venv
        — not the full runtime sidecar — to spawn auxiliary processes
        (e.g. cowork's artifact backend relaunch). Cheap when the venv
        already exists; falls back to a fresh `uv venv` / `python -m venv`
        otherwise.
        """
        self._ensure_venv()
        return self.venv_python()

    def _verify_venv_python(self) -> bool:
        self._last_verify_error = None
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
            ok = result.returncode == 0 and "ok" in result.stdout.decode("utf-8", errors="replace")
            if not ok:
                stderr = result.stderr.decode("utf-8", errors="replace").strip()
                self._last_verify_error = f"exit {result.returncode}" + (f": {stderr}" if stderr else "")
            return ok
        except Exception as exc:
            self._last_verify_error = str(exc)
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
                # UTF-8 explicitly: a plain open() encodes with the host locale
                # (e.g. GBK on Chinese Windows), but the child reads .pth files
                # as UTF-8 under UTF-8 mode — a mismatch corrupts non-ASCII
                # site-packages paths (same class of bug as the boot script,
                # ENG-824).
                with open(pth_path, "w", encoding="utf-8") as f:
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
            with open(req_path, "w", encoding="utf-8") as f:
                for pkg in sorted(self._installed_packages):
                    f.write(pkg + "\n")
        except OSError:
            pass

    def _load_requirements(self) -> None:
        if not self._venv_dir:
            return
        req_path = os.path.join(self._venv_dir, "requirements.txt")
        try:
            with open(req_path, encoding="utf-8") as f:
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
            with open(ver_path, "w", encoding="utf-8") as f:
                f.write(f"{sys.version_info.major}.{sys.version_info.minor}\n")
        except OSError:
            pass

    def _check_python_version(self) -> bool:
        if not self._venv_dir:
            return False
        ver_path = os.path.join(self._venv_dir, ".python_version")
        try:
            with open(ver_path, encoding="utf-8") as f:
                saved = f.read().strip()
            expected = f"{sys.version_info.major}.{sys.version_info.minor}"
            return saved == expected
        except FileNotFoundError:
            return False

    async def start(self) -> None:
        """Write the boot script to a temp file and launch the subprocess."""
        self._ensure_venv()

        boot_code = _read_boot_script()
        fd, path = tempfile.mkstemp(suffix=".py", prefix="anton_scratchpad_")
        os.write(fd, boot_code.encode("utf-8"))
        os.close(fd)
        self._boot_path = path

        # Force UTF-8 in the child (ENG-824).
        env = _utf8_env(os.environ)
        if self._coding_model:
            env["ANTON_SCRATCHPAD_MODEL"] = self._coding_model
        if self._coding_provider:
            env["ANTON_SCRATCHPAD_PROVIDER"] = self._coding_provider
        # Propagate provider credentials from the ANTON_* names into the SDK
        # names the scratchpad's nested get_llm() expects.
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
            # Host-aware (ENG-436): api.mindshub.ai serves /v1, legacy
            # mdb.ai serves /api/v1. The previous hardcoded /api/v1 was
            # wrong for mindshub. Mirrors config/settings.py +
            # cowork-server minds_chat_base_url.
            _minds_base = env["ANTON_MINDS_URL"].rstrip("/")
            if _minds_base.endswith("/v1"):
                env["OPENAI_BASE_URL"] = _minds_base
            elif "mdb.ai" in _minds_base:
                env["OPENAI_BASE_URL"] = f"{_minds_base}/api/v1"
            else:
                env["OPENAI_BASE_URL"] = f"{_minds_base}/v1"
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

        # scratchpad_boot no longer auto-installs on ModuleNotFoundError, so
        # this var (and _read_result's install-span handling below) is dead;
        # left wired rather than reworking the kill-window logic in one pass.
        env["ANTON_CELL_INSTALL_TIMEOUT"] = str(CoreSettings().cell_install_timeout)

        # Namespace snapshot path (ENG-1124). The boot script reads
        # ANTON_SCRATCHPAD_SESSION_PATH and nothing ever set it, so it fell back to a
        # hardcoded "/anton_scratchpad_session.pkl" — the filesystem root, which no
        # Cowork process can write to. Every save failed and every failure was
        # discarded, so state never survived a turn. This is the only place that knows
        # the pad name, so it is where the path is composed.
        snapshot = self._session_snapshot_path(create=True)
        if snapshot is not None:
            env["ANTON_SCRATCHPAD_SESSION_PATH"] = str(snapshot)
        else:
            # Leaving it unset makes the boot script *report* that state will not
            # persist, rather than silently pretending it does.
            env.pop("ANTON_SCRATCHPAD_SESSION_PATH", None)

        # The workspace root, for resolving agent-authored modules when the snapshot
        # is loaded (ENG-1366). Python puts the *script's* directory on `sys.path`,
        # never the cwd, so a helper the agent wrote into the workspace and imported
        # on turn 1 (via its own `sys.path.insert`) is unimportable in the fresh
        # process that loads the snapshot — and dill stores those objects by
        # reference to their defining module, so the import failure discarded the
        # whole namespace.
        #
        # Set ONLY from the explicit workspace arg, never from `os.getcwd()`: with no
        # workspace bound the cwd is whatever launched the parent server (cowork's own
        # install directory), which must never go on the scratchpad's `sys.path`.
        # Persistence is process-global (`ANTON_SCRATCHPAD_PERSIST_SESSION`), so the
        # loader is reachable from contexts that never opted in — the credential probe
        # among them — and this keeps it inert for all of them.
        if self._explicit_workspace_path is not None:
            env["ANTON_SCRATCHPAD_WORKSPACE_PATH"] = str(
                Path(self._explicit_workspace_path).resolve()
            )
        else:
            env.pop("ANTON_SCRATCHPAD_WORKSPACE_PATH", None)

        _anton_root = str(Path(__file__).resolve().parent.parent.parent.parent)
        python_path = env.get("PYTHONPATH", "")
        if _anton_root not in python_path:
            env["PYTHONPATH"] = _anton_root + (
                os.pathsep + python_path if python_path else ""
            )

        # Pin the subprocess cwd to the workspace root so bare-relative
        # paths in scratchpad code (`open("data.csv")`, `os.listdir(".")`,
        # `subprocess.run(["git", "status"])`) operate on the project,
        # not whatever directory the parent server happened to launch
        # from. Falls back to inheriting the parent's cwd when no
        # workspace is bound (older CLI flows + tests that pass
        # `workspace_path=None`) — we use the EXPLICIT constructor
        # arg here, not `_workspace_path`, because the latter has a
        # `~/.anton` fallback that we never want to cd into. Env
        # vars and absolute-path APIs (data vault, get_llm,
        # create_artifact-returned paths) are cwd-independent, so
        # this only changes the relative-IO surface.
        proc_cwd = (
            str(self._explicit_workspace_path)
            if self._explicit_workspace_path is not None
            else None
        )
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self._venv_python,
                path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=proc_cwd,
                start_new_session=(sys.platform != "win32"),
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            self._nuke_venv()
            raise RuntimeError(
                f"Failed to start scratchpad: {exc}. "
                "The Python venv has been deleted and will be recreated on next attempt."
            ) from exc

    async def resume(self) -> None:
        """Restart a dead process WITHOUT discarding its namespace snapshot.

        Recovery from something the agent did not ask for — a watchdog kill, a
        crash — so the last COMPLETED cell's state (dumped to disk after every
        cell, ENG-1124) is exactly what should come back. `start()` below
        already reloads the snapshot whenever ANTON_SCRATCHPAD_SESSION_PATH
        points at one, so there is no separate load path to write here — the
        only thing this adds over a bare restart is NOT calling
        `_discard_session_snapshot()` first, which is what `reset()` does.

        Called automatically by `execute_streaming()` when it finds the
        process dead (see `_auto_resume`) — the agent never has to ask for
        this explicitly.
        """
        await self._stop_process()
        await self.start()

    async def reset(self) -> None:
        """Kill the process, clear cells, and restart — discarding all state,
        including the on-disk namespace snapshot.

        Deliberately different from `resume()`: `reset` is the agent's
        explicit "wipe everything" ask (or the auto-resume fallback below
        giving up on resume()), so the snapshot has to go too. Since the
        namespace is now snapshotted to disk (ENG-1124), leaving it in place
        would mean `start()` below reloads it and `reset` silently stops
        resetting anything. Also clears the auto-resume death counter —
        whatever was wrong before, a reset is a clean slate either way.
        """
        await self._stop_process()
        self.cells.clear()
        self._discard_session_snapshot()
        self._consecutive_deaths = 0
        if not self._verify_venv_python():
            self._nuke_venv()
        await self.start()

    async def _auto_resume(self) -> bool:
        """Best-effort recovery from a dead scratchpad process (ENG-1273).

        Called from the top of `execute_streaming()` when the process isn't
        running. Tries `resume()` first, so the pad comes back with the
        namespace as of the last completed cell intact. After
        `_MAX_CONSECUTIVE_AUTO_RESUMES` deaths in a row with `resume()`
        itself never once producing a live process in between, `resume()`
        clearly isn't fixing whatever is wrong, so this falls back to a
        full `reset()` instead of retrying `resume()` forever.

        The counter tracks resume() FAILING to come back alive, not cells
        getting killed: a resume that succeeds — proven by the process
        being alive right after — zeroes it immediately, even if the cell
        that runs next gets killed for its own reasons (e.g. running over
        its own time budget). Otherwise a batch of several legitimately
        slow cells in a row would trip the fallback and wipe state for a
        reason that has nothing to do with resume() failing (final-review
        finding, ENG-1273).

        Returns whether the process is running afterward. Never raises — a
        `resume()`/`reset()` failure (e.g. the venv itself won't come up) is
        reported via `_last_resume_error` and a False return, so the caller
        can degrade to an error Cell instead of crashing the turn.
        """
        if self._consecutive_deaths >= self._MAX_CONSECUTIVE_AUTO_RESUMES:
            try:
                await self.reset()
            except Exception as exc:
                self._last_resume_error = str(exc)
                return False
            if self._proc is None or self._proc.returncode is not None:
                self._last_resume_error = "process not running after reset"
                return False
            self._consecutive_deaths = 0
            self._pending_recovery_note = (
                f"Scratchpad kept dying after {self._MAX_CONSECUTIVE_AUTO_RESUMES} "
                "resume attempts, so it was fully reset (all state cleared, "
                "including the saved namespace) before running this cell."
            )
            return True

        self._consecutive_deaths += 1
        try:
            await self.resume()
        except Exception as exc:
            self._last_resume_error = str(exc)
            return False
        if self._proc is None or self._proc.returncode is not None:
            self._last_resume_error = "process not running after resume"
            return False
        self._consecutive_deaths = 0
        return True

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
        self._discard_session_snapshot()

    async def execute_streaming(
        self,
        code: str,
        *,
        description: str = "",
        estimated_time: str = "",
        estimated_seconds: int = 0,
    ):
        """Async generator: yields progress strings then a final Cell."""
        recovery_note: str | None = None
        if self._proc is None or self._proc.returncode is not None:
            # ENG-1273: a dead process (watchdog kill, crash, or anything
            # else) recovers automatically here rather than making the agent
            # discover it and call reset — which used to be the only path,
            # and which throws away the very state ENG-1124 started saving.
            if not await self._auto_resume():
                yield Cell(
                    code=code,
                    stdout="",
                    stderr="",
                    error=(
                        "Scratchpad process is not running and could not be "
                        "recovered automatically "
                        f"({self._last_resume_error or 'unknown error'}). "
                        "Use reset to restart with a clean state."
                    ),
                    description=description,
                    estimated_time=estimated_time,
                )
                return
            # Grabbed once and cleared immediately: whatever happens to THIS
            # cell (succeeds, errors, or is killed again), the note must
            # reach the agent exactly once, attached to this cell's result —
            # never left dangling on a later, unrelated one.
            recovery_note = self._pending_recovery_note
            self._pending_recovery_note = None

        # Fresh salvage state per cell: _read_result accumulates the worker's
        # stdout chunks here so a kill/crash can still report partial output.
        self._salvage: list[str] = []
        self._salvage_truncated = False

        payload = code + "\n" + CELL_DELIM + "\n"
        self._proc.stdin.write(_encode_cell_payload(payload))  # type: ignore[union-attr]
        await self._proc.stdin.drain()  # type: ignore[union-attr]

        total_timeout, inactivity_timeout = compute_timeouts(estimated_seconds)

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
            if self._snapshot_configured():
                state_note = (
                    "This cell's own progress is lost, but the scratchpad's "
                    "namespace as of the last completed cell was already "
                    "saved to disk — the next exec call restores it "
                    "automatically, so you do not need to call reset for "
                    "this. Use reset only if you want to deliberately wipe "
                    "all state instead."
                )
            else:
                state_note = (
                    "This cell's state is lost, and this scratchpad has no "
                    "session persistence configured, so nothing survives "
                    "the restart. Use reset if you want a clean slate — "
                    "either way the next exec call starts from an empty "
                    "namespace."
                )
            error_msg = (
                f"{exc}. {state_note}\n\n"
                "If a database query was running, it may still be executing server-side.\n"
                "To check and cancel: run SHOW PROCESSLIST (MySQL) or\n"
                "SELECT * FROM information_schema.processlist WHERE status='running' "
                "and cancel with KILL <id>.\n"
                "For Snowflake: use SHOW RUNNING QUERIES and "
                "SELECT SYSTEM$CANCEL_ALL_QUERIES(<session_id>)."
            )
            salvaged = "".join(self._salvage)
            if salvaged:
                if self._salvage_truncated:
                    salvaged = "(truncated to most recent output)\n" + salvaged
                error_msg += (
                    "\n\nPartial output from before the kill was recovered "
                    "and is shown in stdout (current to within one heartbeat "
                    "interval) — use it to determine which side effects "
                    "already happened."
                )
            cell = Cell(
                code=code,
                stdout=salvaged,
                stderr="",
                error=error_msg,
                description=description,
                estimated_time=estimated_time,
                logs=recovery_note or "",
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
                logs=recovery_note or "",
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

        logs = result_data.get("logs", "")
        if recovery_note:
            logs = f"{recovery_note}\n\n{logs}" if logs else recovery_note
        cell = Cell(
            code=code,
            stdout=result_data.get("stdout", ""),
            stderr=result_data.get("stderr", ""),
            error=result_data.get("error"),
            description=description,
            estimated_time=estimated_time,
            logs=logs,
        )
        self.cells.append(cell)
        # The process proved itself alive by completing this round trip —
        # whatever earlier death streak led here (if any) is over. This is
        # in addition to (not instead of) the reset inside _auto_resume()
        # when a resume succeeds — that one covers a cell that gets killed
        # right after a successful resume; this one covers the steady-state
        # case where the pad was never dead to begin with.
        if self._proc is not None and self._proc.returncode is None:
            self._consecutive_deaths = 0
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
        last_notice = 0.0
        last_output = 0.0

        # Dead since scratchpad_boot dropped its in-cell auto-installer: the
        # markers this deferral watches for are never emitted anymore. Left
        # in place rather than unpicking it from the kill-window logic below.
        install_budget = float(s.cell_install_timeout)
        installing: str | None = None
        install_started = 0.0
        pre_install_total = total_timeout
        pre_install_inactivity = current_inactivity

        # A budget kill with zero salvaged output is ambiguous: a stuck call
        # and silent heavy work look identical from outside, so the message
        # says so instead of implying "too heavy" (the confident wrong guess
        # that taught the ENG-578 per-item pattern). The phrase routes to its
        # own nudge — lockstep constraint, see the silence-kill raise below.
        def _total_timeout_message() -> str:
            if installing:
                return (
                    f"Cell killed during auto-install of '{installing}' — the "
                    f"install ran past its {install_budget:.0f}s budget and "
                    "grace window without reporting a result"
                )
            base = f"Cell timed out after {total_timeout:.0f}s total"
            if not self._salvage:
                return base + (
                    " without producing any output — either a call is stuck "
                    "or the work is heavier than estimated"
                )
            return base

        while True:
            elapsed = _time.monotonic() - start
            remaining_total = total_timeout - elapsed
            if remaining_total <= 0:
                raise asyncio.TimeoutError(_total_timeout_message())

            line_timeout = min(current_inactivity, remaining_total)
            try:
                raw = await asyncio.wait_for(
                    self._proc.stdout.readline(),  # type: ignore[union-attr]
                    timeout=line_timeout,
                )
            except asyncio.TimeoutError:
                elapsed_now = _time.monotonic() - start
                if elapsed_now >= total_timeout - 0.5:
                    raise asyncio.TimeoutError(_total_timeout_message()) from None
                # Wording is load-bearing in THREE places (ENG-578 lockstep):
                # _select_resilience_nudge routes on "auto-install"/
                # "liveness"/"timed out"/"without producing any output", the
                # ACC kill-loop detector classifies on the same phrases, and
                # observe_scratchpad_cell records only the FIRST 120 chars as
                # the ACC reason — the routing keywords must stay inside that
                # slice. Change all of them together.
                if installing:
                    raise asyncio.TimeoutError(
                        f"Cell killed during auto-install of '{installing}' — "
                        f"no liveness signal for {current_inactivity:.0f}s: "
                        "the worker process died or the installer is wedged; "
                        "the package is likely not installed"
                    ) from None
                # Only two things actually reach this timer now: a dead worker
                # (EOF follows shortly) or one pinned below Python by a native
                # call holding the GIL — a userland deadlock or spin keeps
                # heartbeating and runs to the total budget instead.
                raise asyncio.TimeoutError(
                    f"Cell killed after {current_inactivity:.0f}s without a "
                    "liveness signal from the scratchpad worker — the worker "
                    "process died, or a native call is stuck holding it below "
                    "Python. Deliberate sleeps and blocking calls are kept "
                    "alive automatically, so this is NOT caused by "
                    "quiet-but-working code"
                ) from None

            if not raw:
                # Crash/EOF: attach whatever stdout chunks were salvaged —
                # the process died with side effects possibly already done.
                yield {
                    "stdout": "".join(self._salvage),
                    "stderr": "",
                    "error": (
                        f"Process exited unexpectedly while auto-installing "
                        f"'{installing}'."
                        if installing
                        else "Process exited unexpectedly."
                    ),
                }
                return

            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")

            if line.startswith(HEARTBEAT_MARKER):
                # Liveness only: arrival already re-armed the readline timer.
                # Do NOT extend current_inactivity here — a bare heartbeat is
                # a machine signal, not evidence of progress; only an
                # explicit progress() call earns that (ENG-1324). This
                # branch only runs when the cell shipped no new output
                # (otherwise STDOUT_CHUNK_MARKER fires instead), so a chatty
                # cell never reaches it.
                if (
                    elapsed - last_output >= _QUIET_NOTICE_AFTER
                    and elapsed - last_notice >= _QUIET_NOTICE_EVERY
                ):
                    last_notice = elapsed
                    if installing:
                        yield (
                            f"still running — installing '{installing}', "
                            f"{elapsed / 60:.0f}m elapsed of "
                            f"~{total_timeout / 60:.0f}m budget"
                        )
                    else:
                        yield (
                            f"still running — {elapsed / 60:.0f}m elapsed of "
                            f"~{total_timeout / 60:.0f}m budget"
                        )
                continue

            if line.startswith(STDOUT_CHUNK_MARKER):
                # Salvage: accumulate silently (arrival is also liveness).
                # Keep the TAIL when over budget — after a kill the newest
                # output ("sent 47/50") is the valuable part, unlike normal
                # stdout truncation which keeps the head.
                try:
                    chunk = json.loads(line[len(STDOUT_CHUNK_MARKER) :].strip())
                except json.JSONDecodeError:
                    chunk = ""
                if isinstance(chunk, str) and chunk:
                    last_output = elapsed
                    self._salvage.append(chunk)
                    total = sum(len(c) for c in self._salvage)
                    while total > _SALVAGE_MAX and len(self._salvage) > 1:
                        total -= len(self._salvage.pop(0))
                        self._salvage_truncated = True
                continue

            if line.startswith(PROGRESS_MARKER):
                last_output = elapsed
                current_inactivity = max(
                    current_inactivity, float(s.cell_inactivity_after_progress)
                )
                message = line[len(PROGRESS_MARKER) :].strip()
                yield message
                continue

            if line.startswith(INSTALL_START_MARKER):
                # `elapsed` is stale by up to the readline wait — recompute,
                # or the install loses that much of its budget.
                now = _time.monotonic() - start
                installing = line[len(INSTALL_START_MARKER) :].strip()
                install_started = now
                pre_install_total = total_timeout
                pre_install_inactivity = current_inactivity
                total_timeout = max(
                    total_timeout, now + install_budget + _INSTALL_GRACE
                )
                current_inactivity = max(
                    current_inactivity, install_budget + _INSTALL_GRACE
                )
                yield f"Installing {installing}..."
                continue

            if line.startswith(INSTALL_END_MARKER):
                if installing is not None:
                    # Install time doesn't count against the cell's budget.
                    now = _time.monotonic() - start
                    total_timeout = pre_install_total + (now - install_started)
                    current_inactivity = pre_install_inactivity
                    installing = None
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
            # Same UTF-8 mode as the scratchpad process, so pip/uv output on a
            # non-UTF-8 host locale doesn't come back as mojibake (ENG-824).
            env=_utf8_env(os.environ),
        )
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=_install_timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"Install timed out after {_install_timeout}s."
        output = stdout.decode("utf-8", errors="replace")
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


def local_scratchpad_runtime_factory(
    *,
    name: str,
    coding_provider: str,
    coding_model: str,
    coding_api_key: str,
    coding_base_url: str,
    cells: list[Cell] | None,
    workspace_path: Path | None,
    session_id: str | None = None,
) -> ScratchpadRuntime:
    return LocalScratchpadRuntime(
        name=name,
        coding_provider=coding_provider,
        coding_model=coding_model,
        coding_api_key=coding_api_key,
        coding_base_url=coding_base_url,
        cells=cells,
        workspace_path=workspace_path,
        session_id=session_id,
    )
