"""The Landlock policy agent code runs under, and the launcher call that applies it.

A host that turns the sandbox on builds one frozen ``CellSandbox`` per session
and passes it as ``ChatSessionConfig.cell_sandbox``. Every spawn that runs
agent code (the cell, the venv health check, package installs and the artifact
verifier's two checks) then starts through ``landlock_launcher.py``: the
launcher locks itself to the policy's paths with Landlock and execs the target
in place, so the process id, process group and exit code stay the target's.
With no policy, nothing here runs.

Standard library and pydantic only, so a host can probe the kernel and run the
self-test without importing the rest of anton.
"""

from __future__ import annotations

import bisect
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, model_validator

logger = logging.getLogger(__name__)

# The launcher repeats these four values (it runs under -S and cannot import
# anton); tests/test_landlock_launcher.py keeps the two copies equal.
LAUNCHER_API_VERSION: int = 1
MIN_LANDLOCK_ABI: int = 3
SANDBOX_REFUSED_EXIT: int = 87
SANDBOX_REFUSED_MARKER: str = "anton-sandbox: refused:"

# What the agent relays when it asks for an app backend on a sandboxed host.
BACKENDS_OFF_MESSAGE = (
    "App backends are off on this deployment because agent code runs in a sandbox."
)

# The launcher's own environment. The target's environment travels in the
# policy descriptor, so nothing from it (an LD_PRELOAD, a secret) reaches the
# launcher before the lock or shows in its /proc/<pid>/environ.
_LAUNCHER_ENV: dict[str, str] = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"}

# No grant may equal or contain one of these: each holds other people's data,
# the server's own files, or the whole system.
_DENIED_GRANT_ROOTS = (
    "/", "/home", "/opt", "/opt/cowork", "/tmp", "/var/tmp", "/dev", "/etc", "/proc", "/sys",
)

# Rights by the names the launcher maps to Landlock bits.
_READ = ("read_file", "read_dir")
_READ_EXEC = ("execute", "read_file", "read_dir")
_WRITE = ("write_file", "remove_dir", "remove_file", "make_dir", "make_reg", "truncate")
_PROJECT_RIGHTS = _READ_EXEC + _WRITE + ("refer",)
# Execute as in the project, so a file renamed between HOME, TMPDIR and the
# project keeps its rights, and a tool a build or a cache puts there runs.
_HOME_RIGHTS = _READ_EXEC + _WRITE + ("refer",)
# multiprocessing.Manager binds its listener socket under TMPDIR.
_TMP_RIGHTS = _HOME_RIGHTS + ("make_sock",)
_SHARED_RIGHTS = _READ + _WRITE

_SYSTEM_EXEC_DIRS = ("/usr", "/lib", "/lib64", "/bin", "/sbin")
_SYSTEM_READ_DIRS = ("/proc", "/sys", "/etc/ssl", "/etc/ca-certificates")
# pip's vendored distro opens /etc/debian_version and /etc/os-release to build
# its user agent, and lets anything but a missing file escape as a crash.
_SYSTEM_READ_FILES = (
    "/etc/passwd", "/etc/group", "/etc/nsswitch.conf", "/etc/hosts", "/etc/resolv.conf",
    "/etc/host.conf", "/etc/gai.conf", "/etc/services", "/etc/protocols", "/etc/mime.types",
    "/etc/localtime", "/etc/ld.so.cache", "/etc/debian_version", "/etc/os-release",
    "/etc/lsb-release",
)
# The host's certificate settings the cell inherits, each granted read: a
# bundle file, or the folder SSL_CERT_DIR names.
_CA_FILE_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "PIP_CERT", "CURL_CA_BUNDLE")
_CA_DIR_VARS = ("SSL_CERT_DIR",)
# Each (variable, value) that got no read rule and was already logged. The
# rules are built on every spawn, so one bad setting is logged once per process.
_LOGGED_CA_PROBLEMS: set[tuple[str, str]] = set()
_DEVICE_RULES = (
    ("/dev/null", ("read_file", "write_file")),
    ("/dev/zero", ("read_file", "write_file")),
    ("/dev/full", ("read_file", "write_file")),
    ("/dev/urandom", ("read_file",)),
    ("/dev/random", ("read_file",)),
    # multiprocessing's locks and shared memory create named files here. No
    # read_dir, so a cell cannot list other people's segment names.
    ("/dev/shm", ("read_file", "write_file", "make_reg", "remove_file", "truncate")),
)

# The child environment is an allowlist. These names pass from the parent.
_ENV_NAMES = frozenset({
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "ANTON_LANGFUSE_HEADERS",
    "ANTON_MINDS_SSL_VERIFY", "ANTON_CELL_INSTALL_TIMEOUT", "PYTHONPATH", "PYTHONUTF8", "PATH",
    "LANG", "LC_ALL", "LC_CTYPE", "TZ", "NO_PROXY", "no_proxy", "HTTP_PROXY", "http_proxy",
    "HTTPS_PROXY", "https_proxy", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE", "PIP_CERT",
})
_ENV_PREFIXES = ("ANTON_OPENAI_", "ANTON_ANTHROPIC_", "ANTON_SCRATCHPAD_", "DS_", "PIP_")
# These never pass, whatever else names them.
_ENV_NEVER_NAMES = frozenset({"DATABASE_URI", "MINDS_API_KEY", "ANTON_MINDS_API_KEY", "ANTON_UV_PATH"})
_ENV_NEVER_PREFIXES = ("COWORK_", "LD_")
# Each PIP_* variable already logged for lost credentials. The environment is
# built on every spawn, so each variable is logged once per process.
_LOGGED_PIP_CREDENTIALS: set[str] = set()
# A workspace .env key passes unless it could repoint the interpreter, the
# loader, the per-person folders or the certificate trust.
_OVERLAY_DENIED_NAMES = frozenset({
    "HOME", "TMPDIR", "TMP", "TEMP", "PATH", "USER", "LOGNAME", "MPLCONFIGDIR", "PIP_CACHE_DIR",
    "REQUESTS_CA_BUNDLE", "PIP_CERT", "CURL_CA_BUNDLE", "DATABASE_URI", "MINDS_API_KEY",
    "ANTON_MINDS_API_KEY",
})
_OVERLAY_DENIED_PREFIXES = ("LD_", "PYTHON", "XDG_", "SSL_", "COWORK_")


class SandboxUnavailable(RuntimeError):
    """The launcher refused to start agent code, or Landlock is unusable here.

    The class name leads the message, so a host or agent that shows only
    ``str(exc)`` still names the refusal.
    """

    def __str__(self) -> str:
        return f"SandboxUnavailable: {super().__str__()}"


@dataclass(frozen=True)
class LandlockRule:
    """One path the launcher grants, with the rights it grants there."""

    path: str
    access: tuple[str, ...]
    # A missing required path refuses the launch; a missing optional one is skipped.
    required: bool = False
    # A person's folder: the launcher treats a path that is a link, or sits
    # below one, like a missing path. System rules follow links, since /bin
    # and /lib are links on Debian.
    refuse_links: bool = False


@dataclass
class SandboxedSpawn:
    """The launcher invocation for one spawn.

    Pass ``argv``, ``env=env`` and ``pass_fds=pass_fds`` to the spawn call and
    call ``close()`` in a ``finally`` once it returns, so the parent's copies
    of the descriptors the launcher gets never outlive the spawn. Once the
    process has ended, ``refusal()`` says whether the launcher refused.
    """

    argv: list[str]
    env: dict[str, str]
    pass_fds: tuple[int, ...]
    # The read end of the launcher's refusal pipe. The launcher writes its
    # reason there before any target code runs, and the pipe closes when it
    # execs the target, so nothing the target prints or exits with reads as
    # a refusal.
    refusal_fd: int
    _closed: bool = field(default=False, repr=False)
    _refusal_read: bool = field(default=False, repr=False)
    _refusal: SandboxUnavailable | None = field(default=None, repr=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for fd in self.pass_fds:
            try:
                os.close(fd)
            except OSError:
                pass

    def refusal(self) -> SandboxUnavailable | None:
        """The launcher's refusal, or None when it started the target.

        Call it once the spawned process has ended: one still running reads
        as no refusal. Closes the pipe, and later calls return the same answer.
        """
        if self._refusal_read:
            return self._refusal
        self._refusal_read = True
        chunks: list[bytes] = []
        try:
            while chunk := os.read(self.refusal_fd, 1 << 16):
                chunks.append(chunk)
        except OSError:
            pass
        finally:
            os.close(self.refusal_fd)
        text = b"".join(chunks).decode("utf-8", errors="replace").strip()
        if text:
            self._refusal = SandboxUnavailable(text)
        return self._refusal


def _norm(path: Path | str) -> str:
    return os.path.normpath(str(path))


def _below(root: str) -> str:
    """The start every path below ``root`` shares: ``root`` and one separator."""
    return root if root.endswith(os.sep) else root + os.sep


def _same_or_inside(path: str, root: str) -> bool:
    """Whether ``path`` equals ``root`` or sits below it."""
    return path == root or path.startswith(_below(root))


def _holders(path: str) -> set[str]:
    """Every ``root`` for which ``_same_or_inside(path, root)`` holds: ``path``
    itself, and each start of it that ends just before or at a separator."""
    holders = {path}
    for i, char in enumerate(path):
        if char == os.sep:
            holders.add(path[:i])
            holders.add(path[: i + 1])
    return holders


def _overlaps(
    grant: Path | str, root: Path | str, *, resolve: Callable[[Path | str], str] = os.path.realpath
) -> bool:
    """Whether ``grant`` equals or contains ``root``, as written or once resolved."""
    if _same_or_inside(_norm(root), _norm(grant)):
        return True
    return _same_or_inside(resolve(root), resolve(grant))


class _PathResolver:
    """``os.path.realpath`` for many paths that share folders, each folder resolved once.

    realpath walks every folder above a path, each time it's called. A
    policy's roots mostly share folders: every person's partition sits in
    the one shared root. So this keeps each path it resolved, and resolves a
    new name below a known folder with one lstat. A name that isn't a link
    resolves to the resolved folder joined with it, the step realpath takes
    there too. A link resolves through realpath itself, and so does a path
    that isn't absolute and normalized.

    That step holds only below a link realpath fully resolved. Where a link
    loops or can't be followed, realpath stops resolving and keeps the rest
    of the path as written, and abspath then folds a ``..`` in it. So a path
    below such a link goes through realpath whole.
    """

    def __init__(self) -> None:
        self._resolved: dict[str, str] = {}
        # Paths at or below a link realpath couldn't follow (a loop, or a
        # target it can't reach).
        self._unfollowed: set[str] = set()

    def __call__(self, path: Path | str) -> str:
        text = str(path)
        known = self._resolved.get(text)
        if known is not None:
            return known
        if not os.path.isabs(text) or os.path.normpath(text) != text:
            resolved = os.path.realpath(text)
            self._resolved[text] = resolved
            return resolved
        return self._walk(text)

    def _walk(self, path: str) -> str:
        """``path``, absolute and normalized, resolved from the nearest folder
        above it that is already resolved, one name at a time."""
        below: list[str] = []
        known = path
        while known not in self._resolved:
            parent, name = os.path.split(known)
            if not name:
                self._resolved[known] = os.path.realpath(known)
                break
            below.append(name)
            known = parent
        resolved = self._resolved[known]
        for name in reversed(below):
            parent, known = known, os.path.join(known, name)
            if parent in self._unfollowed:
                self._unfollowed.add(known)
                resolved = os.path.realpath(known)
            else:
                resolved = self._resolve_name(known, os.path.join(resolved, name))
            self._resolved[known] = resolved
        return resolved

    def _resolve_name(self, path: str, joined: str) -> str:
        """``path`` resolved, where ``joined`` is its resolved parent joined
        with its last name."""
        try:
            is_link = stat.S_ISLNK(os.lstat(joined).st_mode)
        except OSError:
            # realpath treats a name it cannot lstat as no link, too.
            return joined
        if not is_link:
            return joined
        try:
            return os.path.realpath(joined, strict=True)
        except OSError:
            self._unfollowed.add(path)
            return os.path.realpath(path)


class _PathIndex:
    """Many paths, each as written and as resolved, to ask about all at once.

    ``_overlaps`` answers for one pair. This answers for every path together:
    in a sorted list, every path that starts with a folder and a separator
    sits in one run, so one bisect finds whether any path equals or sits
    below a folder. A hit names the first path in the order given whose pair
    overlaps, the one a pair-by-pair loop would name.

    A path realpath can't resolve (a NUL byte, or a relative one under a
    working folder that is gone) raises when it is resolved. The pairwise
    loop raised only on reaching it, after the paths before it, so with one
    such path the index answers pair by pair, in order, as that loop did.
    """

    def __init__(self, paths: Iterable[Path | str], *, resolve: _PathResolver) -> None:
        self._resolve = resolve
        self._paths = list(dict.fromkeys(str(path) for path in paths))
        self._written = frozenset(_norm(path) for path in self._paths)
        try:
            self._real = frozenset(resolve(path) for path in self._paths)
        except (OSError, ValueError):
            self._pairwise = True
            self._real = frozenset()
        else:
            self._pairwise = False
        self._written_sorted = sorted(self._written)
        self._real_sorted = sorted(self._real)

    def first_inside(self, folder: Path | str) -> str | None:
        """The first path that ``folder`` equals or contains, or None."""
        if self._pairwise:
            return next((path for path in self._paths if _overlaps(folder, path, resolve=self._resolve)), None)
        written, real = _norm(folder), self._resolve(folder)
        hit = _first_at_or_below(self._written_sorted, written) or _first_at_or_below(self._real_sorted, real)
        if hit is None:
            return None
        return next((path for path in self._paths if _overlaps(folder, path, resolve=self._resolve)), hit)

    def first_holding(self, path: Path | str) -> str | None:
        """The first path that equals or contains ``path``, or None."""
        if self._pairwise:
            return next((held for held in self._paths if _overlaps(held, path, resolve=self._resolve)), None)
        written, real = _norm(path), self._resolve(path)
        hit = next((holder for holder in _holders(written) if holder in self._written), None)
        if hit is None:
            hit = next((holder for holder in _holders(real) if holder in self._real), None)
        if hit is None:
            return None
        return next((held for held in self._paths if _overlaps(held, path, resolve=self._resolve)), hit)


def _first_at_or_below(paths: list[str], root: str) -> str | None:
    """A path in the sorted ``paths`` that equals ``root`` or sits below it, or None."""
    i = bisect.bisect_left(paths, root)
    if i < len(paths) and paths[i] == root:
        return root
    start = _below(root)
    i = bisect.bisect_left(paths, start)
    if i < len(paths) and paths[i].startswith(start):
        return paths[i]
    return None


def _env_never(key: str) -> bool:
    return key in _ENV_NEVER_NAMES or key.startswith(_ENV_NEVER_PREFIXES)


def overlay_key_allowed(key: str) -> bool:
    """Whether a workspace ``.env`` key may reach sandboxed agent code."""
    if _env_never(key):
        return False
    return key not in _OVERLAY_DENIED_NAMES and not key.startswith(_OVERLAY_DENIED_PREFIXES)


def _env_key_allowed(key: str, overlay_keys: frozenset[str]) -> bool:
    if key in overlay_keys:
        # Its value may be the workspace .env's, so the overlay rule decides,
        # even for a name the allowlist would pass from the server.
        return overlay_key_allowed(key)
    if _env_never(key):
        return False
    return key in _ENV_NAMES or key.startswith(_ENV_PREFIXES)


def _url_without_credentials(word: str) -> str:
    """``word`` without its userinfo and query when it is a URL, else ``word`` unchanged.

    Scheme, host, port, path and fragment stay. Raises ValueError for a URL
    ``urlsplit`` cannot parse.
    """
    parts = urlsplit(word)
    if not parts.scheme or not parts.netloc:
        return word
    if "@" not in parts.netloc and not parts.query:
        return word
    host = parts.netloc.rpartition("@")[2]
    return urlunsplit((parts.scheme, host, parts.path, "", parts.fragment))


def _pip_value_for_agent_code(key: str, value: str) -> str | None:
    """A PIP_* value with no password or token in any URL, or None to drop the variable.

    A setting that names an index, a links page or a proxy can carry a
    password in a URL's userinfo or a token in its query string. Installs and
    the health check run the venv's own Python, which a cell can rewrite or
    extend with a .pth file, so no spawn under a policy gets either part, and
    an index or proxy that needs them can't be used there. pip splits a list
    setting such as PIP_EXTRA_INDEX_URL on whitespace, so each word is judged
    alone; a value with nothing to remove comes back as it was. A URL
    ``urlsplit`` rejects could hide either part, so its variable is dropped.
    """
    words = value.split()
    try:
        kept = [_url_without_credentials(word) for word in words]
    except ValueError:
        stripped = None
    else:
        stripped = value if kept == words else " ".join(kept)
    if stripped != value and key not in _LOGGED_PIP_CREDENTIALS:
        _LOGGED_PIP_CREDENTIALS.add(key)
        if stripped is None:
            logger.warning("sandbox: dropped %s, which holds a URL that cannot be parsed", key)
        else:
            logger.warning(
                "sandbox: removed the credentials from the URLs in %s; an index or proxy "
                "that needs them can't be used under the sandbox",
                key,
            )
    return stripped


class CellSandbox(BaseModel):
    """The folders one session's agent code may reach.

    The host builds one per session. The launcher adds the system rules
    (``rules()``), so a host names only the person's own folders.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Read, write and execute: the venv, saved cell state and working directory.
    project_dir: Path
    # Read, write and execute: the per-person HOME with its caches.
    home_dir: Path
    # Read, write and execute: the per-person TMPDIR.
    tmp_dir: Path
    # Read and write files and folders, no renames in or out, no special files.
    shared_dir: Path | None = None
    read_only_dirs: tuple[Path, ...] = ()
    read_only_files: tuple[Path, ...] = ()
    # Host roots no grant may equal or contain, besides the fixed system ones.
    forbidden_roots: tuple[Path, ...] = ()
    # Other people's partitions. No grant may equal or contain one, and no
    # writable grant may sit inside one; a read-only grant inside one (a
    # skill its owner shares) is allowed.
    other_partitions: tuple[Path, ...] = ()
    # USER and LOGNAME in the child environment.
    user_name: str

    @model_validator(mode="after")
    def _check_grants(self) -> CellSandbox:
        if not self.user_name or "\0" in self.user_name:
            raise ValueError("user_name must be a non-empty name")
        for root in (*self.forbidden_roots, *self.other_partitions):
            _require_normal_absolute(root)
        # One resolver for the whole check, so each path is resolved once.
        resolve = _PathResolver()
        denied = _PathIndex(self._denied_roots(), resolve=resolve)
        for grant in self._grants():
            _require_normal_absolute(grant)
            root = denied.first_inside(grant)
            if root is not None:
                raise ValueError(f"grant {grant} equals or contains {root}")
        private = self._writable_dirs()
        partitions = _PathIndex(self.other_partitions, resolve=resolve)
        for grant in private:
            partition = partitions.first_holding(grant)
            if partition is not None:
                raise ValueError(f"writable grant {grant} sits in another person's partition {partition}")
        for i, first in enumerate(private):
            for second in private[i + 1:]:
                if _overlaps(first, second, resolve=resolve) or _overlaps(second, first, resolve=resolve):
                    raise ValueError(f"grants {first} and {second} overlap")
        return self

    def _denied_roots(self) -> list[str]:
        return [
            *_DENIED_GRANT_ROOTS,
            str(Path.home()),
            *(str(r) for r in self.forbidden_roots),
            *(str(p) for p in self.other_partitions),
        ]

    def _grants(self) -> list[Path]:
        return [*self._writable_dirs(), *self.read_only_dirs, *self.read_only_files]

    def _writable_dirs(self) -> list[Path]:
        dirs = [self.project_dir, self.home_dir, self.tmp_dir]
        if self.shared_dir is not None:
            dirs.append(self.shared_dir)
        return dirs

    def allows_read(self, path: Path | str) -> bool:
        """Whether the person's agent code may read ``path`` under this policy.

        True when ``path``, once resolved, equals or sits below a grant. A grant
        that is a link, or sits below one, admits nothing, since ``launch()``
        skips it. The system paths ``rules()`` adds are not grants: a host-side
        reader asks about the person's own files, never about /etc or /proc.
        """
        target = os.path.realpath(path)
        return any(
            _same_or_inside(target, str(grant))
            for grant in self._grants()
            if os.path.realpath(grant) == str(grant)
        )

    def _fixed_env(self) -> dict[str, str]:
        home = str(self.home_dir)
        tmp = str(self.tmp_dir)
        return {
            "HOME": home,
            "TMPDIR": tmp,
            "TMP": tmp,
            "TEMP": tmp,
            "USER": self.user_name,
            "LOGNAME": self.user_name,
            "MPLCONFIGDIR": str(self.home_dir / ".config" / "matplotlib"),
            "XDG_CACHE_HOME": str(self.home_dir / ".cache"),
            "XDG_CONFIG_HOME": str(self.home_dir / ".config"),
            "XDG_DATA_HOME": str(self.home_dir / ".local" / "share"),
            "PIP_CACHE_DIR": str(self.home_dir / ".cache" / "pip"),
        }

    def child_env(
        self, env: Mapping[str, str], *, overlay_keys: Iterable[str] = ()
    ) -> dict[str, str]:
        """The allowlisted part of ``env``, with the per-person folders set.

        ``overlay_keys`` names the workspace ``.env`` keys in ``env``. Each
        passes unless it could repoint the interpreter, the loader, the folders
        or the certificate trust, even where the allowlist names it. An entry
        execve would reject (an empty key, a key with ``=`` or NUL, a value
        with NUL) is dropped and logged, so it cannot fail every spawn after
        the lock. Every URL in a ``PIP_*`` value loses its userinfo and query
        (``_pip_value_for_agent_code``).
        """
        overlay = frozenset(overlay_keys)
        child: dict[str, str] = {}
        for key, value in env.items():
            if not _env_key_allowed(key, overlay):
                continue
            if not key or "=" in key or "\0" in key or "\0" in value:
                logger.warning("sandbox: dropped the environment entry %r, which execve refuses", key)
                continue
            if key.startswith("PIP_"):
                stripped = _pip_value_for_agent_code(key, value)
                if stripped is None:
                    continue
                value = stripped
            child[key] = value
        child.update(self._fixed_env())
        return child

    def rules(self, *, host_env: Mapping[str, str] | None = None) -> list[LandlockRule]:
        """Every rule the launcher adds: the system paths, then this policy's grants."""
        host_env = os.environ if host_env is None else host_env
        rules: list[LandlockRule] = []
        exec_dirs: list[str] = [*_SYSTEM_EXEC_DIRS, *_interpreter_dirs()]
        for path in _unique(exec_dirs):
            rules.append(LandlockRule(path, _READ_EXEC))
        for path in _SYSTEM_READ_DIRS:
            rules.append(LandlockRule(path, _READ))
        for path in _SYSTEM_READ_FILES:
            rules.append(LandlockRule(path, ("read_file",)))
        rules.extend(self._ca_rules(host_env))
        for path, access in _DEVICE_RULES:
            rules.append(LandlockRule(path, access))
        rules.append(LandlockRule(str(self.project_dir), _PROJECT_RIGHTS, required=True, refuse_links=True))
        rules.append(LandlockRule(str(self.home_dir), _HOME_RIGHTS, required=True, refuse_links=True))
        rules.append(LandlockRule(str(self.tmp_dir), _TMP_RIGHTS, required=True, refuse_links=True))
        if self.shared_dir is not None:
            rules.append(LandlockRule(str(self.shared_dir), _SHARED_RIGHTS, refuse_links=True))
        for path in self.read_only_dirs:
            rules.append(LandlockRule(str(path), _READ, refuse_links=True))
        for path in self.read_only_files:
            rules.append(LandlockRule(str(path), ("read_file",), refuse_links=True))
        return rules

    def _ca_rules(self, host_env: Mapping[str, str]) -> list[LandlockRule]:
        """Read rules for the host's certificate bundle files and SSL_CERT_DIR.

        A value is held to the same roots as a grant and must be a regular
        file (a folder for SSL_CERT_DIR); anything else is skipped and logged,
        since a folder named as a bundle would let every cell read beneath it.
        """
        rules: list[LandlockRule] = []
        seen: set[str] = set(_SYSTEM_READ_FILES)
        denied: _PathIndex | None = None
        for name in (*_CA_FILE_VARS, *_CA_DIR_VARS):
            value = host_env.get(name, "")
            if not value:
                continue
            path = os.path.normpath(value)
            if path in seen:
                continue
            is_dir = name in _CA_DIR_VARS
            if denied is None:
                denied = _PathIndex(self._denied_roots(), resolve=_PathResolver())
            problem = self._ca_path_problem(path, is_dir=is_dir, denied=denied)
            if problem is not None:
                if (name, value) not in _LOGGED_CA_PROBLEMS:
                    _LOGGED_CA_PROBLEMS.add((name, value))
                    logger.warning("sandbox: no read rule for %s=%s: %s", name, value, problem)
                continue
            seen.add(path)
            rules.append(LandlockRule(path, _READ if is_dir else ("read_file",)))
        return rules

    def _ca_path_problem(self, path: str, *, is_dir: bool, denied: _PathIndex) -> str | None:
        """Why the certificate path ``path`` gets no read rule, or None when it gets one.

        ``denied`` is this policy's denied roots (``_denied_roots``).
        """
        if not os.path.isabs(path):
            return "it is not an absolute path"
        root = denied.first_inside(path)
        if root is not None:
            return f"it equals or contains {root}"
        try:
            mode = os.stat(path).st_mode
        except OSError as exc:
            return exc.strerror or str(exc)
        if is_dir and not stat.S_ISDIR(mode):
            return "it is not a folder"
        if not is_dir and not stat.S_ISREG(mode):
            return "it is not a regular file"
        return None

    def launch(self, argv: Sequence[str], *, env: Mapping[str, str]) -> SandboxedSpawn:
        """The launcher invocation that runs ``argv`` under this policy.

        ``env`` is the target's environment from ``child_env``. It travels with
        the rules in a descriptor the launcher reads and closes before the
        lock, never in the launcher's argv or its own environment.
        """
        target = [str(arg) for arg in argv]
        if not target or not os.path.isabs(target[0]):
            raise ValueError("the sandboxed program must be an absolute path")
        leaked = sorted(key for key in env if _env_never(key))
        if leaked:
            raise ValueError(f"launch() takes an environment from child_env(); it holds {', '.join(leaked)}")
        python = launcher_python()
        # The launcher runs unlocked from these, so a cell that could write
        # one would run its own code before the next lock.
        before_the_lock = {
            "the launcher's interpreter": python,
            "the launcher": str(launcher_path()),
            "the standard library": sysconfig.get_path("stdlib"),
        }
        resolve = _PathResolver()
        for folder in self._writable_dirs():
            for label, path in before_the_lock.items():
                if path and _overlaps(folder, path, resolve=resolve):
                    raise SandboxUnavailable(f"{label} {path} sits under the writable grant {folder}")
        # The launcher grants these folders read and execute on every spawn,
        # so they are held to the same roots as a host's grants.
        denied = _PathIndex(self._denied_roots(), resolve=resolve)
        for prefix in _interpreter_dirs():
            root = denied.first_inside(prefix)
            if root is not None:
                raise SandboxUnavailable(f"the interpreter folder {prefix} contains {root}")
        rules = self.rules()
        optional = [*self.read_only_dirs, *self.read_only_files]
        if self.shared_dir is not None:
            optional.append(self.shared_dir)
        for path in optional:
            if not path.exists():
                logger.info("sandbox grant skipped, path missing: %s", path)
            elif os.path.realpath(path) != str(path):
                logger.warning("sandbox grant skipped, path is a link or sits below one: %s", path)
        payload = json.dumps({
            "version": LAUNCHER_API_VERSION,
            "rules": [
                {"path": r.path, "access": list(r.access), "required": r.required, "refuse_links": r.refuse_links}
                for r in rules
            ],
            "env": {**env, **self._fixed_env()},
        }).encode("utf-8")
        fd = _payload_fd(payload)
        try:
            refusal_read, refusal_write = os.pipe()
        except OSError as exc:
            os.close(fd)
            raise SandboxUnavailable(f"could not create the launcher's refusal pipe: {exc}") from exc
        os.set_blocking(refusal_read, False)
        argv_out = [
            python, "-I", "-S", str(launcher_path()),
            "--policy-fd", str(fd), "--refusal-fd", str(refusal_write), "--", *target,
        ]
        return SandboxedSpawn(
            argv=argv_out, env=dict(_LAUNCHER_ENV), pass_fds=(fd, refusal_write), refusal_fd=refusal_read
        )


def cell_sandbox_of(pool: object) -> CellSandbox | None:
    """The sandbox policy a scratchpad pool carries as ``cell_sandbox``, or None.

    The attribute is optional on a pool, so a pool without one, or with None,
    runs agent code unlocked. Any other value raises TypeError: a policy that
    arrives as something else (a dict, a model from a second copy of anton)
    must not be read as no policy.
    """
    policy = getattr(pool, "cell_sandbox", None)
    if policy is None or isinstance(policy, CellSandbox):
        return policy
    raise TypeError(f"cell_sandbox is {type(policy).__name__}, not a CellSandbox")


def _require_normal_absolute(path: Path) -> None:
    text = str(path)
    if not path.is_absolute() or ".." in path.parts or os.path.normpath(text) != text:
        raise ValueError(f"{text!r} must be an absolute, normalized path")


def _unique(paths: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for path in paths:
        if path and path not in seen:
            seen.append(path)
    return seen


def _interpreter_dirs() -> list[str]:
    """The host interpreter's prefixes, its site-packages and anton's own root."""
    import site

    anton_root = str(Path(__file__).resolve().parents[3])
    try:
        site_dirs = list(site.getsitepackages())
    except AttributeError:
        site_dirs = []
    return _unique([sys.prefix, sys.base_prefix, *site_dirs, anton_root])


def _payload_fd(data: bytes) -> int:
    """A descriptor positioned at the start of ``data``, for the launcher to read."""
    try:
        memfd_create = getattr(os, "memfd_create", None)
        if memfd_create is not None:
            fd = memfd_create("anton-sandbox-policy", os.MFD_CLOEXEC)
        else:
            # Landlock exists only on Linux, and the launcher refuses anywhere
            # else, so an unlinked temporary file carries the payload there.
            fd, name = tempfile.mkstemp(prefix="anton-sandbox-policy-")
            os.unlink(name)
    except OSError as exc:
        raise SandboxUnavailable(f"could not create the launcher's policy descriptor: {exc}") from exc
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.lseek(fd, 0, os.SEEK_SET)
    except BaseException:
        os.close(fd)
        raise
    return fd


def launcher_path() -> Path:
    """The launcher module file, run as a script by ``launcher_python()``."""
    return Path(__file__).with_name("landlock_launcher.py")


def launcher_python() -> str:
    """The host's own interpreter, which runs the launcher before the lock."""
    if not sys.executable:
        raise SandboxUnavailable("the host interpreter's path is unknown (sys.executable is empty)")
    return sys.executable


def landlock_abi() -> int:
    """The running kernel's Landlock ABI version, or 0 when it is unavailable.

    0 covers a non-Linux host, a kernel without Landlock (ENOSYS), Landlock
    disabled at boot (EOPNOTSUPP) and a seccomp filter that blocks the call
    (EPERM). Only asks the kernel; restricts nothing and never raises.
    """
    if not sys.platform.startswith("linux"):
        return 0
    try:
        import ctypes

        syscall = ctypes.CDLL(None, use_errno=True).syscall
        syscall.restype = ctypes.c_long
        # landlock_create_ruleset(NULL, 0, LANDLOCK_CREATE_RULESET_VERSION)
        abi = syscall(ctypes.c_long(444), None, ctypes.c_long(0), ctypes.c_long(1))
    except Exception:
        return 0
    return int(abi) if abi > 0 else 0


_SELF_TEST_CHECK = """\
import errno, sys
with open(sys.argv[1], "w") as f:
    f.write("written")
try:
    open(sys.argv[2]).read()
except OSError as exc:
    print("denied", errno.errorcode.get(exc.errno, exc.errno))
else:
    print("read")
"""


def self_test(*, scratch_dir: Path) -> None:
    """Prove in a child process that the launcher allows a grant and denies its sibling.

    Builds a policy over a fresh folder under ``scratch_dir``, launches a
    child that writes inside the grant and reads a file beside it, and raises
    ``SandboxUnavailable`` unless the write lands and the read gets EACCES.
    Never locks the caller. Removes the folder afterwards.
    """
    # Resolved, since the launcher refuses a grant that sits below a link.
    base = Path(os.path.realpath(tempfile.mkdtemp(prefix="anton-sandbox-selftest-", dir=scratch_dir)))
    try:
        project, home, tmp = base / "project", base / "home", base / "tmp"
        for folder in (project, home, tmp):
            folder.mkdir()
        written = project / "written.txt"
        sibling = base / "sibling.txt"
        sibling.write_text("outside the grant\n", encoding="utf-8")
        policy = CellSandbox(
            project_dir=project, home_dir=home, tmp_dir=tmp, user_name="anton-sandbox-selftest"
        )
        spawn = policy.launch(
            [launcher_python(), "-I", "-S", "-c", _SELF_TEST_CHECK, str(written), str(sibling)],
            env=policy.child_env({}),
        )
        try:
            result = subprocess.run(
                spawn.argv,
                env=spawn.env,
                pass_fds=spawn.pass_fds,
                cwd=str(project),
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxUnavailable(f"the self-test could not run the launcher: {exc}") from exc
        finally:
            spawn.close()
            refusal = spawn.refusal()
        if refusal is not None:
            raise refusal
        stdout = result.stdout.decode("utf-8", errors="replace").strip()
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise SandboxUnavailable(f"the self-test child exited {result.returncode}: {stderr}")
        if not written.is_file():
            raise SandboxUnavailable("the self-test child could not write inside its grant")
        if stdout != "denied EACCES":
            raise SandboxUnavailable(f"the self-test child was not denied a file outside its grant ({stdout!r})")
    finally:
        shutil.rmtree(base, ignore_errors=True)


def validate_pad_name(name: str) -> str:
    """Return ``name`` when it names one folder inside the venvs base, else raise ValueError.

    A pad name becomes a folder under ``<workspace>/.anton/scratchpad-venvs``,
    and the host deletes and recreates it. So the name must be one plain
    folder name on every platform: not empty, no NUL, no ``:`` (a Windows
    drive or NTFS stream), not an absolute path, no ``..``, no ``/`` or ``\\``
    (one pad's venv would sit inside another's, and rebuilding the outer one
    would delete it), not only dots, and no trailing dot or space (Windows
    drops them, so ``x.`` would name pad ``x``'s folder).
    """
    if not isinstance(name, str) or not name:
        raise ValueError("A scratchpad name must not be empty.")
    if "\0" in name:
        raise ValueError("A scratchpad name must not contain a NUL character.")
    if ":" in name:
        raise ValueError(f"A scratchpad name must not contain a drive or stream separator (:): {name!r}.")
    posix, windows = PurePosixPath(name), PureWindowsPath(name)
    if posix.is_absolute() or windows.anchor:
        raise ValueError(f"A scratchpad name must not be an absolute path: {name!r}.")
    if ".." in posix.parts or ".." in windows.parts:
        raise ValueError(f"A scratchpad name must not contain '..': {name!r}.")
    if "/" in name or "\\" in name:
        raise ValueError(f"A scratchpad name must be one folder name, without '/' or '\\': {name!r}.")
    if not name.strip("."):
        raise ValueError(f"A scratchpad name must name a folder: {name!r}.")
    if name.endswith((".", " ")):
        raise ValueError(f"A scratchpad name must not end with '.' or a space: {name!r}.")
    return name
