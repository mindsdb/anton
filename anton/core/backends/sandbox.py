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

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

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
_HOME_RIGHTS = _READ + _WRITE + ("refer",)
# multiprocessing.Manager binds its listener socket under TMPDIR.
_TMP_RIGHTS = _HOME_RIGHTS + ("make_sock",)
_SHARED_RIGHTS = _READ + _WRITE

_SYSTEM_EXEC_DIRS = ("/usr", "/lib", "/lib64", "/bin", "/sbin")
_SYSTEM_READ_DIRS = ("/proc", "/sys", "/etc/ssl", "/etc/ca-certificates")
_SYSTEM_READ_FILES = (
    "/etc/passwd", "/etc/group", "/etc/nsswitch.conf", "/etc/hosts", "/etc/resolv.conf",
    "/etc/host.conf", "/etc/gai.conf", "/etc/services", "/etc/protocols", "/etc/mime.types",
    "/etc/localtime", "/etc/ld.so.cache",
)
_CA_BUNDLE_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "PIP_CERT")
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
# A workspace .env key passes unless it could repoint the interpreter, the
# loader, the per-person folders or the certificate trust.
_OVERLAY_DENIED_NAMES = frozenset({
    "HOME", "TMPDIR", "TMP", "TEMP", "PATH", "USER", "LOGNAME", "MPLCONFIGDIR", "PIP_CACHE_DIR",
    "REQUESTS_CA_BUNDLE", "PIP_CERT", "DATABASE_URI", "MINDS_API_KEY", "ANTON_MINDS_API_KEY",
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


@dataclass
class SandboxedSpawn:
    """The launcher invocation for one spawn.

    Pass ``argv``, ``env=env`` and ``pass_fds=pass_fds`` to the spawn call and
    call ``close()`` in a ``finally`` once it returns, so the parent's copy of
    the policy descriptor never outlives the spawn.
    """

    argv: list[str]
    env: dict[str, str]
    pass_fds: tuple[int, ...]
    _closed: bool = field(default=False, repr=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for fd in self.pass_fds:
            try:
                os.close(fd)
            except OSError:
                pass


def _norm(path: Path | str) -> str:
    return os.path.normpath(str(path))


def _same_or_inside(path: str, root: str) -> bool:
    """Whether ``path`` equals ``root`` or sits below it."""
    if path == root:
        return True
    prefix = root if root.endswith(os.sep) else root + os.sep
    return path.startswith(prefix)


def _overlaps(grant: Path | str, root: Path | str) -> bool:
    """Whether ``grant`` equals or contains ``root``, as written or once resolved."""
    if _same_or_inside(_norm(root), _norm(grant)):
        return True
    return _same_or_inside(os.path.realpath(root), os.path.realpath(grant))


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


class CellSandbox(BaseModel):
    """The folders one session's agent code may reach.

    The host builds one per session. The launcher adds the system rules
    (``rules()``), so a host names only the person's own folders.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Read, write and execute: the venv, saved cell state and working directory.
    project_dir: Path
    # Read and write, no execute: the per-person HOME with its caches.
    home_dir: Path
    # Read and write, no execute: the per-person TMPDIR.
    tmp_dir: Path
    # Read and write files and folders, no renames in or out, no special files.
    shared_dir: Path | None = None
    read_only_dirs: tuple[Path, ...] = ()
    read_only_files: tuple[Path, ...] = ()
    # Host roots no grant may equal or contain, besides the fixed system ones.
    forbidden_roots: tuple[Path, ...] = ()
    # USER and LOGNAME in the child environment.
    user_name: str

    @model_validator(mode="after")
    def _check_grants(self) -> CellSandbox:
        if not self.user_name or "\0" in self.user_name:
            raise ValueError("user_name must be a non-empty name")
        for root in self.forbidden_roots:
            _require_normal_absolute(root)
        for grant in self._grants():
            _require_normal_absolute(grant)
            for root in self._denied_roots():
                if _overlaps(grant, root):
                    raise ValueError(f"grant {grant} equals or contains {root}")
        private = self._writable_dirs()
        for i, first in enumerate(private):
            for second in private[i + 1:]:
                if _overlaps(first, second) or _overlaps(second, first):
                    raise ValueError(f"grants {first} and {second} overlap")
        return self

    def _denied_roots(self) -> list[str]:
        return [*_DENIED_GRANT_ROOTS, str(Path.home()), *(str(r) for r in self.forbidden_roots)]

    def _grants(self) -> list[Path]:
        return [*self._writable_dirs(), *self.read_only_dirs, *self.read_only_files]

    def _writable_dirs(self) -> list[Path]:
        dirs = [self.project_dir, self.home_dir, self.tmp_dir]
        if self.shared_dir is not None:
            dirs.append(self.shared_dir)
        return dirs

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
        or the certificate trust, even where the allowlist names it.
        """
        overlay = frozenset(overlay_keys)
        child = {k: v for k, v in env.items() if _env_key_allowed(k, overlay)}
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
        ca_files = [host_env.get(name, "") for name in _CA_BUNDLE_VARS]
        for path in _unique([*_SYSTEM_READ_FILES, *(p for p in ca_files if p)]):
            rules.append(LandlockRule(path, ("read_file",)))
        for path, access in _DEVICE_RULES:
            rules.append(LandlockRule(path, access))
        rules.append(LandlockRule(str(self.project_dir), _PROJECT_RIGHTS, required=True))
        rules.append(LandlockRule(str(self.home_dir), _HOME_RIGHTS, required=True))
        rules.append(LandlockRule(str(self.tmp_dir), _TMP_RIGHTS, required=True))
        if self.shared_dir is not None:
            rules.append(LandlockRule(str(self.shared_dir), _SHARED_RIGHTS))
        for path in self.read_only_dirs:
            rules.append(LandlockRule(str(path), _READ))
        for path in self.read_only_files:
            rules.append(LandlockRule(str(path), ("read_file",)))
        return rules

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
        for folder in self._writable_dirs():
            if _overlaps(folder, python):
                raise SandboxUnavailable(
                    f"the launcher's interpreter {python} sits under the writable grant {folder}"
                )
        # The launcher grants these folders read and execute on every spawn,
        # so they are held to the same roots as a host's grants.
        for prefix in _interpreter_dirs():
            for root in self._denied_roots():
                if _overlaps(prefix, root):
                    raise SandboxUnavailable(f"the interpreter folder {prefix} contains {root}")
        rules = self.rules()
        optional = [*self.read_only_dirs, *self.read_only_files]
        if self.shared_dir is not None:
            optional.append(self.shared_dir)
        for path in optional:
            if not path.exists():
                logger.info("sandbox grant skipped, path missing: %s", path)
        payload = json.dumps({
            "version": LAUNCHER_API_VERSION,
            "rules": [
                {"path": r.path, "access": list(r.access), "required": r.required} for r in rules
            ],
            "env": {**env, **self._fixed_env()},
        }).encode("utf-8")
        fd = _payload_fd(payload)
        argv_out = [python, "-I", "-S", str(launcher_path()), "--policy-fd", str(fd), "--", *target]
        return SandboxedSpawn(argv=argv_out, env=dict(_LAUNCHER_ENV), pass_fds=(fd,))


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


def refusal_from(returncode: int | None, output: bytes | str) -> SandboxUnavailable | None:
    """The launcher's refusal in a finished spawn's exit code and stderr, if any."""
    if returncode != SANDBOX_REFUSED_EXIT:
        return None
    text = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else output
    for line in text.splitlines():
        if line.startswith(SANDBOX_REFUSED_MARKER):
            return SandboxUnavailable(line.strip())
    return None


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
    base = Path(tempfile.mkdtemp(prefix="anton-sandbox-selftest-", dir=scratch_dir))
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
        refusal = refusal_from(result.returncode, result.stderr)
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
    """Return ``name`` when it names a folder inside the venvs base, else raise ValueError.

    A pad name becomes a path under ``<workspace>/.anton/scratchpad-venvs``,
    and the host deletes and recreates the folder it names. So an empty name,
    a NUL, an absolute path (POSIX or Windows), a ``..`` part, or a name that
    resolves to the base itself is refused.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("A scratchpad name must not be empty.")
    if "\0" in name:
        raise ValueError("A scratchpad name must not contain a NUL character.")
    posix, windows = PurePosixPath(name), PureWindowsPath(name)
    if posix.is_absolute() or windows.anchor:
        raise ValueError(f"A scratchpad name must not be an absolute path: {name!r}.")
    if ".." in posix.parts or ".." in windows.parts:
        raise ValueError(f"A scratchpad name must not contain '..': {name!r}.")
    if not [part for part in windows.parts if part != "."]:
        raise ValueError(f"A scratchpad name must name a folder: {name!r}.")
    return name
