"""Lock this process with Landlock, then replace it with the target program.

Run as ``<host python> -I -S landlock_launcher.py --policy-fd N --refusal-fd M --
<target argv>``. Reads the policy (rules and the target's environment) as
JSON from descriptor N and closes it, sets no_new_privs, restricts itself to
the policy's paths, closes every inherited descriptor above 2 except M and
execs the target in place. The process id, process group and exit code are
then the target's own.

Every refusal writes one line that starts with SANDBOX_REFUSED_MARKER to
descriptor M and to stderr, and exits SANDBOX_REFUSED_EXIT, before any target
code runs. M is close-on-exec, so a target the launcher starts never holds
it: only the launcher can report a refusal there.

Standard library only: under -S it cannot import anton, so the four values
below repeat anton.core.backends.sandbox's, and a test keeps them equal.
"""

import ctypes
import errno
import json
import os
import stat
import sys

LAUNCHER_API_VERSION = 1
MIN_LANDLOCK_ABI = 3
SANDBOX_REFUSED_EXIT = 87
SANDBOX_REFUSED_MARKER = "anton-sandbox: refused:"

_SYS_CREATE_RULESET = 444
_SYS_ADD_RULE = 445
_SYS_RESTRICT_SELF = 446
_CREATE_RULESET_VERSION = 1
_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38
_SCOPE_ABSTRACT_UNIX_SOCKET = 1
_SCOPE_SIGNAL = 2
# The newest ABI whose rights this file knows. A newer kernel handles these.
_KNOWN_ABI = 11

# Filesystem rights by name: (bit, the ABI that introduced it).
_RIGHTS = {
    "execute": (1 << 0, 1),
    "write_file": (1 << 1, 1),
    "read_file": (1 << 2, 1),
    "read_dir": (1 << 3, 1),
    "remove_dir": (1 << 4, 1),
    "remove_file": (1 << 5, 1),
    "make_char": (1 << 6, 1),
    "make_dir": (1 << 7, 1),
    "make_reg": (1 << 8, 1),
    "make_sock": (1 << 9, 1),
    "make_fifo": (1 << 10, 1),
    "make_block": (1 << 11, 1),
    "make_sym": (1 << 12, 1),
    "refer": (1 << 13, 2),
    "truncate": (1 << 14, 3),
    "ioctl_dev": (1 << 15, 5),
    "resolve_unix": (1 << 16, 9),
}
# The rights a rule on a file (not a directory) may carry.
_FILE_RIGHTS = sum(_RIGHTS[n][0] for n in ("execute", "write_file", "read_file", "truncate", "ioctl_dev"))


class _RulesetAttr(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
        ("scoped", ctypes.c_uint64),
    ]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class _Refused(Exception):
    """A reason to refuse, written after the marker."""


def _refuse(reason, refusal_fd=None):
    line = f"{SANDBOX_REFUSED_MARKER} {reason}\n".encode("utf-8", "replace")
    for fd in (refusal_fd, 2):
        if fd is None:
            continue
        try:
            os.write(fd, line)
        except OSError:
            pass
    os._exit(SANDBOX_REFUSED_EXIT)


def _parse_args(args):
    if len(args) < 6 or args[0] != "--policy-fd" or args[2] != "--refusal-fd" or args[4] != "--":
        raise _Refused("usage: --policy-fd N --refusal-fd M -- PROGRAM [ARGS...]")
    try:
        policy_fd, refusal_fd = int(args[1]), int(args[3])
    except ValueError:
        raise _Refused(f"--policy-fd and --refusal-fd take descriptor numbers, not {args[1]!r} and {args[3]!r}") from None
    if refusal_fd <= 2:
        raise _Refused(f"--refusal-fd must be above 2, not {refusal_fd}")
    return policy_fd, refusal_fd, args[5:]


def _read_policy(fd):
    chunks = []
    try:
        while True:
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(fd)
    policy = json.loads(b"".join(chunks))
    if not isinstance(policy, dict) or policy.get("version") != LAUNCHER_API_VERSION:
        raise _Refused(f"policy version {policy.get('version')!r} is not {LAUNCHER_API_VERSION}")
    if not isinstance(policy.get("rules"), list) or not isinstance(policy.get("env"), dict):
        raise _Refused("the policy needs a rules list and an env object")
    return policy


def _syscall(libc, number, *args):
    result = libc.syscall(ctypes.c_long(number), *args)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def _probe_abi(libc):
    try:
        return _syscall(libc, _SYS_CREATE_RULESET, None, ctypes.c_long(0), ctypes.c_long(_CREATE_RULESET_VERSION))
    except OSError as exc:
        reasons = {
            errno.ENOSYS: "the kernel has no Landlock (ENOSYS), or an old container seccomp profile hides it",
            errno.EOPNOTSUPP: "Landlock is built in but off (EOPNOTSUPP); add landlock to the lsm= boot parameter",
            errno.EPERM: "a seccomp filter blocks Landlock (EPERM); Docker Engine 20.10.18 or later allows it",
        }
        raise _Refused(reasons.get(exc.errno, f"the Landlock probe failed: {exc.strerror}")) from None


def _lock(rules):
    if not sys.platform.startswith("linux"):
        raise _Refused(f"Landlock needs Linux, and this is {sys.platform}")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    zero = ctypes.c_ulong(0)
    if libc.prctl(ctypes.c_int(_PR_SET_NO_NEW_PRIVS), ctypes.c_ulong(1), zero, zero, zero) != 0:
        raise _Refused(f"prctl(PR_SET_NO_NEW_PRIVS) failed: {os.strerror(ctypes.get_errno())}")
    abi = _probe_abi(libc)
    if abi < MIN_LANDLOCK_ABI:
        raise _Refused(f"the kernel's Landlock ABI is {abi}, below {MIN_LANDLOCK_ABI} (Linux 6.2)")
    abi = min(abi, _KNOWN_ABI)
    handled = sum(bit for bit, since in _RIGHTS.values() if since <= abi)
    attr = _RulesetAttr(handled, 0, _SCOPE_ABSTRACT_UNIX_SOCKET | _SCOPE_SIGNAL if abi >= 6 else 0)
    # The struct's size at the running ABI: the fields it lacks stay out.
    size = 8 if abi <= 3 else 16 if abi <= 5 else 24
    try:
        ruleset = _syscall(libc, _SYS_CREATE_RULESET, ctypes.byref(attr), ctypes.c_long(size), ctypes.c_long(0))
    except OSError as exc:
        raise _Refused(f"landlock_create_ruleset failed: {exc.strerror}") from None
    for rule in rules:
        _add_rule(libc, ruleset, rule, handled)
    try:
        _syscall(libc, _SYS_RESTRICT_SELF, ctypes.c_long(ruleset), ctypes.c_long(0))
    except OSError as exc:
        raise _Refused(f"landlock_restrict_self failed: {exc.strerror}") from None


def _reached_through_a_link(fd, path):
    """Whether the opened ``path`` is a symlink, or a folder above it is one."""
    if stat.S_ISLNK(os.fstat(fd).st_mode):
        return True
    try:
        return os.readlink(f"/proc/self/fd/{fd}") != path
    except OSError:
        return os.path.realpath(path) != path


def _add_rule(libc, ruleset, rule, handled):
    path, required = rule["path"], bool(rule.get("required"))
    refuse_links = bool(rule.get("refuse_links"))
    allowed = 0
    for name in rule["access"]:
        if name not in _RIGHTS:
            raise _Refused(f"unknown right {name!r} for {path}")
        allowed |= _RIGHTS[name][0]
    allowed &= handled
    flags = os.O_PATH | os.O_CLOEXEC | (os.O_NOFOLLOW if refuse_links else 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if required:
            raise _Refused(f"required path {path} cannot be opened: {exc.strerror}") from None
        return
    try:
        # A person's folder through a link would grant wherever the link
        # points, maybe another person's folder.
        if refuse_links and _reached_through_a_link(fd, path):
            if required:
                raise _Refused(f"required path {path} is a link or sits below one")
            return
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            allowed &= _FILE_RIGHTS
        if not allowed:
            return
        beneath = _PathBeneathAttr(allowed, fd)
        try:
            _syscall(
                libc, _SYS_ADD_RULE, ctypes.c_long(ruleset), ctypes.c_long(_RULE_PATH_BENEATH),
                ctypes.byref(beneath), ctypes.c_long(0),
            )
        except OSError as exc:
            raise _Refused(f"landlock_add_rule failed for {path}: {exc.strerror}") from None
    finally:
        os.close(fd)


def _close_inherited_fds(keep):
    """Close every descriptor above 2 but ``keep``, which closes when the target starts."""
    try:
        limit = os.sysconf("SC_OPEN_MAX")
    except (OSError, ValueError):
        limit = 1 << 16
    limit = limit if limit > 3 else 1 << 16
    os.closerange(3, keep)
    os.closerange(keep + 1, max(limit, keep + 1))
    os.set_inheritable(keep, False)


def _exec(target, env, refusal_fd):
    try:
        os.execve(target[0], target, env)
    except ValueError as exc:
        _refuse(f"execve refused the target's environment: {exc}", refusal_fd)
    except OSError as exc:
        denied = exc.errno in (errno.EACCES, errno.EPERM)
        # Landlock does not mediate access(2), so it answers from the file's
        # own permissions: a target they forbid is a broken file, not a refusal.
        if denied and os.access(target[0], os.X_OK):
            _refuse(f"the sandbox denies running {target[0]}: {exc.strerror}", refusal_fd)
        os.write(2, f"anton-sandbox: cannot run {target[0]}: {exc.strerror}\n".encode("utf-8", "replace"))
        os._exit(126 if denied else 127)


def main(args):
    refusal_fd = None
    try:
        policy_fd, refusal_fd, target = _parse_args(args)
        policy = _read_policy(policy_fd)
        env = {str(k): str(v) for k, v in policy["env"].items()}
        _lock(policy["rules"])
    except _Refused as exc:
        _refuse(str(exc), refusal_fd)
    except Exception as exc:
        _refuse(f"{type(exc).__name__}: {exc}", refusal_fd)
    _close_inherited_fds(refusal_fd)
    _exec(target, env, refusal_fd)


if __name__ == "__main__":
    main(sys.argv[1:])
