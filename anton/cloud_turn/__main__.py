"""`python -m anton.cloud_turn` — the pod entrypoint.

Process contract:
  stdin : a single TurnRequestV1 JSON document, EOF-terminated (bounded read)
  stdout: versioned JSONL TurnEventV1 records ONLY — nothing else, ever
  stderr: diagnostic logs + full tracebacks
  exit  : 0 = turn completed
          2 = request could not be validated (no turn.started emitted)
          3 = valid request reached execution but failed

stdout isolation is enforced at the OS file-descriptor level (see
``_isolated_protocol_stdout``): the original FD 1 is duplicated to a private,
non-inheritable descriptor used ONLY for protocol events, and process FD 1 is
redirected to stderr. So ``print()``, ``os.write(1, ...)``, native-library
writes, and child-process stdout can never corrupt the events stream.

`python -m anton.cloud_turn capabilities` prints a stable machine-readable
manifest of what this milestone actually enables.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys

from anton.cloud_turn import protocol
from anton.cloud_turn.errors import classify_error
from anton.cloud_turn.protocol import (
    SUPPORTED_CONTENT_BLOCK_TYPES,
    SUPPORTED_MESSAGE_ROLES,
    SUPPORTED_PROTOCOL_VERSIONS,
    TurnFailedV1,
    event_line,
    parse_request,
)
from anton.cloud_turn.runner import run_turn

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_BAD_REQUEST = 2
EXIT_FAILED = 3

#: The cloud-turn implementation version (independent of the wire protocol
#: version). Bump on behavioural changes to the process boundary.
CLOUD_TURN_IMPL_VERSION = "1.0"


def _capabilities() -> dict:
    """Stable, machine-readable manifest of what THIS milestone enables.

    Reports only behaviour that is actually on — not merely code that exists.
    """
    from anton.cloud_turn.session import CLOUD_TOOL_ALLOWLIST, _MODEL_ALLOWLIST_ENV

    try:
        import anton

        anton_version = getattr(anton, "__version__", "unknown")
    except Exception:
        anton_version = "unknown"

    provider_sdks: dict[str, bool] = {}
    for name in ("anthropic", "openai"):
        try:
            __import__(name)
            provider_sdks[name] = True
        except Exception:
            provider_sdks[name] = False

    model_allowlist = [
        m.strip() for m in os.environ.get(_MODEL_ALLOWLIST_ENV, "").split(",") if m.strip()
    ]

    return {
        "ok": True,
        "entrypoint": "anton.cloud_turn",
        "cloud_turn_version": CLOUD_TURN_IMPL_VERSION,
        "anton_version": anton_version,
        "protocol_versions": list(SUPPORTED_PROTOCOL_VERSIONS),
        "tools": sorted(CLOUD_TOOL_ALLOWLIST),
        "content_block_types": sorted(SUPPORTED_CONTENT_BLOCK_TYPES),
        "message_roles": sorted(SUPPORTED_MESSAGE_ROLES),
        "model_override": {
            # A request may request a model, but only one on the trusted pod-side
            # allowlist is honoured; empty allowlist = default-deny.
            "supported": True,
            "policy": "trusted_allowlist",
            "default": "deny",
            "allowlist_size": len(model_allowlist),
        },
        "memory": {"personal": False, "workspace": False},
        "connectors": False,
        "data_vault": False,
        "scratchpad_execution": True,
        "web_tools": False,
        "provider_sdks": provider_sdks,
        # The capability FLAG names (all default OFF this milestone).
        "capabilities": list(protocol.CapabilitiesV1.model_fields.keys()),
    }


@contextlib.contextmanager
def _isolated_protocol_stdout():
    """OS-level stdout isolation. Yields an ``emit(event)`` that writes JSONL to
    the saved protocol descriptor; everything else (FD 1) is redirected to
    stderr.

    Works wherever ``os.dup2`` is available (POSIX + Windows for FDs 0/1/2),
    which covers the Linux pod, macOS dev, and CI.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    stderr_fd = sys.stderr.fileno()

    # Private, non-inheritable copy of the real protocol channel (original FD 1).
    protocol_fd = os.dup(1)
    os.set_inheritable(protocol_fd, False)
    # Redirect process FD 1 → stderr: any write to fd 1 (print, os.write(1, …),
    # native libs, inherited child stdout) now lands on stderr, never protocol.
    os.dup2(stderr_fd, 1)
    saved_sys_stdout = sys.stdout
    sys.stdout = sys.stderr  # Python-level print() → stderr too
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    def emit(event) -> None:
        data = (event_line(event) + "\n").encode("utf-8")
        view = memoryview(data)
        while view:  # os.write may do a partial write; loop until fully flushed
            n = os.write(protocol_fd, view)
            view = view[n:]

    try:
        yield emit
    finally:
        # Flush diagnostics, then close the protocol descriptor. os.write went
        # straight to the kernel, so there is no userspace buffer to lose.
        with contextlib.suppress(Exception):
            sys.stderr.flush()
        sys.stdout = saved_sys_stdout
        os.close(protocol_fd)


def _safe_ids(raw: str) -> tuple[str | None, str | None]:
    """Best-effort identifiers from an UNVALIDATED request body.

    Returns ``None`` for any id that is absent or not a string — we never invent
    a valid-looking identifier for a request we could not validate (malformed
    JSON, empty stdin, missing/mistyped ids all yield ``None``)."""
    try:
        data = json.loads(raw)
    except Exception:
        return None, None
    if not isinstance(data, dict):
        return None, None
    rid = data.get("run_id")
    aid = data.get("attempt_id")
    return (
        rid if isinstance(rid, str) else None,
        aid if isinstance(aid, str) else None,
    )


def _run(raw: str, emit, session_builder=None) -> int:
    """Parse the request and drive the turn. The testable core — no FD or stdin
    handling, so unit tests can inject ``raw`` + a capturing ``emit``."""
    try:
        request = parse_request(raw)
    except Exception as exc:
        # Invalid request: one structured failure event, NO turn.started
        # (sequence 1, the only event of the run). Ids are None when the request
        # supplied none usable.
        run_id, attempt_id = _safe_ids(raw)
        logger.exception("invalid cloud-turn request run_id=%s", run_id)
        emit(
            TurnFailedV1(
                run_id=run_id,
                attempt_id=attempt_id,
                sequence=1,
                error=classify_error(exc),
            )
        )
        return EXIT_BAD_REQUEST
    return asyncio.run(run_turn(request, emit, session_builder=session_builder))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] == "capabilities":
        print(json.dumps(_capabilities()))
        return 0

    with _isolated_protocol_stdout() as emit:
        # Bounded read: never pull an unbounded request into memory. Read at most
        # MAX_REQUEST_BYTES+1 (chars ≤ bytes in UTF-8); parse_request enforces
        # the precise byte limit.
        raw = sys.stdin.read(protocol.MAX_REQUEST_BYTES + 1)
        return _run(raw, emit)


if __name__ == "__main__":
    sys.exit(main())
