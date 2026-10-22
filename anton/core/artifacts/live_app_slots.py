"""Launch a full-stack artifact's backend in an app slot outside this process.

A deployment that runs agent code in a sandbox starts no backend in its own
process (`backend_launcher.launch_artifact_backend` refuses). With
`ANTON_LIVE_APP_SLOTS` set it runs live apps in fixed app slots instead: each
slot is a separate container that holds one app at a time, restarts empty
before it takes another, and serves the app at its own address, which only the
person who launched it opens. A slot holds the app's files, the `DS_*` values
of the connections the operator approved for live apps, and nothing of this
process.

anton owns the launch: which files and values go, the approval card before a
launch that changes what the app runs or reaches, and which slot takes the
app. The host owns who the launcher is and records each launch, through the
`LiveAppHost` on the session; the slot program owns running the app. Every
model here is the wire format the slot program mirrors field by field.

Settings are read at call time, as the render check reads its URL:

- `ANTON_LIVE_APP_SLOTS`: the slots' control addresses in slot order.
- `ANTON_LIVE_APP_CONNECTIONS_FILE`: the approved connections, as JSON.
- `ANTON_LIVE_APP_ADDRESS`: the app's address, `{n}` standing for the slot.
"""
from __future__ import annotations

import asyncio
import base64
import errno
import hashlib
import json
import logging
import os
import re
import threading
import weakref
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol
from urllib.parse import SplitResult, urlsplit, urlunsplit

import httpx2 as httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from anton.core.artifacts.store import BACKEND_LOG_FILENAME
from anton.core.backends.sandbox import cell_sandbox_of
from anton.core.utils.scratchpad import RefusalCause, refusal_why
from anton.utils.datasources import scrub_output
from anton.utils.nofollow import read_bytes_nofollow, write_text_nofollow

if TYPE_CHECKING:
    from anton.core.artifacts.models import DatasourceRef


_log = logging.getLogger(__name__)

SLOTS_ENV_VAR = "ANTON_LIVE_APP_SLOTS"
CONNECTIONS_FILE_ENV_VAR = "ANTON_LIVE_APP_CONNECTIONS_FILE"
ADDRESS_ENV_VAR = "ANTON_LIVE_APP_ADDRESS"
DEFAULT_ADDRESS = "http://app{n}.view.localhost:8080/"
# The stack's HTTP relay: a slot reaches an approved API only through it.
RELAY_HOST = "live-apps-relay"

# What a launch sends, and what a duplicated app copies: one list for both.
LAUNCH_FILES = ("backend.py", "requirements.txt", "state_manifest.json")
LAUNCH_DIRS = ("static",)
MAX_LAUNCH_FILES = 500
MAX_LAUNCH_FILE_BYTES = 8 * 1024 * 1024
MAX_LAUNCH_TOTAL_BYTES = 64 * 1024 * 1024
# The slot's rule for a `DS_*` name it puts in the app's environment.
ENV_NAME = re.compile(r"^DS_[A-Z0-9_]+__[A-Z0-9][A-Z0-9_]*$")

# How long anton waits on a slot: a status read, the restart a reset starts
# (the slot program exits no sooner than 10 seconds after its boot), a
# launch, which waits up to 20 seconds for the app's port, and a probe,
# which GETs its paths one after another with 5 seconds for each.
STATUS_TIMEOUT_SECONDS = 2.0
RESET_TIMEOUT_SECONDS = 45.0
RESET_POLL_SECONDS = 0.5
LAUNCH_TIMEOUT_SECONDS = 60.0
PROBE_PATH_SECONDS = 5.0
MAX_REPLY_BYTES = 1024 * 1024
# The most paths one `POST /probe` takes, and the longest path it takes.
MAX_PROBE_PATHS = 32
MAX_PROBE_PATH_LENGTH = 512

_SLOT_ADDRESS = re.compile(r"^http://[A-Za-z0-9.-]+:(?P<port>\d{1,5})$")

ConnectionKind = Literal["db", "http-read", "http-write"]
MissingReason = Literal["not_approved", "not_saved", "other_host", "field_names"]
SlotState = Literal["empty", "running", "exited"]
SlotStopReason = Literal["idle", "max_time", "app_exited", "reset", "terminated"]
# What a launch card approves, in card order. The `files` hash is no part of it.
ChangedPart = Literal["backend", "requirements", "connections"]

MISSING_REASON_TEXT: dict[MissingReason, str] = {
    "not_approved": "not approved for live apps by the operator",
    "not_saved": "no saved connection by that engine and name",
    "other_host": "saved with a host other than the approved one",
    "field_names": "saved with a field name a live app can't receive (only letters, digits and _)",
}


class LiveAppConfigError(ValueError):
    """A live-apps setting is missing or malformed. The message names the setting, never a value."""


LaunchFailure = Literal[
    "no_host", "config", "launch_files", "launch_not_approved", "no_slot", "slots_full", "slots_changed",
    "slot_busy", "slot_reset", "slot_unavailable", "slot_refused", "missing_package", "app_not_ready",
    "record_failed",
]


class LiveAppLaunchError(Exception):
    """A launch that did not happen; `str(error)` is the sentence the agent
    relays. `cause` says why, for the caller to choose what follows: the
    tool handler maps it to a classified tool reason."""

    def __init__(self, message: str, *, cause: LaunchFailure, log_tail: str = "") -> None:
        super().__init__(message)
        self.cause = cause
        self.log_tail = log_tail


class SlotUnavailable(Exception):
    """A slot gave no usable answer: unreachable, timed out, or a reply that doesn't parse."""


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def live_apps_enabled() -> bool:
    """Whether this deployment runs live apps in app slots: `ANTON_LIVE_APP_SLOTS` is set and not blank."""
    return bool(os.environ.get(SLOTS_ENV_VAR, "").strip())


def live_app_slot_addresses() -> tuple[str, ...]:
    """The slots' control addresses in slot order; slot `n` is entry `n - 1`.

    Empty when live apps are off. Raises `LiveAppConfigError` for an entry
    that isn't `http://<host>:<port>`, so a typo refuses launches rather than
    dropping a slot.
    """
    raw = os.environ.get(SLOTS_ENV_VAR, "").strip()
    if not raw:
        return ()
    # A blank entry stays, so it refuses rather than renumbering the slots
    # after it; one trailing comma is allowed.
    addresses = tuple(entry.strip() for entry in raw.removesuffix(",").split(","))
    for position, address in enumerate(addresses, start=1):
        match = _SLOT_ADDRESS.match(address)
        if match is None or not 1 <= int(match["port"]) <= 65535:
            raise LiveAppConfigError(
                f"{SLOTS_ENV_VAR} entry {position} is not an http://<host>:<port> address."
            )
    return addresses


def app_address(slot: int) -> str:
    """The address slot `slot` serves its app at."""
    template = os.environ.get(ADDRESS_ENV_VAR, "").strip() or DEFAULT_ADDRESS
    if "{n}" not in template:
        raise LiveAppConfigError(f"{ADDRESS_ENV_VAR} has no {{n}} for the slot number.")
    return template.replace("{n}", str(slot))


class ApprovedConnection(BaseModel):
    """One connection the operator approved for live apps."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    engine: str = Field(min_length=1)
    name: str = Field(min_length=1)
    kind: ConnectionKind
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    # The relay's port for this entry; `http-*` entries only.
    relay_port: int | None = Field(default=None, ge=1, le=65535)
    # An `http-write` entry whose host a write-approval rule names: the
    # operator's approval overrides the hold chat applies to that host.
    overrides_write_hold: bool = False
    # An `http-read` entry's POST paths the relay lets through: as listed,
    # or only when every GraphQL operation in the body is a query. The relay
    # enforces them; the launch card names them.
    post_paths: tuple[str, ...] = ()
    graphql_paths: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _http_fields(self) -> ApprovedConnection:
        is_http = self.kind != "db"
        if is_http and self.relay_port is None:
            raise ValueError(f"{self.engine}/{self.name}: an {self.kind} connection needs a relay_port")
        if not is_http and self.relay_port is not None:
            raise ValueError(f"{self.engine}/{self.name}: a db connection takes no relay_port")
        if self.overrides_write_hold and self.kind != "http-write":
            raise ValueError(f"{self.engine}/{self.name}: only an http-write connection overrides the hold")
        if (self.post_paths or self.graphql_paths) and self.kind != "http-read":
            raise ValueError(f"{self.engine}/{self.name}: only an http-read connection lists POST paths")
        if any(not path.startswith("/") for path in (*self.post_paths, *self.graphql_paths)):
            raise ValueError(f"{self.engine}/{self.name}: a POST path starts with /")
        if self.host != self.host.strip().lower() or "/" in self.host or ":" in self.host:
            raise ValueError(f"{self.engine}/{self.name}: host must be a lower-case name with no scheme or port")
        return self

    def matches(self, ref: DatasourceRef) -> bool:
        return self.engine.lower() == ref.engine.lower() and self.name.lower() == ref.name.lower()


class ApprovedConnections(BaseModel):
    """The approved-connections file that `ANTON_LIVE_APP_CONNECTIONS_FILE` names; the host renders it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1]
    connections: tuple[ApprovedConnection, ...] = ()

    @model_validator(mode="after")
    def _unique(self) -> ApprovedConnections:
        seen: set[tuple[str, str]] = set()
        # One relay port per entry: a shared one would send one connection's
        # values to the other entry's upstream.
        by_relay_port: dict[int, ApprovedConnection] = {}
        for entry in self.connections:
            key = (entry.engine.lower(), entry.name.lower())
            if key in seen:
                raise ValueError(f"{entry.engine}/{entry.name} is approved twice")
            seen.add(key)
            if entry.relay_port is not None:
                other = by_relay_port.setdefault(entry.relay_port, entry)
                if other is not entry:
                    raise ValueError(
                        f"{other.engine}/{other.name} and {entry.engine}/{entry.name} share relay_port {entry.relay_port}"
                    )
        return self

    def find(self, ref: DatasourceRef) -> ApprovedConnection | None:
        return next((entry for entry in self.connections if entry.matches(ref)), None)


def load_approved_connections() -> ApprovedConnections:
    """Read the approved-connections file. Raises `LiveAppConfigError` when it
    is unset, unreadable or invalid, so no connection passes on a broken file."""
    path = os.environ.get(CONNECTIONS_FILE_ENV_VAR, "").strip()
    if not path:
        raise LiveAppConfigError(f"{CONNECTIONS_FILE_ENV_VAR} is not set.")
    try:
        # Bytes: pydantic reads them as UTF-8, so a file that isn't takes the
        # "is not valid" path below.
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise LiveAppConfigError(f"{CONNECTIONS_FILE_ENV_VAR} can't be read ({type(exc).__name__}).") from exc
    try:
        return ApprovedConnections.model_validate_json(raw)
    except ValidationError as exc:
        # Each problem's place and message: they name engines, names and
        # fields, never a saved value, and the file holds none.
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" if error["loc"] else error["msg"]
            for error in exc.errors()
        )
        raise LiveAppConfigError(f"{CONNECTIONS_FILE_ENV_VAR} is not valid: {problems}") from exc


# ---------------------------------------------------------------------------
# The host: who launches, and the record of each launch
# ---------------------------------------------------------------------------


class LiveAppLauncher(BaseModel):
    """The person whose turn launches an app."""

    model_config = ConfigDict(frozen=True)

    subject: str = Field(min_length=1, max_length=255)
    email: str = Field(min_length=3, max_length=254)


class LiveAppLaunch(BaseModel):
    """What the host records once a slot accepts a launch."""

    model_config = ConfigDict(frozen=True)

    slot: int
    slug: str
    artifact_dir: str
    address: str
    boot_id: str


class LiveAppHost(Protocol):
    """What the host hands a session that may launch live apps
    (`ChatSessionConfig.live_app_host`). A method that raises fails the launch."""

    launcher: LiveAppLauncher
    launch_token: str

    async def claim(self, slot: int) -> None:
        """Called only when the launch replaces an app slot `slot` is
        running, before anton resets it: the host stops that slot's running
        record, so nobody opens the slot until `launched`. A free slot
        (empty, or its app exited) gets no claim."""

    async def launched(self, launch: LiveAppLaunch) -> None:
        """Called after the slot accepted the launch."""

    async def launch_failed(self, slot: int) -> None:
        """Called once when a launch that ran `claim(slot)` doesn't finish:
        the slot refused or failed it, a Stop cancelled it, or `launched`
        raised. The app `claim` stopped is gone and nothing runs in the slot,
        so the host tells its owner. Raising here changes nothing."""


# ---------------------------------------------------------------------------
# The slot's wire format
# ---------------------------------------------------------------------------


class _Wire(BaseModel):
    # The slot may add fields; anton reads the ones it knows. A reply's
    # values stay out of error text, since the warnings log these errors.
    model_config = ConfigDict(extra="ignore", frozen=True, hide_input_in_errors=True)


class SlotConnection(_Wire):
    """A connection the app was given, by engine, name and kind; never a value."""

    engine: str
    name: str
    kind: ConnectionKind
    # The card said this connection's writes override chat's hold.
    overrides_write_hold: bool = False


class AppHashes(_Wire):
    """Lowercase hex SHA-256 of what a launch changes that the owner approves."""

    backend: str
    requirements: str
    connections: str
    # Every file the launch sent: it never asks the card, and a relaunch
    # after a cell that left all of them as they were is skipped.
    files: str


class SlotApp(_Wire):
    artifact: str
    # The launch's `app_id`, echoed: which folder the app came from.
    app_id: str
    launcher: LiveAppLauncher
    started_at: str
    expires_at: str
    # The app's latest request, or None before its first.
    last_request_at: str | None = None
    # Why the app stopped, while the slot is `exited`.
    stop_reason: SlotStopReason | None = None
    hashes: AppHashes
    env_names: tuple[str, ...] = ()
    connections: tuple[SlotConnection, ...] = ()


class SlotStatus(_Wire):
    """A slot's `GET /status` reply."""

    version: Literal[1]
    boot_id: str = Field(min_length=1)
    booted_at: str
    state: SlotState
    app: SlotApp | None = None


class LaunchFile(_Wire):
    path: str
    content_b64: str


class LaunchRequest(_Wire):
    """A slot's `POST /launch` body. The `DS_*` values travel here and nowhere else."""

    version: Literal[1] = 1
    artifact: str
    app_id: str
    launcher: LiveAppLauncher
    files: tuple[LaunchFile, ...]
    env: dict[str, str]
    connections: tuple[SlotConnection, ...]


class LaunchReply(_Wire):
    state: Literal["running"]
    boot_id: str
    started_at: str
    log_tail: str = ""


class SlotError(_Wire):
    """Every error reply's body. `detail` never holds a submitted value."""

    error: str
    field: str | None = None
    detail: str = ""
    log_tail: str = ""


class ProbeResult(_Wire):
    path: str
    status: int


class ProbeReply(_Wire):
    results: tuple[ProbeResult, ...]
    log_tail: str = ""


class SlotRefused(Exception):
    """A slot answered a request with an error reply."""

    def __init__(self, status_code: int, body: SlotError) -> None:
        super().__init__(f"{status_code} {body.error}")
        self.status_code = status_code
        self.body = body


class SlotClient:
    """One slot's control API. Blocking: callers run it in a thread."""

    def __init__(self, address: str, *, token: str) -> None:
        self.address = address
        self._token = token

    def status(self, *, timeout: float = STATUS_TIMEOUT_SECONDS) -> SlotStatus:
        return SlotStatus.model_validate_json(self._call("GET", "/status", timeout=timeout))

    def reset(self) -> None:
        self._call("POST", "/reset", timeout=STATUS_TIMEOUT_SECONDS, body={})

    def launch(self, request: LaunchRequest) -> LaunchReply:
        return LaunchReply.model_validate_json(
            self._call("POST", "/launch", timeout=LAUNCH_TIMEOUT_SECONDS, body=request.model_dump(mode="json"))
        )

    def probe(self, paths: Sequence[str]) -> ProbeReply:
        # The slot GETs the paths one after another, 5 seconds at most each.
        timeout = PROBE_PATH_SECONDS * len(paths) + STATUS_TIMEOUT_SECONDS
        return ProbeReply.model_validate_json(
            self._call("POST", "/probe", timeout=timeout, body={"paths": list(paths)})
        )

    def _call(self, method: str, path: str, *, timeout: float, body: dict | None = None) -> bytes:
        """The reply's body for a 2xx answer. Raises `SlotRefused` for an
        error reply and `SlotUnavailable` for everything else. Nothing here
        logs or raises the request's body."""
        content = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Authorization": f"Bearer {self._token}"}
        if content is not None:
            headers["Content-Type"] = "application/json"
        try:
            # trust_env=False: a slot is a sibling container on an internal
            # network, never behind the egress proxy a deployment may set.
            with httpx.Client(timeout=timeout, trust_env=False) as client:
                with client.stream(method, f"{self.address}{path}", content=content, headers=headers) as response:
                    reply = bytearray()
                    for chunk in response.iter_bytes():
                        reply += chunk
                        if len(reply) > MAX_REPLY_BYTES:
                            raise SlotUnavailable(f"slot {self.address} answered more than anton reads")
                    status_code = response.status_code
        except httpx.TimeoutException as exc:
            raise SlotUnavailable(f"slot {self.address} gave no answer within {timeout:.0f} s") from exc
        except httpx.HTTPError as exc:
            raise SlotUnavailable(f"slot {self.address} could not be reached ({type(exc).__name__})") from exc
        if 200 <= status_code < 300:
            return bytes(reply)
        try:
            error = SlotError.model_validate_json(bytes(reply))
        except ValidationError:
            error = SlotError(error=f"http_{status_code}")
        raise SlotRefused(status_code, error)


def probe_path_accepted(path: str) -> bool:
    """Whether a slot's `POST /probe` takes `path`: it refuses the whole
    probe for one path it doesn't."""
    return (
        path.startswith("/") and not path.startswith("//") and len(path) <= MAX_PROBE_PATH_LENGTH
        and "#" not in path and not any(ch.isspace() for ch in path)
    )


# ---------------------------------------------------------------------------
# What a launch sends
# ---------------------------------------------------------------------------


class LaunchFileData(NamedTuple):
    path: str
    data: bytes


def _bad_name(rel: str) -> bool:
    parts = rel.split("/")
    return "\\" in rel or "\0" in rel or rel.startswith("/") or any(p in ("", ".", "..") for p in parts)


# Why a launch file couldn't be read, by the no-follow read's errno; the
# refusal names the path and this, never the content.
_READ_REFUSALS: dict[int, str] = {
    errno.ELOOP: "is a link, and live apps send no links",
    errno.EMLINK: "has another hard link, and live apps send only plain files",
    errno.EINVAL: "is not a plain file, and live apps send only plain files",
    errno.EACCES: "can't be read",
}


def read_launch_files(folder: Path) -> list[LaunchFileData]:
    """The files a launch sends from `folder`, a resolved path, read with
    no-follow opens from the filesystem root, so a link agent code swaps in
    anywhere on the way makes the read fail instead of leading it elsewhere.

    Raises `LiveAppLaunchError` for a symlink, a hard-linked or non-regular
    file, a name the slot would refuse, or a file or total over the caps.
    The error names the path, never what the file holds.
    """
    def refuse(rel: str, why: str) -> LiveAppLaunchError:
        return LiveAppLaunchError(f"Not launched: `{rel}` {why}.", cause="launch_files")

    def too_many() -> LiveAppLaunchError:
        return LiveAppLaunchError(
            f"Not launched: the app has more than the {MAX_LAUNCH_FILES} files a launch sends.", cause="launch_files",
        )

    found: list[str] = []
    for name in LAUNCH_FILES:
        if os.path.lexists(folder / name):
            found.append(name)
    if "backend.py" not in found:
        raise LiveAppLaunchError("Not launched: the artifact has no `backend.py`.", cause="launch_files")
    for top in LAUNCH_DIRS:
        root = folder / top
        if os.path.islink(root):
            raise refuse(top, "is a link, and live apps send no links")
        if not root.is_dir():
            continue
        for current, dirnames, filenames in os.walk(root, followlinks=False):
            for name in (*dirnames, *filenames):
                path = Path(current) / name
                rel = path.relative_to(folder).as_posix()
                if _bad_name(rel):
                    raise refuse(rel, "has a name live apps can't send")
                if path.is_symlink():
                    raise refuse(rel, "is a link, and live apps send no links")
            found.extend((Path(current) / name).relative_to(folder).as_posix() for name in sorted(filenames))
            # Counted as it walks, so a huge tree stops at the cap.
            if len(found) > MAX_LAUNCH_FILES:
                raise too_many()
    files: list[LaunchFileData] = []
    total = 0
    for rel in found:
        try:
            data = read_bytes_nofollow(folder.anchor, folder / rel, limit=MAX_LAUNCH_FILE_BYTES)
        except OSError as exc:
            if exc.errno == errno.EFBIG:
                raise refuse(rel, f"is over the {MAX_LAUNCH_FILE_BYTES // (1024 * 1024)} MB a launch sends per file") from exc
            raise refuse(rel, _READ_REFUSALS.get(exc.errno, "can't be read, and live apps send only plain files")) from exc
        total += len(data)
        if total > MAX_LAUNCH_TOTAL_BYTES:
            raise LiveAppLaunchError(
                f"Not launched: the app's files are over the {MAX_LAUNCH_TOTAL_BYTES // (1024 * 1024)} MB a launch sends.",
                cause="launch_files",
            )
        files.append(LaunchFileData(rel, data))
    return files


class MissingConnection(BaseModel):
    """A declared connection the app didn't get. Never a value."""

    model_config = ConfigDict(frozen=True)

    engine: str
    name: str
    reason: MissingReason


def _missing(ref: DatasourceRef, why: MissingReason) -> MissingConnection:
    return MissingConnection(engine=ref.engine, name=ref.name, reason=why)


@dataclass(frozen=True)
class GrantedConnection:
    """A declared connection the app gets, with what the card says about it."""

    slot: SlotConnection
    approved: ApprovedConnection


@dataclass
class ResolvedConnections:
    env: dict[str, str] = field(default_factory=dict)
    granted: list[GrantedConnection] = field(default_factory=list)
    missing: list[MissingConnection] = field(default_factory=list)


def _default_port(scheme: str) -> int:
    return 443 if scheme == "https" else 80


def _http_url(value: str) -> SplitResult | None:
    """`value` split as an http(s) URL with a host, else None."""
    try:
        parts = urlsplit(value.strip())
        if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
            return None
        parts.port  # noqa: B018 - raises ValueError for a port that isn't a number
    except ValueError:
        return None
    return parts


def _points_at(parts: SplitResult, approved: ApprovedConnection) -> bool:
    # Only an absent port takes the scheme's default; an explicit 0 is no match.
    port = parts.port if parts.port is not None else _default_port(parts.scheme.lower())
    return parts.hostname.lower() == approved.host and port == approved.port


def _to_relay(parts: SplitResult, approved: ApprovedConnection) -> str:
    """The URL pointed at the relay's port for `approved`, its path and query kept."""
    userinfo = parts.netloc.rpartition("@")[0]
    netloc = f"{userinfo}@{RELAY_HOST}:{approved.relay_port}" if userinfo else f"{RELAY_HOST}:{approved.relay_port}"
    return urlunsplit(("http", netloc, parts.path, parts.query, ""))


def _db_points_at(env: dict[str, str], prefix: str, approved: ApprovedConnection) -> bool:
    host = env.get(f"{prefix}__HOST")
    if host is None or host.strip().lower() != approved.host:
        return False
    port = env.get(f"{prefix}__PORT")
    if port is None:
        return True
    try:
        return int(port.strip()) == approved.port
    except ValueError:
        return False


def resolve_connections(
    vault, datasources: Sequence[DatasourceRef], approved: ApprovedConnections, *, slug: str
) -> ResolvedConnections:
    """The `DS_*` a launch sends: for each declared connection that the
    operator approved and the person saved pointing at the approved host.
    Every other declared connection is named in `missing`.

    Resolved per connection with `build_datasource_env`, the resolver both
    in-process launch paths use, so a slot gets exactly what the local
    launcher would, minus what isn't approved.
    """
    from anton.core.artifacts.backend_launcher import build_datasource_env

    resolved = ResolvedConnections()
    for ref in datasources or ():
        entry = approved.find(ref)
        if entry is None:
            resolved.missing.append(_missing(ref, "not_approved"))
            continue
        env = build_datasource_env(vault, [ref], slug=slug)
        if not env:
            resolved.missing.append(_missing(ref, "not_saved"))
            continue
        if any(not ENV_NAME.fullmatch(name) for name in env):
            # The slot refuses the whole launch for one such name.
            resolved.missing.append(_missing(ref, "field_names"))
            continue
        if entry.kind == "db":
            if not _db_points_at(env, ref.env_prefix, entry):
                resolved.missing.append(_missing(ref, "other_host"))
                continue
        else:
            rewritten = dict(env)
            points = False
            for key, value in env.items():
                parts = _http_url(value)
                if parts is not None and _points_at(parts, entry):
                    rewritten[key] = _to_relay(parts, entry)
                    points = True
            if not points:
                resolved.missing.append(_missing(ref, "other_host"))
                continue
            env = rewritten
        resolved.env.update(env)
        resolved.granted.append(
            GrantedConnection(
                slot=SlotConnection(
                    engine=ref.engine, name=ref.name, kind=entry.kind, overrides_write_hold=entry.overrides_write_hold,
                ),
                approved=entry,
            )
        )
    return resolved


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _listed_hash(listed: list) -> str:
    return _sha256(json.dumps(sorted(listed), separators=(",", ":")).encode("utf-8"))


def app_hashes(files: Sequence[LaunchFileData], connections: Sequence[SlotConnection]) -> AppHashes:
    """The hashes a slot reports for an app, computed from what a launch would send."""
    by_path = {f.path: f.data for f in files}
    return AppHashes(
        backend=_sha256(by_path.get("backend.py", b"")),
        requirements=_sha256(by_path.get("requirements.txt", b"")),
        connections=_listed_hash([[c.engine, c.name, c.kind, c.overrides_write_hold] for c in connections]),
        files=_listed_hash([[f.path, _sha256(f.data)] for f in files]),
    )


def app_identity(folder: Path) -> str:
    """A launch's `app_id`: the SHA-256 of the artifact folder's resolved
    path, so two artifacts of one person with one slug never count as one app."""
    return _sha256(os.fsencode(os.path.realpath(folder)))


# ---------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------

_CHANGED_PART_NAMES: dict[ChangedPart, str] = {
    "backend": "`backend.py`", "requirements": "`requirements.txt`", "connections": "its connections",
}


def _access_words(entry: ApprovedConnection) -> str:
    if entry.kind == "db":
        return "reads, and writes if its database role allows"
    if entry.kind == "http-read":
        # The POSTs the relay lets through, since the owner approves them here too.
        allowed = [", ".join(entry.post_paths)] if entry.post_paths else []
        if entry.graphql_paths:
            allowed.append(", ".join(entry.graphql_paths) + ", queries only")
        return f"reads (POST to {'; '.join(allowed)})" if allowed else "reads"
    if entry.overrides_write_hold:
        return "reads and writes, overriding the write-approval hold"
    return "reads and writes"


def changed_parts(running: AppHashes, new: AppHashes) -> list[ChangedPart]:
    """Which approved parts differ between the running app and the new launch, in card order."""
    return [part for part in _CHANGED_PART_NAMES if getattr(running, part) != getattr(new, part)]


def shown_time(iso: str) -> str:
    """An ISO 8601 time from a slot as `HH:MM UTC`, or as given when it doesn't parse."""
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%H:%M UTC")
    except ValueError:
        return iso


def launch_card_prompt(
    *,
    slug: str,
    granted: Sequence[GrantedConnection],
    changed: Sequence[ChangedPart],
    running: bool,
    replaces: SlotApp | None = None,
) -> str:
    """The launch card: the app, each connection with what it may do, what
    changed since the owner last approved it, and the owner's other app it
    replaces when every slot is taken."""
    lines = ["**Launch this live app?**", ""]
    if granted:
        lines += [f"`{slug}` would run as you, with these connections:", ""]
        lines += [f"- `{g.slot.engine}/{g.slot.name}`: {_access_words(g.approved)}" for g in granted]
        lines.append("")
    else:
        lines += [f"`{slug}` would run as you, with no connections.", ""]
    if changed:
        lines += ["Since you last approved it, these changed: " + ", ".join(_CHANGED_PART_NAMES[p] for p in changed) + ".", ""]
    overriding = list(dict.fromkeys(g.approved.host for g in granted if g.approved.overrides_write_hold))
    if overriding:
        hosts = ", ".join(f"`{host}`" for host in overriding)
        lines += [
            f"Chat holds writes to {hosts} for your approval. This app's writes there aren't held: "
            "your operator approved them for apps, and each write is logged.",
            "",
        ]
    if replaces is not None:
        lines += [
            f"Every app slot is in use, so this stops your app `{replaces.artifact}`, "
            f"running since {shown_time(replaces.started_at)}, and takes its slot.",
            "",
        ]
    if running or replaces is not None:
        lines.append("It starts only if you approve. Cancel, Skip or no answer leaves your running apps as they are.")
    else:
        lines.append("It starts only if you approve. Cancel, Skip or no answer starts nothing.")
    return "\n".join(lines)


# What came of a launch card: "stopped" when the turn ended while it waited.
LaunchStatus = RefusalCause | Literal["approved", "stopped"]


class LaunchDecision(BaseModel):
    """One logged launch card: the app, its connections by name, and what came of it."""

    model_config = ConfigDict(frozen=True)

    slug: str
    connections: tuple[str, ...]
    changed: tuple[ChangedPart, ...]
    replaces: str | None = None
    status: LaunchStatus


# ---------------------------------------------------------------------------
# The launch
# ---------------------------------------------------------------------------

NO_HOST_MESSAGE = "Not launched: live apps need a host that records who launches them, and this session has none."
NO_SLOT_MESSAGE = "Not launched: no live-app slot answered. Ask the operator to check the stack's app slots."


def not_approved_message(cause: RefusalCause, *, running: bool) -> str:
    """A launch card that didn't approve: who or what stopped it, and what stays."""
    if cause == "cancelled":
        lead = "Not launched: the person didn't approve the launch."
    else:
        lead = f"Not launched: the launch needs the person's approval, and {refusal_why(cause)}."
    return f"{lead} {'Their running apps are unchanged.' if running else 'Nothing was started.'}"


def missing_clause(missing: Sequence[MissingConnection]) -> str:
    """The sentence a launch result adds for the declared connections the app
    didn't get, by engine, name and reason, never a value; "" when none."""
    if not missing:
        return ""
    named = ", ".join(f"`{m.engine}/{m.name}` ({MISSING_REASON_TEXT[m.reason]})" for m in missing)
    return f" Not passed to the app: {named}. Ask the person to add or rename the connection, or their operator to approve it."


class LiveAppLaunchResult(BaseModel):
    """A launch that happened, or, with `unchanged`, a relaunch that wasn't
    needed because the slot already runs exactly these files."""

    model_config = ConfigDict(frozen=True)

    slot: int
    address: str
    slug: str
    missing: tuple[MissingConnection, ...] = ()
    probe: tuple[ProbeResult, ...] = ()
    # Why the probe couldn't run, when the slot gave it no usable answer.
    probe_error: str = ""
    log_tail: str = ""
    unchanged: bool = False

    def message(self) -> str:
        return (
            f"Live app `{self.slug}` is running at `{self.address}`. "
            "Give the person this address in backticks; only they can open it."
        ) + missing_clause(self.missing)


def _slot_locks() -> dict[str, asyncio.Lock]:
    """This event loop's per-slot locks. Launches from one process never
    interleave on a slot between `claim` and `launched`."""
    loop = asyncio.get_running_loop()
    locks = _LOCKS_BY_LOOP.get(loop)
    if locks is None:
        locks = _LOCKS_BY_LOOP[loop] = {}
    return locks


_LOCKS_BY_LOOP: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Lock]] = (
    weakref.WeakKeyDictionary()
)


async def _read_status(client: SlotClient) -> SlotStatus | None:
    try:
        return await asyncio.to_thread(client.status)
    except (SlotUnavailable, SlotRefused, ValidationError):
        _log.warning("live app slot %s gave no usable status", client.address, exc_info=True)
        return None


async def _read_statuses(clients: Sequence[SlotClient]) -> list[SlotStatus | None]:
    return list(await asyncio.gather(*(_read_status(c) for c in clients)))


def _holds(status: SlotStatus | None, app_id: str, launcher: LiveAppLauncher) -> bool:
    return (
        status is not None and status.state == "running" and status.app is not None
        and status.app.app_id == app_id and status.app.launcher.subject == launcher.subject
    )


def running_slot(statuses: Sequence[SlotStatus | None], app_id: str, launcher: LiveAppLauncher) -> int | None:
    """The index of the slot that runs the app `app_id` for `launcher`, or None."""
    return next((i for i, s in enumerate(statuses) if _holds(s, app_id, launcher)), None)


@dataclass(frozen=True)
class SlotPlan:
    """The slot a launch takes, and the launcher's own other app it replaces, if any."""

    index: int
    replaces: SlotApp | None = None


def _free(status: SlotStatus) -> bool:
    # An exited app's slot is about to restart empty.
    return status.app is None or status.state != "running"


def _held_by_others(statuses: Sequence[SlotStatus | None]) -> str:
    """Who holds which slot, and which gave no answer, for the refusal the agent relays."""
    held = [
        f"slot {i + 1} gave no answer" if s is None or s.app is None else
        f"slot {i + 1} runs `{s.app.artifact}` for {s.app.launcher.email} since {shown_time(s.app.started_at)}"
        for i, s in enumerate(statuses)
    ]
    expiries = sorted(s.app.expires_at for s in statuses if s is not None and s.app is not None)
    latest = f", and the first frees itself by {shown_time(expiries[0])} at the latest" if expiries else ""
    return (
        "Not launched: no app slot is free for you: " + "; ".join(held) + ". "
        f"Tell the person they can wait for a slot to free itself (an app stops after a while with no requests{latest}), "
        "or ask the operator for more slots."
    )


def plan_slot(statuses: Sequence[SlotStatus | None], app_id: str, launcher: LiveAppLauncher) -> SlotPlan:
    """The slot a launch takes. The app's own slot when one still runs it;
    else the lowest free slot, which after a stop may not be the slot the app
    had; else the launcher's own app that started longest ago, which the card
    names. Nobody else's app is ever replaced: with every slot held by other
    people, raises `LiveAppLaunchError` naming who holds what."""
    own = running_slot(statuses, app_id, launcher)
    if own is not None:
        return SlotPlan(own)
    answering = [(i, s) for i, s in enumerate(statuses) if s is not None]
    if not answering:
        raise LiveAppLaunchError(NO_SLOT_MESSAGE, cause="no_slot")
    free = [i for i, s in answering if _free(s)]
    if free:
        return SlotPlan(free[0])
    mine = [(i, s) for i, s in answering if s.app.launcher.subject == launcher.subject]
    if mine:
        index, status = min(mine, key=lambda pair: (pair[1].app.started_at, pair[0]))
        return SlotPlan(index, replaces=status.app)
    raise LiveAppLaunchError(_held_by_others(statuses), cause="slots_full")


def _same_plan(new: SlotPlan, approved: SlotPlan) -> bool:
    """Whether `new` replaces nothing the card didn't name."""
    if new.replaces is None:
        return True
    return (
        approved.replaces is not None and new.index == approved.index
        and new.replaces.artifact == approved.replaces.artifact
        and new.replaces.started_at == approved.replaces.started_at
    )


def _still_as_planned(seen: SlotStatus, current: SlotStatus | None) -> bool:
    """Whether the slot is still the one the plan saw. An exited app's slot
    restarts by itself, so the same slot back empty under a new boot is too."""
    if current is None:
        return False
    if seen.state == "exited" and current.state == "empty":
        return True
    return current.boot_id == seen.boot_id and current.state == seen.state


async def _wait_for_restart(client: SlotClient, old_boot_id: str) -> SlotStatus:
    """Poll until the slot is back from a reset: a new boot id and empty. A
    slot that can't be reached meanwhile is still restarting."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + RESET_TIMEOUT_SECONDS
    while loop.time() < deadline:
        await asyncio.sleep(RESET_POLL_SECONDS)
        try:
            status = await asyncio.to_thread(client.status)
        except SlotUnavailable:
            continue
        except (SlotRefused, ValidationError):
            _log.warning("live app slot %s gave no usable status while restarting", client.address, exc_info=True)
            continue
        if status.boot_id != old_boot_id and status.state == "empty":
            return status
    _log.warning("live app slot %s did not come back empty within %.0f s", client.address, RESET_TIMEOUT_SECONDS)
    raise LiveAppLaunchError(
        "Not launched: the app slot did not come back empty after its restart. Ask the operator to check it.",
        cause="slot_reset",
    )


def _launch_failure(error: SlotRefused) -> LiveAppLaunchError:
    body = error.body
    tail = scrub_output(body.log_tail)
    if body.error == "missing_package":
        return LiveAppLaunchError(
            f"Not launched: the app slot lacks the package `{body.detail}`, and live apps can't install packages.",
            cause="missing_package",
        )
    if body.error == "app_not_ready":
        return LiveAppLaunchError(
            f"Error: the backend did not start in the app slot ({body.detail or 'no detail'}).\nLog tail:\n{tail}",
            cause="app_not_ready", log_tail=tail,
        )
    named = f" (field `{body.field}`)" if body.field else ""
    return LiveAppLaunchError(
        f"Not launched: the app slot refused the launch: {body.error}{named}.", cause="slot_refused",
    )


class _LaunchSetup(NamedTuple):
    host: LiveAppHost
    addresses: tuple[str, ...]
    approved: ApprovedConnections


def _launch_setup(session) -> _LaunchSetup:
    """What every launch from `session` needs. Raises `LiveAppLaunchError`
    when none can succeed: no host, or a broken setting. Checked before any
    slot is touched, since an address that fails after a launch would leave
    an app running that nobody recorded."""
    host: LiveAppHost | None = getattr(session, "_live_app_host", None)
    if host is None:
        raise LiveAppLaunchError(NO_HOST_MESSAGE, cause="no_host")
    try:
        addresses = live_app_slot_addresses()
        approved = load_approved_connections()
        app_address(1)
    except LiveAppConfigError as exc:
        raise LiveAppLaunchError(f"Not launched: {exc} Ask the operator to fix it.", cause="config") from exc
    return _LaunchSetup(host, addresses, approved)


def live_app_launch_blocker(session) -> LiveAppLaunchError | None:
    """Why no launch from `session` can succeed, or None: no host, or a
    broken setting. The generator asks before it builds a fullstack app."""
    try:
        _launch_setup(session)
    except LiveAppLaunchError as exc:
        return exc
    return None


async def launch_live_app(
    session,
    *,
    slug: str,
    folder: Path,
    datasources: Sequence[DatasourceRef],
    probe_paths: Sequence[str] = (),
    only_if_changed: bool = False,
) -> LiveAppLaunchResult:
    """Launch `slug` into an app slot, through the owner's card when it
    changes what the app runs or reaches. Raises `LiveAppLaunchError`, whose
    text the agent relays, when nothing launched; nothing ever starts in this
    process. With `only_if_changed`, a running app whose files are all as
    they were isn't relaunched: the result says `unchanged`."""
    from anton.core.utils.scratchpad import ask_approval_card

    host, addresses, approved = _launch_setup(session)
    # This process reads the files outside the sandbox, so under a policy it
    # reads only what the person's own cells may read, as the render check
    # does: the folder's resolved path is approved, then read without links.
    policy = cell_sandbox_of(getattr(session, "_scratchpads", None))
    try:
        # Inside the try: resolving a name that agent code is swapping can
        # itself fail, in realpath and in the policy's own resolve.
        folder = Path(os.path.realpath(folder))
        allowed = policy is None or policy.allows_read(folder)
    except OSError as exc:
        raise LiveAppLaunchError(
            "Not launched: the app's folder could not be resolved; it changed while the launch read it.",
            cause="launch_files",
        ) from exc
    if not allowed:
        raise LiveAppLaunchError(
            "Not launched: the app's folder is outside the files this conversation may read.", cause="launch_files",
        )
    files = await asyncio.to_thread(read_launch_files, folder)
    resolved = resolve_connections(getattr(session, "_data_vault", None), datasources, approved, slug=slug)
    connections = tuple(g.slot for g in resolved.granted)
    hashes = app_hashes(files, connections)
    app_id = app_identity(folder)
    clients = [SlotClient(address, token=host.launch_token) for address in addresses]
    launcher = host.launcher

    statuses = await _read_statuses(clients)
    plan = plan_slot(statuses, app_id, launcher)
    own = running_slot(statuses, app_id, launcher)
    running_hashes = statuses[own].app.hashes if own is not None else None
    if only_if_changed and running_hashes == hashes:
        return LiveAppLaunchResult(
            slot=own + 1, address=app_address(own + 1), slug=slug, missing=tuple(resolved.missing), unchanged=True,
        )
    changed = changed_parts(running_hashes, hashes) if running_hashes is not None else []
    if running_hashes is None or changed or plan.replaces is not None:
        running = own is not None or plan.replaces is not None
        prompt = launch_card_prompt(
            slug=slug, granted=resolved.granted, changed=changed, running=running, replaces=plan.replaces,
        )
        decision = LaunchDecision(
            slug=slug,
            connections=tuple(f"{c.engine}/{c.name}:{c.kind}" for c in connections),
            changed=tuple(changed),
            replaces=plan.replaces.artifact if plan.replaces is not None else None,
            status="approved",
        )
        try:
            cause = await ask_approval_card(
                session, prompt,
                approve_detail="Start the app with these connections", cancel_detail="Leave things as they are",
                log_label="live app launch",
            )
        except asyncio.CancelledError:
            _log_launch_decision(decision.model_copy(update={"status": "stopped"}))
            raise
        _log_launch_decision(decision.model_copy(update={"status": cause or "approved"}))
        if cause is not None:
            raise LiveAppLaunchError(not_approved_message(cause, running=running), cause="launch_not_approved")

    request = LaunchRequest(
        artifact=slug,
        app_id=app_id,
        launcher=launcher,
        files=tuple(LaunchFile(path=f.path, content_b64=base64.b64encode(f.data).decode("ascii")) for f in files),
        env=resolved.env,
        connections=connections,
    )
    # A slot another launch took between the read and the lock is picked
    # again once, and a slot that answers 409 is picked again once. The new
    # plan may not replace an app the card didn't name.
    approved_plan = plan
    changed_left = busy_left = 1
    # The app `claim` stopped, and its slot: a launch that then fails says so.
    claimed: SlotApp | None = None
    claimed_slot = 0
    try:
        while True:
            index = plan.index
            client, seen = clients[index], statuses[index]
            slot, address = index + 1, app_address(index + 1)
            async with _slot_locks().setdefault(client.address, asyncio.Lock()):
                current = await _read_status(client)
                if current is None and seen.state == "exited":
                    # The exited app's slot is restarting by itself.
                    current = await _wait_for_restart(client, seen.boot_id)
                reply = None
                if _still_as_planned(seen, current):
                    if current.state == "running":
                        # Only a running app has a record to clear. A free
                        # slot's older record opens nothing, since the host's
                        # access check also asks the slot who runs it now,
                        # and claiming one here could clear the record
                        # another process's launch just wrote.
                        await _claim(host, client, slot)
                        claimed, claimed_slot = current.app, slot
                    if current.state != "empty":
                        await asyncio.to_thread(_reset, client)
                        current = await _wait_for_restart(client, current.boot_id)
                    reply = await _send_launch(
                        client, request, folder=folder, boot_id=current.boot_id, app_id=app_id, launcher=launcher,
                    )
                    if reply is not None:
                        await _record_launched(host, client, LiveAppLaunch(
                            slot=slot, slug=slug, artifact_dir=str(folder), address=address, boot_id=reply.boot_id,
                        ))
                        break
                    busy = True
                else:
                    busy = False
            if busy and busy_left:
                busy_left -= 1
            elif not busy and changed_left:
                changed_left -= 1
            else:
                raise LiveAppLaunchError(
                    "Not launched: other launches kept taking the app slot. Try again in a moment.", cause="slot_busy",
                )
            statuses = await _read_statuses(clients)
            plan = plan_slot(statuses, app_id, launcher)
            if not _same_plan(plan, approved_plan):
                raise LiveAppLaunchError(
                    "Not launched: the app slots changed while the card waited. Ask to launch it again.",
                    cause="slots_changed",
                )
    except BaseException as exc:
        if claimed is None:
            raise
        await _report_launch_failed(host, claimed_slot)
        if not isinstance(exc, LiveAppLaunchError):
            raise
        raise LiveAppLaunchError(
            f"{exc} Your app `{claimed.artifact}` that slot {claimed_slot} was running has stopped for this "
            "launch, and nothing runs there now.",
            cause=exc.cause, log_tail=exc.log_tail,
        ) from exc

    tail, probe, probe_error = reply.log_tail, (), ""
    if probe_paths:
        try:
            probed = await asyncio.to_thread(client.probe, probe_paths)
        except SlotRefused as exc:
            # A slot whose app stopped answers none of the paths; any other
            # refusal says nothing about the app, so the result says why.
            probe = tuple(ProbeResult(path=p, status=0) for p in probe_paths)
            if exc.body.error != "not_running":
                probe_error = f"the slot refused the probe: {exc.body.error}"
        except (SlotUnavailable, ValidationError) as exc:
            _log.warning("live app slot %s: the probe failed", client.address, exc_info=True)
            probe = tuple(ProbeResult(path=p, status=0) for p in probe_paths)
            probe_error = f"the slot gave no usable answer ({type(exc).__name__})"
        else:
            # One result per path sent, in order; a path the slot left out didn't answer.
            answered = {r.path: r.status for r in probed.results}
            tail, probe = probed.log_tail, tuple(ProbeResult(path=p, status=answered.get(p, 0)) for p in probe_paths)
    tail = scrub_output(tail)
    _write_log(folder, tail)
    return LiveAppLaunchResult(
        slot=slot, address=address, slug=slug, missing=tuple(resolved.missing), probe=probe,
        probe_error=probe_error, log_tail=tail,
    )


def _log_launch_decision(decision: LaunchDecision) -> None:
    # WARNING: hosts log at WARNING by default, and this line is the record
    # of an approved or refused launch.
    _log.warning("live app launch: %s", decision.model_dump_json())


async def _send_launch(
    client: SlotClient, request: LaunchRequest, *, folder: Path, boot_id: str, app_id: str, launcher: LiveAppLauncher,
) -> LaunchReply | None:
    """`POST /launch`. None when the slot answered 409 `slot_busy`. A launch
    the slot may have started but anton can't record, because a Stop
    cancelled it or the answer didn't parse, is reset once it lands, so no
    unrecorded app keeps the slot."""
    sent = asyncio.ensure_future(asyncio.to_thread(client.launch, request))
    try:
        return await asyncio.shield(sent)
    except asyncio.CancelledError:
        # The thread finishes the POST whatever happens here.
        sent.add_done_callback(lambda _: _in_thread(
            _reset_if_ours, client, boot_id=boot_id, app_id=app_id, launcher=launcher,
        ))
        raise
    except SlotRefused as exc:
        if exc.body.error == "slot_busy":
            return None
        failure = _launch_failure(exc)
        _write_log(folder, failure.log_tail)
        raise failure from None
    except (SlotUnavailable, ValidationError) as exc:
        await asyncio.shield(asyncio.to_thread(
            _reset_if_ours, client, boot_id=boot_id, app_id=app_id, launcher=launcher,
        ))
        raise LiveAppLaunchError(
            f"Not launched: the app slot gave no usable answer ({type(exc).__name__}).", cause="slot_unavailable",
        ) from None


async def _record_launched(host: LiveAppHost, client: SlotClient, launch: LiveAppLaunch) -> None:
    """Record a launch the slot accepted. When that fails or is stopped, the
    app is reset: nobody could open an app with no record."""
    try:
        await _record(host.launched(launch))
    except BaseException:
        await asyncio.shield(asyncio.to_thread(_reset_quietly, client))
        raise


async def _record(step: Awaitable[None]) -> None:
    """Await one of the host's record calls; a failure fails the launch."""
    try:
        await step
    except Exception as exc:
        # The type only: the host's message may name the person.
        _log.warning("live app launch record failed (%s)", type(exc).__name__)
        raise LiveAppLaunchError(
            "Not launched: the stack couldn't record the launch. Ask the operator to check it.",
            cause="record_failed",
        ) from exc


async def _claim(host: LiveAppHost, client: SlotClient, slot: int) -> None:
    """`host.claim(slot)`, which a Stop doesn't cut short. A host may finish
    the claim's record after the Stop, and that record then says the app was
    replaced while it still runs where nobody can open it. So a claim that
    went through after a Stop resets the slot and tells the host the launch
    failed before the Stop goes on. Raises `LiveAppLaunchError` when the
    claim fails. The caller holds the slot's lock."""
    claiming = asyncio.ensure_future(_record(host.claim(slot)))
    try:
        await asyncio.shield(claiming)
    except asyncio.CancelledError:
        while not claiming.done():
            try:
                await asyncio.wait({claiming})
            except asyncio.CancelledError:
                continue
        if not claiming.cancelled() and claiming.exception() is None:
            await _stop_claimed(host, client, slot)
        raise


async def _stop_claimed(host: LiveAppHost, client: SlotClient, slot: int) -> None:
    """Reset the slot whose app `claim` stopped and tell the host the launch
    failed, the report going out even when a Stop lands during the reset."""
    try:
        await asyncio.shield(asyncio.to_thread(_reset_quietly, client))
    finally:
        await _report_launch_failed(host, slot)


async def _report_launch_failed(host: LiveAppHost, slot: int) -> None:
    """Tell the host that the app `claim` stopped in `slot` is gone and
    nothing replaced it. Never raises an `Exception`."""
    try:
        await asyncio.shield(host.launch_failed(slot))
    except Exception as exc:
        # The type only, as for the other records.
        _log.warning("live app launch_failed record failed (%s)", type(exc).__name__)


def _in_thread(target, *args, **kwargs) -> None:
    """Run `target` in a daemon thread: from a done-callback, after the
    turn that started the launch has gone."""
    threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True).start()


def _reset_if_ours(client: SlotClient, *, boot_id: str, app_id: str, launcher: LiveAppLauncher) -> None:
    """Reset the slot when it runs this launch's app in the boot anton
    launched into, and nothing else. Never raises."""
    try:
        status = client.status()
    except (SlotUnavailable, SlotRefused, ValidationError):
        _log.warning("live app slot %s: no status to check before a reset", client.address, exc_info=True)
        return
    app = status.app
    if status.boot_id == boot_id and app is not None and app.app_id == app_id and app.launcher.subject == launcher.subject:
        _reset_quietly(client)


def _reset_quietly(client: SlotClient) -> None:
    try:
        client.reset()
    except (SlotUnavailable, SlotRefused):
        _log.warning("live app slot %s: the reset after a failed launch failed", client.address, exc_info=True)


def _reset(client: SlotClient) -> None:
    try:
        client.reset()
    except SlotUnavailable:
        # A slot whose app exited may already be restarting; the wait that
        # follows sees it come back, or refuses after its timeout.
        _log.warning("live app slot %s gave no answer to its reset", client.address)
    except SlotRefused as exc:
        raise LiveAppLaunchError(
            "Not launched: the app slot refused its restart. Ask the operator to check it.", cause="slot_reset",
        ) from exc


def _write_log(folder: Path, tail: str) -> None:
    """The slot's scrubbed log tail as the artifact's `backend.log`, written
    from the filesystem root without links, as the launch reads its files."""
    try:
        write_text_nofollow(folder.anchor, folder / BACKEND_LOG_FILENAME, tail)
    except OSError:
        _log.warning("could not write %s for %s", BACKEND_LOG_FILENAME, folder, exc_info=True)


class _RunningApp(NamedTuple):
    host: LiveAppHost
    client: SlotClient
    app_id: str
    slot: int


async def _running_app(session, folder: Path) -> _RunningApp | None:
    """The slot that runs the app from `folder` for this session's launcher,
    or None when none does or the slots can't say."""
    host: LiveAppHost | None = getattr(session, "_live_app_host", None)
    if host is None:
        return None
    try:
        clients = [SlotClient(a, token=host.launch_token) for a in live_app_slot_addresses()]
        app_id = app_identity(folder)
    except (LiveAppConfigError, OSError):
        return None
    index = running_slot(await _read_statuses(clients), app_id, host.launcher)
    return None if index is None else _RunningApp(host, clients[index], app_id, index + 1)


async def running_app_address(session, *, folder: Path) -> str | None:
    """The address a slot serves the app from `folder` at for this session's
    launcher, or None when none runs it. Never raises an `Exception`."""
    running = await _running_app(session, folder)
    if running is None:
        return None
    try:
        return app_address(running.slot)
    except LiveAppConfigError:
        return None


async def stop_live_app(session, *, folder: Path) -> None:
    """Stop the app a slot runs from `folder` for this session's launcher, if
    one does: a generation that gave up on it. The host hears of it as a
    launch that failed. Never raises an `Exception`."""
    running = await _running_app(session, folder)
    if running is None:
        return
    host, client = running.host, running.client
    async with _slot_locks().setdefault(client.address, asyncio.Lock()):
        if not _holds(await _read_status(client), running.app_id, host.launcher):
            return
        try:
            await _claim(host, client, running.slot)
        except LiveAppLaunchError:
            return
        await _stop_claimed(host, client, running.slot)


def launch_paths_changed(before: dict[str, object], after: dict[str, object]) -> bool:
    """Whether a file a launch sends differs between two `{relative path: stamp}` maps."""
    def sent(rel: str) -> bool:
        return rel in LAUNCH_FILES or any(rel.startswith(f"{top}/") for top in LAUNCH_DIRS)

    keys = {rel for rel in (*before, *after) if sent(rel)}
    return any(before.get(rel) != after.get(rel) for rel in keys)


async def relaunch_changed_apps(session, store, slugs: Sequence[str]) -> list[str]:
    """Relaunch each of `slugs` that a slot runs for this session's launcher,
    after a cell changed its files. One line per app for the cell's result;
    none for an app whose files all stayed as they were. Never raises."""
    host: LiveAppHost | None = getattr(session, "_live_app_host", None)
    if host is None or not slugs:
        return []
    try:
        clients = [SlotClient(a, token=host.launch_token) for a in live_app_slot_addresses()]
    except LiveAppConfigError:
        return []
    statuses = await _read_statuses(clients)
    lines: list[str] = []
    for slug in slugs:
        try:
            folder = store.folder_for(slug)
            app_id = app_identity(folder)
        except (OSError, ValueError):
            continue
        if running_slot(statuses, app_id, host.launcher) is None:
            continue
        try:
            artifact = store.open(slug)
        except Exception:
            artifact = None
        if artifact is None:
            lines.append(
                f"Live app `{slug}` was not relaunched: its metadata.json can't be read. "
                "Fix it, then call `launch_backend`."
            )
            continue
        try:
            result = await launch_live_app(
                session, slug=slug, folder=folder, datasources=artifact.datasources, only_if_changed=True,
            )
        except Exception as exc:
            if not isinstance(exc, LiveAppLaunchError):
                _log.warning("relaunching live app %s failed", slug, exc_info=True)
            reason = str(exc) if isinstance(exc, LiveAppLaunchError) else "The relaunch failed."
            # The slot as it is now says whether the app still runs.
            if running_slot(await _read_statuses(clients), app_id, host.launcher) is None:
                lines.append(
                    f"Live app `{slug}` is stopped: its relaunch failed. {reason} "
                    "Fix it, then call `launch_backend` to start it again."
                )
            else:
                lines.append(f"Live app `{slug}` was not relaunched. {reason}")
            continue
        if result.unchanged:
            # Nothing to relaunch; a connection it lacks is still worth a line.
            if result.missing:
                lines.append(f"Live app `{slug}` already runs these files at `{result.address}`." + missing_clause(result.missing))
            continue
        lines.append(
            f"Relaunched live app `{slug}` at `{result.address}` with this cell's changes." + missing_clause(result.missing)
        )
    return lines
