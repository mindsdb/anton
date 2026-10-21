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
import weakref
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol
from urllib.parse import SplitResult, urlsplit, urlunsplit

import httpx2 as httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from anton.core.artifacts.store import BACKEND_LOG_FILENAME
from anton.core.backends.sandbox import cell_sandbox_of
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

# How long anton waits on a slot: a status read, the restart a reset starts
# (the slot program exits no sooner than 10 seconds after its boot), and a
# launch, which waits up to 20 seconds for the app's port.
STATUS_TIMEOUT_SECONDS = 2.0
RESET_TIMEOUT_SECONDS = 45.0
RESET_POLL_SECONDS = 0.5
LAUNCH_TIMEOUT_SECONDS = 60.0
PROBE_TIMEOUT_SECONDS = 30.0
MAX_REPLY_BYTES = 1024 * 1024

_SLOT_ADDRESS = re.compile(r"^http://[A-Za-z0-9.-]+:(?P<port>\d{1,5})$")

ConnectionKind = Literal["db", "http-read", "http-write"]
MissingReason = Literal["not_approved", "not_saved", "other_host"]
SlotState = Literal["empty", "running", "exited"]

MISSING_REASON_TEXT: dict[MissingReason, str] = {
    "not_approved": "not approved for live apps by the operator",
    "not_saved": "no saved connection by that engine and name",
    "other_host": "saved with a host other than the approved one",
}


class LiveAppConfigError(Exception):
    """A live-apps setting is missing or malformed. The message names the setting, never a value."""


class LiveAppLaunchError(Exception):
    """A launch that did not happen; `str(error)` is the sentence the agent relays."""

    def __init__(self, message: str, *, reason: str, log_tail: str = "") -> None:
        super().__init__(message)
        self.reason = reason
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
    raw = os.environ.get(SLOTS_ENV_VAR, "")
    addresses = tuple(entry.strip() for entry in raw.split(",") if entry.strip())
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
    """The approved-connections file the stack renders from `LIVE_APPS_CONNECTIONS`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1]
    connections: tuple[ApprovedConnection, ...] = ()

    @model_validator(mode="after")
    def _unique(self) -> ApprovedConnections:
        seen: set[tuple[str, str]] = set()
        for entry in self.connections:
            key = (entry.engine.lower(), entry.name.lower())
            if key in seen:
                raise ValueError(f"{entry.engine}/{entry.name} is approved twice")
            seen.add(key)
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
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise LiveAppConfigError(f"{CONNECTIONS_FILE_ENV_VAR} can't be read ({type(exc).__name__}).") from exc
    try:
        return ApprovedConnections.model_validate_json(text)
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
        """Called before anton resets slot `slot` or sends it a launch: the
        host clears the slot's record, so nobody opens it until `launched`."""

    async def launched(self, launch: LiveAppLaunch) -> None:
        """Called after the slot accepted the launch."""


# ---------------------------------------------------------------------------
# The slot's wire format
# ---------------------------------------------------------------------------


class _Wire(BaseModel):
    # The slot may add fields; anton reads the ones it knows.
    model_config = ConfigDict(extra="ignore", frozen=True)


class SlotConnection(_Wire):
    """A connection the app was given, by engine, name and kind; never a value."""

    engine: str
    name: str
    kind: ConnectionKind


class AppHashes(_Wire):
    """Lowercase hex SHA-256 of what a launch changes that the owner approves."""

    backend: str
    requirements: str
    connections: str


class SlotApp(_Wire):
    artifact: str
    launcher: LiveAppLauncher
    started_at: str
    expires_at: str
    # The app's latest request, or None before its first.
    last_request_at: str | None = None
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
        return ProbeReply.model_validate_json(
            self._call("POST", "/probe", timeout=PROBE_TIMEOUT_SECONDS, body={"paths": list(paths)})
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


def fetch_slot_status(address: str, *, token: str, timeout: float = STATUS_TIMEOUT_SECONDS) -> SlotStatus:
    """One slot's status, for a host that checks who runs which slot.
    Blocking. Raises `SlotUnavailable` or `SlotRefused`."""
    try:
        return SlotClient(address, token=token).status(timeout=timeout)
    except ValidationError as exc:
        raise SlotUnavailable(f"slot {address} answered a status that doesn't parse") from exc


# ---------------------------------------------------------------------------
# What a launch sends
# ---------------------------------------------------------------------------


class LaunchFileData(NamedTuple):
    path: str
    data: bytes


def _bad_name(rel: str) -> bool:
    parts = rel.split("/")
    return "\\" in rel or "\0" in rel or rel.startswith("/") or any(p in ("", ".", "..") for p in parts)


def read_launch_files(folder: Path) -> list[LaunchFileData]:
    """The files a launch sends from `folder`, a resolved path, read with
    no-follow opens from the filesystem root, so a link agent code swaps in
    anywhere on the way makes the read fail instead of leading it elsewhere.

    Raises `LiveAppLaunchError` for a symlink, a hard-linked or non-regular
    file, a name the slot would refuse, or a file or total over the caps.
    The error names the path, never what the file holds.
    """
    def refuse(rel: str, why: str) -> LiveAppLaunchError:
        return LiveAppLaunchError(f"Not launched: `{rel}` {why}.", reason="launch_files")

    found: list[str] = []
    for name in LAUNCH_FILES:
        if os.path.lexists(folder / name):
            found.append(name)
    if "backend.py" not in found:
        raise LiveAppLaunchError("Not launched: the artifact has no `backend.py`.", reason="launch_files")
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
    if len(found) > MAX_LAUNCH_FILES:
        raise LiveAppLaunchError(
            f"Not launched: the app has {len(found)} files, more than the {MAX_LAUNCH_FILES} a launch sends.",
            reason="launch_files",
        )
    files: list[LaunchFileData] = []
    total = 0
    for rel in found:
        try:
            data = read_bytes_nofollow(folder.anchor, folder / rel, limit=MAX_LAUNCH_FILE_BYTES)
        except OSError as exc:
            if exc.errno == errno.EFBIG:
                raise refuse(rel, f"is over the {MAX_LAUNCH_FILE_BYTES // (1024 * 1024)} MB a launch sends per file") from exc
            raise refuse(rel, "is a link, a linked or special file, or unreadable, and live apps send only plain files") from exc
        total += len(data)
        if total > MAX_LAUNCH_TOTAL_BYTES:
            raise LiveAppLaunchError(
                f"Not launched: the app's files are over the {MAX_LAUNCH_TOTAL_BYTES // (1024 * 1024)} MB a launch sends.",
                reason="launch_files",
            )
        files.append(LaunchFileData(rel, data))
    return files


class MissingConnection(BaseModel):
    """A declared connection the app didn't get. Never a value."""

    model_config = ConfigDict(frozen=True)

    engine: str
    name: str
    reason: MissingReason


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
    port = parts.port or _default_port(parts.scheme.lower())
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
            resolved.missing.append(MissingConnection(engine=ref.engine, name=ref.name, reason="not_approved"))
            continue
        env = build_datasource_env(vault, [ref], slug=slug)
        if not env:
            resolved.missing.append(MissingConnection(engine=ref.engine, name=ref.name, reason="not_saved"))
            continue
        if entry.kind == "db":
            if not _db_points_at(env, ref.env_prefix, entry):
                resolved.missing.append(MissingConnection(engine=ref.engine, name=ref.name, reason="other_host"))
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
                resolved.missing.append(MissingConnection(engine=ref.engine, name=ref.name, reason="other_host"))
                continue
            env = rewritten
        resolved.env.update(env)
        resolved.granted.append(
            GrantedConnection(slot=SlotConnection(engine=ref.engine, name=ref.name, kind=entry.kind), approved=entry)
        )
    return resolved


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def app_hashes(files: Sequence[LaunchFileData], connections: Sequence[SlotConnection]) -> AppHashes:
    """The hashes a slot reports for an app, computed from what a launch would send."""
    by_path = {f.path: f.data for f in files}
    listed = sorted([c.engine, c.name, c.kind] for c in connections)
    return AppHashes(
        backend=_sha256(by_path.get("backend.py", b"")),
        requirements=_sha256(by_path.get("requirements.txt", b"")),
        connections=_sha256(json.dumps(listed, separators=(",", ":")).encode("utf-8")),
    )


# ---------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------

_CHANGED_PART_NAMES = {"backend": "`backend.py`", "requirements": "`requirements.txt`", "connections": "its connections"}


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


def changed_parts(running: AppHashes, new: AppHashes) -> list[str]:
    """Which approved parts differ between the running app and the new launch, in card order."""
    return [part for part in ("backend", "requirements", "connections") if getattr(running, part) != getattr(new, part)]


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
    changed: Sequence[str],
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


class LaunchDecision(BaseModel):
    """One logged launch card: the app, its connections by name, and what came of it."""

    model_config = ConfigDict(frozen=True)

    slug: str
    connections: tuple[str, ...]
    changed: tuple[str, ...]
    replaces: str | None = None
    status: str


# ---------------------------------------------------------------------------
# The launch
# ---------------------------------------------------------------------------

NO_HOST_MESSAGE = "Not launched: live apps need a host that records who launches them, and this session has none."
NO_SLOT_MESSAGE = "Not launched: no live-app slot answered. Ask the operator to check the stack's app slots."
NOT_APPROVED_MESSAGE = "Not launched: the person didn't approve the launch. The app that was running is unchanged."


class LiveAppLaunchResult(BaseModel):
    """A launch that happened."""

    model_config = ConfigDict(frozen=True)

    slot: int
    address: str
    slug: str
    missing: tuple[MissingConnection, ...] = ()
    probe: tuple[ProbeResult, ...] = ()
    log_tail: str = ""

    def message(self) -> str:
        text = (
            f"Live app `{self.slug}` is running at `{self.address}`. "
            "Give the person this address in backticks; only they can open it."
        )
        if self.missing:
            named = ", ".join(
                f"`{m.engine}/{m.name}` ({MISSING_REASON_TEXT[m.reason]})" for m in self.missing
            )
            text += (
                f" Not passed to the app: {named}. Ask the person to add or rename the "
                "connection, or their operator to approve it."
            )
        return text


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


def _holds(status: SlotStatus | None, slug: str, launcher: LiveAppLauncher) -> bool:
    return (
        status is not None and status.state == "running" and status.app is not None
        and status.app.artifact == slug and status.app.launcher.subject == launcher.subject
    )


def running_slot(statuses: Sequence[SlotStatus | None], slug: str, launcher: LiveAppLauncher) -> int | None:
    """The index of the slot that runs `slug` for `launcher`, or None."""
    return next((i for i, s in enumerate(statuses) if _holds(s, slug, launcher)), None)


@dataclass(frozen=True)
class SlotPlan:
    """The slot a launch takes, and the launcher's own other app it replaces, if any."""

    index: int
    replaces: SlotApp | None = None


def _free(status: SlotStatus) -> bool:
    # An exited app's slot is about to restart empty.
    return status.app is None or status.state != "running"


def _held_by_others(statuses: Sequence[SlotStatus | None]) -> str:
    """Who holds which slot, for the refusal the agent relays."""
    held = [
        f"slot {i + 1} runs `{s.app.artifact}` for {s.app.launcher.email} since {shown_time(s.app.started_at)}"
        for i, s in enumerate(statuses) if s is not None and s.app is not None
    ]
    expiries = sorted(s.app.expires_at for s in statuses if s is not None and s.app is not None)
    latest = f", and the first frees itself by {shown_time(expiries[0])} at the latest" if expiries else ""
    return (
        "Not launched: every app slot runs another person's app: " + "; ".join(held) + ". "
        f"Tell the person they can wait for a slot to free itself (an app stops after a while with no requests{latest}), "
        "or ask the operator for more slots."
    )


def plan_slot(statuses: Sequence[SlotStatus | None], slug: str, launcher: LiveAppLauncher) -> SlotPlan:
    """The slot a launch takes. The app's own slot when one still runs it;
    else the lowest free slot; else the launcher's own app that started
    longest ago, which the card names. Nobody else's app is ever replaced:
    with every slot held by other people, raises `LiveAppLaunchError` naming
    who holds what."""
    own = running_slot(statuses, slug, launcher)
    if own is not None:
        return SlotPlan(own)
    answering = [(i, s) for i, s in enumerate(statuses) if s is not None]
    if not answering:
        raise LiveAppLaunchError(NO_SLOT_MESSAGE, reason="no_slot")
    free = [i for i, s in answering if _free(s)]
    if free:
        return SlotPlan(free[0])
    mine = [(i, s) for i, s in answering if s.app.launcher.subject == launcher.subject]
    if mine:
        index, status = min(mine, key=lambda pair: (pair[1].app.started_at, pair[0]))
        return SlotPlan(index, replaces=status.app)
    raise LiveAppLaunchError(_held_by_others(statuses), reason="slots_full")


def _same_plan(new: SlotPlan, approved: SlotPlan) -> bool:
    """Whether `new` replaces nothing the card didn't name."""
    if new.replaces is None:
        return True
    return (
        approved.replaces is not None and new.index == approved.index
        and new.replaces.artifact == approved.replaces.artifact
        and new.replaces.started_at == approved.replaces.started_at
    )


async def _wait_for_restart(client: SlotClient, old_boot_id: str) -> SlotStatus:
    """Poll until the slot is back from a reset: a new boot id and empty."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + RESET_TIMEOUT_SECONDS
    while loop.time() < deadline:
        await asyncio.sleep(RESET_POLL_SECONDS)
        status = await _read_status(client)
        if status is not None and status.boot_id != old_boot_id and status.state == "empty":
            return status
    raise LiveAppLaunchError(
        "Not launched: the app slot did not come back empty after its restart. Ask the operator to check it.",
        reason="slot_reset",
    )


def _launch_failure(error: SlotRefused) -> LiveAppLaunchError:
    body = error.body
    tail = scrub_output(body.log_tail)
    if body.error == "missing_package":
        return LiveAppLaunchError(
            f"Not launched: the app slot lacks the package `{body.detail}`, and live apps can't install packages.",
            reason="missing_package",
        )
    if body.error == "app_not_ready":
        return LiveAppLaunchError(
            f"Error: the backend did not start in the app slot ({body.detail or 'no detail'}).\nLog tail:\n{tail}",
            reason="app_not_ready", log_tail=tail,
        )
    named = f" (field `{body.field}`)" if body.field else ""
    return LiveAppLaunchError(
        f"Not launched: the app slot refused the launch: {body.error}{named}.", reason="slot_refused",
    )


async def launch_live_app(
    session,
    *,
    slug: str,
    folder: Path,
    datasources: Sequence[DatasourceRef],
    probe_paths: Sequence[str] = (),
) -> LiveAppLaunchResult:
    """Launch `slug` into an app slot, through the owner's card when it
    changes what the app runs or reaches. Raises `LiveAppLaunchError`, whose
    text the agent relays, when nothing launched; nothing ever starts in this
    process."""
    from anton.core.utils.scratchpad import ask_approval_card

    host: LiveAppHost | None = getattr(session, "_live_app_host", None)
    if host is None:
        raise LiveAppLaunchError(NO_HOST_MESSAGE, reason="no_host")
    try:
        addresses = live_app_slot_addresses()
        approved = load_approved_connections()
    except LiveAppConfigError as exc:
        raise LiveAppLaunchError(f"Not launched: {exc} Ask the operator to fix it.", reason="config") from exc
    # This process reads the files outside the sandbox, so under a policy it
    # reads only what the person's own cells may read, as the render check
    # does: the folder's resolved path is approved, then read without links.
    folder = Path(os.path.realpath(folder))
    policy = cell_sandbox_of(getattr(session, "_scratchpads", None))
    if policy is not None and not policy.allows_read(folder):
        raise LiveAppLaunchError(
            "Not launched: the app's folder is outside the files this conversation may read.", reason="launch_files",
        )
    files = await asyncio.to_thread(read_launch_files, folder)
    resolved = resolve_connections(getattr(session, "_data_vault", None), datasources, approved, slug=slug)
    connections = tuple(g.slot for g in resolved.granted)
    hashes = app_hashes(files, connections)
    clients = [SlotClient(address, token=host.launch_token) for address in addresses]
    launcher = host.launcher

    statuses = await _read_statuses(clients)
    plan = plan_slot(statuses, slug, launcher)
    own = running_slot(statuses, slug, launcher)
    running_hashes = statuses[own].app.hashes if own is not None else None
    if running_hashes != hashes or plan.replaces is not None:
        changed = changed_parts(running_hashes, hashes) if running_hashes is not None else []
        prompt = launch_card_prompt(
            slug=slug, granted=resolved.granted, changed=changed, running=own is not None, replaces=plan.replaces,
        )
        cause = await ask_approval_card(
            session, prompt,
            approve_detail="Start the app with these connections", cancel_detail="Leave things as they are",
        )
        decision = LaunchDecision(
            slug=slug,
            connections=tuple(f"{c.engine}/{c.name}:{c.kind}" for c in connections),
            changed=tuple(changed),
            replaces=plan.replaces.artifact if plan.replaces is not None else None,
            status=cause or "approved",
        )
        # WARNING: hosts log at WARNING by default, and this line is the
        # record of an approved or refused launch.
        _log.warning("live app launch: %s", decision.model_dump_json())
        if cause is not None:
            raise LiveAppLaunchError(NOT_APPROVED_MESSAGE, reason="launch_not_approved")

    request = LaunchRequest(
        artifact=slug,
        launcher=launcher,
        files=tuple(LaunchFile(path=f.path, content_b64=base64.b64encode(f.data).decode("ascii")) for f in files),
        env=resolved.env,
        connections=connections,
    )
    # Twice at most: a slot another launch took between the read and the
    # lock is read again, once. The new plan may not replace an app the card
    # didn't name.
    approved_plan = plan
    for attempt in range(2):
        if attempt:
            statuses = await _read_statuses(clients)
            plan = plan_slot(statuses, slug, launcher)
            if not _same_plan(plan, approved_plan):
                raise LiveAppLaunchError(
                    "Not launched: the app slots changed while the card waited. Ask to launch it again.",
                    reason="slots_changed",
                )
        index = plan.index
        client, seen = clients[index], statuses[index]
        lock = _slot_locks().setdefault(client.address, asyncio.Lock())
        async with lock:
            current = await _read_status(client)
            if current is None or current.boot_id != seen.boot_id or current.state != seen.state:
                continue
            slot = index + 1
            await host.claim(slot)
            if current.state != "empty":
                await asyncio.to_thread(_reset, client)
                current = await _wait_for_restart(client, current.boot_id)
            try:
                reply = await asyncio.to_thread(client.launch, request)
            except SlotRefused as exc:
                if exc.body.error == "slot_busy":
                    continue
                failure = _launch_failure(exc)
                _write_log(folder, failure.log_tail)
                raise failure from None
            except (SlotUnavailable, ValidationError) as exc:
                raise LiveAppLaunchError(
                    f"Not launched: the app slot gave no usable answer ({type(exc).__name__}).", reason="slot_unavailable",
                ) from None
            address = app_address(slot)
            await host.launched(LiveAppLaunch(
                slot=slot, slug=slug, artifact_dir=str(folder), address=address, boot_id=reply.boot_id,
            ))
        break
    else:
        raise LiveAppLaunchError(
            "Not launched: other launches kept taking the app slot. Try again in a moment.", reason="slot_busy",
        )

    tail, probe = reply.log_tail, ()
    if probe_paths:
        try:
            probed = await asyncio.to_thread(client.probe, probe_paths)
            tail, probe = probed.log_tail, probed.results
        except (SlotUnavailable, SlotRefused, ValidationError):
            _log.warning("live app slot %s: the probe failed", client.address, exc_info=True)
            probe = tuple(ProbeResult(path=p, status=0) for p in probe_paths)
    tail = scrub_output(tail)
    _write_log(folder, tail)
    return LiveAppLaunchResult(
        slot=slot, address=address, slug=slug, missing=tuple(resolved.missing), probe=tuple(probe), log_tail=tail,
    )


def _reset(client: SlotClient) -> None:
    try:
        client.reset()
    except (SlotUnavailable, SlotRefused) as exc:
        raise LiveAppLaunchError(
            "Not launched: the app slot refused its restart. Ask the operator to check it.", reason="slot_reset",
        ) from exc


def _write_log(folder: Path, tail: str) -> None:
    """The slot's scrubbed log tail as the artifact's `backend.log`."""
    try:
        write_text_nofollow(folder, folder / BACKEND_LOG_FILENAME, tail)
    except OSError:
        _log.warning("could not write %s for %s", BACKEND_LOG_FILENAME, folder, exc_info=True)


def launch_paths_changed(before: dict[str, object], after: dict[str, object]) -> bool:
    """Whether a file a launch sends differs between two `{relative path: stamp}` maps."""
    def sent(rel: str) -> bool:
        return rel in LAUNCH_FILES or any(rel.startswith(f"{top}/") for top in LAUNCH_DIRS)

    keys = {rel for rel in (*before, *after) if sent(rel)}
    return any(before.get(rel) != after.get(rel) for rel in keys)


async def relaunch_changed_apps(session, store, slugs: Sequence[str]) -> list[str]:
    """Relaunch each of `slugs` that a slot runs for this session's launcher,
    after a cell changed its files. One line per app for the cell's result.
    Never raises."""
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
        if running_slot(statuses, slug, host.launcher) is None:
            continue
        artifact = store.open(slug)
        if artifact is None:
            continue
        try:
            result = await launch_live_app(
                session, slug=slug, folder=store.folder_for(slug), datasources=artifact.datasources,
            )
        except LiveAppLaunchError as exc:
            lines.append(f"Live app `{slug}` was not relaunched. {exc}")
            continue
        except Exception:
            _log.warning("relaunching live app %s failed", slug, exc_info=True)
            lines.append(f"Live app `{slug}` was not relaunched: the relaunch failed.")
            continue
        lines.append(f"Relaunched live app `{slug}` at `{result.address}` with this cell's changes.")
    return lines
