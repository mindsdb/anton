"""Live apps in app slots: settings, what a launch sends, the card, the slot
choice and the launch itself, against an in-memory slot."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from anton.core.artifacts import live_app_slots as las
from anton.core.artifacts.live_app_slots import (
    AppHashes,
    ApprovedConnection,
    ApprovedConnections,
    GrantedConnection,
    LaunchReply,
    LiveAppConfigError,
    LiveAppLauncher,
    LiveAppLaunchError,
    ProbeReply,
    ProbeResult,
    SlotApp,
    SlotConnection,
    SlotError,
    SlotRefused,
    SlotStatus,
    SlotUnavailable,
)
from anton.core.artifacts.models import DatasourceRef
from anton.core.utils.scratchpad import ApprovalChoice
from anton.diagnostics import make_formatter

_SLOTS = ("http://live-app-1:8001", "http://live-app-2:8001", "http://live-app-3:8001")
_TOKEN = "t" * 64
_ME = LiveAppLauncher(subject="sub-ana", email="ana@example.com")
_OTHER = LiveAppLauncher(subject="sub-bo", email="bo@example.com")
_PG_PASSWORD = "pg-sentinel-value-7f3a"
_API_KEY = "api-sentinel-value-91c2"
# A value the log scrubber redacts, as the in-process launcher's tests use.
_LOGGED_TOKEN = "FAKE0tok3nZr1xK2mN4pL7vT9wY3sB6dF"


# ── Fakes ────────────────────────────────────────────────────────────────────


class FakeVault:
    """`env_for` as the data vault answers it: namespaced `DS_*` or None."""

    def __init__(self, saved: dict[tuple[str, str], dict[str, str]]):
        self._saved = saved

    def env_for(self, engine: str, name: str, *, flat: bool = False):
        fields = self._saved.get((engine, name))
        if fields is None:
            return None
        prefix = DatasourceRef(engine=engine, name=name).env_prefix
        return {f"{prefix}__{key}": value for key, value in fields.items()}


class FakeHost:
    def __init__(self, launcher: LiveAppLauncher = _ME):
        self.launcher = launcher
        self.launch_token = _TOKEN
        self.events: list[tuple] = []
        self.refused = 0
        self.available = 0

    def launch_lock(self):
        return las._slot_locks().setdefault("test-admission", asyncio.Lock())

    async def slots_full(self):
        self.refused += 1

    async def slots_available(self):
        self.available += 1

    async def claim(self, slot: int) -> None:
        self.events.append(("claim", slot))

    async def launched(self, launch) -> None:
        self.events.append(("launched", launch))

    async def launch_failed(self, slot: int) -> None:
        self.events.append(("launch_failed", slot))


_APP_ID = re.compile(r"^[0-9a-f]{64}$")
_OLD_BOOT, _DOWN = "old", "down"


class FakeSlot:
    """One slot's control API, in memory: the rules the slot program keeps.
    A reset doesn't come back at once: for `restart_window` polls the slot
    answers with its old boot, or not at all, as the real slot's restart does."""

    def __init__(self, n: int):
        self.n = n
        self.boot = 1
        self.state = "empty"
        self.app: SlotApp | None = None
        self.down = False
        self.requests: list[las.LaunchRequest] = []
        self.resets = 0
        self.probed: list[tuple[str, ...]] = []
        self.refuse_launch: SlotRefused | None = None
        self.refuse_probe: Exception | None = None
        self.probe_status = 200
        self.log_tail = "started"
        self.restart_window: tuple[str, ...] = (_OLD_BOOT, _DOWN)
        self.comes_back = True
        self._restart_left: list[str] = []
        self._restarting = False

    @property
    def boot_id(self) -> str:
        return f"boot-{self.n}-{self.boot}"

    def status(self, *, timeout: float = 2.0) -> SlotStatus:
        if self.down:
            raise SlotUnavailable(f"slot {self.n} is down")
        if self._restarting:
            if self._restart_left:
                if self._restart_left.pop(0) == _DOWN:
                    raise SlotUnavailable(f"slot {self.n} is restarting")
                return SlotStatus(version=1, boot_id=self.boot_id, booted_at="2026-10-21T04:00:00Z",
                                  state=self.state, app=self.app)
            if not self.comes_back:
                raise SlotUnavailable(f"slot {self.n} never came back")
            self._restarting = False
            self.boot += 1
            self.state, self.app = "empty", None
        return SlotStatus(version=1, boot_id=self.boot_id, booted_at="2026-10-21T04:00:00Z",
                          state=self.state, app=self.app)

    def reset(self) -> None:
        self.resets += 1
        self._restarting, self._restart_left = True, list(self.restart_window)

    def run(self, artifact: str, launcher: LiveAppLauncher, *, started_at: str, hashes: AppHashes | None = None,
            app_id: str | None = None, state: str = "running"):
        """Put an app in this slot as a launch would have."""
        self.state = state
        self.app = SlotApp(
            artifact=artifact, app_id=app_id or f"id-{artifact}", launcher=launcher, started_at=started_at,
            expires_at="2026-10-21T12:00:00Z",
            hashes=hashes or AppHashes(backend="x", requirements="x", connections="x", files="x"),
        )

    def launch(self, request: las.LaunchRequest) -> LaunchReply:
        if self.refuse_launch is not None:
            raise self.refuse_launch
        if self.state != "empty" or self._restarting:
            raise SlotRefused(409, SlotError(error="slot_busy"))
        if not _APP_ID.fullmatch(request.app_id):
            raise SlotRefused(400, SlotError(error="invalid_field", field="app_id"))
        self.requests.append(request)
        files = [las.LaunchFileData(f.path, base64.b64decode(f.content_b64)) for f in request.files]
        self.run(request.artifact, request.launcher, started_at="2026-10-21T05:00:00Z",
                 hashes=las.app_hashes(files, request.connections), app_id=request.app_id)
        return LaunchReply(state="running", boot_id=self.boot_id, started_at="2026-10-21T05:00:00Z",
                           log_tail=self.log_tail)

    def probe(self, paths) -> ProbeReply:
        self.probed.append(tuple(paths))
        if self.refuse_probe is not None:
            raise self.refuse_probe
        return ProbeReply(results=tuple(ProbeResult(path=p, status=self.probe_status) for p in paths),
                          log_tail=self.log_tail)


class Slots(list):
    """The three fake slots. `tokens` is every token a slot client was built with."""

    tokens: list[str]


@pytest.fixture
def slots(monkeypatch) -> Slots:
    fakes = Slots(FakeSlot(n) for n in (1, 2, 3))
    fakes.tokens = []
    by_address = dict(zip(_SLOTS, fakes))

    def client(address: str, *, token: str):
        fakes.tokens.append(token)
        fake = by_address[address]
        fake.address = address
        return fake

    monkeypatch.setenv(las.SLOTS_ENV_VAR, ",".join(_SLOTS))
    monkeypatch.setattr(las, "SlotClient", client)
    monkeypatch.setattr(las, "RESET_POLL_SECONDS", 0)
    return fakes


@pytest.fixture
def approved(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "connections.json"
    path.write_text(json.dumps({"version": 1, "connections": [
        {"engine": "postgres", "name": "demo", "kind": "db", "host": "samples.mindsdb.com", "port": 5432},
        {"engine": "rest", "name": "httpbin", "kind": "http-read", "host": "httpbin.org", "port": 443,
         "relay_port": 10001},
    ]}), encoding="utf-8")
    monkeypatch.setenv(las.CONNECTIONS_FILE_ENV_VAR, str(path))
    return path


class Cards(list):
    """The prompts of every card asked. `answer` is what each card returns:
    None approves, a refusal cause refuses."""

    answer: str | None = None
    pick: int = 0
    options: tuple = ()


@pytest.fixture
def cards(monkeypatch) -> Cards:
    asked = Cards()

    async def ask(session, prompt, *, approve_detail, cancel_detail, log_label="write approval"):
        asked.append(prompt)
        return asked.answer

    import anton.core.utils.scratchpad as scratchpad

    async def choose(session, prompt, *, options, **kwargs):
        asked.append(prompt)
        asked.options = options
        return ApprovalChoice(options[asked.pick].value, None) if asked.answer is None else ApprovalChoice(
            None, asked.answer,
        )

    monkeypatch.setattr(scratchpad, "ask_approval_card", ask)
    monkeypatch.setattr(scratchpad, "ask_approval_choice", choose)
    return asked


def _artifact(root: Path, slug: str = "sales-demo", *, backend: str = "app = None\n") -> Path:
    folder = root / slug
    (folder / "static" / "js").mkdir(parents=True)
    (folder / "backend.py").write_text(backend, encoding="utf-8")
    (folder / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    (folder / "static" / "index.html").write_text("<html><body></body></html>", encoding="utf-8")
    (folder / "static" / "js" / "app.js").write_text("// app", encoding="utf-8")
    (folder / "prd.md").write_text("# PRD", encoding="utf-8")
    (folder / "backend.log").write_text("old log", encoding="utf-8")
    (folder / ".published.json").write_text("{}", encoding="utf-8")
    (folder / "metadata.json").write_text("{}", encoding="utf-8")
    return folder


def _session(host: FakeHost | None = None, saved: dict | None = None):
    saved = saved if saved is not None else {
        ("postgres", "demo"): {"HOST": "samples.mindsdb.com", "PORT": "5432", "PASSWORD": _PG_PASSWORD},
        ("rest", "httpbin"): {"BASE_URL": "https://httpbin.org/anything?x=1", "API_KEY": _API_KEY},
    }
    return SimpleNamespace(_live_app_host=host if host is not None else FakeHost(), _data_vault=FakeVault(saved))


_DECLARED = (DatasourceRef(engine="postgres", name="demo"), DatasourceRef(engine="rest", name="httpbin"))


def _runs_from(slot: FakeSlot, folder: Path, launcher: LiveAppLauncher = _ME, *,
               started_at: str = "2026-10-21T04:10:00Z", state: str = "running") -> None:
    """Put the app built from `folder` in `slot`, as an earlier launch of it would have."""
    slot.run(folder.name, launcher, started_at=started_at, app_id=las.app_identity(folder), state=state)


# ── Settings ─────────────────────────────────────────────────────────────────


def test_live_apps_are_off_without_the_slot_setting(monkeypatch):
    monkeypatch.delenv(las.SLOTS_ENV_VAR, raising=False)
    assert las.live_apps_enabled() is False
    assert las.live_app_slot_addresses() == ()
    monkeypatch.setenv(las.SLOTS_ENV_VAR, "  ")
    assert las.live_apps_enabled() is False


def test_slot_addresses_keep_their_order(monkeypatch):
    monkeypatch.setenv(las.SLOTS_ENV_VAR, " http://live-app-2:8001 ,http://live-app-1:8001,")
    assert las.live_apps_enabled() is True
    assert las.live_app_slot_addresses() == ("http://live-app-2:8001", "http://live-app-1:8001")


@pytest.mark.parametrize("entry", ["live-app-1:8001", "https://live-app-1:8001", "http://live-app-1",
                                   "http://live-app-1:8001/x", "http://live-app-1:99999"])
def test_a_malformed_slot_address_refuses_and_names_the_setting(monkeypatch, entry):
    monkeypatch.setenv(las.SLOTS_ENV_VAR, f"http://live-app-1:8001,{entry}")
    with pytest.raises(LiveAppConfigError, match=f"{las.SLOTS_ENV_VAR} entry 2"):
        las.live_app_slot_addresses()


@pytest.mark.parametrize("value, position", [
    ("http://live-app-1:8001,,http://live-app-3:8001", 2),
    (",http://live-app-2:8001", 1),
])
def test_a_blank_slot_entry_refuses_rather_than_renumbering(monkeypatch, value, position):
    """Slot n is the nth entry: a blank one dropped would move every app after
    it to another slot's number, and so another person's address."""
    monkeypatch.setenv(las.SLOTS_ENV_VAR, value)
    with pytest.raises(LiveAppConfigError, match=f"{las.SLOTS_ENV_VAR} entry {position}"):
        las.live_app_slot_addresses()


def test_a_bad_setting_is_a_value_error_for_hosts_that_catch_one(monkeypatch):
    monkeypatch.setenv(las.SLOTS_ENV_VAR, "not-an-address")
    with pytest.raises(ValueError):
        las.live_app_slot_addresses()


def test_the_app_address_comes_from_the_slot_number(monkeypatch):
    monkeypatch.delenv(las.ADDRESS_ENV_VAR, raising=False)
    assert las.app_address(2) == "http://app2.view.localhost:8080/"
    monkeypatch.setenv(las.ADDRESS_ENV_VAR, "http://app{n}.demo.test:9000/")
    assert las.app_address(3) == "http://app3.demo.test:9000/"


def test_the_approved_connections_file_loads(approved):
    loaded = las.load_approved_connections()
    assert [(c.engine, c.kind, c.relay_port) for c in loaded.connections] == [
        ("postgres", "db", None), ("rest", "http-read", 10001),
    ]


def test_a_read_only_api_may_list_post_and_graphql_paths(tmp_path, monkeypatch):
    path = tmp_path / "connections.json"
    path.write_text(json.dumps({"version": 1, "connections": [
        {"engine": "rest", "name": "planner", "kind": "http-read", "host": "planner.example.com", "port": 443,
         "relay_port": 10001, "post_paths": ["/search"], "graphql_paths": ["/graphql"]},
    ]}), encoding="utf-8")
    monkeypatch.setenv(las.CONNECTIONS_FILE_ENV_VAR, str(path))
    (entry,) = las.load_approved_connections().connections
    assert entry.post_paths == ("/search",) and entry.graphql_paths == ("/graphql",)


@pytest.mark.parametrize("entry, problem", [
    ({"engine": "rest", "name": "a", "kind": "http-read", "host": "a.example.com", "port": 443}, "relay_port"),
    ({"engine": "pg", "name": "a", "kind": "db", "host": "db.example.com", "port": 5432, "relay_port": 1}, "relay_port"),
    ({"engine": "pg", "name": "a", "kind": "write", "host": "db.example.com", "port": 5432}, "kind"),
    ({"engine": "pg", "name": "a", "kind": "db", "host": "DB.example.com", "port": 5432}, "lower-case"),
    ({"engine": "pg", "name": "a", "kind": "db", "host": "db.example.com:5432", "port": 5432}, "lower-case"),
    ({"engine": "rest", "name": "a", "kind": "http-read", "host": "a.example.com", "port": 443, "relay_port": 1,
      "overrides_write_hold": True}, "only an http-write"),
    ({"engine": "rest", "name": "a", "kind": "http-write", "host": "a.example.com", "port": 443, "relay_port": 1,
      "graphql_paths": ["/graphql"]}, "only an http-read"),
    ({"engine": "rest", "name": "a", "kind": "http-read", "host": "a.example.com", "port": 443, "relay_port": 1,
      "post_paths": ["search"]}, "starts with /"),
    ({"engine": "rest", "name": "a", "kind": "http-read", "host": "a.example.com", "port": 443, "relay_port": 1,
      "get_paths": ["/x"]}, "get_paths"),
])
def test_a_bad_approved_connection_refuses_the_whole_file(tmp_path, monkeypatch, entry, problem):
    path = tmp_path / "connections.json"
    path.write_text(json.dumps({"version": 1, "connections": [entry]}), encoding="utf-8")
    monkeypatch.setenv(las.CONNECTIONS_FILE_ENV_VAR, str(path))
    with pytest.raises(LiveAppConfigError, match=las.CONNECTIONS_FILE_ENV_VAR) as raised:
        las.load_approved_connections()
    assert problem in str(raised.value)


def test_a_connection_approved_twice_refuses_the_file(tmp_path, monkeypatch):
    entry = {"engine": "pg", "name": "a", "kind": "db", "host": "db.example.com", "port": 5432}
    path = tmp_path / "connections.json"
    path.write_text(json.dumps({"version": 1, "connections": [entry, {**entry, "name": "A"}]}), encoding="utf-8")
    monkeypatch.setenv(las.CONNECTIONS_FILE_ENV_VAR, str(path))
    with pytest.raises(LiveAppConfigError, match="approved twice"):
        las.load_approved_connections()


def test_two_entries_on_one_relay_port_refuse_the_file(tmp_path, monkeypatch):
    """One relay port per entry: a shared one would carry one connection's
    values to the other entry's upstream."""
    path = tmp_path / "connections.json"
    path.write_text(json.dumps({"version": 1, "connections": [
        {"engine": "rest", "name": "a", "kind": "http-read", "host": "a.example.com", "port": 443, "relay_port": 10001},
        {"engine": "rest", "name": "b", "kind": "http-write", "host": "b.example.com", "port": 443, "relay_port": 10001},
    ]}), encoding="utf-8")
    monkeypatch.setenv(las.CONNECTIONS_FILE_ENV_VAR, str(path))
    with pytest.raises(LiveAppConfigError, match="rest/a and rest/b share relay_port 10001"):
        las.load_approved_connections()


@pytest.mark.parametrize("raw", [b"\xff", b'{"version": 1, "connections": [{"engine": "pg", "name": "d\xe9mo"}]}'])
def test_a_file_that_isnt_utf8_refuses_as_an_invalid_file(tmp_path, monkeypatch, raw):
    path = tmp_path / "connections.json"
    path.write_bytes(raw)
    monkeypatch.setenv(las.CONNECTIONS_FILE_ENV_VAR, str(path))
    with pytest.raises(LiveAppConfigError, match=f"{las.CONNECTIONS_FILE_ENV_VAR} is not valid"):
        las.load_approved_connections()


def test_the_connections_file_names_antons_own_setting():
    assert las.CONNECTIONS_FILE_ENV_VAR in las.ApprovedConnections.__doc__


def test_a_missing_approved_connections_file_refuses(tmp_path, monkeypatch):
    monkeypatch.delenv(las.CONNECTIONS_FILE_ENV_VAR, raising=False)
    with pytest.raises(LiveAppConfigError, match="is not set"):
        las.load_approved_connections()
    monkeypatch.setenv(las.CONNECTIONS_FILE_ENV_VAR, str(tmp_path / "absent.json"))
    with pytest.raises(LiveAppConfigError, match="can't be read"):
        las.load_approved_connections()


# ── What a launch sends ──────────────────────────────────────────────────────


def test_a_launch_sends_only_the_launch_list(tmp_path):
    folder = _artifact(tmp_path)
    sent = {f.path for f in las.read_launch_files(folder)}
    assert sent == {"backend.py", "requirements.txt", "static/index.html", "static/js/app.js"}


def test_a_launch_needs_a_backend(tmp_path):
    folder = _artifact(tmp_path)
    (folder / "backend.py").unlink()
    with pytest.raises(LiveAppLaunchError, match="no `backend.py`"):
        las.read_launch_files(folder)


def test_a_symlinked_file_refuses_the_launch_without_its_content(tmp_path):
    folder = _artifact(tmp_path)
    secret = tmp_path / "outside.txt"
    secret.write_text(_PG_PASSWORD, encoding="utf-8")
    os.symlink(secret, folder / "static" / "leak.txt")
    with pytest.raises(LiveAppLaunchError) as raised:
        las.read_launch_files(folder)
    assert "`static/leak.txt` is a link" in str(raised.value)
    assert _PG_PASSWORD not in str(raised.value)


def test_a_symlinked_backend_refuses_the_launch(tmp_path):
    folder = _artifact(tmp_path)
    real = tmp_path / "real_backend.py"
    real.write_text("app = None\n", encoding="utf-8")
    (folder / "backend.py").unlink()
    os.symlink(real, folder / "backend.py")
    with pytest.raises(LiveAppLaunchError, match="`backend.py`"):
        las.read_launch_files(folder)


def test_a_symlinked_static_folder_refuses_the_launch(tmp_path):
    folder = _artifact(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    os.symlink(elsewhere, folder / "static" / "more")
    with pytest.raises(LiveAppLaunchError, match="`static/more` is a link"):
        las.read_launch_files(folder)


def test_a_hard_linked_file_refuses_the_launch(tmp_path):
    folder = _artifact(tmp_path)
    os.link(folder / "static" / "index.html", tmp_path / "second-name.html")
    with pytest.raises(LiveAppLaunchError, match="`static/index.html` has another hard link"):
        las.read_launch_files(folder)


def test_a_backslash_in_a_name_refuses_the_launch_without_its_content(tmp_path):
    folder = _artifact(tmp_path)
    (folder / "static" / "a\\b.js").write_text(_PG_PASSWORD, encoding="utf-8")
    with pytest.raises(LiveAppLaunchError) as raised:
        las.read_launch_files(folder)
    assert "`static/a\\b.js` has a name live apps can't send" in str(raised.value)
    assert _PG_PASSWORD not in str(raised.value)


def test_a_linked_static_folder_itself_refuses_the_launch(tmp_path):
    folder = _artifact(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "secret.js").write_text(_PG_PASSWORD, encoding="utf-8")
    import shutil

    shutil.rmtree(folder / "static")
    os.symlink(elsewhere, folder / "static")
    with pytest.raises(LiveAppLaunchError, match="`static` is a link"):
        las.read_launch_files(folder)


@pytest.mark.parametrize("linked", ["folder", "parent"])
def test_the_log_write_never_follows_a_link(tmp_path, linked):
    """The log is written from the filesystem root without links, as the
    files are read: a link at the folder or above it sends it nowhere else."""
    folder = _artifact(tmp_path / "artifacts")
    other = _artifact(tmp_path / "other", "theirs")
    (other / "backend.log").write_text("theirs", encoding="utf-8")
    resolved = Path(os.path.realpath(folder))
    if linked == "folder":
        import shutil

        shutil.rmtree(folder)
        os.symlink(other, folder)
    else:
        moved = tmp_path / "moved"
        os.rename(tmp_path / "artifacts", moved)
        os.symlink(other.parent, tmp_path / "artifacts")
        (other.parent / "sales-demo").mkdir(exist_ok=True)
        (other.parent / "sales-demo" / "backend.log").write_text("theirs", encoding="utf-8")
    las._write_log(resolved, "the app's text")
    assert (other / "backend.log").read_text(encoding="utf-8") == "theirs"
    if linked == "parent":
        assert (other.parent / "sales-demo" / "backend.log").read_text(encoding="utf-8") == "theirs"


def test_the_log_write_lands_in_a_plain_folder(tmp_path):
    folder = Path(os.path.realpath(_artifact(tmp_path)))
    las._write_log(folder, "normal log")
    assert (folder / "backend.log").read_text(encoding="utf-8") == "normal log"


def test_a_file_over_the_cap_refuses_the_launch(tmp_path, monkeypatch):
    folder = _artifact(tmp_path)
    monkeypatch.setattr(las, "MAX_LAUNCH_FILE_BYTES", 10)
    with pytest.raises(LiveAppLaunchError, match="per file"):
        las.read_launch_files(folder)


def test_files_over_the_total_cap_refuse_the_launch(tmp_path, monkeypatch):
    folder = _artifact(tmp_path)
    monkeypatch.setattr(las, "MAX_LAUNCH_TOTAL_BYTES", 30)
    with pytest.raises(LiveAppLaunchError, match="app's files are over"):
        las.read_launch_files(folder)


def test_too_many_files_refuse_the_launch(tmp_path, monkeypatch):
    folder = _artifact(tmp_path)
    monkeypatch.setattr(las, "MAX_LAUNCH_FILES", 3)
    with pytest.raises(LiveAppLaunchError, match="more than the 3"):
        las.read_launch_files(folder)


def test_the_walk_stops_at_the_cap(tmp_path, monkeypatch):
    """A tree far over the cap is refused at the folder that crosses it, not
    after the whole tree is listed."""
    folder = _artifact(tmp_path)
    for n in range(5):
        (folder / "static" / f"d{n}").mkdir()
        (folder / "static" / f"d{n}" / "f.js").write_text("", encoding="utf-8")
    monkeypatch.setattr(las, "MAX_LAUNCH_FILES", 4)
    visited: list[str] = []
    real_walk = os.walk

    def walk(top, **kw):
        for entry in real_walk(top, **kw):
            visited.append(entry[0])
            yield entry

    monkeypatch.setattr(las.os, "walk", walk)
    with pytest.raises(LiveAppLaunchError, match="more than the 4"):
        las.read_launch_files(folder)
    assert len(visited) < 7


# ── Which connections a launch sends ─────────────────────────────────────────


def _approved_set() -> ApprovedConnections:
    return ApprovedConnections(version=1, connections=(
        ApprovedConnection(engine="postgres", name="demo", kind="db", host="samples.mindsdb.com", port=5432),
        ApprovedConnection(engine="rest", name="httpbin", kind="http-read", host="httpbin.org", port=443,
                           relay_port=10001),
    ))


def test_an_approved_database_on_its_approved_host_is_sent():
    vault = FakeVault({("postgres", "demo"): {"HOST": "Samples.MindsDB.com ", "PORT": "5432", "PASSWORD": "pw"}})
    resolved = las.resolve_connections(vault, _DECLARED[:1], _approved_set(), slug="a")
    assert resolved.env == {"DS_POSTGRES_DEMO__HOST": "Samples.MindsDB.com ", "DS_POSTGRES_DEMO__PORT": "5432",
                            "DS_POSTGRES_DEMO__PASSWORD": "pw"}
    assert [g.slot for g in resolved.granted] == [SlotConnection(engine="postgres", name="demo", kind="db")]
    assert resolved.missing == []


def test_an_api_url_points_at_the_relay_with_its_path_and_query_kept():
    vault = FakeVault({("rest", "httpbin"): {
        "BASE_URL": "https://httpbin.org/anything/v1?x=1", "TOKEN_URL": "https://auth.example.com/token",
        "API_KEY": _API_KEY,
    }})
    resolved = las.resolve_connections(vault, _DECLARED[1:], _approved_set(), slug="a")
    assert resolved.env == {
        "DS_REST_HTTPBIN__BASE_URL": "http://live-apps-relay:10001/anything/v1?x=1",
        # Another host's URL passes unchanged: a slot can't reach it anyway.
        "DS_REST_HTTPBIN__TOKEN_URL": "https://auth.example.com/token",
        "DS_REST_HTTPBIN__API_KEY": _API_KEY,
    }


def test_an_api_url_keeps_its_user_info_through_the_relay():
    vault = FakeVault({("rest", "httpbin"): {"URL": "https://user:pw@httpbin.org:443"}})
    resolved = las.resolve_connections(vault, _DECLARED[1:], _approved_set(), slug="a")
    assert resolved.env == {"DS_REST_HTTPBIN__URL": "http://user:pw@live-apps-relay:10001"}


@pytest.mark.parametrize("ref, saved, reason", [
    (DatasourceRef(engine="snowflake", name="sales"), {("snowflake", "sales"): {"PASSWORD": _PG_PASSWORD}},
     "not_approved"),
    (DatasourceRef(engine="postgres", name="demo"), {}, "not_saved"),
    (DatasourceRef(engine="postgres", name="demo"),
     {("postgres", "demo"): {"HOST": "evil.example.com", "PASSWORD": _PG_PASSWORD}}, "other_host"),
    (DatasourceRef(engine="postgres", name="demo"),
     {("postgres", "demo"): {"HOST": "samples.mindsdb.com", "PORT": "6543", "PASSWORD": _PG_PASSWORD}}, "other_host"),
    (DatasourceRef(engine="postgres", name="demo"), {("postgres", "demo"): {"PASSWORD": _PG_PASSWORD}}, "other_host"),
    (DatasourceRef(engine="rest", name="httpbin"),
     {("rest", "httpbin"): {"BASE_URL": "https://httpbin.org:8443/", "API_KEY": _PG_PASSWORD}}, "other_host"),
    (DatasourceRef(engine="rest", name="httpbin"),
     {("rest", "httpbin"): {"BASE_URL": "https://evil.example.com/", "API_KEY": _PG_PASSWORD}}, "other_host"),
    # Only an absent port takes the scheme's default.
    (DatasourceRef(engine="rest", name="httpbin"),
     {("rest", "httpbin"): {"BASE_URL": "https://httpbin.org:0/", "API_KEY": _PG_PASSWORD}}, "other_host"),
    # The slot refuses any other `DS_*` name, and with it the whole launch.
    (DatasourceRef(engine="rest", name="httpbin"),
     {("rest", "httpbin"): {"BASE_URL": "https://httpbin.org/", "api-key": _PG_PASSWORD}}, "field_names"),
])
def test_a_connection_left_out_is_named_without_a_value(ref, saved, reason):
    resolved = las.resolve_connections(FakeVault(saved), [ref], _approved_set(), slug="a")
    assert resolved.env == {}
    assert resolved.granted == []
    assert [m.model_dump() for m in resolved.missing] == [{"engine": ref.engine, "name": ref.name, "reason": reason}]
    result = las.LiveAppLaunchResult(slot=1, address="http://app1.view.localhost:8080/", slug="a",
                                     missing=tuple(resolved.missing))
    assert f"`{ref.engine}/{ref.name}` ({las.MISSING_REASON_TEXT[reason]})" in result.message()
    assert _PG_PASSWORD not in result.message()


def test_an_api_approved_for_writes_points_at_its_relay_and_carries_its_override():
    approved = ApprovedConnections(version=1, connections=(
        ApprovedConnection(engine="rest", name="planner", kind="http-write", host="planner.example.com", port=443,
                           relay_port=10003, overrides_write_hold=True),
    ))
    ref = DatasourceRef(engine="rest", name="planner")
    vault = FakeVault({("rest", "planner"): {"BASE_URL": "https://planner.example.com/api/v1?region=eu"}})
    resolved = las.resolve_connections(vault, [ref], approved, slug="a")
    assert resolved.env == {"DS_REST_PLANNER__BASE_URL": "http://live-apps-relay:10003/api/v1?region=eu"}
    assert [g.slot for g in resolved.granted] == [
        SlotConnection(engine="rest", name="planner", kind="http-write", overrides_write_hold=True),
    ]
    elsewhere = FakeVault({("rest", "planner"): {"BASE_URL": "https://evil.example.com/api"}})
    assert las.resolve_connections(elsewhere, [ref], approved, slug="a").granted == []


def test_engine_and_name_match_the_approval_ignoring_case():
    vault = FakeVault({("Postgres", "Demo"): {"HOST": "samples.mindsdb.com"}})
    resolved = las.resolve_connections(vault, [DatasourceRef(engine="Postgres", name="Demo")], _approved_set(), slug="a")
    assert [g.approved.name for g in resolved.granted] == ["demo"]


# ── What the owner approves ──────────────────────────────────────────────────


def _files(**overrides: bytes) -> list[las.LaunchFileData]:
    files = {"backend.py": b"app = None\n", "requirements.txt": b"fastapi\n", "static/index.html": b"<html>"}
    files.update(overrides)
    return [las.LaunchFileData(path, data) for path, data in files.items()]


def test_a_backend_change_shows_in_its_hash():
    before, after = las.app_hashes(_files(), []), las.app_hashes(_files(**{"backend.py": b"app = 1\n"}), [])
    assert las.changed_parts(before, after) == ["backend"]


def test_the_connections_hash_ignores_order_and_sees_a_new_kind():
    a = SlotConnection(engine="pg", name="a", kind="db")
    b = SlotConnection(engine="rest", name="b", kind="http-read")
    assert las.app_hashes(_files(), [a, b]) == las.app_hashes(_files(), [b, a])
    widened = SlotConnection(engine="rest", name="b", kind="http-write")
    assert las.changed_parts(las.app_hashes(_files(), [a, b]), las.app_hashes(_files(), [a, widened])) == ["connections"]


def test_an_override_of_the_write_hold_is_part_of_the_approval():
    plain = SlotConnection(engine="rest", name="planner", kind="http-write")
    overriding = SlotConnection(engine="rest", name="planner", kind="http-write", overrides_write_hold=True)
    assert las.changed_parts(las.app_hashes(_files(), [plain]), las.app_hashes(_files(), [overriding])) == [
        "connections"
    ]


def test_an_absent_requirements_file_hashes_as_empty():
    files = [f for f in _files() if f.path != "requirements.txt"]
    assert las.app_hashes(files, []).requirements == hashlib.sha256(b"").hexdigest()


def test_the_hashes_follow_the_slot_contract():
    """Written out from the launch contract, not from `app_hashes`: the slot
    computes the same values from what it received, and a drift here would
    ask the card on every relaunch."""
    files = [las.LaunchFileData("backend.py", b"app = None\n"), las.LaunchFileData("static/index.html", b"<html>")]
    conns = [
        SlotConnection(engine="rest", name="b", kind="http-write", overrides_write_hold=True),
        SlotConnection(engine="Postgres", name="demo", kind="db"),
    ]
    got = las.app_hashes(files, conns)
    backend_sha = hashlib.sha256(b"app = None\n").hexdigest()
    page_sha = hashlib.sha256(b"<html>").hexdigest()
    assert got.backend == backend_sha
    assert got.requirements == hashlib.sha256(b"").hexdigest()
    assert got.connections == hashlib.sha256(
        b'[["Postgres","demo","db",false],["rest","b","http-write",true]]'
    ).hexdigest()
    assert got.files == hashlib.sha256(
        f'[["backend.py","{backend_sha}"],["static/index.html","{page_sha}"]]'.encode()
    ).hexdigest()


def test_a_static_change_changes_only_the_files_hash():
    before, after = las.app_hashes(_files(), []), las.app_hashes(_files(**{"static/index.html": b"<html>v2"}), [])
    assert las.changed_parts(before, after) == [] and before.files != after.files


def test_the_app_id_is_the_folders_resolved_path(tmp_path):
    folder = _artifact(tmp_path)
    assert las.app_identity(folder) == hashlib.sha256(os.fsencode(os.path.realpath(folder))).hexdigest()
    assert las.app_identity(folder) != las.app_identity(_artifact(tmp_path / "other"))


_CARD_CASES = yaml.safe_load((Path(__file__).parent / "fixtures" / "live_app_cards.yaml").read_text())["cases"]


@pytest.mark.parametrize("case", _CARD_CASES, ids=[c["name"] for c in _CARD_CASES])
def test_the_launch_card_reads_as_written(case):
    granted = [
        GrantedConnection(slot=SlotConnection(engine=c["engine"], name=c["name"], kind=c["kind"]),
                          approved=ApprovedConnection(**c))
        for c in case["connections"]
    ]
    replaces = None
    if case.get("replaces"):
        replaces = SlotApp(launcher=_ME, expires_at="2026-10-21T12:00:00Z", app_id="id-x",
                           hashes=AppHashes(backend="x", requirements="x", connections="x", files="x"),
                           **case["replaces"])
    prompt = las.launch_card_prompt(slug=case["slug"], granted=granted, changed=case["changed"],
                                    running=case["running"], replaces=replaces)
    assert prompt == case["card"].rstrip("\n")


# ── Which slot a launch takes ────────────────────────────────────────────────


def _status(n: int, *, state: str = "empty", app: SlotApp | None = None) -> SlotStatus:
    return SlotStatus(version=1, boot_id=f"b{n}", booted_at="2026-10-21T04:00:00Z", state=state, app=app)


def _app(artifact: str, launcher: LiveAppLauncher, started_at: str) -> SlotApp:
    return SlotApp(artifact=artifact, app_id=f"id-{artifact}", launcher=launcher, started_at=started_at,
                   expires_at="2026-10-21T12:00:00Z",
                   hashes=AppHashes(backend="x", requirements="x", connections="x", files="x"))


def test_a_relaunch_takes_the_slot_that_still_runs_the_app():
    statuses = [_status(1), _status(2, state="running", app=_app("sales", _ME, "2026-10-21T04:10:00Z"))]
    assert las.plan_slot(statuses, "id-sales", _ME) == las.SlotPlan(1)


def test_the_same_slug_from_another_folder_is_another_app():
    """Two of one person's artifacts with one slug, in two projects, never
    count as one app: the launch takes a free slot and stops neither."""
    statuses = [_status(1, state="running", app=_app("sales", _ME, "2026-10-21T04:10:00Z")), _status(2)]
    assert las.plan_slot(statuses, "id-another-folder", _ME) == las.SlotPlan(1)


def test_a_relaunch_after_a_stop_takes_the_lowest_free_slot():
    """A stopped app's slot restarts empty and keeps no record of it, so the
    relaunch is a launch like any other: its address can change."""
    assert las.plan_slot([_status(1), _status(2)], "id-sales", _ME) == las.SlotPlan(0)


def test_a_new_app_takes_the_lowest_free_slot():
    statuses = [_status(1, state="running", app=_app("x", _OTHER, "2026-10-21T04:10:00Z")), None,
                _status(3), _status(4, state="exited", app=_app("y", _OTHER, "2026-10-21T04:11:00Z"))]
    assert las.plan_slot(statuses, "id-sales", _ME) == las.SlotPlan(2)


def test_an_exited_app_frees_its_slot():
    statuses = [_status(1, state="exited", app=_app("y", _OTHER, "2026-10-21T04:11:00Z"))]
    assert las.plan_slot(statuses, "id-sales", _ME) == las.SlotPlan(0)


def test_with_no_free_slot_a_launch_requires_the_person_to_pick_their_app():
    mine_new = _app("newer", _ME, "2026-10-21T04:30:00Z")
    mine_old = _app("older", _ME, "2026-10-21T04:20:00Z")
    statuses = [_status(1, state="running", app=_app("x", _OTHER, "2026-10-21T04:00:00Z")),
                _status(2, state="running", app=mine_new), _status(3, state="running", app=mine_old)]
    with pytest.raises(las.SlotSelectionRequired) as asked:
        las.plan_slot(statuses, "id-sales", _ME)
    assert [p.index for p in asked.value.choices] == [1, 2]
    assert [p.replaces.artifact for p in asked.value.choices] == ["newer", "older"]


def test_a_launch_never_replaces_another_persons_app():
    statuses = [_status(n, state="running", app=_app(f"app{n}", _OTHER, f"2026-10-21T04:0{n}:00Z")) for n in (1, 2, 3)]
    with pytest.raises(LiveAppLaunchError) as raised:
        las.plan_slot(statuses, "id-sales", _ME)
    assert raised.value.cause == "slots_full"
    text = str(raised.value)
    assert text.startswith("Not launched: no app slot is free for you: slot 1 runs `app1` for bo@example.com")
    assert "slot 1 runs `app1` for bo@example.com since 04:01 UTC" in text
    assert "slot 3 runs `app3` for bo@example.com since 04:03 UTC" in text
    assert "by 12:00 UTC at the latest" in text
    assert "ask the operator for more slots" in text


def test_a_refusal_names_a_slot_that_gave_no_answer():
    statuses = [_status(1, state="running", app=_app("app1", _OTHER, "2026-10-21T04:01:00Z")), None]
    with pytest.raises(LiveAppLaunchError) as raised:
        las.plan_slot(statuses, "id-sales", _ME)
    assert "slot 1 runs `app1` for bo@example.com since 04:01 UTC; slot 2 gave no answer" in str(raised.value)


def test_with_no_slot_answering_a_launch_is_refused():
    with pytest.raises(LiveAppLaunchError) as raised:
        las.plan_slot([None, None, None], "id-sales", _ME)
    assert raised.value.cause == "no_slot"


# ── The launch ───────────────────────────────────────────────────────────────


async def test_a_first_launch_asks_the_card_then_sends_only_approved_values(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    session = _session()
    declared = (*_DECLARED, DatasourceRef(engine="snowflake", name="sales"))
    result = await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=declared,
                                       probe_paths=["/api/health"])

    assert len(cards) == 1
    assert "- `postgres/demo`: reads, and writes if its database role allows" in cards[0]
    assert "- `rest/httpbin`: reads" in cards[0]
    request = slots[0].requests[0]
    assert {f.path for f in request.files} == {"backend.py", "requirements.txt", "static/index.html",
                                              "static/js/app.js"}
    assert request.env["DS_POSTGRES_DEMO__PASSWORD"] == _PG_PASSWORD
    assert request.env["DS_REST_HTTPBIN__BASE_URL"] == "http://live-apps-relay:10001/anything?x=1"
    assert not any(name.startswith("DS_SNOWFLAKE") for name in request.env)
    assert request.launcher == _ME
    assert request.app_id == las.app_identity(folder)
    assert [(c.engine, c.kind) for c in request.connections] == [("postgres", "db"), ("rest", "http-read")]
    # Every slot client carries the host's launch token.
    assert slots.tokens and set(slots.tokens) == {_TOKEN}

    assert result.slot == 1 and result.address == "http://app1.view.localhost:8080/"
    assert [m.reason for m in result.missing] == ["not_approved"]
    assert slots[0].probed == [("/api/health",)]
    message = result.message()
    assert "`http://app1.view.localhost:8080/`" in message and "127.0.0.1" not in message
    assert _PG_PASSWORD not in message and _API_KEY not in message

    # A free slot has no record to clear, so only the launch is recorded.
    (event,) = session._live_app_host.events
    assert event[0] == "launched"
    assert event[1].model_dump() == {"slot": 1, "slug": "sales-demo", "artifact_dir": str(folder),
                                     "address": "http://app1.view.localhost:8080/", "boot_id": "boot-1-1"}


async def test_the_launch_writes_the_scrubbed_log_tail_and_no_port(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    slots[0].log_tail = f"{{'access_token': '{_LOGGED_TOKEN}'}}"
    await las.launch_live_app(_session(), slug="sales-demo", folder=folder, datasources=())
    log = (folder / "backend.log").read_text(encoding="utf-8")
    assert "access_token" in log and _LOGGED_TOKEN not in log
    assert (folder / "metadata.json").read_text(encoding="utf-8") == "{}"


async def test_a_refused_card_leaves_the_running_app_alone(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    _runs_from(slots[0], folder)
    cards.answer = "cancelled"
    session = _session()
    with pytest.raises(LiveAppLaunchError) as raised:
        await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=())
    assert raised.value.cause == "launch_not_approved"
    assert str(raised.value) == "Not launched: the person didn't approve the launch. Their running apps are unchanged."
    assert session._live_app_host.events == []
    assert slots[0].resets == 0 and slots[0].requests == []
    assert slots[0].app.artifact == "sales-demo"


def _logged_decisions(caplog) -> list[dict]:
    """Each launch decision this module logged, as a host's structured formatter writes it."""
    formatter = make_formatter()
    return [json.loads(formatter.format(record)) for record in caplog.records
            if record.name == las.__name__ and record.funcName == "_log_launch_decision"]


async def test_an_approved_launch_logs_its_app_and_connections_through_a_structured_host(
        tmp_path, slots, approved, cards, caplog):
    folder = _artifact(tmp_path)
    with caplog.at_level(logging.WARNING, logger=las.__name__):
        await las.launch_live_app(_session(), slug="sales-demo", folder=folder, datasources=_DECLARED)

    (line,) = _logged_decisions(caplog)
    assert (line["event"], line["outcome"]) == ("live_app_launch", "approved")
    # cowork-server's formatter also keeps the slug as a field of its own.
    (record,) = [r for r in caplog.records if r.funcName == "_log_launch_decision"]
    assert record.artifact_slug == "sales-demo"
    decision = json.loads(line["message"].removeprefix("live app launch: "))
    assert decision["slug"] == "sales-demo" and decision["status"] == "approved"
    assert decision["connections"] == ["postgres/demo:db", "rest/httpbin:http-read"]
    assert _PG_PASSWORD not in line["message"] and _API_KEY not in line["message"]
    assert _ME.email not in json.dumps(line)


async def test_a_refused_launch_logs_its_outcome_through_a_structured_host(tmp_path, slots, approved, cards,
                                                                        caplog):
    cards.answer = "cancelled"
    with caplog.at_level(logging.WARNING, logger=las.__name__):
        with pytest.raises(LiveAppLaunchError):
            await las.launch_live_app(_session(), slug="sales-demo", folder=_artifact(tmp_path), datasources=())

    (line,) = _logged_decisions(caplog)
    assert (line["event"], line["outcome"]) == ("live_app_launch", "cancelled")
    decision = json.loads(line["message"].removeprefix("live app launch: "))
    assert (decision["slug"], decision["status"]) == ("sales-demo", "cancelled")


async def test_a_static_change_relaunches_without_a_card(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    session = _session()
    await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=_DECLARED[:1])
    (folder / "static" / "index.html").write_text("<html><body>v2</body></html>", encoding="utf-8")
    result = await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=_DECLARED[:1])
    assert len(cards) == 1
    assert slots[0].resets == 1 and len(slots[0].requests) == 2
    assert result.slot == 1


async def test_a_backend_change_relaunches_through_the_card(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    session = _session()
    await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=())
    (folder / "backend.py").write_text("app = 'v2'\n", encoding="utf-8")
    await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=())
    assert len(cards) == 2
    assert "Since you last approved it, these changed: `backend.py`." in cards[1]
    assert "leaves your running apps as they are" in cards[1]


async def test_a_requirements_change_relaunches_through_the_card(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    session = _session()
    await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=())
    (folder / "requirements.txt").write_text("fastapi\nhttpx\n", encoding="utf-8")
    await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=())
    assert len(cards) == 2
    assert "Since you last approved it, these changed: `requirements.txt`." in cards[1]


async def test_a_new_connection_relaunches_through_the_card(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    session = _session()
    await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=_DECLARED[:1])
    await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=_DECLARED)
    assert len(cards) == 2
    assert "these changed: its connections." in cards[1]


async def test_a_full_house_of_my_apps_replaces_my_selected_newer_app(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    slots[0].run("mine-new", _ME, started_at="2026-10-21T04:30:00Z")
    slots[1].run("mine-old", _ME, started_at="2026-10-21T04:20:00Z")
    slots[2].run("theirs", _OTHER, started_at="2026-10-21T04:00:00Z")
    result = await las.launch_live_app(_session(), slug="sales-demo", folder=folder, datasources=())
    assert "Choose which of your running apps" in cards[0]
    assert [o.label for o in cards.options] == ["Stop mine-new", "Stop mine-old"]
    assert "04:30 UTC" in cards.options[0].detail and "last request none yet" in cards.options[0].detail
    assert result.slot == 1
    assert slots[0].resets == 1 and slots[2].resets == 0 and slots[1].resets == 0
    assert slots[2].app.artifact == "theirs"


async def test_every_slot_held_by_others_refuses_before_any_card(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    for n, slot in enumerate(slots, start=1):
        slot.run(f"theirs-{n}", _OTHER, started_at=f"2026-10-21T04:0{n}:00Z")
    session = _session()
    with pytest.raises(LiveAppLaunchError) as raised:
        await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=())
    assert raised.value.cause == "slots_full"
    assert cards == [] and session._live_app_host.events == []
    assert session._live_app_host.refused == 1
    assert all(slot.resets == 0 for slot in slots)


async def test_a_slot_taken_while_the_card_waited_is_never_replaced_unasked(tmp_path, slots, approved, monkeypatch):
    """Slot 1 is free when the card asks; another person's launch takes it
    meanwhile, and the only other choice would stop an app the card didn't name."""
    folder = _artifact(tmp_path)
    slots[1].run("mine", _ME, started_at="2026-10-21T04:20:00Z")
    slots[2].run("theirs", _OTHER, started_at="2026-10-21T04:00:00Z")

    async def ask(session, prompt, **_):
        slots[0].run("someone-else", _OTHER, started_at="2026-10-21T04:59:00Z")
        slots[0].boot += 1
        return None

    import anton.core.utils.scratchpad as scratchpad

    monkeypatch.setattr(scratchpad, "ask_approval_card", ask)
    session = _session()
    with pytest.raises(LiveAppLaunchError) as raised:
        await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=())
    assert raised.value.cause == "slots_changed"
    assert slots[1].app.artifact == "mine" and slots[1].resets == 0
    assert session._live_app_host.events == []


def _taken_by_another_process(slot: FakeSlot) -> None:
    """Make `slot` answer 409 once: another process takes it between the lock and the launch."""
    def launch_once_busy(request):
        slot.run("someone-else", _OTHER, started_at="2026-10-21T04:59:00Z")
        slot.boot += 1
        raise SlotRefused(409, SlotError(error="slot_busy"))

    slot.launch = launch_once_busy  # type: ignore[method-assign]


async def test_a_busy_slot_is_picked_again_once(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    _taken_by_another_process(slots[0])
    result = await las.launch_live_app(_session(), slug="sales-demo", folder=folder, datasources=())
    assert result.slot == 2 and len(slots[1].requests) == 1
    assert slots[0].app.artifact == "someone-else" and slots[0].resets == 0


async def test_a_second_busy_slot_stops_the_launch(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    _taken_by_another_process(slots[0])
    _taken_by_another_process(slots[1])
    with pytest.raises(LiveAppLaunchError) as raised:
        await las.launch_live_app(_session(), slug="sales-demo", folder=folder, datasources=())
    assert raised.value.cause == "slot_busy"
    assert slots[2].requests == []


async def test_a_changed_slot_and_a_busy_one_each_get_their_retry(tmp_path, slots, approved, monkeypatch):
    """Slot 1 changes while the card waits, then slot 2 answers 409: each
    is picked again once, so the launch still lands in slot 3."""
    folder = _artifact(tmp_path)
    _taken_by_another_process(slots[1])

    async def ask(session, prompt, **_):
        slots[0].run("someone-else", _OTHER, started_at="2026-10-21T04:59:00Z")
        slots[0].boot += 1
        return None

    import anton.core.utils.scratchpad as scratchpad

    monkeypatch.setattr(scratchpad, "ask_approval_card", ask)
    result = await las.launch_live_app(_session(), slug="sales-demo", folder=folder, datasources=())
    assert result.slot == 3


async def test_a_missing_package_is_named(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    slots[0].refuse_launch = SlotRefused(422, SlotError(error="missing_package", field="requirements.txt",
                                                        detail="polars"))
    with pytest.raises(LiveAppLaunchError, match="lacks the package `polars`") as raised:
        await las.launch_live_app(_session(), slug="sales-demo", folder=folder, datasources=())
    assert raised.value.cause == "missing_package"


async def test_an_app_that_does_not_start_returns_its_scrubbed_log(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    slots[0].refuse_launch = SlotRefused(422, SlotError(
        error="app_not_ready", detail="no port after 20 s", log_tail=f"{{'access_token': '{_LOGGED_TOKEN}'}}",
    ))
    with pytest.raises(LiveAppLaunchError) as raised:
        await las.launch_live_app(_session(), slug="sales-demo", folder=folder, datasources=())
    assert raised.value.cause == "app_not_ready"
    assert _LOGGED_TOKEN not in str(raised.value)
    assert _LOGGED_TOKEN not in (folder / "backend.log").read_text(encoding="utf-8")


async def test_a_slot_error_names_the_field_never_the_value(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    slots[0].refuse_launch = SlotRefused(400, SlotError(error="bad_env_name", field="env.DS_X", detail="name"))
    with pytest.raises(LiveAppLaunchError) as raised:
        await las.launch_live_app(_session(), slug="sales-demo", folder=folder, datasources=_DECLARED)
    assert "field `env.DS_X`" in str(raised.value)
    assert _PG_PASSWORD not in str(raised.value) and _API_KEY not in str(raised.value)


async def test_no_slot_answering_starts_nothing(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    for slot in slots:
        slot.down = True
    with pytest.raises(LiveAppLaunchError) as raised:
        await las.launch_live_app(_session(), slug="sales-demo", folder=folder, datasources=())
    assert raised.value.cause == "no_slot"
    assert cards == []


async def test_a_session_without_a_host_launches_nothing(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    session = SimpleNamespace(_data_vault=None)
    with pytest.raises(LiveAppLaunchError) as raised:
        await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=())
    assert raised.value.cause == "no_host"
    assert all(slot.requests == [] for slot in slots)


async def test_a_broken_setting_refuses_the_launch(tmp_path, slots, approved, cards, monkeypatch):
    folder = _artifact(tmp_path)
    approved.write_text("{not json", encoding="utf-8")
    with pytest.raises(LiveAppLaunchError) as raised:
        await las.launch_live_app(_session(), slug="sales-demo", folder=folder, datasources=())
    assert raised.value.cause == "config"
    assert las.CONNECTIONS_FILE_ENV_VAR in str(raised.value)


async def test_a_failing_claim_stops_the_launch_before_the_slot_resets(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    _runs_from(slots[0], folder)

    class RefusingHost(FakeHost):
        async def claim(self, slot: int) -> None:
            raise RuntimeError("record store down for ana@example.com")

    with pytest.raises(LiveAppLaunchError) as raised:
        await las.launch_live_app(_session(RefusingHost()), slug="sales-demo", folder=folder, datasources=())
    assert raised.value.cause == "record_failed"
    assert "ana@example.com" not in str(raised.value)
    assert slots[0].resets == 0 and slots[0].requests == []


async def test_a_failing_launch_record_stops_the_unrecorded_app(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)

    class RefusingHost(FakeHost):
        async def launched(self, launch) -> None:
            raise RuntimeError("record store down")

    with pytest.raises(LiveAppLaunchError) as raised:
        await las.launch_live_app(_session(RefusingHost()), slug="sales-demo", folder=folder, datasources=())
    assert raised.value.cause == "record_failed"
    assert len(slots[0].requests) == 1 and slots[0].resets == 1


async def test_replacing_a_running_app_claims_its_slot_first(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    _runs_from(slots[0], folder)
    session = _session()
    await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=())
    assert [e[0] for e in session._live_app_host.events] == ["claim", "launched"]
    assert session._live_app_host.events[0] == ("claim", 1)


async def test_an_exited_slot_is_launched_without_a_claim(tmp_path, slots, approved, cards):
    """An exited app's record opens nothing, so only the launch is recorded."""
    folder = _artifact(tmp_path)
    slots[0].run("gone", _OTHER, started_at="2026-10-21T04:00:00Z", state="exited")
    session = _session()
    result = await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=())
    assert result.slot == 1 and slots[0].resets == 1
    assert [e[0] for e in session._live_app_host.events] == ["launched"]


async def test_an_exited_slot_mid_restart_is_waited_for(tmp_path, slots, approved, cards):
    """The exited app's slot program exits and Docker restarts it: no answer
    under the lock means the slot is coming back, not that it's taken."""
    folder = _artifact(tmp_path)
    slots[0].run("gone", _OTHER, started_at="2026-10-21T04:00:00Z", state="exited")
    reads = {"n": 0}
    status = slots[0].status

    def restarting(**kw):
        reads["n"] += 1
        if reads["n"] == 2:
            # The read under the lock: the program just exited.
            slots[0].reset()
            raise SlotUnavailable("restarting")
        return status(**kw)

    slots[0].status = restarting  # type: ignore[method-assign]
    result = await las.launch_live_app(_session(), slug="sales-demo", folder=folder, datasources=())
    assert result.slot == 1 and len(slots[0].requests) == 1


async def test_two_launches_into_one_slot_take_turns(tmp_path, slots, approved, cards):
    """The per-slot lock: a second launch of the app waits until the first
    has recorded its launch, then reads the slot again."""
    folder = _artifact(tmp_path)
    _runs_from(slots[0], folder)
    release = asyncio.Event()

    class SlowHost(FakeHost):
        async def claim(self, slot: int) -> None:
            await super().claim(slot)
            if [e[0] for e in self.events].count("claim") == 1:
                await release.wait()

    host = SlowHost()
    session = _session(host)
    first = asyncio.create_task(las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=()))
    while not host.events:
        await asyncio.sleep(0.01)
    second = asyncio.create_task(las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=()))
    await asyncio.sleep(0.2)
    assert [e[0] for e in host.events] == ["claim"], "the second launch waits on the slot's lock"
    release.set()
    await asyncio.gather(first, second)
    assert [e[0] for e in host.events] == ["claim", "launched", "claim", "launched"]
    assert slots[1].requests == [] and len(slots[0].requests) == 2


async def test_a_restart_waits_for_a_new_boot_not_an_old_empty_answer(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    _runs_from(slots[0], folder)
    slots[0].state = "running"
    session = _session()
    slots[0].restart_window = (_OLD_BOOT, _DOWN, _OLD_BOOT, _DOWN)
    await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=())
    (_, launched) = session._live_app_host.events[-1]
    assert launched.boot_id == "boot-1-2" and len(slots[0].requests) == 1


async def test_a_slot_that_never_comes_back_refuses_and_says_the_app_stopped(tmp_path, slots, approved, cards,
                                                                            monkeypatch):
    folder = _artifact(tmp_path)
    _runs_from(slots[0], folder)
    slots[0].comes_back = False
    monkeypatch.setattr(las, "RESET_TIMEOUT_SECONDS", 0.05)
    session = _session()
    with pytest.raises(LiveAppLaunchError) as raised:
        await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=())
    assert raised.value.cause == "slot_reset"
    assert "Your app `sales-demo` that slot 1 was running has stopped for this launch" in str(raised.value)
    assert [e[0] for e in session._live_app_host.events] == ["claim", "launch_failed"]


async def test_a_launch_that_fails_after_its_claim_tells_the_host(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    _runs_from(slots[0], folder)
    slots[0].refuse_launch = SlotRefused(422, SlotError(error="missing_package", field="requirements.txt",
                                                        detail="polars"))
    session = _session()
    with pytest.raises(LiveAppLaunchError) as raised:
        await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=())
    assert raised.value.cause == "missing_package"
    assert "has stopped for this launch, and nothing runs there now" in str(raised.value)
    assert session._live_app_host.events == [("claim", 1), ("launch_failed", 1)]


async def test_a_failing_launch_failed_record_changes_nothing(tmp_path, slots, approved, cards, caplog):
    folder = _artifact(tmp_path)
    _runs_from(slots[0], folder)
    slots[0].refuse_launch = SlotRefused(422, SlotError(error="slot_refused_x"))

    class DownHost(FakeHost):
        async def launch_failed(self, slot: int) -> None:
            raise RuntimeError("records down for ana@example.com")

    with caplog.at_level(logging.WARNING, logger=las.__name__):
        with pytest.raises(LiveAppLaunchError) as raised:
            await las.launch_live_app(_session(DownHost()), slug="sales-demo", folder=folder, datasources=())
    assert raised.value.cause == "slot_refused"
    records = [r for r in caplog.records if r.name == las.__name__ and "launch_failed" in r.getMessage()]
    assert records and "ana@example.com" not in records[0].getMessage()


async def test_a_launch_the_slot_started_but_anton_cant_read_is_reset(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    accept = slots[0].launch

    def started_then_garbled(request):
        accept(request)
        raise SlotUnavailable("the answer was cut off")

    slots[0].launch = started_then_garbled  # type: ignore[method-assign]
    session = _session()
    with pytest.raises(LiveAppLaunchError) as raised:
        await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=())
    assert raised.value.cause == "slot_unavailable"
    assert slots[0].resets == 1 and session._live_app_host.events == []


async def test_a_stop_during_the_launch_resets_the_app_it_started(tmp_path, slots, approved, cards):
    """A Stop can't stop the thread that sends the launch: the app it starts
    is reset when the POST lands, so nothing unrecorded keeps the slot."""
    folder = _artifact(tmp_path)
    gate, entered = threading.Event(), threading.Event()
    accept = slots[0].launch

    def slow_launch(request):
        entered.set()
        gate.wait(5)
        return accept(request)

    slots[0].launch = slow_launch  # type: ignore[method-assign]
    session = _session()
    task = asyncio.create_task(las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=()))
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    assert not task.done()
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert slots[0].resets == 1 and session._live_app_host.events == []


async def test_a_stop_while_recording_the_launch_resets_it(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    started, release = asyncio.Event(), asyncio.Event()

    class HangingHost(FakeHost):
        async def launched(self, launch) -> None:
            started.set()
            await release.wait()

    task = asyncio.create_task(las.launch_live_app(_session(HangingHost()), slug="sales-demo", folder=folder,
                                                   datasources=()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert slots[0].resets == 1


@pytest.mark.parametrize("step", ["launch", "stop"])
async def test_a_stop_while_claiming_resets_the_app_the_claim_stopped(tmp_path, slots, approved, cards, step):
    """A host may finish a claim's record after the Stop, as a database write
    it began does. The app that record calls replaced is reset, and the host
    hears the launch failed, so nothing runs that nobody can open."""
    folder = _artifact(tmp_path)
    _runs_from(slots[0], folder)
    started, release = asyncio.Event(), asyncio.Event()

    class SlowHost(FakeHost):
        async def claim(self, slot: int) -> None:
            started.set()
            try:
                await release.wait()
            finally:
                # The record lands whatever happens to the caller.
                self.events.append(("claim", slot))

    session = _session(SlowHost())
    if step == "launch":
        call = las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=())
    else:
        call = las.stop_live_app(session, folder=folder)
    task = asyncio.create_task(call)
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert slots[0].resets == 1 and slots[0].requests == []
    assert session._live_app_host.events == [("claim", 1), ("launch_failed", 1)]


async def test_an_address_without_a_slot_number_refuses_before_any_slot(tmp_path, slots, approved, cards,
                                                                         monkeypatch):
    folder = _artifact(tmp_path)
    monkeypatch.setenv(las.ADDRESS_ENV_VAR, "http://apps.example.test:8080/")
    session = _session()
    with pytest.raises(LiveAppLaunchError) as raised:
        await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=())
    assert raised.value.cause == "config"
    assert las.ADDRESS_ENV_VAR in str(raised.value)
    assert all(slot.requests == [] and slot.resets == 0 for slot in slots)
    assert cards == [] and session._live_app_host.events == []


# ── After a cell ─────────────────────────────────────────────────────────────


def test_only_launch_files_count_as_a_change():
    before = {"backend.py": 1, "static/a.js": 1, "prd.md": 1}
    assert las.launch_paths_changed(before, {**before, "prd.md": 2}) is False
    assert las.launch_paths_changed(before, {**before, "static/a.js": 2}) is True
    assert las.launch_paths_changed(before, {**before, "static/new.css": 1}) is True
    assert las.launch_paths_changed(before, {"static/a.js": 1, "prd.md": 1}) is True


def _store(root: Path, **datasources):
    return SimpleNamespace(
        open=lambda slug: SimpleNamespace(datasources=datasources.get(slug, [])),
        folder_for=lambda slug: root / slug,
    )


async def test_a_cell_relaunches_only_the_launchers_running_apps(tmp_path, slots, approved, cards):
    root = tmp_path / "artifacts"
    mine, theirs, idle = _artifact(root, "mine"), _artifact(root, "theirs"), _artifact(root, "idle")
    _runs_from(slots[0], mine)
    _runs_from(slots[1], theirs, _OTHER)
    store = SimpleNamespace(
        open=lambda slug: SimpleNamespace(datasources=[]),
        folder_for=lambda slug: root / slug,
    )
    lines = await las.relaunch_changed_apps(_session(), store, ["mine", "theirs", "idle"])
    assert lines == ["Relaunched live app `mine` at `http://app1.view.localhost:8080/` with this cell's changes."]
    assert slots[0].resets == 1 and slots[1].resets == 0
    assert mine.exists() and theirs.exists() and idle.exists()


async def test_a_refused_relaunch_says_so_on_the_cell(tmp_path, slots, approved, cards):
    root = tmp_path / "artifacts"
    _runs_from(slots[0], _artifact(root, "mine"))
    cards.answer = "cancelled"
    lines = await las.relaunch_changed_apps(_session(), _store(root), ["mine"])
    assert lines == [
        "Live app `mine` was not relaunched. Not launched: the person didn't approve the launch. "
        "Their running apps are unchanged."
    ]


async def test_a_cell_that_left_the_files_as_they_were_relaunches_nothing(tmp_path, slots, approved, cards):
    root = tmp_path / "artifacts"
    folder = _artifact(root, "mine")
    session = _session()
    await las.launch_live_app(session, slug="mine", folder=folder, datasources=())
    # Rewritten with the same bytes: the stamps change, the files don't.
    (folder / "static" / "index.html").write_text("<html><body></body></html>", encoding="utf-8")
    assert await las.relaunch_changed_apps(session, _store(root), ["mine"]) == []
    assert slots[0].resets == 0 and len(slots[0].requests) == 1


async def test_a_relaunch_names_the_connections_it_left_out(tmp_path, slots, approved, cards):
    root = tmp_path / "artifacts"
    _runs_from(slots[0], _artifact(root, "mine"))
    store = _store(root, mine=[DatasourceRef(engine="snowflake", name="sales")])
    (line,) = await las.relaunch_changed_apps(_session(), store, ["mine"])
    assert line.startswith("Relaunched live app `mine` at `http://app1.view.localhost:8080/`")
    assert "`snowflake/sales` (not approved for live apps by the operator)" in line


async def test_a_relaunch_that_failed_after_the_reset_says_the_app_stopped(tmp_path, slots, approved, cards):
    root = tmp_path / "artifacts"
    _runs_from(slots[0], _artifact(root, "mine"))
    slots[0].refuse_launch = SlotRefused(422, SlotError(error="missing_package", detail="polars"))
    (line,) = await las.relaunch_changed_apps(_session(), _store(root), ["mine"])
    assert line.startswith("Live app `mine` is stopped: its relaunch failed.")
    assert "call `launch_backend` to start it again" in line


async def test_an_app_whose_metadata_cant_be_read_gets_a_line(tmp_path, slots, approved, cards):
    root = tmp_path / "artifacts"
    _runs_from(slots[0], _artifact(root, "mine"))
    store = SimpleNamespace(open=lambda slug: None, folder_for=lambda slug: root / slug)
    (line,) = await las.relaunch_changed_apps(_session(), store, ["mine"])
    assert "its metadata.json can't be read" in line and slots[0].resets == 0


async def test_a_relaunch_takes_only_the_app_from_that_folder(tmp_path, slots, approved, cards):
    """The same slug in another project is another app: a cell there leaves
    this one running."""
    _runs_from(slots[0], _artifact(tmp_path / "project-a", "sales"))
    other = tmp_path / "project-b"
    _artifact(other, "sales")
    assert await las.relaunch_changed_apps(_session(), _store(other), ["sales"]) == []
    assert slots[0].resets == 0


async def test_stopping_an_app_stops_only_that_folders_app(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    _runs_from(slots[0], folder)
    _runs_from(slots[1], _artifact(tmp_path / "theirs"), _OTHER)
    session = _session()
    assert await las.stop_live_app(session, folder=folder) is None
    assert slots[0].resets == 1 and slots[1].resets == 0
    assert session._live_app_host.events == [("claim", 1), ("launch_failed", 1)]
    assert await las.stop_live_app(_session(), folder=tmp_path / "nothing-runs-here") is None


async def test_a_stop_whose_lock_timed_out_names_the_slot_the_app_may_still_run_in(tmp_path, slots, approved, cards):
    """Other launches held the host's lock past its limit: the stop touches
    nothing and says which slot's app may still run."""
    import contextlib

    class BusyHost(FakeHost):
        @contextlib.asynccontextmanager
        async def launch_lock(self):
            raise LiveAppLaunchError("Live app slots are busy. Try launching again shortly.", cause="slot_busy")
            yield

    folder = _artifact(tmp_path)
    _runs_from(slots[1], folder)
    host = BusyHost()
    assert await las.stop_live_app(_session(host), folder=folder) == 2
    assert slots[1].resets == 0 and slots[1].app.artifact == folder.name
    assert host.events == []


async def test_a_stop_whose_slot_gives_no_status_under_the_lock_names_the_slot(tmp_path, slots, approved, cards):
    """The slot stopped answering while the stop waited for the host's lock:
    the stop can't tell whether the app still runs, so it touches nothing
    and names the slot."""
    import contextlib

    class SilentHost(FakeHost):
        @contextlib.asynccontextmanager
        async def launch_lock(self):
            slots[1].down = True
            yield

    folder = _artifact(tmp_path)
    _runs_from(slots[1], folder)
    host = SilentHost()
    assert await las.stop_live_app(_session(host), folder=folder) == 2
    assert slots[1].resets == 0 and slots[1].app.artifact == folder.name
    assert host.events == []


async def test_a_stop_during_the_reset_still_tells_the_host_the_app_stopped(tmp_path, slots, approved, cards):
    """The claim already stopped the app's record: without `launch_failed`
    its owner would never hear the app is gone."""
    folder = _artifact(tmp_path)
    _runs_from(slots[0], folder)
    entered, gate = threading.Event(), threading.Event()
    reset = slots[0].reset

    def slow_reset():
        entered.set()
        gate.wait(5)
        reset()

    slots[0].reset = slow_reset  # type: ignore[method-assign]
    session = _session()
    task = asyncio.create_task(las.stop_live_app(session, folder=folder))
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(100):
        if slots[0].resets:
            break
        await asyncio.sleep(0.02)
    assert slots[0].resets == 1
    assert session._live_app_host.events == [("claim", 1), ("launch_failed", 1)]


async def test_the_running_address_is_that_folders_apps(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    _runs_from(slots[1], folder)
    _runs_from(slots[0], _artifact(tmp_path / "theirs"), _OTHER)
    assert await las.running_app_address(_session(), folder=folder) == "http://app2.view.localhost:8080/"
    assert await las.running_app_address(_session(), folder=tmp_path / "nothing-runs-here") is None
    assert await las.running_app_address(SimpleNamespace(), folder=folder) is None


# ── What the launch says ─────────────────────────────────────────────────────


@pytest.mark.parametrize("cause, running, text", [
    ("cancelled", True, "Not launched: the person didn't approve the launch. Their running apps are unchanged."),
    ("cancelled", False, "Not launched: the person didn't approve the launch. Nothing was started."),
    ("timeout", False,
     "Not launched: the launch needs the person's approval, and no one answered in time. Nothing was started."),
    ("unavailable", True,
     "Not launched: the launch needs the person's approval, and there is no way to ask them here. "
     "Their running apps are unchanged."),
])
def test_a_card_that_didnt_approve_says_why(cause, running, text):
    assert las.not_approved_message(cause, running=running) == text


async def test_a_first_launch_refused_says_nothing_was_started(tmp_path, slots, approved, cards):
    cards.answer = "timeout"
    with pytest.raises(LiveAppLaunchError) as raised:
        await las.launch_live_app(_session(), slug="sales-demo", folder=_artifact(tmp_path), datasources=())
    assert raised.value.cause == "launch_not_approved"
    assert str(raised.value).endswith("no one answered in time. Nothing was started.")


def test_the_launch_blocker_names_why_no_launch_can_succeed(slots, approved, monkeypatch):
    assert las.live_app_launch_blocker(_session()) is None
    assert las.live_app_launch_blocker(SimpleNamespace(_data_vault=None)).cause == "no_host"
    monkeypatch.setenv(las.PER_PERSON_ENV_VAR, "0")
    assert las.live_app_launch_blocker(_session()).cause == "launches_disabled"
    monkeypatch.delenv(las.PER_PERSON_ENV_VAR)
    monkeypatch.delenv(las.CONNECTIONS_FILE_ENV_VAR)
    assert las.live_app_launch_blocker(_session()).cause == "config"


class _OlderHost:
    """A host from before slots were shared: launch records only."""

    def __init__(self):
        self.launcher = _ME
        self.launch_token = _TOKEN
        self.events: list[tuple] = []

    async def claim(self, slot: int) -> None:
        self.events.append(("claim", slot))

    async def launched(self, launch) -> None:
        self.events.append(("launched", launch))

    async def launch_failed(self, slot: int) -> None:
        self.events.append(("launch_failed", slot))


async def test_an_older_host_refuses_the_launch_and_its_stop_does_not_crash(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    host = _OlderHost()
    session = _session(host)
    blocker = las.live_app_launch_blocker(session)
    assert blocker.cause == "no_host" and str(blocker) == las.OLD_HOST_MESSAGE
    with pytest.raises(LiveAppLaunchError) as raised:
        await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=())
    assert raised.value.cause == "no_host" and "Ask the operator to update it" in str(raised.value)
    assert cards == [] and all(s.requests == [] for s in slots)
    _runs_from(slots[1], folder)
    assert await las.stop_live_app(session, folder=folder) == 2
    assert slots[1].resets == 0 and host.events == []


# ── The card through the gate ────────────────────────────────────────────────


def _recording_elicit(monkeypatch, status: str):
    import anton.core.interaction.elicit as elicit_module

    asked: list = []

    async def elicit(session, question_id, request, *, counted):
        asked.append(request)
        from anton.core.interaction.elicit import AskAnswer

        return AskAnswer(status=status, values=("approve",) if status == "answered" else ())

    monkeypatch.setattr(elicit_module, "elicit", elicit)
    return asked


async def test_the_launch_card_offers_approve_and_cancel_with_no_default(monkeypatch):
    from anton.core.utils.scratchpad import ask_approval_card

    asked = _recording_elicit(monkeypatch, "answered")
    session = SimpleNamespace(elicitor=None, _write_gate=None)
    assert await ask_approval_card(session, "**Launch this live app?**", approve_detail="Start the app with these "
                                   "connections", cancel_detail="Leave things as they are") is None
    (request,) = asked
    assert [(o.value, o.label, o.detail) for o in request.options] == [
        ("approve", "Approve", "Start the app with these connections"),
        ("cancel", "Cancel", "Leave things as they are"),
    ]
    assert request.allow_custom is False and not request.default_value


async def test_after_one_unanswered_card_the_next_is_refused_unasked(monkeypatch):
    from anton.core.utils.scratchpad import WriteGate, ask_approval_card

    asked = _recording_elicit(monkeypatch, "timeout")
    session = SimpleNamespace(elicitor=None, _write_gate=WriteGate())
    kw = {"approve_detail": "a", "cancel_detail": "c"}
    assert await ask_approval_card(session, "first", **kw) == "timeout"
    assert await ask_approval_card(session, "second", **kw) == "unanswered_earlier"
    assert len(asked) == 1


# ── The wire ─────────────────────────────────────────────────────────────────


class _SlotServer(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def log_message(self, *args):  # quiet
        pass

    def _reply(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.seen.append({"path": self.path, "auth": self.headers.get("Authorization")})
        if self.headers.get("Authorization") != f"Bearer {_TOKEN}":
            return self._reply(401, {"error": "unauthorized"})
        self._reply(200, {"version": 1, "boot_id": "b1", "booted_at": "2026-10-21T04:00:00Z", "state": "empty",
                          "app": None, "a_field_anton_doesnt_know": 1})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        self._reply(400, {"error": "bad_env_name", "field": "env.DS_X__Y", "detail": "not a DS_ name"})


@pytest.fixture
def slot_server():
    _SlotServer.seen = []
    server = HTTPServer(("127.0.0.1", 0), _SlotServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_the_client_sends_the_token_and_reads_the_status(slot_server):
    status = las.SlotClient(slot_server, token=_TOKEN).status()
    assert status.state == "empty" and status.boot_id == "b1"
    assert _SlotServer.seen[-1] == {"path": "/status", "auth": f"Bearer {_TOKEN}"}
    with pytest.raises(SlotRefused) as raised:
        las.SlotClient(slot_server, token="wrong").status()
    assert raised.value.status_code == 401


def test_a_launch_body_carries_the_values_and_an_error_never_does(slot_server):
    request = las.LaunchRequest(
        artifact="a", app_id="0" * 64, launcher=_ME, files=(las.LaunchFile(path="backend.py", content_b64="YQ=="),),
        env={"DS_X__Y": _PG_PASSWORD}, connections=(),
    )
    with pytest.raises(SlotRefused) as raised:
        las.SlotClient(slot_server, token=_TOKEN).launch(request)
    assert _SlotServer.seen[-1]["body"]["env"] == {"DS_X__Y": _PG_PASSWORD}
    assert raised.value.body.field == "env.DS_X__Y"
    assert _PG_PASSWORD not in str(raised.value)


def test_an_unreachable_slot_is_unavailable():
    with pytest.raises(SlotUnavailable):
        las.SlotClient("http://127.0.0.1:9", token=_TOKEN).status(timeout=0.5)


def test_the_probe_waits_five_seconds_for_each_path(monkeypatch):
    """The slot GETs the paths one after another, 5 seconds at most each."""
    waited: list[float] = []

    def call(self, method, path, *, timeout, body=None):
        waited.append(timeout)
        return json.dumps({"results": []}).encode()

    monkeypatch.setattr(las.SlotClient, "_call", call)
    las.SlotClient("http://live-app-1:8001", token=_TOKEN).probe([f"/p{n}" for n in range(32)])
    assert waited == [5.0 * 32 + las.STATUS_TIMEOUT_SECONDS]


@pytest.mark.parametrize("path, accepted", [
    ("/api/health", True), ("/api/items?x=1", True), ("//evil.example.com/x", False), ("/a b", False),
    ("/a#b", False), ("api/health", False), ("/" + "a" * 512, False),
])
def test_only_paths_the_slot_takes_are_probed(path, accepted):
    assert las.probe_path_accepted(path) is accepted


async def test_a_stopped_app_answers_no_path(tmp_path, slots, approved, cards):
    slots[0].refuse_probe = SlotRefused(409, SlotError(error="not_running"))
    result = await las.launch_live_app(_session(), slug="sales-demo", folder=_artifact(tmp_path), datasources=(),
                                       probe_paths=["/api/health", "/api/items"])
    assert [p.status for p in result.probe] == [0, 0] and result.probe_error == ""


async def test_a_probe_the_slot_cant_run_says_so(tmp_path, slots, approved, cards):
    slots[0].refuse_probe = SlotUnavailable("timed out")
    result = await las.launch_live_app(_session(), slug="sales-demo", folder=_artifact(tmp_path), datasources=(),
                                       probe_paths=["/api/health"])
    assert result.probe_error == "the slot gave no usable answer (SlotUnavailable)"
    assert [p.status for p in result.probe] == [0]


async def test_a_path_the_slot_left_out_reads_as_no_answer(tmp_path, slots, approved, cards):
    slots[0].probe = lambda paths: ProbeReply(results=(ProbeResult(path="/api/items", status=200),))
    result = await las.launch_live_app(_session(), slug="sales-demo", folder=_artifact(tmp_path), datasources=(),
                                       probe_paths=["/api/health", "/api/items"])
    assert [(p.path, p.status) for p in result.probe] == [("/api/health", 0), ("/api/items", 200)]


def test_status_keeps_the_slots_stop_reason():
    body = {"version": 1, "boot_id": "a", "booted_at": "2026-10-21T04:00:00Z", "state": "exited", "app": {
        "artifact": "s", "app_id": "0" * 64, "launcher": {"subject": "x", "email": "ana@example.com"},
        "started_at": "t", "expires_at": "t", "stop_reason": "idle",
        "hashes": {"backend": "a", "requirements": "b", "connections": "c", "files": "d"},
    }}
    assert SlotStatus.model_validate_json(json.dumps(body)).app.stop_reason == "idle"


def test_a_malformed_reply_never_echoes_its_values():
    """A version-skewed slot's reply that doesn't parse names the field, never
    what it held: the warnings log these errors."""
    body = {"version": 1, "boot_id": "a", "booted_at": "t", "state": "running", "app": {
        "artifact": "s", "app_id": "0" * 64, "launcher": {"email": "ana@example.com"}, "started_at": "t",
        "expires_at": "t", "hashes": {"backend": "a", "requirements": "b", "connections": "c", "files": "d"},
    }}
    with pytest.raises(las.ValidationError) as raised:
        SlotStatus.model_validate_json(json.dumps(body))
    assert "ana@example.com" not in str(raised.value)


async def test_a_folder_outside_the_persons_files_is_never_read(tmp_path, slots, approved, cards):
    """A cell that turns its artifact folder into a link to files only the
    server may read gets a refusal; nothing from there reaches a slot."""
    from anton.core.backends.sandbox import CellSandbox

    project, home, tmp = tmp_path / "project", tmp_path / "home", tmp_path / "tmp"
    for folder in (project, home, tmp):
        folder.mkdir()
    server_only = _artifact(tmp_path / "server", "secrets")
    (project / "artifacts").mkdir()
    os.symlink(server_only, project / "artifacts" / "sales-demo")
    session = _session()
    session._scratchpads = SimpleNamespace(
        cell_sandbox=CellSandbox(project_dir=project, home_dir=home, tmp_dir=tmp, user_name="bob"),
    )
    with pytest.raises(LiveAppLaunchError, match="outside the files this conversation may read"):
        await las.launch_live_app(session, slug="sales-demo", folder=project / "artifacts" / "sales-demo",
                                  datasources=())
    assert all(slot.requests == [] for slot in slots)


async def test_a_folder_inside_the_persons_files_launches(tmp_path, slots, approved, cards):
    from anton.core.backends.sandbox import CellSandbox

    project, home, tmp = tmp_path / "project", tmp_path / "home", tmp_path / "tmp"
    for folder in (project, home, tmp):
        folder.mkdir()
    folder = _artifact(project / "artifacts")
    session = _session()
    session._scratchpads = SimpleNamespace(
        cell_sandbox=CellSandbox(project_dir=project, home_dir=home, tmp_dir=tmp, user_name="bob"),
    )
    result = await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=())
    assert result.slot == 1 and slots[0].requests


@pytest.mark.parametrize("value", ["-1", "4", "1.5", "x", "１"])
def test_per_person_cap_refuses_invalid_values(monkeypatch, value):
    monkeypatch.setenv(las.PER_PERSON_ENV_VAR, value)
    with pytest.raises(LiveAppConfigError, match=las.PER_PERSON_ENV_VAR):
        las.live_apps_per_person(3)


def test_per_person_cap_defaults_to_slot_count_and_accepts_zero(monkeypatch):
    monkeypatch.delenv(las.PER_PERSON_ENV_VAR, raising=False)
    assert las.live_apps_per_person(15) == 15
    monkeypatch.setenv(las.PER_PERSON_ENV_VAR, "0")
    assert las.live_apps_per_person(15) == 0


async def test_disabled_launch_touches_no_app_even_on_relaunch(tmp_path, slots, approved, monkeypatch, cards):
    folder = _artifact(tmp_path)
    _runs_from(slots[0], folder)
    monkeypatch.setenv(las.PER_PERSON_ENV_VAR, "0")
    with pytest.raises(LiveAppLaunchError, match="operator turned") as raised:
        await las.launch_live_app(_session(), slug="sales-demo", folder=folder, datasources=())
    assert raised.value.cause == "launches_disabled"
    assert not cards and all(s.resets == 0 and s.requests == [] for s in slots)


async def test_cap_replaces_selected_newer_app_even_with_a_free_slot(tmp_path, slots, approved, cards, monkeypatch):
    monkeypatch.setenv(las.PER_PERSON_ENV_VAR, "2")
    slots[0].run("clock", _ME, started_at="2026-10-21T04:00:00Z")
    slots[1].run("counter", _ME, started_at="2026-10-21T04:10:00Z")
    cards.pick = 1
    result = await las.launch_live_app(_session(), slug="notes", folder=_artifact(tmp_path, "notes"), datasources=())
    assert result.slot == 2
    assert slots[0].app.artifact == "clock" and slots[0].resets == 0
    assert slots[1].app.artifact == "notes" and slots[1].resets == 1
    assert slots[2].app is None and slots[2].requests == []


@pytest.mark.parametrize("cause", ["cancelled", "timeout", "unavailable", "error", "unanswered_earlier"])
async def test_unapproved_replacement_changes_nothing(tmp_path, slots, approved, cards, monkeypatch, cause):
    monkeypatch.setenv(las.PER_PERSON_ENV_VAR, "1")
    slots[0].run("clock", _ME, started_at="2026-10-21T04:00:00Z")
    cards.answer = cause
    with pytest.raises(LiveAppLaunchError) as raised:
        await las.launch_live_app(_session(), slug="notes", folder=_artifact(tmp_path, "notes"), datasources=())
    assert raised.value.cause == "launch_not_approved"
    assert all(s.resets == 0 and s.requests == [] for s in slots)


async def test_stale_selected_launch_is_not_replaced(tmp_path, slots, approved, monkeypatch):
    from anton.core.utils import scratchpad
    monkeypatch.setenv(las.PER_PERSON_ENV_VAR, "1")
    slots[0].run("clock", _ME, started_at="2026-10-21T04:00:00Z")
    async def choose(session, prompt, *, options, **kwargs):
        slots[0].run("colleague", _OTHER, started_at="2026-10-21T04:01:00Z")
        slots[0].boot += 1
        return ApprovalChoice(options[0].value, None)
    monkeypatch.setattr(scratchpad, "ask_approval_choice", choose)
    with pytest.raises(LiveAppLaunchError) as raised:
        await las.launch_live_app(_session(), slug="notes", folder=_artifact(tmp_path, "notes"), datasources=())
    assert raised.value.cause == "slots_changed"
    assert slots[0].app.artifact == "colleague"
    assert all(s.resets == 0 and s.requests == [] for s in slots)


async def test_a_pick_made_while_the_same_app_started_elsewhere_replaces_nothing(tmp_path, slots, approved,
                                                                                  monkeypatch):
    """A second tab launches the same app while the first tab's card waits:
    the first tab's pick would start a second copy, so nothing stops."""
    from anton.core.utils import scratchpad
    monkeypatch.setenv(las.PER_PERSON_ENV_VAR, "2")
    folder = _artifact(tmp_path, "notes")
    slots[0].run("clock", _ME, started_at="2026-10-21T04:00:00Z")
    slots[1].run("counter", _ME, started_at="2026-10-21T04:10:00Z")

    async def choose(session, prompt, *, options, **kwargs):
        # The second tab replaced counter with this same app meanwhile.
        slots[1].boot += 1
        _runs_from(slots[1], folder, started_at="2026-10-21T04:20:00Z")
        return ApprovalChoice(options[0].value, None)

    monkeypatch.setattr(scratchpad, "ask_approval_choice", choose)
    with pytest.raises(LiveAppLaunchError) as raised:
        await las.launch_live_app(_session(), slug="notes", folder=folder, datasources=())
    assert raised.value.cause == "slots_changed"
    assert slots[0].app.artifact == "clock" and slots[0].resets == 0
    assert slots[1].app.artifact == "notes" and slots[1].requests == []


async def test_two_people_racing_for_one_free_slot_leave_the_second_refused_by_name(tmp_path, slots, approved,
                                                                                    monkeypatch):
    """Both pass the card seeing one free slot. The first to launch takes it;
    the second is refused as slots full, naming the winner's app and email,
    and its host records the refusal so it hears when a slot frees."""
    from anton.core.utils import scratchpad
    third = LiveAppLauncher(subject="sub-cy", email="cy@example.com")
    slots[1].run("clock", _OTHER, started_at="2026-10-21T04:00:00Z")
    slots[2].run("counter", _OTHER, started_at="2026-10-21T04:10:00Z")
    both = asyncio.Event()
    waiting = 0

    async def ask(session, prompt, **kwargs):
        nonlocal waiting
        waiting += 1
        if waiting == 2:
            both.set()
        await both.wait()

    monkeypatch.setattr(scratchpad, "ask_approval_card", ask)
    hosts = (FakeHost(_ME), FakeHost(third))
    results = await asyncio.gather(*(
        las.launch_live_app(_session(host), slug=name, folder=_artifact(tmp_path, name), datasources=())
        for host, name in zip(hosts, ("notes-c", "notes-d"))
    ), return_exceptions=True)
    won = [i for i, r in enumerate(results) if isinstance(r, las.LiveAppLaunchResult)]
    lost = [i for i, r in enumerate(results) if isinstance(r, LiveAppLaunchError)]
    assert len(won) == 1 and len(lost) == 1
    (winner,), (loser,) = won, lost
    refusal = results[loser]
    assert refusal.cause == "slots_full"
    text = str(refusal)
    assert f"slot 1 runs `notes-{'cd'[winner]}` for {hosts[winner].launcher.email}" in text
    assert "bo@example.com" in text
    assert (hosts[winner].refused, hosts[loser].refused) == (0, 1)
    assert slots[0].app.launcher == hosts[winner].launcher
    assert all(slot.resets == 0 for slot in slots)


async def test_simultaneous_same_person_launches_cannot_exceed_cap(tmp_path, slots, approved, monkeypatch):
    from anton.core.utils import scratchpad
    monkeypatch.setenv(las.PER_PERSON_ENV_VAR, "1")
    both = asyncio.Event()
    waiting = 0
    async def ask(session, prompt, **kwargs):
        nonlocal waiting
        waiting += 1
        if waiting == 2:
            both.set()
        await both.wait()
    monkeypatch.setattr(scratchpad, "ask_approval_card", ask)
    results = await asyncio.gather(*(
        las.launch_live_app(_session(), slug=name, folder=_artifact(tmp_path, name), datasources=())
        for name in ("clock", "counter")
    ), return_exceptions=True)
    assert sum(isinstance(r, las.LiveAppLaunchResult) for r in results) == 1
    assert sum(isinstance(r, LiveAppLaunchError) and r.cause == "slots_changed" for r in results) == 1
    assert sum(s.app is not None for s in slots) == 1
    assert all(s.resets == 0 for s in slots)


def test_refusal_uses_conditional_idle_deadline_before_hard_max():
    app = _app("clock", _OTHER, "2026-10-21T04:00:00Z").model_copy(update={
        "idle_expires_at": "2026-10-21T04:03:00Z",
    })
    with pytest.raises(LiveAppLaunchError) as raised:
        las.plan_slot([_status(1, state="running", app=app)], "new", _ME)
    assert "04:03 UTC if no further requests arrive" in str(raised.value)
    assert "maximum lifetime ends by 12:00 UTC" in str(raised.value)


def test_trusted_approval_supports_fifteen_apps_without_widening_model_questions():
    from anton.core.interaction.elicit import ApprovalRequest, AskOption, AskRequest, validate_request
    options = tuple(AskOption(value=str(i), label=str(i)) for i in range(16))
    assert validate_request(ApprovalRequest(prompt="Pick", options=options))
    assert not validate_request(AskRequest(prompt="Pick", options=options))
    assert not validate_request(ApprovalRequest(prompt="Pick", options=options, default_value="0"))
    assert not validate_request(ApprovalRequest(prompt="Pick", options=options, allow_custom=True))
    assert not validate_request(ApprovalRequest(prompt="Pick", options=options + (AskOption(value="17", label="x"),)))


@pytest.mark.parametrize("answer", ["last", "cancel", "forged", "timeout", "skip", "typed"])
async def test_fifteen_owned_apps_use_the_real_approval_helper(monkeypatch, answer):
    import importlib
    from anton.core.interaction.elicit import AskAnswer, validate_request

    module = importlib.import_module("anton.core.interaction.elicit")
    choices = tuple(las.SlotPlan(
        i, replaces=_app(f"app-{i}", _ME, "2026-10-21T04:00:00Z"), boot_id=str(i),
    ) for i in range(15))

    async def elicit(session, question_id, request, *, counted):
        assert validate_request(request) and not counted
        assert len(request.options) == 16
        assert [option.label for option in request.options[:-1]] == [f"Stop app-{i}" for i in range(15)]
        assert request.options[-1].value == "cancel"
        assert all("last request none yet" in option.detail for option in request.options[:-1])
        assert not request.default_value and not request.allow_custom
        if answer == "timeout":
            return AskAnswer(status="timeout")
        if answer == "skip":
            return AskAnswer(status="cancelled")
        value = request.options[-2].value if answer in ("last", "typed") else answer
        return AskAnswer(status="answered", values=(value,), text="yes" if answer == "typed" else "")

    monkeypatch.setattr(module, "elicit", elicit)
    session = SimpleNamespace(elicitor=None, _write_gate=None)
    if answer == "last":
        assert await las._choose_replacement(session, choices, prompt="Launch?") == choices[-1]
    else:
        with pytest.raises(LiveAppLaunchError) as raised:
            await las._choose_replacement(session, choices, prompt="Launch?")
        assert raised.value.cause == "launch_not_approved"


def test_unknown_slot_cannot_hide_another_app_when_cap_is_lower():
    with pytest.raises(LiveAppLaunchError, match="running-app count") as raised:
        las.plan_slot([_status(1), None], "new", _ME, per_person=1)
    assert raised.value.cause == "no_slot"


def test_an_unknown_slot_that_cant_reach_the_cap_leaves_a_free_slot_free_to_take():
    """A slot restarting after an idle stop doesn't refuse a person who holds
    no apps: one unknown slot can't bring them to a cap of two."""
    assert las.plan_slot([_status(1), None, _status(3)], "new", _ME, per_person=2) == las.SlotPlan(0)


def test_an_unknown_slot_that_could_reach_the_cap_refuses():
    statuses = [_status(1, state="running", app=_app("clock", _ME, "2026-10-21T04:00:00Z")), None, _status(3)]
    with pytest.raises(LiveAppLaunchError, match="running-app count") as raised:
        las.plan_slot(statuses, "new", _ME, per_person=2)
    assert raised.value.cause == "no_slot"


async def test_exited_capacity_notifies_waiters_before_another_launch_consumes_it(tmp_path, slots, approved, cards):
    for slot in slots:
        slot.run("colleague", _OTHER, started_at="2026-10-21T04:00:00Z")
    slots[0].state = "exited"

    class Host(FakeHost):
        async def launched(self, launch):
            assert self.available >= 1
            await super().launched(launch)

    host = Host()
    result = await las.launch_live_app(_session(host), slug="notes", folder=_artifact(tmp_path), datasources=())
    assert result.slot == 1 and host.available == 1
    assert all(slot.resets == 0 for slot in slots[1:])


async def test_a_launch_into_an_empty_slot_notifies_waiters_before_it_records(tmp_path, slots, approved, cards):
    """A slot restarted empty before any poll saw it: the launch that finds it
    tells the host a slot is free before it takes the slot."""
    slots[1].run("colleague", _OTHER, started_at="2026-10-21T04:00:00Z")
    slots[2].run("colleague-2", _OTHER, started_at="2026-10-21T04:01:00Z")

    class Host(FakeHost):
        async def launched(self, launch):
            assert self.available >= 1
            await super().launched(launch)

    host = Host()
    result = await las.launch_live_app(_session(host), slug="notes", folder=_artifact(tmp_path), datasources=())
    assert result.slot == 1 and host.available >= 1
    assert host.events[-1][0] == "launched"


@pytest.mark.parametrize("cause", ["cancelled", "timeout", "unavailable", "error", "unanswered_earlier", "stopped"])
async def test_replacement_logs_every_refusal_and_stop(tmp_path, slots, approved, monkeypatch, caplog, cause):
    from anton.core.utils import scratchpad

    monkeypatch.setenv(las.PER_PERSON_ENV_VAR, "1")
    slots[0].run("clock", _ME, started_at="2026-10-21T04:00:00Z")

    async def choose(*args, **kwargs):
        if cause == "stopped":
            raise asyncio.CancelledError
        return ApprovalChoice(None, cause)

    monkeypatch.setattr(scratchpad, "ask_approval_choice", choose)
    error = asyncio.CancelledError if cause == "stopped" else LiveAppLaunchError
    with caplog.at_level(logging.WARNING, logger=las.__name__), pytest.raises(error):
        await las.launch_live_app(_session(), slug="notes", folder=_artifact(tmp_path), datasources=())
    (line,) = _logged_decisions(caplog)
    decision = json.loads(line["message"].removeprefix("live app launch: "))
    assert decision["status"] == cause
    assert decision["slug"] == "notes" and decision["replaces"] is None
    assert all(slot.resets == 0 and not slot.requests for slot in slots)


async def test_stop_holds_cross_worker_admission_until_reset_finishes(tmp_path, slots, approved, cards, monkeypatch):
    from contextvars import ContextVar

    admission = asyncio.Lock()
    worker_locks = ContextVar("worker_locks", default={})
    monkeypatch.setattr(las, "_slot_locks", worker_locks.get)
    entered, release = asyncio.Event(), asyncio.Event()
    folder = _artifact(tmp_path, "clock")
    _runs_from(slots[0], folder)
    for slot in slots[1:]:
        slot.run("other-held", _ME, started_at="2026-10-21T04:00:00Z")

    class StopHost(FakeHost):
        def launch_lock(self):
            return admission

        async def claim(self, slot):
            entered.set()
            await release.wait()
            await super().claim(slot)

    class OtherHost(FakeHost):
        def launch_lock(self):
            return admission

    worker_locks.set({})
    stopping = asyncio.create_task(las.stop_live_app(_session(StopHost()), folder=folder))
    await entered.wait()
    # The supervisor can expire an app while the old stop records its claim.
    slots[0].state = "exited"
    worker_locks.set({})
    launching = asyncio.create_task(las.launch_live_app(
        _session(OtherHost(_OTHER)), slug="new", folder=_artifact(tmp_path, "new"), datasources=(),
    ))
    try:
        assert admission.locked()
        await asyncio.sleep(0)
        assert not launching.done()
    finally:
        release.set()
    await stopping
    result = await launching
    assert result.slot == 1 and slots[0].app.launcher == _OTHER
    assert slots[0].app.artifact == "new"


@pytest.mark.parametrize("step", ["post", "replace_reset", "stop_reset"])
async def test_repeated_stop_keeps_admission_until_mutating_thread_settles(
        tmp_path, slots, approved, cards, step):
    admission = asyncio.Lock()
    entered, release = threading.Event(), threading.Event()
    folder = _artifact(tmp_path)

    class Host(FakeHost):
        def launch_lock(self):
            return admission

    if step != "post":
        _runs_from(slots[0], folder)
    method = "launch" if step == "post" else "reset"
    original = getattr(slots[0], method)

    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    setattr(slots[0], method, slow)
    session = _session(Host())
    call = las.stop_live_app(session, folder=folder) if step == "stop_reset" else las.launch_live_app(
        session, slug="sales-demo", folder=folder, datasources=(),
    )
    task = asyncio.create_task(call)
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    try:
        assert not task.done() and admission.locked()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not admission.locked()
    assert slots[0].resets >= 1


async def test_record_receives_one_stop_and_finishes_cleanup_before_unlock(tmp_path, slots, approved, cards):
    admission = asyncio.Lock()
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Host(FakeHost):
        def launch_lock(self):
            return admission

        async def launched(self, launch):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
                self.events.append(("record_cancelled", launch.slot))
                raise

    host = Host()
    task = asyncio.create_task(las.launch_live_app(
        _session(host), slug="notes", folder=_artifact(tmp_path), datasources=(),
    ))
    await entered.wait()
    task.cancel()
    await cancelled.wait()
    task.cancel()
    await asyncio.sleep(0)
    try:
        assert not task.done() and admission.locked()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert host.events == [("record_cancelled", 1)]
    assert slots[0].resets == 1 and not admission.locked()
