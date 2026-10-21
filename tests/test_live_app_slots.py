"""Live apps in app slots: settings, what a launch sends, the card, the slot
choice and the launch itself, against an in-memory slot."""
from __future__ import annotations

import base64
import json
import os
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

    async def claim(self, slot: int) -> None:
        self.events.append(("claim", slot))

    async def launched(self, launch) -> None:
        self.events.append(("launched", launch))


class FakeSlot:
    """One slot's control API, in memory: the rules the slot program keeps."""

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
        self.probe_status = 200
        self.log_tail = "started"

    @property
    def boot_id(self) -> str:
        return f"boot-{self.n}-{self.boot}"

    def status(self, *, timeout: float = 2.0) -> SlotStatus:
        if self.down:
            raise SlotUnavailable(f"slot {self.n} is down")
        return SlotStatus(version=1, boot_id=self.boot_id, booted_at="2026-10-21T04:00:00Z",
                          state=self.state, app=self.app)

    def reset(self) -> None:
        self.resets += 1
        self.boot += 1
        self.state, self.app = "empty", None

    def run(self, artifact: str, launcher: LiveAppLauncher, *, started_at: str, hashes: AppHashes | None = None):
        """Put an app in this slot as a launch would have."""
        self.state = "running"
        self.app = SlotApp(
            artifact=artifact, launcher=launcher, started_at=started_at, expires_at="2026-10-21T12:00:00Z",
            hashes=hashes or AppHashes(backend="x", requirements="x", connections="x"),
        )

    def launch(self, request: las.LaunchRequest) -> LaunchReply:
        if self.refuse_launch is not None:
            raise self.refuse_launch
        if self.state != "empty":
            raise SlotRefused(409, SlotError(error="slot_busy"))
        self.requests.append(request)
        files = [las.LaunchFileData(f.path, base64.b64decode(f.content_b64)) for f in request.files]
        self.run(request.artifact, request.launcher, started_at="2026-10-21T05:00:00Z",
                 hashes=las.app_hashes(files, request.connections))
        return LaunchReply(state="running", boot_id=self.boot_id, started_at="2026-10-21T05:00:00Z",
                           log_tail=self.log_tail)

    def probe(self, paths) -> ProbeReply:
        self.probed.append(tuple(paths))
        return ProbeReply(results=tuple(ProbeResult(path=p, status=self.probe_status) for p in paths),
                          log_tail=self.log_tail)


@pytest.fixture
def slots(monkeypatch) -> list[FakeSlot]:
    fakes = [FakeSlot(n) for n in (1, 2, 3)]
    by_address = dict(zip(_SLOTS, fakes))
    tokens: list[str] = []

    def client(address: str, *, token: str):
        tokens.append(token)
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


@pytest.fixture
def cards(monkeypatch) -> Cards:
    asked = Cards()

    async def ask(session, prompt, *, approve_detail, cancel_detail):
        asked.append(prompt)
        return asked.answer

    import anton.core.utils.scratchpad as scratchpad

    monkeypatch.setattr(scratchpad, "ask_approval_card", ask)
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
    with pytest.raises(LiveAppLaunchError, match="`static/index.html`"):
        las.read_launch_files(folder)


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


def test_engine_and_name_match_the_approval_ignoring_case():
    vault = FakeVault({("Postgres", "Demo"): {"HOST": "samples.mindsdb.com"}})
    resolved = las.resolve_connections(vault, [DatasourceRef(engine="Postgres", name="Demo")], _approved_set(), slug="a")
    assert [g.approved.name for g in resolved.granted] == ["demo"]


# ── What the owner approves ──────────────────────────────────────────────────


def _files(**overrides: bytes) -> list[las.LaunchFileData]:
    files = {"backend.py": b"app = None\n", "requirements.txt": b"fastapi\n", "static/index.html": b"<html>"}
    files.update(overrides)
    return [las.LaunchFileData(path, data) for path, data in files.items()]


def test_a_static_change_leaves_the_approved_hashes_alone():
    assert las.app_hashes(_files(), []) == las.app_hashes(_files(**{"static/index.html": b"<html>v2"}), [])


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
    import hashlib

    assert las.app_hashes(files, []).requirements == hashlib.sha256(b"").hexdigest()


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
        replaces = SlotApp(launcher=_ME, expires_at="2026-10-21T12:00:00Z",
                           hashes=AppHashes(backend="x", requirements="x", connections="x"), **case["replaces"])
    prompt = las.launch_card_prompt(slug=case["slug"], granted=granted, changed=case["changed"],
                                    running=case["running"], replaces=replaces)
    assert prompt == case["card"].rstrip("\n")


# ── Which slot a launch takes ────────────────────────────────────────────────


def _status(n: int, *, state: str = "empty", app: SlotApp | None = None) -> SlotStatus:
    return SlotStatus(version=1, boot_id=f"b{n}", booted_at="2026-10-21T04:00:00Z", state=state, app=app)


def _app(artifact: str, launcher: LiveAppLauncher, started_at: str) -> SlotApp:
    return SlotApp(artifact=artifact, launcher=launcher, started_at=started_at, expires_at="2026-10-21T12:00:00Z",
                   hashes=AppHashes(backend="x", requirements="x", connections="x"))


def test_a_relaunch_takes_the_slot_that_still_runs_the_app():
    statuses = [_status(1), _status(2, state="running", app=_app("sales", _ME, "2026-10-21T04:10:00Z"))]
    assert las.plan_slot(statuses, "sales", _ME) == las.SlotPlan(1)


def test_a_new_app_takes_the_lowest_free_slot():
    statuses = [_status(1, state="running", app=_app("x", _OTHER, "2026-10-21T04:10:00Z")), None,
                _status(3), _status(4, state="exited", app=_app("y", _OTHER, "2026-10-21T04:11:00Z"))]
    assert las.plan_slot(statuses, "sales", _ME) == las.SlotPlan(2)


def test_an_exited_app_frees_its_slot():
    statuses = [_status(1, state="exited", app=_app("y", _OTHER, "2026-10-21T04:11:00Z"))]
    assert las.plan_slot(statuses, "sales", _ME) == las.SlotPlan(0)


def test_with_no_free_slot_a_launch_replaces_the_launchers_own_oldest_app():
    mine_new = _app("newer", _ME, "2026-10-21T04:30:00Z")
    mine_old = _app("older", _ME, "2026-10-21T04:20:00Z")
    statuses = [_status(1, state="running", app=_app("x", _OTHER, "2026-10-21T04:00:00Z")),
                _status(2, state="running", app=mine_new), _status(3, state="running", app=mine_old)]
    assert las.plan_slot(statuses, "sales", _ME) == las.SlotPlan(2, replaces=mine_old)


def test_a_launch_never_replaces_another_persons_app():
    statuses = [_status(n, state="running", app=_app(f"app{n}", _OTHER, f"2026-10-21T04:0{n}:00Z")) for n in (1, 2, 3)]
    with pytest.raises(LiveAppLaunchError) as raised:
        las.plan_slot(statuses, "sales", _ME)
    assert raised.value.cause == "slots_full"
    text = str(raised.value)
    assert "slot 1 runs `app1` for bo@example.com since 04:01 UTC" in text
    assert "slot 3 runs `app3` for bo@example.com since 04:03 UTC" in text
    assert "by 12:00 UTC at the latest" in text
    assert "ask the operator for more slots" in text


def test_a_refusal_names_a_slot_that_gave_no_answer():
    statuses = [_status(1, state="running", app=_app("app1", _OTHER, "2026-10-21T04:01:00Z")), None]
    with pytest.raises(LiveAppLaunchError) as raised:
        las.plan_slot(statuses, "sales", _ME)
    assert "slot 1 runs `app1` for bo@example.com since 04:01 UTC; slot 2 gave no answer" in str(raised.value)


def test_with_no_slot_answering_a_launch_is_refused():
    with pytest.raises(LiveAppLaunchError) as raised:
        las.plan_slot([None, None, None], "sales", _ME)
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
    assert [(c.engine, c.kind) for c in request.connections] == [("postgres", "db"), ("rest", "http-read")]

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
    slots[0].run("sales-demo", _ME, started_at="2026-10-21T04:10:00Z")
    cards.answer = "cancelled"
    session = _session()
    with pytest.raises(LiveAppLaunchError) as raised:
        await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=())
    assert raised.value.cause == "launch_not_approved"
    assert session._live_app_host.events == []
    assert slots[0].resets == 0 and slots[0].requests == []
    assert slots[0].app.artifact == "sales-demo"


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


async def test_a_new_connection_relaunches_through_the_card(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    session = _session()
    await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=_DECLARED[:1])
    await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=_DECLARED)
    assert len(cards) == 2
    assert "these changed: its connections." in cards[1]


async def test_a_full_house_of_my_apps_replaces_my_oldest_after_the_card(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    slots[0].run("mine-new", _ME, started_at="2026-10-21T04:30:00Z")
    slots[1].run("mine-old", _ME, started_at="2026-10-21T04:20:00Z")
    slots[2].run("theirs", _OTHER, started_at="2026-10-21T04:00:00Z")
    result = await las.launch_live_app(_session(), slug="sales-demo", folder=folder, datasources=())
    assert "this stops your app `mine-old`, running since 04:20 UTC" in cards[0]
    assert result.slot == 2
    assert slots[1].resets == 1 and slots[2].resets == 0 and slots[0].resets == 0
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


async def test_a_busy_slot_is_picked_again_once(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    busy = SlotRefused(409, SlotError(error="slot_busy"))
    slots[0].refuse_launch = busy
    original = slots[0].launch

    def launch_once_busy(request):
        # Another process took slot 1 between the lock and the launch.
        slots[0].refuse_launch = None
        slots[0].run("someone-else", _OTHER, started_at="2026-10-21T04:59:00Z")
        slots[0].boot += 1
        raise busy

    slots[0].launch = launch_once_busy  # type: ignore[method-assign]
    result = await las.launch_live_app(_session(), slug="sales-demo", folder=folder, datasources=())
    assert result.slot == 2
    assert slots[1].requests and original.__self__ is slots[0]


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
    slots[0].run("sales-demo", _ME, started_at="2026-10-21T04:10:00Z")

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
    assert len(slots[0].requests) == 1 and slots[0].resets == 1 and slots[0].app is None


async def test_replacing_a_running_app_claims_its_slot_first(tmp_path, slots, approved, cards):
    folder = _artifact(tmp_path)
    slots[0].run("sales-demo", _ME, started_at="2026-10-21T04:10:00Z")
    session = _session()
    await las.launch_live_app(session, slug="sales-demo", folder=folder, datasources=())
    assert [e[0] for e in session._live_app_host.events] == ["claim", "launched"]
    assert session._live_app_host.events[0] == ("claim", 1)


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


async def test_a_cell_relaunches_only_the_launchers_running_apps(tmp_path, slots, approved, cards):
    root = tmp_path / "artifacts"
    mine, theirs, idle = _artifact(root, "mine"), _artifact(root, "theirs"), _artifact(root, "idle")
    slots[0].run("mine", _ME, started_at="2026-10-21T04:10:00Z")
    slots[1].run("theirs", _OTHER, started_at="2026-10-21T04:10:00Z")
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
    _artifact(root, "mine")
    slots[0].run("mine", _ME, started_at="2026-10-21T04:10:00Z")
    cards.answer = "cancelled"
    store = SimpleNamespace(open=lambda slug: SimpleNamespace(datasources=[]), folder_for=lambda slug: root / slug)
    lines = await las.relaunch_changed_apps(_session(), store, ["mine"])
    assert lines == [f"Live app `mine` was not relaunched. {las.NOT_APPROVED_MESSAGE}"]


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
    status = las.fetch_slot_status(slot_server, token=_TOKEN)
    assert status.state == "empty" and status.boot_id == "b1"
    assert _SlotServer.seen[-1] == {"path": "/status", "auth": f"Bearer {_TOKEN}"}
    with pytest.raises(SlotRefused) as raised:
        las.fetch_slot_status(slot_server, token="wrong")
    assert raised.value.status_code == 401


def test_a_launch_body_carries_the_values_and_an_error_never_does(slot_server):
    request = las.LaunchRequest(
        artifact="a", launcher=_ME, files=(las.LaunchFile(path="backend.py", content_b64="YQ=="),),
        env={"DS_X__Y": _PG_PASSWORD}, connections=(),
    )
    with pytest.raises(SlotRefused) as raised:
        las.SlotClient(slot_server, token=_TOKEN).launch(request)
    assert _SlotServer.seen[-1]["body"]["env"] == {"DS_X__Y": _PG_PASSWORD}
    assert raised.value.body.field == "env.DS_X__Y"
    assert _PG_PASSWORD not in str(raised.value)


def test_an_unreachable_slot_is_unavailable():
    with pytest.raises(SlotUnavailable):
        las.fetch_slot_status("http://127.0.0.1:9", token=_TOKEN, timeout=0.5)


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
