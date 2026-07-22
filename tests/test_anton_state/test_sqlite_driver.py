import time
import pytest
from anton_state.schema import Attr, StateSchema
from anton_state.errors import ConditionalCheckFailed, StateValidationError
from anton_state.sqlite_driver import SQLiteDriver

M = StateSchema(
    pk=Attr(name="pk"),
    sk=Attr(name="sk"),
    ttl_attribute="expires_at",
)


@pytest.fixture
def drv(tmp_path):
    return SQLiteDriver(str(tmp_path / "state.db"), M)


def test_put_get_roundtrip(drv):
    drv.put({"pk": "u1", "sk": "profile", "name": "Alice"}, if_not_exists=False, if_version=None)
    got = drv.get("u1", "profile", consistent=True)
    assert got["pk"] == "u1" and got["sk"] == "profile" and got["name"] == "Alice"
    assert "_v" in got  # server-managed version present


def test_get_missing_returns_none(drv):
    assert drv.get("nope", "x", consistent=True) is None


def test_wal_enabled(drv):
    import sqlite3
    con = sqlite3.connect(drv.path)
    mode = con.execute("PRAGMA journal_mode").fetchone()[0]
    con.close()
    assert mode.lower() == "wal"


def test_if_not_exists_conflict(drv):
    drv.put({"pk": "u1", "sk": "s"}, if_not_exists=True, if_version=None)
    with pytest.raises(ConditionalCheckFailed):
        drv.put({"pk": "u1", "sk": "s"}, if_not_exists=True, if_version=None)


def test_put_assigns_monotonic_version(drv):
    drv.put({"pk": "c", "sk": "g", "n": 1}, if_not_exists=True, if_version=None)
    v0 = drv.get("c", "g", consistent=True)["_v"]
    assert v0 > 1_000_000_000_000  # epoch-millis, NOT a reset-to-1 (prevents ABA)
    # unconditional REPLACE assigns a fresh, non-decreasing version
    drv.put({"pk": "c", "sk": "g", "n": 2}, if_not_exists=False, if_version=None)
    assert drv.get("c", "g", consistent=True)["_v"] >= v0


def test_put_if_version_checks_and_bumps(drv):
    drv.put({"pk": "c", "sk": "g", "n": 1}, if_not_exists=True, if_version=None)
    v0 = drv.get("c", "g", consistent=True)["_v"]
    with pytest.raises(ConditionalCheckFailed):
        drv.put({"pk": "c", "sk": "g", "n": 9}, if_not_exists=False, if_version=v0 + 999)
    drv.put({"pk": "c", "sk": "g", "n": 2}, if_not_exists=False, if_version=v0)
    assert drv.get("c", "g", consistent=True)["_v"] == v0 + 1


def test_put_overwrites_client_supplied_version(drv):
    drv.put({"pk": "c", "sk": "g", "n": 1, "_v": 999}, if_not_exists=True, if_version=None)
    assert drv.get("c", "g", consistent=True)["_v"] != 999  # client _v ignored (server sets it)


def test_ttl_hidden_on_read_but_zombie_blocks_conditional(drv):
    past = time.time() - 10
    drv.put({"pk": "u1", "sk": "s", "expires_at": past}, if_not_exists=False, if_version=None)
    # expired → invisible on read
    assert drv.get("u1", "s", consistent=True) is None
    # but physically present → if_not_exists fails (parity with the DynamoDB zombie)
    with pytest.raises(ConditionalCheckFailed):
        drv.put({"pk": "u1", "sk": "s"}, if_not_exists=True, if_version=None)


def test_query_prefix_and_ttl_filter(drv):
    drv.put({"pk": "u1", "sk": "msg#1", "t": "a"}, if_not_exists=False, if_version=None)
    drv.put({"pk": "u1", "sk": "msg#2", "t": "b"}, if_not_exists=False, if_version=None)
    drv.put({"pk": "u1", "sk": "note#1"}, if_not_exists=False, if_version=None)
    drv.put({"pk": "u1", "sk": "msg#old", "expires_at": time.time() - 5}, if_not_exists=False, if_version=None)
    rows = drv.query("u1", sk_prefix="msg#", filters=None, consistent=True, limit=None)
    sks = sorted(r["sk"] for r in rows)
    assert sks == ["msg#1", "msg#2"]


def test_query_prefix_is_literal_not_like(drv):
    # An underscore in the prefix must NOT act as a LIKE wildcard.
    drv.put({"pk": "u1", "sk": "user_settings#a"}, if_not_exists=False, if_version=None)
    drv.put({"pk": "u1", "sk": "userXsettings#b"}, if_not_exists=False, if_version=None)
    rows = drv.query("u1", sk_prefix="user_settings#", filters=None, consistent=True, limit=None)
    assert [r["sk"] for r in rows] == ["user_settings#a"]


def test_query_filters_equality(drv):
    drv.put({"pk": "u1", "sk": "a", "kind": "x"}, if_not_exists=False, if_version=None)
    drv.put({"pk": "u1", "sk": "b", "kind": "y"}, if_not_exists=False, if_version=None)
    rows = drv.query("u1", sk_prefix=None, filters={"kind": "x"}, consistent=True, limit=None)
    assert [r["sk"] for r in rows] == ["a"]


def test_increment_atomic(drv):
    assert drv.increment("c", "g", field="n", by=1) == 1   # creates item
    assert drv.increment("c", "g", field="n", by=2) == 3
    assert drv.get("c", "g", consistent=True)["n"] == 3


def test_update_set_and_add(drv):
    drv.put({"pk": "c", "sk": "g", "n": 1}, if_not_exists=True, if_version=None)
    item = drv.update("c", "g", set_fields={"name": "x"}, add_fields={"n": 4}, if_version=None)
    assert item["name"] == "x" and item["n"] == 5


def test_put_validates(drv):
    with pytest.raises(StateValidationError):
        drv.put({"sk": "s"}, if_not_exists=False, if_version=None)  # no pk
