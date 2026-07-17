import time
import pytest
from anton_state.schema import Attr, Index, StateSchema
from anton_state.errors import ConditionalCheckFailed, StateValidationError
from anton_state.sqlite_driver import SQLiteDriver

M = StateSchema(
    pk=Attr(name="pk"),
    sk=Attr(name="sk"),
    gsis=[Index(name="by_user", pk=Attr(name="user_id"))],
    ttl_attribute="expires_at",
)


@pytest.fixture
def drv(tmp_path):
    return SQLiteDriver(str(tmp_path / "state.db"), M)


def test_put_get_roundtrip(drv):
    drv.put({"pk": "u1", "sk": "profile", "name": "Alice"}, if_not_exists=False, if_version=None)
    assert drv.get("u1", "profile", consistent=True) == {"pk": "u1", "sk": "profile", "name": "Alice"}


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


def test_optimistic_lock_version(drv):
    drv.put({"pk": "u1", "sk": "s", "_v": 1, "n": 0}, if_not_exists=False, if_version=None)
    # correct version succeeds
    drv.put({"pk": "u1", "sk": "s", "_v": 2, "n": 1}, if_not_exists=False, if_version=1)
    # stale version fails
    with pytest.raises(ConditionalCheckFailed):
        drv.put({"pk": "u1", "sk": "s", "_v": 3, "n": 2}, if_not_exists=False, if_version=1)


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
    rows = drv.query("u1", sk_prefix="msg#", index=None, filters=None, consistent=True, limit=None)
    sks = sorted(r["sk"] for r in rows)
    assert sks == ["msg#1", "msg#2"]


def test_query_filters_equality(drv):
    drv.put({"pk": "u1", "sk": "a", "kind": "x"}, if_not_exists=False, if_version=None)
    drv.put({"pk": "u1", "sk": "b", "kind": "y"}, if_not_exists=False, if_version=None)
    rows = drv.query("u1", sk_prefix=None, index=None, filters={"kind": "x"}, consistent=True, limit=None)
    assert [r["sk"] for r in rows] == ["a"]


def test_query_by_gsi(drv):
    drv.put({"pk": "p1", "sk": "s1", "user_id": "u9"}, if_not_exists=False, if_version=None)
    drv.put({"pk": "p2", "sk": "s2", "user_id": "u9"}, if_not_exists=False, if_version=None)
    drv.put({"pk": "p3", "sk": "s3", "user_id": "u8"}, if_not_exists=False, if_version=None)
    rows = drv.query("u9", sk_prefix=None, index="by_user", filters=None, consistent=False, limit=None)
    assert sorted(r["pk"] for r in rows) == ["p1", "p2"]


def test_unknown_index_raises_validation(drv):
    with pytest.raises(StateValidationError):
        drv.query("u1", sk_prefix=None, index="by_ghost", filters=None, consistent=False, limit=None)


def test_put_validates(drv):
    with pytest.raises(StateValidationError):
        drv.put({"sk": "s"}, if_not_exists=False, if_version=None)  # no pk
