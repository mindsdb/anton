import pytest
from anton_state.schema import Attr, StateSchema
from anton_state.factory import open_store
from anton_state.odm import Collection

M = StateSchema(pk=Attr(name="pk"), sk=Attr(name="sk"))


@pytest.fixture
def store(tmp_path):
    return open_store(M, state=None, local_path=str(tmp_path / "s.db"))


async def test_put_get(store):
    c = Collection(store, "task")
    await c.put("proj1", "t1", {"title": "do it"})
    got = await c.get("proj1", "t1")
    assert got["title"] == "do it"


async def test_list_only_collection_prefix(store):
    tasks = Collection(store, "task")
    notes = Collection(store, "note")
    await tasks.put("p1", "a", {"n": 1})
    await tasks.put("p1", "b", {"n": 2})
    await notes.put("p1", "a", {"x": 9})
    items = await tasks.list("p1")
    assert sorted(i["_key"] for i in items) == ["a", "b"]


async def test_delete(store):
    c = Collection(store, "task")
    await c.put("p1", "t1", {"n": 1})
    await c.delete("p1", "t1")
    assert await c.get("p1", "t1") is None


async def test_put_rejects_reserved_attrs_in_value(store):
    from anton_state.errors import StateValidationError
    c = Collection(store, "task")
    with pytest.raises(StateValidationError):
        await c.put("p1", "t1", {"pk": "hijack"})
