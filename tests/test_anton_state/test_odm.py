import pytest
from anton_state.schema import Attr, StateSchema
from anton_state.factory import open_store
from anton_state.odm import Collection

M = StateSchema(pk=Attr(name="pk"), sk=Attr(name="sk"))


@pytest.fixture
def store(tmp_path):
    return open_store(M, state=None, local_path=str(tmp_path / "s.db"))


async def test_put_get_explicit_pk(store):
    c = Collection(store, "task")
    await c.put("t1", {"title": "do it"}, pk="proj1")
    got = await c.get("t1", pk="proj1")
    assert got["title"] == "do it" and got["_key"] == "t1"


async def test_default_pk_roundtrip(store):
    todos = Collection(store, "todos")
    await todos.put("a", {"text": "buy milk"})  # pk defaulted
    got = await todos.get("a")
    assert got["text"] == "buy milk" and got["_key"] == "a"
    lst = await todos.list()
    assert len(lst) == 1


async def test_list_only_collection_prefix(store):
    tasks = Collection(store, "task")
    notes = Collection(store, "note")
    await tasks.put("a", {"n": 1}, pk="p1")
    await tasks.put("b", {"n": 2}, pk="p1")
    await notes.put("a", {"x": 9}, pk="p1")
    items = await tasks.list(pk="p1")
    assert sorted(i["_key"] for i in items) == ["a", "b"]


async def test_delete(store):
    c = Collection(store, "task")
    await c.put("t1", {"n": 1}, pk="p1")
    await c.delete("t1", pk="p1")
    assert await c.get("t1", pk="p1") is None


async def test_increment(store):
    counters = Collection(store, "counters")
    assert await counters.increment("visits", field="n") == 1
    assert await counters.increment("visits", field="n", by=2) == 3


async def test_update(store):
    c = Collection(store, "task")
    await c.put("t1", {"n": 1}, pk="p1")
    item = await c.update("t1", set_fields={"done": True}, add_fields={"n": 4}, pk="p1")
    assert item["done"] is True and item["n"] == 5


async def test_put_rejects_reserved_attrs_in_value(store):
    from anton_state.errors import StateValidationError
    c = Collection(store, "task")
    with pytest.raises(StateValidationError):
        await c.put("t1", {"pk": "hijack"}, pk="p1")
