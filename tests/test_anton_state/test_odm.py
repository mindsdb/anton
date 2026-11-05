import pytest
from anton_state.schema import Attr, StateSchema
from anton_state.factory import open_store
from anton_state.errors import ConditionalCheckFailed, StateValidationError
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


async def test_put_accepts_item_read_back_with_get(store):
    games = Collection(store, "game")
    await games.put("current", {"moves": [], "board": {"A1": None}})
    game = await games.get("current")
    game["moves"].append("A1")
    game["board"]["A1"] = "X"
    await games.put("current", game)
    got = await games.get("current")
    assert got["moves"] == ["A1"] and got["board"]["A1"] == "X"


async def test_put_read_back_with_if_version_rejects_stale_version(store):
    games = Collection(store, "game")
    await games.put("current", {"moves": []}, pk="room1")
    game = await games.get("current", pk="room1")
    game["moves"].append("A1")
    await games.put("current", game, pk="room1", if_version=game["_v"])
    game["moves"].append("B2")
    with pytest.raises(ConditionalCheckFailed):
        await games.put("current", game, pk="room1", if_version=game["_v"])


async def test_put_accepts_read_back_with_custom_key_names(tmp_path):
    schema = StateSchema(pk=Attr(name="tenant"), sk=Attr(name="id"))
    store = open_store(schema, state=None, local_path=str(tmp_path / "s.db"))
    games = Collection(store, "game")
    await games.put("current", {"moves": []}, pk="t1")
    game = await games.get("current", pk="t1")
    game["moves"].append("A1")
    await games.put("current", game, pk="t1")
    assert (await games.get("current", pk="t1"))["moves"] == ["A1"]


async def test_put_leaves_callers_value_unchanged(store):
    c = Collection(store, "task")
    await c.put("t1", {"n": 1})
    item = await c.get("t1")
    before = dict(item)
    await c.put("t1", item)
    assert item == before


@pytest.mark.parametrize("value", [
    {"_key": "b"},
    {"sk": "task#b"},
    {"pk": "other"},
])
async def test_put_rejects_key_attrs_of_another_target(store, value):
    c = Collection(store, "task")
    with pytest.raises(StateValidationError, match="do not match"):
        await c.put("a", value)


async def test_put_rejects_item_copied_under_another_key(store):
    c = Collection(store, "task")
    await c.put("a", {"n": 1})
    with pytest.raises(StateValidationError, match="do not match"):
        await c.put("b", await c.get("a"))


async def test_put_rejects_item_copied_to_another_partition(store):
    c = Collection(store, "task")
    await c.put("a", {"n": 1}, pk="room1")
    with pytest.raises(StateValidationError, match="do not match"):
        await c.put("a", await c.get("a", pk="room1"), pk="room2")
