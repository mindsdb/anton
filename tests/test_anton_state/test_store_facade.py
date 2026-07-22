import pytest
from anton_state.base import Store
from anton_state.schema import Attr, StateSchema


class FakeDriver:
    def __init__(self):
        self.calls = []

    def get(self, pk, sk, *, consistent):
        self.calls.append(("get", pk, sk, consistent))
        return {"pk": pk}

    def put(self, item, *, if_not_exists, if_version):
        self.calls.append(("put", item, if_not_exists, if_version))

    def delete(self, pk, sk, *, if_version):
        self.calls.append(("delete", pk, sk, if_version))

    def query(self, pk, *, sk_prefix, filters, consistent, limit):
        self.calls.append(("query", pk, sk_prefix, filters, consistent, limit))
        return [{"pk": pk}]

    def increment(self, pk, sk, *, field, by):
        self.calls.append(("increment", pk, sk, field, by))
        return 7

    def update(self, pk, sk, *, set_fields, add_fields, if_version):
        self.calls.append(("update", pk, sk, set_fields, add_fields, if_version))
        return {"pk": pk}


M = StateSchema(pk=Attr(name="pk"), sk=Attr(name="sk"))


async def test_get_delegates_with_default_consistent_true():
    d = FakeDriver()
    store = Store(d, M)
    res = await store.get("u1", "profile")
    assert res == {"pk": "u1"}
    assert d.calls[0] == ("get", "u1", "profile", True)


async def test_query_defaults():
    d = FakeDriver()
    store = Store(d, M)
    res = await store.query("u1", sk_prefix="msg#")
    assert res == [{"pk": "u1"}]
    assert d.calls[0][0] == "query"


async def test_put_rejects_mutually_exclusive_conditions():
    from anton_state.errors import StateValidationError
    store = Store(FakeDriver(), M)
    with pytest.raises(StateValidationError):
        await store.put({"pk": "u1", "sk": "s"}, if_not_exists=True, if_version=1)


async def test_query_has_no_index_kwarg():
    store = Store(FakeDriver(), M)
    with pytest.raises(TypeError):
        await store.query("u1", index="byUser")  # index removed in v1


async def test_increment_delegates_and_returns():
    d = FakeDriver()
    store = Store(d, M)
    v = await store.increment("c", "global", field="n", by=2)
    assert v == 7
    assert d.calls[-1] == ("increment", "c", "global", "n", 2)


async def test_update_delegates():
    d = FakeDriver()
    store = Store(d, M)
    item = await store.update("c", "g", set_fields={"a": 1}, add_fields={"n": 1})
    assert item == {"pk": "c"}
    assert d.calls[-1] == ("update", "c", "g", {"a": 1}, {"n": 1}, None)
