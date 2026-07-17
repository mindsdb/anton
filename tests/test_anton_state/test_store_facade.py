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

    def query(self, pk, *, sk_prefix, index, filters, consistent, limit):
        self.calls.append(("query", pk, sk_prefix, index, filters, consistent, limit))
        return [{"pk": pk}]


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
    import pytest
    from anton_state.errors import StateValidationError
    store = Store(FakeDriver(), M)
    with pytest.raises(StateValidationError):
        await store.put({"pk": "u1", "sk": "s"}, if_not_exists=True, if_version=1)
