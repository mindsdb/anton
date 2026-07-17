import pytest
from anton_state.schema import Attr, Index, StateSchema
from anton_state.factory import open_store
from anton_state.errors import StateValidationError

M = StateSchema(
    pk=Attr(name="pk"), sk=Attr(name="sk"),
    gsis=[Index(name="by_user", pk=Attr(name="user_id"))],
    ttl_attribute="expires_at",
)


def test_backend_and_publish_read_same_schema(tmp_path):
    manifest = tmp_path / "state_manifest.json"
    M.to_manifest(manifest)
    # "publish side" (without executing artifact code)
    publish_schema = StateSchema.from_manifest(manifest)
    # "backend side" — via the factory, from the same file
    store = open_store(state=None, local_path=str(tmp_path / "s.db"), manifest_path=str(manifest))
    assert store.schema == publish_schema == M


async def test_unknown_index_surfaces_through_store(tmp_path):
    manifest = tmp_path / "state_manifest.json"
    M.to_manifest(manifest)
    store = open_store(state=None, local_path=str(tmp_path / "s.db"), manifest_path=str(manifest))
    with pytest.raises(StateValidationError):
        await store.query("u1", index="by_ghost")
