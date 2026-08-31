from anton_state.schema import Attr, StateSchema
from anton_state.factory import open_store

M = StateSchema(
    pk=Attr(name="pk"), sk=Attr(name="sk"),
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
