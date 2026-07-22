import pytest
from anton_state.schema import Attr, Index, StateSchema


def test_minimal_schema_defaults():
    m = StateSchema(pk=Attr(name="pk"))
    assert m.version == 1
    assert m.pk.type == "S"
    assert m.sk is None
    assert m.gsis == []
    assert m.ttl_attribute is None


def test_key_attrs_collects_all_keys():
    m = StateSchema(
        pk=Attr(name="pk"),
        sk=Attr(name="sk"),
        ttl_attribute="expires_at",
    )
    assert m.key_attrs() == {"pk", "sk"}


def test_non_empty_gsis_rejected():
    # v1 shared table: secondary indexes are not supported.
    with pytest.raises(ValueError):
        StateSchema(pk=Attr(name="pk"), sk=Attr(name="sk"),
                    gsis=[Index(name="byUser", pk=Attr(name="user"))])


def test_empty_gsis_ok():
    s = StateSchema(pk=Attr(name="pk"), sk=Attr(name="sk"))
    assert s.gsis == []


def test_non_string_key_type_rejected():
    # v1: string keys only
    with pytest.raises(Exception):
        Attr(name="pk", type="N")


def test_manifest_roundtrip(tmp_path):
    m = StateSchema(pk=Attr(name="pk"), sk=Attr(name="sk"), ttl_attribute="expires_at")
    path = tmp_path / "state_manifest.json"
    m.to_manifest(path)
    loaded = StateSchema.from_manifest(path)
    assert loaded == m


def test_from_manifest_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        StateSchema.from_manifest(tmp_path / "nope.json")
