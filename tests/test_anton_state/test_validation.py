import pytest
from anton_state.schema import Attr, StateSchema
from anton_state.errors import StateValidationError
from anton_state.validation import validate_item, validate_key, MAX_ITEM_BYTES

M = StateSchema(pk=Attr(name="pk"), sk=Attr(name="sk"), ttl_attribute="expires_at")


def test_ok_item_passes():
    validate_item({"pk": "u1", "sk": "profile", "name": "Alice"}, M)


def test_missing_pk_rejected():
    with pytest.raises(StateValidationError):
        validate_item({"sk": "profile"}, M)


def test_empty_string_key_rejected():
    with pytest.raises(StateValidationError):
        validate_item({"pk": "", "sk": "profile"}, M)


def test_unsupported_type_rejected():
    with pytest.raises(StateValidationError):
        validate_item({"pk": "u1", "sk": "s", "when": object()}, M)


def test_oversize_item_rejected():
    big = {"pk": "u1", "sk": "s", "blob": "x" * (MAX_ITEM_BYTES + 10)}
    with pytest.raises(StateValidationError):
        validate_item(big, M)


def test_ttl_must_be_number_epoch():
    with pytest.raises(StateValidationError):
        validate_item({"pk": "u1", "sk": "s", "expires_at": "2026-01-01"}, M)
    validate_item({"pk": "u1", "sk": "s", "expires_at": 1893456000}, M)


def test_validate_key_empty_rejected():
    with pytest.raises(StateValidationError):
        validate_key("", None, M)


def test_underscore_prefixed_user_attr_rejected():
    with pytest.raises(StateValidationError):
        validate_item({"pk": "p", "sk": "s", "_ttl": 1}, M)


def test_reserved_underscore_attrs_allowed():
    # _v (version) and _key (Collection) are the only allowed "_" names.
    validate_item({"pk": "p", "sk": "s", "_v": 3, "_key": "k", "n": 1}, M)
