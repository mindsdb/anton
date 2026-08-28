"""Shared validation — deliberately strict, "like the cloud" (spec §2).

Local SQLite is by nature more permissive than DynamoDB; here we reject what
prod would reject, so invalid data is not silently accepted locally.
"""
from __future__ import annotations

import json

from .errors import StateValidationError
from .schema import StateSchema

MAX_ITEM_BYTES = 400 * 1024

# _pk/_sk/_ttl are physical control attributes added by the broker on the shared
# table; only _v (version) and _key (Collection) are allowed reserved names.
_RESERVED_UNDERSCORE = {"_v", "_key"}


def check_value(value) -> None:
    if isinstance(value, bool) or isinstance(value, (int, float, str)) or value is None:
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise StateValidationError("map keys must be strings")
            check_value(v)
        return
    if isinstance(value, list):
        for v in value:
            check_value(v)
        return
    raise StateValidationError(f"unsupported value type: {type(value).__name__}")


def check_number(value, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StateValidationError(f"{label} must be a number")


def check_size(item: dict) -> None:
    try:
        size = len(json.dumps(item, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError) as e:
        raise StateValidationError(f"item is not JSON-serializable: {e}")
    if size > MAX_ITEM_BYTES:
        raise StateValidationError(f"item too large: {size} > {MAX_ITEM_BYTES} bytes")


def validate_key(pk: str, sk: str | None, schema: StateSchema) -> None:
    if not isinstance(pk, str) or pk == "":
        raise StateValidationError("partition key must be a non-empty string")
    if schema.sk is not None:
        if sk is None or not isinstance(sk, str) or sk == "":
            raise StateValidationError("sort key must be a non-empty string")


def validate_item(item: dict, schema: StateSchema) -> None:
    if not isinstance(item, dict):
        raise StateValidationError("item must be a dict")

    pk_name = schema.pk.name
    if pk_name not in item:
        raise StateValidationError(f"item missing partition key '{pk_name}'")
    sk_name = schema.sk.name if schema.sk else None
    validate_key(item.get(pk_name), item.get(sk_name) if sk_name else None, schema)

    # Empty strings are forbidden only in key attributes (DynamoDB has allowed
    # empty strings in non-key attributes since 2020).
    for key_name in schema.key_attrs():
        if key_name in item and isinstance(item[key_name], str) and item[key_name] == "":
            raise StateValidationError(f"key attribute '{key_name}' must not be empty")

    # Forbid user attrs with a reserved "_" prefix (collide with the broker's
    # physical _pk/_sk/_ttl on the shared table). _v and _key are allowed.
    for name in item:
        if name.startswith("_") and name not in _RESERVED_UNDERSCORE:
            raise StateValidationError(f"attribute '{name}' uses a reserved '_' prefix")

    for value in item.values():
        check_value(value)

    if schema.ttl_attribute and schema.ttl_attribute in item:
        ttl = item[schema.ttl_attribute]
        if isinstance(ttl, bool) or not isinstance(ttl, (int, float)):
            raise StateValidationError(
                f"TTL attribute '{schema.ttl_attribute}' must be a number (epoch seconds)"
            )

    check_size(item)
