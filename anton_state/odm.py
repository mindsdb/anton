"""Thin single-table helper: a logical collection over (pk, sk).

`pk` is the partition scope and defaults to a single partition — simple apps
never think about it; multi-tenant-within-an-artifact apps pass an explicit pk.
The sort key is managed here: sk = "{collection}#{key}".
"""
from __future__ import annotations

from typing import Any

from .base import Item, Store
from .errors import StateValidationError

_DEFAULT_PK = "_"


class Collection:
    def __init__(self, store: Store, name: str):
        self._store = store
        self._name = name
        self._pk = store.schema.pk.name
        self._sk = store.schema.sk.name if store.schema.sk else None
        if self._sk is None:
            raise ValueError("Collection requires a schema with a sort key")
        self._reserved = {self._pk, self._sk, "_key"}

    def _sk_val(self, key: str) -> str:
        return f"{self._name}#{key}"

    async def get(self, key: str, *, pk: str = _DEFAULT_PK) -> Item | None:
        return await self._store.get(pk, self._sk_val(key))

    async def put(
        self, key: str, value: dict, *, pk: str = _DEFAULT_PK,
        if_not_exists: bool = False, if_version: int | None = None,
    ) -> None:
        clash = self._reserved & set(value)
        if clash:
            raise StateValidationError(f"value must not contain reserved attrs: {sorted(clash)}")
        item: dict[str, Any] = dict(value)
        item[self._pk] = pk
        item[self._sk] = self._sk_val(key)
        item["_key"] = key
        await self._store.put(item, if_not_exists=if_not_exists, if_version=if_version)

    async def delete(self, key: str, *, pk: str = _DEFAULT_PK, if_version: int | None = None) -> None:
        await self._store.delete(pk, self._sk_val(key), if_version=if_version)

    async def list(self, *, pk: str = _DEFAULT_PK, limit: int | None = None) -> list[Item]:
        return await self._store.query(pk, sk_prefix=f"{self._name}#", limit=limit)

    async def increment(
        self, key: str, *, field: str, by: int | float = 1, pk: str = _DEFAULT_PK
    ) -> int | float:
        return await self._store.increment(pk, self._sk_val(key), field=field, by=by)

    async def update(
        self, key: str, *, set_fields: dict | None = None, add_fields: dict | None = None,
        pk: str = _DEFAULT_PK, if_version: int | None = None,
    ) -> Item:
        return await self._store.update(
            pk, self._sk_val(key),
            set_fields=set_fields, add_fields=add_fields, if_version=if_version,
        )
