"""Driver protocol and the async Store facade."""
from __future__ import annotations

import asyncio
from typing import Any, Protocol

from .errors import StateValidationError
from .schema import StateSchema

Item = dict[str, Any]

# Reserved version attribute for optimistic locking, set by the driver/broker
# (the request body is not trusted). Single place for all drivers.
_VERSION_ATTR = "_v"


class Driver(Protocol):
    def get(self, pk: str, sk: str | None, *, consistent: bool) -> Item | None: ...

    def put(self, item: Item, *, if_not_exists: bool, if_version: int | None) -> None: ...

    def delete(self, pk: str, sk: str | None, *, if_version: int | None) -> None: ...

    def query(
        self,
        pk: str,
        *,
        sk_prefix: str | None,
        filters: dict[str, Any] | None,
        consistent: bool,
        limit: int | None,
    ) -> list[Item]: ...

    def increment(self, pk: str, sk: str | None, *, field: str, by: int | float) -> int | float: ...

    def update(
        self,
        pk: str,
        sk: str | None,
        *,
        set_fields: dict[str, Any] | None,
        add_fields: dict[str, int | float] | None,
        if_version: int | None,
    ) -> Item: ...


class Store:
    """Async facade: backend routes are `async def`, so sync drivers run in a thread."""

    def __init__(self, driver: Driver, schema: StateSchema):
        self._driver = driver
        self.schema = schema

    async def get(self, pk: str, sk: str | None = None, *, consistent: bool = True) -> Item | None:
        return await asyncio.to_thread(self._driver.get, pk, sk, consistent=consistent)

    async def put(
        self, item: Item, *, if_not_exists: bool = False, if_version: int | None = None
    ) -> None:
        """Write an item. `_v` is managed by the driver/broker; a `_v` present in
        `item` is overwritten. `if_not_exists` and `if_version` are mutually
        exclusive."""
        if if_not_exists and if_version is not None:
            raise StateValidationError("if_not_exists and if_version are mutually exclusive")
        await asyncio.to_thread(
            self._driver.put, item, if_not_exists=if_not_exists, if_version=if_version
        )

    async def delete(self, pk: str, sk: str | None = None, *, if_version: int | None = None) -> None:
        await asyncio.to_thread(self._driver.delete, pk, sk, if_version=if_version)

    async def query(
        self,
        pk: str,
        *,
        sk_prefix: str | None = None,
        filters: dict[str, Any] | None = None,
        consistent: bool = True,
        limit: int | None = None,
    ) -> list[Item]:
        return await asyncio.to_thread(
            self._driver.query,
            pk,
            sk_prefix=sk_prefix,
            filters=filters,
            consistent=consistent,
            limit=limit,
        )

    async def increment(
        self, pk: str, sk: str | None = None, *, field: str, by: int | float = 1
    ) -> int | float:
        """Atomically add `by` to numeric `field` of one item; returns the new value."""
        return await asyncio.to_thread(self._driver.increment, pk, sk, field=field, by=by)

    async def update(
        self,
        pk: str,
        sk: str | None = None,
        *,
        set_fields: dict[str, Any] | None = None,
        add_fields: dict[str, int | float] | None = None,
        if_version: int | None = None,
    ) -> Item:
        """Atomically SET/ADD fields of one item; returns the new item."""
        return await asyncio.to_thread(
            self._driver.update, pk, sk,
            set_fields=set_fields, add_fields=add_fields, if_version=if_version,
        )
