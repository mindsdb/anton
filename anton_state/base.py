"""Driver protocol and the async Store facade."""
from __future__ import annotations

import asyncio
from typing import Any, Protocol

from .errors import StateValidationError
from .schema import StateSchema

Item = dict[str, Any]

# Reserved version attribute for optimistic locking. Single place for both
# drivers (dynamo imports it from here).
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
        index: str | None,
        filters: dict[str, Any] | None,
        consistent: bool,
        limit: int | None,
    ) -> list[Item]: ...


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
        """Write an item.

        Optimistic locking: the version attribute `_v` is incremented by the
        **caller** — read the item, set `item["_v"] = current + 1`, then call
        `put(..., if_version=current)`. The driver only checks the condition.
        `if_not_exists` and `if_version` are **mutually exclusive** (the
        combination is meaningless and would be interpreted differently by the
        drivers) — rejected uniformly.
        """
        if if_not_exists and if_version is not None:
            raise StateValidationError("if_not_exists and if_version are mutually exclusive")
        await asyncio.to_thread(
            self._driver.put, item, if_not_exists=if_not_exists, if_version=if_version
        )

    async def delete(
        self, pk: str, sk: str | None = None, *, if_version: int | None = None
    ) -> None:
        await asyncio.to_thread(self._driver.delete, pk, sk, if_version=if_version)

    async def query(
        self,
        pk: str,
        *,
        sk_prefix: str | None = None,
        index: str | None = None,
        filters: dict[str, Any] | None = None,
        consistent: bool | None = None,
        limit: int | None = None,
    ) -> list[Item]:
        # Base table is strongly consistent by default; GSIs are always eventual.
        # NOTE: the result order of a GSI query is not guaranteed and may differ
        # between drivers (SQLite orders by the base (pk, sk), DynamoDB by the
        # index sk). Do not rely on GSI query ordering.
        eff_consistent = (index is None) if consistent is None else consistent
        return await asyncio.to_thread(
            self._driver.query,
            pk,
            sk_prefix=sk_prefix,
            index=index,
            filters=filters,
            consistent=eff_consistent,
            limit=limit,
        )
