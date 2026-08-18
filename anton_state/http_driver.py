"""Cloud driver: thin HTTP RPC to the trusted state broker (plan #2).

stdlib-only (urllib), synchronous — the async Store facade runs it in a thread.
The namespace is NEVER sent; the broker derives it from the signed token, so
untrusted code cannot address another artifact's data. Reads are retried on
transient failure; mutations are not (their outcome would be unknown).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from . import errors
from .base import Driver  # noqa: F401  (documents intent; Protocol is structural)
from .errors import (
    ConditionalCheckFailed,
    StateThrottled,
    StateUnavailable,
    StateValidationError,
)
from .schema import StateSchema
from .validation import validate_item, validate_key

_TIMEOUT_S = 5
_READ_RETRIES = 2
_READ_OPS = {"get", "query"}


class _WireError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _map_wire(e: _WireError) -> Exception:
    if e.status == 409 or e.code == "conditional":
        return ConditionalCheckFailed(e.message)
    if e.status == 429 or e.code == "throttled":
        return StateThrottled(e.message)
    if e.status == 400 or e.code == "validation":
        return StateValidationError(e.message)
    if e.status == 401 or e.code == "unauthorized":
        # Token rejected: a config/clock problem, DEFINITELY not applied — do not
        # dress it up as "outcome unknown" like a 5xx.
        return StateValidationError(f"state broker rejected the token: {e.message}")
    return StateUnavailable(e.message)


def _map_and_record(e: _WireError) -> Exception:
    """_map_wire, plus recording runner-visible outages (errors.py) so
    artifact_runner can surface them even when this exception never reaches
    Lambda as a FunctionError — see errors._RUNNER_VISIBLE_ERRORS."""
    exc = _map_wire(e)
    errors._record(exc)
    return exc


class HTTPDriver:
    def __init__(self, url: str, token: str, schema: StateSchema):
        self._url = url
        self._token = token
        self.schema = schema
        self._pk = schema.pk.name
        self._sk = schema.sk.name if schema.sk else None
        self._ttl = schema.ttl_attribute

    # --- transport (the only network point; monkeypatched in tests) ---
    def _post(self, op: str, payload: dict) -> dict:
        body = json.dumps({"op": op, **payload}).encode("utf-8")
        req = urllib.request.Request(
            self._url, data=body, method="POST",
            headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                data = json.loads(e.read().decode("utf-8"))
            except Exception:
                data = {}
            raise _WireError(e.code, data.get("error", ""), data.get("message", str(e)))
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise _WireError(503, "unavailable", str(e))

    def _call(self, op: str, payload: dict) -> dict:
        attempts = _READ_RETRIES + 1 if op in _READ_OPS else 1
        last: _WireError | None = None
        for _ in range(attempts):
            try:
                return self._post(op, payload)
            except _WireError as e:
                last = e
                # Only retry reads on transient (5xx/unavailable); never mutations.
                if op in _READ_OPS and (e.status >= 500 or e.code == "unavailable"):
                    continue
                raise _map_and_record(e)
        raise _map_and_record(last)  # type: ignore[arg-type]

    # --- Driver protocol ---
    def get(self, pk: str, sk: str | None, *, consistent: bool) -> dict | None:
        validate_key(pk, sk, self.schema)
        return self._call("get", {"pk": pk, "sk": sk, "consistent": consistent}).get("item")

    def put(self, item: dict, *, if_not_exists: bool, if_version: int | None) -> None:
        validate_item(item, self.schema)
        pk = item[self._pk]
        sk = item.get(self._sk) if self._sk else None
        cond: dict | None = None
        if if_not_exists:
            cond = {"if_not_exists": True}
        elif if_version is not None:
            cond = {"if_version": if_version}
        ttl = item.get(self._ttl) if self._ttl else None
        self._call("put", {"pk": pk, "sk": sk, "item": item, "ttl": ttl, "cond": cond})

    def delete(self, pk: str, sk: str | None, *, if_version: int | None) -> None:
        validate_key(pk, sk, self.schema)
        cond = {"if_version": if_version} if if_version is not None else None
        self._call("delete", {"pk": pk, "sk": sk, "cond": cond})

    def query(self, pk, *, sk_prefix, filters, consistent, limit) -> list[dict]:
        return self._call("query", {
            "pk": pk, "sk_prefix": sk_prefix, "filters": filters,
            "consistent": consistent, "limit": limit,
        }).get("items", [])

    def increment(self, pk, sk, *, field, by) -> int | float:
        validate_key(pk, sk, self.schema)
        return self._call("increment", {"pk": pk, "sk": sk, "field": field, "by": by})["value"]

    def update(self, pk, sk, *, set_fields, add_fields, if_version) -> dict:
        validate_key(pk, sk, self.schema)
        cond = {"if_version": if_version} if if_version is not None else None
        return self._call("update", {
            "pk": pk, "sk": sk, "set": set_fields or {}, "add": add_fields or {}, "cond": cond,
        })["item"]
