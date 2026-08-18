"""Local driver: SQLite, zero-deps, WAL, lazy TTL, conditional writes.

Stores the whole item as JSON in the `body` column; key/index/TTL values are
extracted via json_extract for queries. Physical deletion of expired items is
lazy (a separate sweep); reads filter them out, and conditional writes still
see the "zombie" — so behavior matches DynamoDB (spec §2).
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from .base import DEFAULT_QUERY_LIMIT, MAX_QUERY_LIMIT, _VERSION_ATTR
from .errors import ConditionalCheckFailed, StateThrottled
from .schema import StateSchema
from .validation import check_number, check_size, check_value, validate_item, validate_key


class SQLiteDriver:
    def __init__(self, path: str, schema: StateSchema):
        self.path = path
        self.schema = schema
        self._pk = schema.pk.name
        self._sk = schema.sk.name if schema.sk else None
        self._ttl = schema.ttl_attribute
        self._init_db()
        # Lazy cleanup of accumulated "zombies" on startup (dev server) —
        # otherwise expired items would live forever locally.
        self.sweep_expired()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=5000")
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS items ("
                " pk TEXT NOT NULL, sk TEXT NOT NULL DEFAULT '',"
                " body TEXT NOT NULL, expires_at REAL,"
                " PRIMARY KEY (pk, sk))"
            )

    # --- helpers ---
    def _sk_value(self, sk: str | None) -> str:
        return "" if self._sk is None else (sk or "")

    def _expires_of(self, item: dict) -> float | None:
        if self._ttl and self._ttl in item:
            return float(item[self._ttl])
        return None

    def _row_to_item(self, row: sqlite3.Row) -> dict:
        return json.loads(row["body"])

    # --- Driver protocol ---
    def get(self, pk: str, sk: str | None, *, consistent: bool) -> dict | None:
        validate_key(pk, sk, self.schema)
        now = time.time()
        with self._connect() as con:
            row = con.execute(
                "SELECT body, expires_at FROM items WHERE pk=? AND sk=?",
                (pk, self._sk_value(sk)),
            ).fetchone()
        if row is None:
            return None
        if row["expires_at"] is not None and row["expires_at"] < now:
            return None
        return json.loads(row["body"])

    def put(self, item: dict, *, if_not_exists: bool, if_version: int | None) -> None:
        validate_item(item, self.schema)
        pk = item[self._pk]
        sk = self._sk_value(item.get(self._sk) if self._sk else None)
        con = self._connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            existing = con.execute(
                "SELECT body FROM items WHERE pk=? AND sk=?", (pk, sk)
            ).fetchone()
            cur = json.loads(existing["body"]) if existing else None
            if if_not_exists and existing is not None:
                raise ConditionalCheckFailed(f"item already exists: {pk}/{sk}")
            if if_version is not None:
                if cur is None or cur.get(_VERSION_ATTR) != if_version:
                    raise ConditionalCheckFailed(
                        f"version mismatch on {pk}/{sk}: expected {if_version}"
                    )
            # Server-managed version: never trust a client-supplied _v. Monotonic
            # to prevent ABA lost-update: an optimistic-lock write -> if_version+1;
            # any other put -> epoch-millis (practically always greater than any
            # prior _v, so a stale if_version can't match a re-put generation).
            # Matches the broker (plan #2).
            stored = {k: v for k, v in item.items() if k != _VERSION_ATTR}
            stored[_VERSION_ATTR] = (if_version + 1) if if_version is not None else int(time.time() * 1000)
            body = json.dumps(stored, separators=(",", ":"), ensure_ascii=False)
            expires = self._expires_of(stored)
            con.execute(
                "INSERT INTO items (pk, sk, body, expires_at) VALUES (?,?,?,?) "
                "ON CONFLICT(pk, sk) DO UPDATE SET body=excluded.body, expires_at=excluded.expires_at",
                (pk, sk, body, expires),
            )
            con.commit()
        except sqlite3.OperationalError as e:
            con.rollback()
            raise StateThrottled(str(e)) from e
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def delete(self, pk: str, sk: str | None, *, if_version: int | None) -> None:
        validate_key(pk, sk, self.schema)
        skv = self._sk_value(sk)
        con = self._connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            if if_version is not None:
                existing = con.execute(
                    "SELECT body FROM items WHERE pk=? AND sk=?", (pk, skv)
                ).fetchone()
                cur = json.loads(existing["body"]) if existing else None
                if cur is None or cur.get(_VERSION_ATTR) != if_version:
                    raise ConditionalCheckFailed(
                        f"version mismatch on {pk}/{skv}: expected {if_version}"
                    )
            con.execute("DELETE FROM items WHERE pk=? AND sk=?", (pk, skv))
            con.commit()
        except sqlite3.OperationalError as e:
            con.rollback()
            raise StateThrottled(str(e)) from e
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def query(
        self,
        pk: str,
        *,
        sk_prefix: str | None,
        filters: dict[str, Any] | None,
        consistent: bool,
        limit: int | None,
    ) -> list[dict]:
        now = time.time()
        where = ["(expires_at IS NULL OR expires_at >= ?)", "pk = ?"]
        params: list[Any] = [now, pk]
        if sk_prefix is not None:
            # Literal prefix match (NOT `LIKE`): '%' and '_' in the prefix are
            # LIKE wildcards — an underscore in a collection name like
            # "user_settings#" would over-match — while DynamoDB begins_with is
            # literal. substr keeps local and cloud identical.
            where.append("substr(sk, 1, length(?)) = ?")
            params += [sk_prefix, sk_prefix]

        if filters:
            for k, v in filters.items():
                where.append("json_extract(body, '$.' || ?) = ?")
                params += [k, v]

        # Bounded exactly like the cloud broker: an absent limit means "the
        # default cap", never "every row". Diverging here is the local-vs-
        # published mismatch ENG-704 exists to avoid — a collection over the cap
        # would otherwise return everything locally and silently truncate in prod.
        sql = "SELECT body FROM items WHERE " + " AND ".join(where) + " ORDER BY pk, sk"
        sql += " LIMIT ?"
        params.append(DEFAULT_QUERY_LIMIT if limit is None else min(limit, MAX_QUERY_LIMIT))

        with self._connect() as con:
            rows = con.execute(sql, params).fetchall()
        return [json.loads(r["body"]) for r in rows]

    def _rmw(self, pk, sk, mutate):
        """Read-modify-write one item atomically; `mutate(item)->item` runs inside the txn.

        Partial mutation: unlike put, this does NOT run validate_item and does
        NOT synthesise the logical pk/sk attributes — mirroring the schema-agnostic
        broker (plan #2), where increment/update on an absent item create a minimal
        item (mutated field + _v) without the logical key attributes. Callers
        (increment/update) still type-check their inputs up front, and the
        mutated item is size-capped here, same as put.

        An EXPIRED row reads as absent here, matching get/query (physical
        deletion is lazy, so the "zombie" is still on disk). Without that, an
        expired counter would resume from a value the reads pretend is gone —
        jumping from "missing" straight to old_value + 1. The cloud broker
        enforces the same rule with a TTL condition on update_item.
        """
        skv = self._sk_value(sk)
        con = self._connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT body, expires_at FROM items WHERE pk=? AND sk=?", (pk, skv)
            ).fetchone()
            expired = (row is not None and row["expires_at"] is not None
                       and row["expires_at"] < time.time())
            item = json.loads(row["body"]) if (row and not expired) else {}
            item = mutate(item)
            check_size(item)
            item[_VERSION_ATTR] = item.get(_VERSION_ATTR, 0) + 1
            body = json.dumps(item, separators=(",", ":"), ensure_ascii=False)
            con.execute(
                "INSERT INTO items (pk, sk, body, expires_at) VALUES (?,?,?,?) "
                "ON CONFLICT(pk, sk) DO UPDATE SET body=excluded.body, expires_at=excluded.expires_at",
                (pk, skv, body, self._expires_of(item)),
            )
            con.commit()
            return item
        except sqlite3.OperationalError as e:
            con.rollback()
            raise StateThrottled(str(e)) from e
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def increment(self, pk: str, sk: str | None, *, field: str, by: int | float) -> int | float:
        validate_key(pk, sk, self.schema)
        check_number(by, "increment 'by'")

        def mut(item):
            item[field] = item.get(field, 0) + by
            return item

        return self._rmw(pk, sk, mut)[field]

    def update(self, pk, sk, *, set_fields, add_fields, if_version):
        validate_key(pk, sk, self.schema)
        for v in (set_fields or {}).values():
            check_value(v)
        for k, v in (add_fields or {}).items():
            check_number(v, f"add_fields[{k!r}]")

        def mut(item):
            if if_version is not None and item.get(_VERSION_ATTR, 0) != if_version:
                raise ConditionalCheckFailed(f"version mismatch on {pk}/{sk}: expected {if_version}")
            for k, v in (set_fields or {}).items():
                item[k] = v
            for k, v in (add_fields or {}).items():
                item[k] = item.get(k, 0) + v
            return item

        return self._rmw(pk, sk, mut)

    def sweep_expired(self) -> int:
        """Lazy physical cleanup of expired items (not called on the hot path)."""
        now = time.time()
        with self._connect() as con:
            cur = con.execute(
                "DELETE FROM items WHERE expires_at IS NOT NULL AND expires_at < ?", (now,)
            )
            return cur.rowcount
