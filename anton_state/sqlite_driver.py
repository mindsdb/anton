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

from .base import _VERSION_ATTR
from .errors import ConditionalCheckFailed, StateValidationError
from .schema import StateSchema
from .validation import validate_item, validate_key


class SQLiteDriver:
    def __init__(self, path: str, schema: StateSchema):
        self.path = path
        self.schema = schema
        self._pk = schema.pk.name
        self._sk = schema.sk.name if schema.sk else None
        self._ttl = schema.ttl_attribute
        self._gsi = {g.name: g for g in schema.gsis}
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
        body = json.dumps(item, separators=(",", ":"), ensure_ascii=False)
        expires = self._expires_of(item)
        con = self._connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            existing = con.execute(
                "SELECT body FROM items WHERE pk=? AND sk=?", (pk, sk)
            ).fetchone()
            if if_not_exists and existing is not None:
                raise ConditionalCheckFailed(f"item already exists: {pk}/{sk}")
            if if_version is not None:
                cur = json.loads(existing["body"]) if existing else None
                if cur is None or cur.get(_VERSION_ATTR) != if_version:
                    raise ConditionalCheckFailed(
                        f"version mismatch on {pk}/{sk}: expected {if_version}"
                    )
            con.execute(
                "INSERT INTO items (pk, sk, body, expires_at) VALUES (?,?,?,?) "
                "ON CONFLICT(pk, sk) DO UPDATE SET body=excluded.body, expires_at=excluded.expires_at",
                (pk, sk, body, expires),
            )
            con.commit()
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
        index: str | None,
        filters: dict[str, Any] | None,
        consistent: bool,
        limit: int | None,
    ) -> list[dict]:
        now = time.time()
        where = ["(expires_at IS NULL OR expires_at >= ?)"]
        params: list[Any] = [now]

        if index is not None:
            if index not in self._gsi:
                raise StateValidationError(
                    f"unknown index '{index}' (declared: {sorted(self._gsi)})"
                )
            gsi = self._gsi[index]
            where.append("json_extract(body, '$.' || ?) = ?")
            params += [gsi.pk.name, pk]
            if sk_prefix is not None and gsi.sk is not None:
                where.append("json_extract(body, '$.' || ?) LIKE ? || '%'")
                params += [gsi.sk.name, sk_prefix]
        else:
            where.append("pk = ?")
            params.append(pk)
            if sk_prefix is not None:
                where.append("sk LIKE ? || '%'")
                params.append(sk_prefix)

        if filters:
            for k, v in filters.items():
                where.append("json_extract(body, '$.' || ?) = ?")
                params += [k, v]

        sql = "SELECT body FROM items WHERE " + " AND ".join(where) + " ORDER BY pk, sk"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        with self._connect() as con:
            rows = con.execute(sql, params).fetchall()
        return [json.loads(r["body"]) for r in rows]

    def sweep_expired(self) -> int:
        """Lazy physical cleanup of expired items (not called on the hot path)."""
        now = time.time()
        with self._connect() as con:
            cur = con.execute(
                "DELETE FROM items WHERE expires_at IS NOT NULL AND expires_at < ?", (now,)
            )
            return cur.rowcount
