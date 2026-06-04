from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


# Sentinel used in modify-flow round-trips. The renderer fetches the
# stored record, gets this string in every secret-shaped slot, pre-
# fills the form with it, and on submit any field whose value is
# still this exact sentinel means "leave the existing vault value
# alone" — distinct from an empty string, which means "explicitly
# clear this field". Importers should reference the constant rather
# than re-typing the literal so the two ends stay in sync.
ANTON_VAULT_KEEP = "__anton_vault_keep__"


# Keys we treat as secret when the on-disk record predates the
# `secure_keys` schema. Conservative on purpose — over-masking a
# benign field is harmless (the modify form just asks the user to
# re-enter), under-masking leaks. Any field whose name (case-folded,
# either side of an underscore-cluster) contains one of these tokens
# is considered secret.
_SECRET_KEY_TOKENS = (
    "password",
    "passphrase",
    "secret",
    "token",
    "key",         # catches api_key, private_key, access_key, ssh_key…
    "credential",
    "auth",        # catches auth_token, basic_auth, …
)


def is_secret_key(field_name: str, secure_keys: list[str] | None = None) -> bool:
    """Return True when a stored field should be treated as a secret.

    When the record carries an explicit `secure_keys` list, that list
    is authoritative — exact matches only, no fuzzing. The vault
    record is the source of truth once it's been written under the
    new schema.

    Legacy records (no `secure_keys` on disk) fall back to a name-
    matching heuristic: case-insensitive substring match against
    `_SECRET_KEY_TOKENS`. This is the bridge until every record has
    been re-saved under the new schema.
    """
    if secure_keys is not None:
        return field_name in set(secure_keys)
    name_lc = (field_name or "").lower()
    return any(token in name_lc for token in _SECRET_KEY_TOKENS)


def _sanitize(value: str) -> str:
    """Strip characters unsafe for file names, keep alphanumeric, dash, underscore."""
    return re.sub(r"[^\w\-]", "_", value).strip("_")


def resolve_modify_merge(
    vault: "DataVault",
    engine: str,
    name: str,
    incoming: dict[str, str],
    *,
    spec_secret_keys: list[str] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Apply the modify-flow sentinel merge.

    Renderer pre-fills the form with values from a prior `read_record`
    call; secret slots come back as `ANTON_VAULT_KEEP`. On submit, any
    field whose value is *still* the sentinel means "keep the existing
    vault value" — distinct from an empty string, which means
    "explicitly clear".

    Returns:
      merged_credentials — `incoming` with sentinel slots resolved
        against the existing record. Sentinel entries with no prior
        value are dropped (defensive: prevents the literal sentinel
        from ever landing on disk).
      secure_keys — the union of (a) the prior record's `secure_keys`
        list, (b) the spec-marked secret fields supplied by the
        caller, and (c) the heuristic applied to the merged key set.
        Union-only — once a key is known-secret we never demote it.

    Pure no-op for create paths: there's no prior record so no
    sentinels can survive, and the secure-key set is computed from
    spec + heuristic alone. Callers can use this on every save
    without branching on create-vs-modify.
    """
    prior = vault.read_record(engine, name) if name else None
    prior_fields = (prior or {}).get("fields") or {}
    prior_secure = (prior or {}).get("secure_keys")

    merged: dict[str, str] = {}
    for key, value in incoming.items():
        if value == ANTON_VAULT_KEEP:
            if key in prior_fields:
                merged[key] = prior_fields[key]
            # If there's no prior value the sentinel is meaningless —
            # drop the field rather than persist the literal string.
            continue
        merged[key] = value

    heuristic_secret = {k for k in merged.keys() if is_secret_key(k, secure_keys=None)}
    secure_keys = sorted({
        *(prior_secure or []),
        *(spec_secret_keys or []),
        *heuristic_secret,
    })
    return merged, secure_keys


def _slug_env_prefix(engine: str, name: str) -> str:
    """Return the DS_ prefix for a namespaced connection env var.

    Examples:
      engine="postgres", name="prod_db"  → "DS_POSTGRES_PROD_DB"
      engine="hubspot",  name="main"     → "DS_HUBSPOT_MAIN"
      engine="postgres", name="prod-db.eu" → "DS_POSTGRES_PROD_DB_EU"
    """
    raw = f"{engine}-{name}"
    return "DS_" + re.sub(r"[^\w]", "_", raw).upper()


@runtime_checkable
class DataVault(Protocol):
    """Interface for credential storage backends.

    The local implementation (LocalDataVault) stores JSON files in
    ~/.anton/data_vault/. Cloud implementations can satisfy this protocol
    with any backend (database, secrets manager, etc.) scoped to a user
    or tenant.
    """

    def save(
        self,
        engine: str,
        name: str,
        credentials: dict[str, str],
        *,
        secure_keys: list[str] | None = None,
    ) -> object:
        """Persist credentials for engine/name. Returns an implementation-defined path/key.

        `secure_keys` is the authoritative list of field names the
        record should treat as secret. Optional for backward
        compatibility; absent records are classified by heuristic at
        read time (see `is_secret_key`).
        """
        ...

    def load(self, engine: str, name: str) -> dict[str, str] | None:
        """Return the fields dict for a connection, or None if not found."""
        ...

    def read_record(self, engine: str, name: str) -> dict[str, Any] | None:
        """Return the full on-disk record (engine/name/timestamps/fields/secure_keys)
        for a connection, or None if not found. Distinct from `load`,
        which intentionally returns just the credential fields.
        """
        ...

    def delete(self, engine: str, name: str) -> bool:
        """Remove a connection. Returns True if it existed."""
        ...

    def list_connections(self) -> list[dict[str, str]]:
        """Return [{engine, name, created_at}] for all stored connections."""
        ...

    def inject_env(self, engine: str, name: str, *, flat: bool = False) -> list[str] | None:
        """Load credentials and set DS_* environment variables."""
        ...

    def clear_ds_env(self) -> None:
        """Remove all DS_* variables from os.environ."""
        ...

    def next_connection_number(self, engine: str) -> int:
        """Return the next auto-increment number for an engine (1-based)."""
        ...


class LocalDataVault:
    """File-based credential store in ~/.anton/data_vault/."""

    def __init__(self, vault_dir: Path | None = None) -> None:
        self._dir = vault_dir or Path("~/.anton/data_vault").expanduser()

    def _path_for(self, engine: str, name: str) -> Path:
        return self._dir / f"{_sanitize(engine)}-{_sanitize(name)}"

    def _ensure_dir(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._dir.chmod(0o700)

    def save(
        self,
        engine: str,
        name: str,
        credentials: dict[str, str],
        *,
        secure_keys: list[str] | None = None,
    ) -> Path:
        """Write credentials as JSON atomically. Creates vault dir if needed.

        Forward-compatible: when an older record exists at the same
        path, `created_at` is preserved (this looks like an update,
        not a fresh record). `updated_at` is always stamped. When
        `secure_keys` is provided it's persisted on the record so
        future reads can classify fields without falling back to the
        name-matching heuristic.
        """
        self._ensure_dir()
        path = self._path_for(engine, name)
        now = datetime.now(timezone.utc).isoformat()
        # Preserve created_at across updates so the timestamp keeps
        # its original meaning. New records get now() for both.
        prior = self._read_raw(path)
        created_at = (prior.get("created_at") if prior else None) or now
        data: dict[str, Any] = {
            "engine": engine,
            "name": name,
            "created_at": created_at,
            "updated_at": now,
            "fields": credentials,
        }
        if secure_keys is not None:
            # Stable order so the on-disk JSON diffs cleanly across
            # updates that don't change the secret-set membership.
            data["secure_keys"] = sorted(set(secure_keys))
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.chmod(0o600)
        tmp.rename(path)
        return path

    def load(self, engine: str, name: str) -> dict[str, str] | None:
        """Return the fields dict for a connection, or None if not found."""
        path = self._path_for(engine, name)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("fields", {})
        except (json.JSONDecodeError, OSError):
            return None

    def _read_raw(self, path: Path) -> dict[str, Any] | None:
        """Internal helper — load the full JSON record (or None on miss/error)."""
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def read_record(self, engine: str, name: str) -> dict[str, Any] | None:
        """Return the full on-disk record for a connection, or None if not found.

        Shape:
            {
              "engine": str, "name": str,
              "created_at": str, "updated_at": str | None,
              "fields": dict[str, str],
              "secure_keys": list[str] | None,    # absent on legacy records
            }

        Callers that want classified-fields-with-sentinels should layer
        the modify-flow logic on top — this method intentionally
        returns the raw record so the server endpoint can apply the
        sentinel substitution in one place.
        """
        return self._read_raw(self._path_for(engine, name))

    def delete(self, engine: str, name: str) -> bool:
        """Remove a connection file. Returns True if it existed."""
        path = self._path_for(engine, name)
        if path.is_file():
            path.unlink()
            return True
        return False

    def list_connections(self) -> list[dict[str, str]]:
        """Return [{engine, name, created_at}] for all stored connections."""
        if not self._dir.is_dir():
            return []
        results: list[dict[str, str]] = []
        for path in sorted(self._dir.iterdir()):
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                results.append(
                    {
                        "engine": data.get("engine", ""),
                        "name": data.get("name", ""),
                        "created_at": data.get("created_at", ""),
                    }
                )
            except (json.JSONDecodeError, OSError):
                continue
        return results

    def env_for(self, engine: str, name: str, *, flat: bool = False) -> dict[str, str] | None:
        """Build the DS_* env mapping for a connection WITHOUT mutating os.environ.

        Default (flat=False): namespaced vars, e.g. DS_POSTGRES_PROD_DB__HOST.
        flat=True: legacy flat vars, e.g. DS_HOST — use only during
        single-connection test_snippet execution.

        Returns the {var: value} mapping, or None if connection not found.
        Use this when the env should reach only a specific subprocess (pass
        the result as an explicit `env`); use `inject_env` when the variables
        must be visible in the current process.
        """
        fields = self.load(engine, name)
        if fields is None:
            return None
        env: dict[str, str] = {}
        if flat:
            for key, value in fields.items():
                env[f"DS_{key.upper()}"] = value
        else:
            prefix = _slug_env_prefix(engine, name)
            for key, value in fields.items():
                env[f"{prefix}__{key.upper()}"] = value if isinstance(value, str) else str(value)
        return env

    def inject_env(self, engine: str, name: str, *, flat: bool = False) -> list[str] | None:
        """Load credentials and set DS_* environment variables.

        Default (flat=False): injects namespaced vars, e.g. DS_POSTGRES_PROD_DB__HOST.
        flat=True: injects legacy flat vars, e.g. DS_HOST — use only during
        single-connection test_snippet execution.

        Returns the list of env var names set, or None if connection not found.
        """
        env = self.env_for(engine, name, flat=flat)
        if env is None:
            return None
        os.environ.update(env)
        return list(env)

    def clear_ds_env(self) -> None:
        """Remove all DS_* variables from os.environ."""
        ds_keys = [k for k in os.environ if k.startswith("DS_")]
        for key in ds_keys:
            del os.environ[key]

    def next_connection_number(self, engine: str) -> int:
        """Return the next auto-increment number for an engine (1-based).

        Used when naming partial connections: postgresql-1, postgresql-2, etc.
        """
        prefix = _sanitize(engine) + "-"
        if not self._dir.is_dir():
            return 1
        existing = [
            p.name
            for p in self._dir.iterdir()
            if p.is_file() and p.name.startswith(prefix)
        ]
        max_n = 0
        for fname in existing:
            suffix = fname[len(prefix):]
            if suffix.isdigit():
                max_n = max(max_n, int(suffix))
        return max_n + 1
