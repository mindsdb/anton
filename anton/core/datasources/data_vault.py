from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


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


def _sweep_ds_env_vars() -> None:
    """Remove every DS_* variable from os.environ.

    Shared by every clear_ds_env() implementation (module-level and both
    vault classes) so the sweep logic lives in exactly one place.
    """
    for key in [k for k in os.environ if k.startswith("DS_")]:
        del os.environ[key]


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

    def env_for(self, engine: str, name: str, *, flat: bool = False) -> dict[str, str] | None:
        """Build the DS_* env mapping for a connection, without touching os.environ."""
        ...

    def inject_env(self, engine: str, name: str, *, flat: bool = False) -> list[str] | None:
        """Load credentials and set DS_* environment variables."""
        ...

    def clear_ds_env(self) -> None:
        """Remove all DS_* variables from os.environ."""
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
        tmp.replace(path)
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

        `_`-prefixed fields (bookkeeping: `_user_label`, `_label`,
        `_connector_id`, `_method`, `_picked_files`, ...) are never injected —
        they aren't credentials and nothing reads them from an env var.

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
                if key.startswith("_"):
                    continue
                env[f"DS_{key.upper()}"] = value
        else:
            prefix = _slug_env_prefix(engine, name)
            for key, value in fields.items():
                if key.startswith("_"):
                    continue
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
        _sweep_ds_env_vars()


#: Deployment-side override for auth's base URL (never taken from the wire
#: request — same trust posture as ANTON_CLOUD_WORKSPACE_PATH etc. in
#: cloud_turn/session.py). Lets a PR/staging pod point at a non-prod auth.
ANTON_CLOUD_AUTH_BASE_URL_ENV = "ANTON_CLOUD_AUTH_BASE_URL"
_DEFAULT_AUTH_BASE_URL = "https://auth.mindshub.ai"

#: Non-secret fields the turn-key endpoint returns alongside access_token,
#: same field names LocalDataVault already persists for these engines
#: (cowork-server/cowork/services/connectors/oauth/google.py) — kept
#: identical so scratchpad code sees the same DS_* var names on cloud and
#: desktop.
_TURNKEY_RESPONSE_FIELDS = ("access_token", "account_email", "token_type", "scope", "expires_at")


class TurnKeyDataVault:
    """Read-only DataVault backed by a live call to auth's turn-key token
    endpoint (POST /v1/oauth/{engine}/token), for cloud turns only.

    Connections are exactly what cowork-server resolved at enqueue time
    (the turn's `oauth["connections"]` block) — this class never discovers
    or persists connections on its own; `save`/`delete` are no-ops/errors,
    matching the fact that a cloud turn never edits connections mid-turn.
    """

    def __init__(self, oauth: dict[str, Any]) -> None:
        self._turn_key = str(oauth.get("turn_key") or "")
        self._connections: list[dict[str, str]] = [
            {"engine": str(c["engine"]), "name": str(c["name"])}
            for c in (oauth.get("connections") or [])
            if isinstance(c, dict) and c.get("engine") and c.get("name")
        ]
        self._connection_keys = frozenset((c["engine"], c["name"]) for c in self._connections)
        # ENG-2128: this used to also accept a keyword-only `base_url`
        # override, but the one real call site (cloud_turn/session.py)
        # constructs positionally and never bound it, so a value
        # cowork-server put in the oauth block's own `base_url` field was
        # dead on the wire - the module comment on
        # ANTON_CLOUD_AUTH_BASE_URL_ENV above already states the intended
        # design ("never taken from the wire request"), which the removed
        # kwarg contradicted by existing at all. Removed rather than wired
        # up: the env var is already the correct, working, per-environment
        # source, and nothing needs cowork-server to steer this per-request.
        self._base_url = (
            os.environ.get(ANTON_CLOUD_AUTH_BASE_URL_ENV) or _DEFAULT_AUTH_BASE_URL
        ).rstrip("/")
        # Per-turn cache: the loop in restore_namespaced_env() calls
        # inject_env() once per connection already, but read_record()/load()
        # may also be called for the same connection within the same turn
        # (e.g. system-prompt building) — one token fetch per connection,
        # not one per call.
        self._cache: dict[tuple[str, str], dict[str, str] | None] = {}

    def list_connections(self) -> list[dict[str, str]]:
        """Return [{engine, name, created_at}] — created_at is unknown here,
        so it's always empty; nothing reads it for OAuth connections today."""
        return [{**c, "created_at": ""} for c in self._connections]

    def load(self, engine: str, name: str) -> dict[str, str] | None:
        return self._fetch(engine, name)

    def read_record(self, engine: str, name: str) -> dict[str, Any] | None:
        fields = self._fetch(engine, name)
        if fields is None:
            return None
        return {
            "engine": engine,
            "name": name,
            "created_at": "",
            "updated_at": "",
            "fields": fields,
            "secure_keys": ["access_token"],
        }

    def save(
        self,
        engine: str,
        name: str,
        credentials: dict[str, str],
        *,
        secure_keys: list[str] | None = None,
    ) -> object:
        raise NotImplementedError("TurnKeyDataVault is read-only for the life of a turn")

    def delete(self, engine: str, name: str) -> bool:
        return False

    def env_for(self, engine: str, name: str, *, flat: bool = False) -> dict[str, str] | None:
        """Same contract as LocalDataVault.env_for() — see its docstring."""
        fields = self._fetch(engine, name)
        if fields is None:
            return None
        env: dict[str, str] = {}
        if flat:
            for key, value in fields.items():
                env[f"DS_{key.upper()}"] = value
        else:
            prefix = _slug_env_prefix(engine, name)
            for key, value in fields.items():
                env[f"{prefix}__{key.upper()}"] = value
        return env

    def inject_env(self, engine: str, name: str, *, flat: bool = False) -> list[str] | None:
        env = self.env_for(engine, name, flat=flat)
        if env is None:
            return None
        os.environ.update(env)
        return list(env)

    def clear_ds_env(self) -> None:
        """Remove all DS_* variables from os.environ and drop this vault's
        per-connection token cache, so a later fetch can't return stale
        pre-clear credentials."""
        _sweep_ds_env_vars()
        self._cache.clear()

    def _fetch(self, engine: str, name: str) -> dict[str, str] | None:
        cache_key = (engine, name)
        if cache_key in self._cache:
            return self._cache[cache_key]
        fields = self._fetch_uncached(engine, name)
        self._cache[cache_key] = fields
        return fields

    def _fetch_uncached(self, engine: str, name: str) -> dict[str, str] | None:
        if not self._turn_key:
            logger.warning("TurnKeyDataVault: no turn key available for %s/%s", engine, name)
            return None
        # Defense in depth: don't rely solely on auth's own org-scoped query —
        # never fetch a connection cowork-server didn't list for this turn.
        if (engine, name) not in self._connection_keys:
            logger.warning("TurnKeyDataVault: %s/%s not in this turn's connection list; refusing", engine, name)
            return None
        from anton.minds_client import minds_request  # local: avoid a module-load-time dep from core.datasources

        url = f"{self._base_url}/v1/oauth/{engine}/token"
        # `name` disambiguates when an org has more than one connection for
        # this engine — the endpoint auto-resolves a lone connection without
        # it, but 400s on an ambiguous one. Sent as JSON body, matching
        # auth's request.data.get("name") read.
        payload = json.dumps({"name": name}).encode()
        try:
            raw = minds_request(url, self._turn_key, method="POST", payload=payload, timeout=15)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                # Clean, expected outcome — the connector needs reconnecting.
                # Never a raw error or a hang: this just yields "no creds",
                # same as a connection that was never made.
                logger.info("connector %s/%s needs reconnecting (auth returned HTTP %s)", engine, name, e.code)
            else:
                logger.warning("turn-key token fetch failed for %s/%s: HTTP %s", engine, name, e.code)
            return None
        except Exception:
            logger.warning("turn-key token fetch failed for %s/%s", engine, name, exc_info=True)
            return None

        try:
            data = json.loads(raw.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("turn-key token fetch for %s/%s returned unparseable JSON", engine, name)
            return None
        if not isinstance(data, dict) or not data.get("access_token"):
            logger.warning("turn-key token fetch for %s/%s returned no access_token", engine, name)
            return None
        fields = {k: str(data[k]) for k in _TURNKEY_RESPONSE_FIELDS if data.get(k) is not None}
        # Every credential this vault serves is OAuth-backed by construction
        # (the turn-key endpoint only exists for OAuth connectors) — this is
        # the same signal LocalDataVault's own stored `auth_type` field gives
        # build_datasource_context() on desktop, so synthesize it rather than
        # depend on auth's response carrying it.
        fields["auth_type"] = "oauth"
        picked_files = data.get("_picked_files")
        if picked_files:
            # Auth doesn't send this yet (ENG follow-up), but the shape is
            # ready: same JSON-string-in-a-field convention _parse_picked_files
            # (anton/utils/datasources.py) reads from LocalDataVault.
            fields["_picked_files"] = picked_files if isinstance(picked_files, str) else json.dumps(picked_files)
        return fields
