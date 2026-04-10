from __future__ import annotations

import os
import re
import yaml
from typing import TYPE_CHECKING

from anton.core.datasources.data_vault import DataVault, _slug_env_prefix
from anton.core.datasources.datasource_registry import DatasourceRegistry, _YAML_BLOCK_RE

if TYPE_CHECKING:
    from anton.core.datasources.datasource_registry import DatasourceEngine

# DS_* var names whose values are known to be secret (passwords, tokens, keys).
# Populated at startup and after each successful connect.
_DS_SECRET_VARS: set[str] = set()

# DS_* var names for **ALL** fields of registered engines.
_DS_KNOWN_VARS: set[str] = set()


def _reset_registered_ds_vars() -> None:
    """Clear the DS_* var registries so they can be rebuilt from current vault state."""
    _DS_SECRET_VARS.clear()
    _DS_KNOWN_VARS.clear()


def parse_connection_slug(
    slug: str,
    known_engines: list[str],
    *,
    vault: DataVault | None = None,
) -> tuple[str, str] | None:
    """Split a connection slug into (engine, name) using longest-prefix matching.

    First tries each known registry engine longest-first so that 'sql-server-prod-db' is
    correctly parsed as engine='sql-server', name='prod-db' rather than
    engine='sql', name='server-prod-db'.

    If nothing matches and a vault is supplied, falls back to scanning vault
    connections for an exact slug match — handles custom/unregistered engines.

    Returns None if no match found or name part is empty.
    """
    for engine in sorted(known_engines, key=len, reverse=True):
        prefix = engine + "-"
        if slug.startswith(prefix) and len(slug) > len(prefix):
            return (engine, slug[len(prefix):])

    if vault is not None:
        for conn in vault.list_connections():
            if f"{conn['engine']}-{conn['name']}" == slug:
                return (conn["engine"], conn["name"])

    return None


def register_secret_vars(
    engine_def: "DatasourceEngine", *, engine: str = "", name: str = ""
) -> None:
    """Record which DS_* var names correspond to known/secret fields for engine_def.

    If engine and name are given, registers namespaced vars (DS_ENGINE_NAME__FIELD).
    Otherwise registers flat vars (DS_FIELD) — for temporary test_snippet execution.
    """
    all_fields = list(engine_def.fields)
    for am in engine_def.auth_methods or []:
        all_fields.extend(am.fields)
    for f in all_fields:
        if engine and name:
            prefix = _slug_env_prefix(engine, name)
            key = f"{prefix}__{f.name.upper()}"
        else:
            key = f"DS_{f.name.upper()}"
        _DS_KNOWN_VARS.add(key)
        if f.secret:
            _DS_SECRET_VARS.add(key)


def scrub_credentials(text: str) -> str:
    """Remove secret DS_* values from scratchpad output before it reaches the LLM.

    Only redacts vars registered as secret via _register_secret_vars (driven by
    DatasourceField.secret=true in datasources.md).  Non-secret fields of known
    engines (DS_HOST, DS_PORT, DS_BASE_URL, …) are left readable so the LLM can
    reason about connection errors.  For truly unknown DS_* vars (custom engines
    not yet in the registry) the fallback scrubs any long value — conservative
    but safe.
    """
    for key in _DS_SECRET_VARS:
        value = os.environ.get(key, "")
        if not value:
            continue
        text = text.replace(value, f"[{key}]")
    for key, value in os.environ.items():
        if not key.startswith("DS_") or key in _DS_KNOWN_VARS:
            continue
        # Length guard only for unknown DS_* vars (not registered secrets).
        # Unknown vars are matched heuristically — a short value like "on"
        # or "true" in a DS_ENABLE_X var should not be scrubbed.
        # Registered secret vars bypass this check entirely.
        if not value or len(value) <= 8:
            continue
        text = text.replace(value, f"[{key}]")
    return text


def build_datasource_context(vault: DataVault, active_only: str | None = None) -> str:
    """Build a system-prompt section listing available DS_* env vars by name.

    Shows the LLM what data sources are connected and which environment
    variable names to use — without exposing any credential values.

    If active_only is set, only the matching slug is included.
    """
    try:
        vault = vault or DataVault()
        conns = vault.list_connections()
    except Exception:
        return ""
    if not conns:
        return ""
    lines = ["\n\n## Connected Data Sources"]
    lines.append(
        "Credentials are pre-injected as namespaced DS_<ENGINE_NAME>__<FIELD> "
        "environment variables. Use them directly in scratchpad code "
        "(e.g. DS_POSTGRES_PROD_DB__HOST). "
        "Never read ~/.anton/data_vault/ files directly.\n"
    )
    for c in conns:
        slug = f"{c['engine']}-{c['name']}"
        if active_only and slug != active_only:
            continue
        fields = vault.load(c["engine"], c["name"]) or {}
        prefix = _slug_env_prefix(c["engine"], c["name"])
        var_names = ", ".join(f"{prefix}__{k.upper()}" for k in fields)
        lines.append(f"- `{slug}` ({c['engine']}) → {var_names}")
    return "\n".join(lines)


def restore_namespaced_env(vault: DataVault) -> None:
    """Clear all DS_* vars, then reinject every saved connection as namespaced."""
    _reset_registered_ds_vars()
    vault.clear_ds_env()
    dreg = DatasourceRegistry()
    for conn in vault.list_connections():
        vault.inject_env(conn["engine"], conn["name"])  # flat=False by default
        edef = dreg.get(conn["engine"])
        if edef is not None:
            register_secret_vars(edef, engine=conn["engine"], name=conn["name"])


def remove_engine_block(text: str, slug: str) -> str:
    """Return *text* with any YAML datasource block for *slug* removed."""
    cleaned = []
    prev = 0
    for m in _YAML_BLOCK_RE.finditer(text):
        try:
            data = yaml.safe_load(m.group(3))
            is_dup = isinstance(data, dict) and str(data.get("engine", "")) == slug
        except Exception:
            is_dup = False
        if is_dup:
            pre = text[prev : m.start()].rstrip()
            pre = re.sub(r"\n---\s*$", "", pre)
            cleaned.append(pre)
        else:
            cleaned.append(text[prev : m.end()])
        prev = m.end()
    cleaned.append(text[prev:])
    return "".join(cleaned)
