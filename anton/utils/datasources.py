from __future__ import annotations

import os
import re
import shutil
import yaml
from pathlib import Path
from typing import TYPE_CHECKING

from anton.core.datasources.data_vault import DataVault, LocalDataVault, _slug_env_prefix
from anton.core.datasources.datasource_registry import DatasourceRegistry, _YAML_BLOCK_RE

if TYPE_CHECKING:
    from anton.core.datasources.datasource_registry import DatasourceEngine, DatasourceField

# DS_* var names whose values are known to be secret (passwords, tokens, keys).
# Populated at startup and after each successful connect.
_DS_SECRET_VARS: set[str] = set()

# DS_* var names for **ALL** fields of registered engines.
_DS_KNOWN_VARS: set[str] = set()

# Provider credential env vars injected into the scratchpad execution env for
# the code to USE. Their values must never reach the LLM (ENG-463): a model can
# only emit a secret it can see, and the agent was caught writing a raw `mdb_`
# key into generated code. Scrubbed by exact value (labeled) from anything that
# re-enters model context.
_PROVIDER_SECRET_VARS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "ANTON_OPENAI_API_KEY",
    "ANTON_ANTHROPIC_API_KEY",
    "ANTON_GEMINI_API_KEY",
    "ANTON_MINDS_API_KEY",
)

# Well-known provider key formats — scrubbed by shape so a key is redacted even
# when it didn't come from one of our env vars (e.g. the model already emitted
# it, or it arrived via an unexpected path). Length floors keep this from
# matching short incidental `sk-`/`mdb_` strings.
_SECRET_KEY_PATTERN = re.compile(
    r"mdb_[A-Za-z0-9._-]{10,}"   # MindsHub
    r"|sk-[A-Za-z0-9_-]{20,}"    # OpenAI / Anthropic (sk-, sk-proj-, sk-ant-)
    r"|AIza[A-Za-z0-9_-]{30,}"   # Google / Gemini
)


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
    """Remove secret values from scratchpad/tool output before it reaches the LLM.

    Redacts, in order:
      * DS_* values registered as secret via register_secret_vars (driven by
        DatasourceField.secret=true). Non-secret fields of known engines
        (DS_HOST, DS_PORT, DS_BASE_URL, …) stay readable so the LLM can reason
        about connection errors.
      * Unknown DS_* vars (custom engines not yet in the registry) — any long
        value, conservatively.
      * Provider credentials (ENG-463): the values of _PROVIDER_SECRET_VARS,
        then anything matching a well-known key shape (_SECRET_KEY_PATTERN), so
        a raw `mdb_`/`sk-`/`AIza` key can't reach model context via a tool
        result, traceback, or settings echo.
    """
    for key in _DS_SECRET_VARS:
        value = os.environ.get(key, "")
        if not value:
            continue
        text = re.sub(r'(?<!\w)' + re.escape(value) + r'(?!\w)', f'[{key}]', text)
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
    # Provider keys: exact-value first (informative label), then by shape to
    # catch any that didn't originate from one of our env vars.
    for key in _PROVIDER_SECRET_VARS:
        value = os.environ.get(key, "")
        if not value or len(value) <= 8:
            continue
        text = re.sub(r'(?<!\w)' + re.escape(value) + r'(?!\w)', f'[{key}]', text)
    text = _SECRET_KEY_PATTERN.sub("[REDACTED_API_KEY]", text)
    return text


def build_datasource_context(vault: DataVault, active_only: str | None = None) -> str:
    """Build a system-prompt section listing available DS_* env vars by name.

    Shows the LLM what data sources are connected and which environment
    variable names to use — without exposing any credential values.

    If active_only is set, only the matching slug is included.
    """
    try:
        vault = vault or LocalDataVault()
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
        "Never read the data vault files directly.\n"
        "If you see `[DS_<NAME>]` patterns in scratchpad output, those are "
        "scrub-markers where a secret value was redacted before returning "
        "text to you — the actual value IS injected in the env var. Reference "
        "it by name; never treat the bracket form as a literal credential "
        "or pass it back as a value to any tool.\n"
    )
    for c in conns:
        slug = f"{c['engine']}-{c['name']}"
        if active_only and slug != active_only:
            continue
        # read_record gives fields + secure_keys in one call; fall back to load
        # for any vault backend that doesn't implement it.
        if hasattr(vault, "read_record"):
            record = vault.read_record(c["engine"], c["name"]) or {}
            fields = record.get("fields", {}) or {}
            secure_keys = record.get("secure_keys")
        else:
            fields = vault.load(c["engine"], c["name"]) or {}
            secure_keys = None
        prefix = _slug_env_prefix(c["engine"], c["name"])
        # Skip `_`-prefixed bookkeeping (`_connector_id`, `_method`, `_label`) —
        # they're not credential env vars the agent should reference.
        var_names = ", ".join(
            f"{prefix}__{k.upper()}" for k in fields if not k.startswith("_")
        )
        # Prefer a user/agent-assigned label ("Support"); otherwise the derived
        # non-secret identity (email / host).
        identity = str(fields.get("_label", "")).strip() or _connection_identity(
            fields, secure_keys
        )
        head = f"`{slug}` ({c['engine']})"
        if identity:
            head += f" — {identity}"
        lines.append(f"- {head} → {var_names}")
    return "\n".join(lines)


def _connection_identity(fields: dict, secure_keys: list | None = None) -> str | None:
    """A short, non-secret label so the LLM can tell connections apart — the
    account email, or the database host (+ name).

    Scoped to these inherently non-secret fields on purpose: it is NOT a dump of
    every field (which would surface opaque ``client_id`` / config like
    ``ssl_mode``) and never a secret. Lets the agent pick the right account
    ("send from my support email") even when the connection slug is an old random
    one. Any field a record explicitly marks secret via ``secure_keys`` is
    skipped, defensively, even though email/host are not normally secret.
    """
    secure = set(secure_keys or [])
    # `email` is collected by credential forms; `account_email` is stored by the
    # OAuth flows (from userinfo). Either identifies the account.
    for key in ("email", "account_email"):
        val = str(fields.get(key, "")).strip()
        if val and key not in secure:
            return val
    host = str(fields.get("host", "")).strip()
    if host and "host" not in secure:
        database = str(fields.get("database", "")).strip()
        if database and "database" not in secure:
            return f"{host}/{database}"
        return host
    return None


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


def find_matching_connection(
    vault: DataVault,
    engine_def: "DatasourceEngine",
    credentials: dict[str, str],
) -> str | None:
    """Return the name of an existing connection with matching identity fields.

    Uses the engine's ``name_from`` declaration as the identity signature.
    If those fields exist in both the incoming credentials and a stored
    connection of the same engine (and all values match), the stored name
    is returned so the caller can update in place instead of creating a
    duplicate. Returns None when no match is found, or when the engine has
    no ``name_from`` (custom/ad-hoc engines).
    """
    name_from = engine_def.name_from
    if not name_from:
        return None
    if isinstance(name_from, str):
        name_from = [name_from]
    incoming_sig = tuple(credentials.get(f, "") for f in name_from)
    if not any(incoming_sig):
        return None
    for conn in vault.list_connections():
        if conn["engine"] != engine_def.engine:
            continue
        existing_fields = vault.load(conn["engine"], conn["name"]) or {}
        existing_sig = tuple(existing_fields.get(f, "") for f in name_from)
        if existing_sig == incoming_sig:
            return conn["name"]
    return None


def save_connection(
    vault: DataVault,
    engine_def: "DatasourceEngine",
    name: str,
    credentials: dict[str, str],
    *,
    secure_keys: list[str] | None = None,
) -> str:
    """Persist credentials and refresh DS_* env vars. Returns the connection slug.

    Shared save path used by both /connect and connect_new_datasource tool.

    `secure_keys` is forwarded to the vault so future reads can
    classify fields without falling back to the name-matching
    heuristic. Optional for backward compatibility — callers that
    don't have the connector spec handy (the chat-side tool path,
    for example) can still call this without supplying the list.
    """
    vault.save(engine_def.engine, name, credentials, secure_keys=secure_keys)
    restore_namespaced_env(vault)
    register_secret_vars(engine_def, engine=engine_def.engine, name=name)
    return f"{engine_def.engine}-{name}"


def persist_custom_engine(
    registry: DatasourceRegistry,
    display_name: str,
    fields: list["DatasourceField"],
    test_snippet: str = "",
    pip: str = "",
) -> "DatasourceEngine | None":
    """Append a YAML block for a custom engine to ``~/.anton/datasources.md``.

    Reloads the registry and returns the parsed DatasourceEngine on success.
    Returns None if the newly-written block fails parse validation (caller
    may still fall back to an in-memory engine).
    """
    from anton.core.datasources.datasource_registry import DatasourceEngine

    slug = re.sub(r"[^\w]", "_", display_name.lower()).strip("_")
    field_lines = "\n".join(
        f"  - {{ name: {f.name}, required: {str(f.required).lower()}, "
        f"secret: {str(f.secret).lower()}, "
        f"description: \"{f.description}\" }}"
        for f in fields
    )
    test_snippet_yaml = ""
    if test_snippet:
        indented = "\n".join(f"  {line}" for line in test_snippet.splitlines())
        test_snippet_yaml = f"test_snippet: |\n{indented}\n"

    yaml_block = (
        f"\n---\n\n## {display_name}\n"
        "```yaml\n"
        f"engine: {slug}\n"
        f"display_name: {display_name}\n"
        + (f"pip: {pip}\n" if pip else "")
        + f"fields:\n{field_lines}\n"
        + test_snippet_yaml
        + "```\n"
    )
    user_ds_path = Path("~/.anton/datasources.md").expanduser()
    tmp_path = user_ds_path.with_suffix(".tmp")

    existing = (
        user_ds_path.read_text(encoding="utf-8")
        if user_ds_path.is_file()
        else ""
    )
    existing = remove_engine_block(existing, slug)

    user_ds_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(existing + yaml_block, encoding="utf-8")

    parsed = registry.validate_file(tmp_path)
    if slug not in parsed:
        tmp_path.unlink(missing_ok=True)
        return None

    shutil.move(str(tmp_path), str(user_ds_path))
    registry.reload()
    engine_def = registry.get(slug)
    if engine_def is None:
        engine_def = DatasourceEngine(
            engine=slug,
            display_name=display_name,
            pip=pip,
            fields=list(fields),
            test_snippet=test_snippet,
            custom=True,
        )
    return engine_def


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
