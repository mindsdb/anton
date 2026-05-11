"""Settings — reads and writes Anton configuration.

Anton's current SDK reads configuration from ``ANTON_*`` environment names.
The desktop app keeps using friendly frontend field names, but this module is
the translation layer and the single place that handles legacy env fallbacks.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .cowork_state import update_state, load_state

router = APIRouter()
logger = logging.getLogger(__name__)

GLOBAL_ENV_PATH = Path.home() / ".anton" / ".env"

CANONICAL_KEY_FALLBACKS = {
    "ANTON_ANTHROPIC_API_KEY": ("ANTHROPIC_API_KEY",),
    "ANTON_OPENAI_API_KEY": ("OPENAI_API_KEY",),
}

PROVIDER_LABELS = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "openai-compatible": "OpenAI-compatible",
}

UI_DEFAULTS = {
    "greeting": "Let's knock something off your list",
    "tone": "balanced",
    "defaultModel": "",
    "autoPin": True,
    "showDots": True,
    "accentVariant": "aqua",
}


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            result[key.strip()] = val.strip().strip('"').strip("'")
    except Exception:
        pass
    return result


def _write_dotenv(path: Path, data: dict[str, str], delete_keys: tuple[str, ...] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    existing = _read_dotenv(path)
    for key in delete_keys:
        existing.pop(key, None)
    existing.update({k: v for k, v in data.items() if v})  # don't write empty strings
    lines = [f"{k}={v}" for k, v in existing.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _stage_string_env(
    patch_value: Optional[str],
    env_key: str,
    writes: dict[str, str],
    delete_keys: list[str],
) -> None:
    """Apply an optional text field to dotenv writes, treating empty as delete."""
    if patch_value is None:
        return
    value = patch_value.strip()
    if value:
        writes[env_key] = value
    else:
        delete_keys.append(env_key)


def _get_env(key: str, default: str = "", legacy_keys: tuple[str, ...] | None = None) -> str:
    """Read from env → ~/.anton/.env → legacy env → default."""
    val = os.environ.get(key)
    if val:
        return val
    dotenv = _read_dotenv(GLOBAL_ENV_PATH)
    val = dotenv.get(key)
    if val:
        return val

    for legacy_key in legacy_keys or CANONICAL_KEY_FALLBACKS.get(key, ()):
        val = os.environ.get(legacy_key)
        if val:
            return val
        val = dotenv.get(legacy_key)
        if val:
            return val
    return default


def _bool_env(key: str, default: bool) -> bool:
    return _get_env(key, str(default).lower()).strip().lower() in {"1", "true", "yes", "on"}


def _masked(key: str) -> str:
    return "***" if _get_env(key) else ""


def _migrate_legacy_keys() -> list[str]:
    """Copy legacy API keys into Anton's canonical names without deleting them."""
    dotenv = _read_dotenv(GLOBAL_ENV_PATH)
    writes: dict[str, str] = {}
    for canonical, legacy_keys in CANONICAL_KEY_FALLBACKS.items():
        if os.environ.get(canonical) or dotenv.get(canonical):
            continue
        for legacy_key in legacy_keys:
            legacy_value = os.environ.get(legacy_key) or dotenv.get(legacy_key)
            if legacy_value:
                writes[canonical] = legacy_value
                break

    if writes:
        _write_dotenv(GLOBAL_ENV_PATH, writes)
        os.environ.update(writes)
    return list(writes.keys())


def get_config_status() -> dict[str, Any]:
    """Return safe readiness metadata for health checks and setup gates."""
    migrated = _migrate_legacy_keys()
    provider = _get_env("ANTON_PLANNING_PROVIDER", "anthropic")
    model = _get_env("ANTON_PLANNING_MODEL", "claude-sonnet-4-6")

    anthropic_key = bool(_get_env("ANTON_ANTHROPIC_API_KEY"))
    openai_key = bool(_get_env("ANTON_OPENAI_API_KEY"))
    openai_base_url = bool(_get_env("ANTON_OPENAI_BASE_URL"))
    minds_key = bool(_get_env("ANTON_MINDS_API_KEY"))

    missing: list[str] = []
    if provider == "anthropic" and not anthropic_key:
        missing.append("ANTON_ANTHROPIC_API_KEY")
    elif provider == "openai" and not openai_key:
        missing.append("ANTON_OPENAI_API_KEY")
    elif provider == "openai-compatible":
        if not (openai_key or minds_key):
            missing.append("ANTON_OPENAI_API_KEY or ANTON_MINDS_API_KEY")
        if not (openai_base_url or minds_key):
            missing.append("ANTON_OPENAI_BASE_URL")

    ready = len(missing) == 0
    return {
        "config_ready": ready,
        "config_error": "" if ready else f"Configure {', '.join(missing)} for {PROVIDER_LABELS.get(provider, provider)}.",
        "provider": provider,
        "model": model,
        "provider_label": PROVIDER_LABELS.get(provider, provider),
        "migrated": migrated,
    }


# Minds Cloud uses sentinel model names (`_reason_`, `_code_`) that
# only its OpenAI-compatible router resolves. If the user onboards via
# Minds and later switches `ANTON_PLANNING_PROVIDER` to anthropic /
# openai / gemini, these sentinels linger in cowork preferences and
# the UI sends them on every request — every request then 404s
# because the new provider doesn't know what `_reason_` is.
def _is_minds_sentinel(model_id: str | None) -> bool:
    return bool(model_id) and model_id.startswith("_") and model_id.endswith("_")


def _ui_settings() -> dict[str, Any]:
    prefs = load_state().get("preferences", {})
    if not isinstance(prefs, dict):
        prefs = {}
    merged = dict(UI_DEFAULTS)
    merged.update({key: value for key, value in prefs.items() if key in UI_DEFAULTS})

    env_planning_model = _get_env("ANTON_PLANNING_MODEL", "claude-sonnet-4-6")
    env_planning_provider = _get_env("ANTON_PLANNING_PROVIDER", "anthropic")

    # Drop a stale Minds sentinel the moment the active provider can't
    # resolve it. Falls through to the env's planning model, which is
    # the source of truth for "what model will the request actually
    # talk to." Without this guard the cowork UI keeps showing
    # `_reason_` indefinitely after a provider switch.
    saved_default = merged.get("defaultModel") or ""
    if _is_minds_sentinel(saved_default) and env_planning_provider != "openai-compatible":
        # Persist the cleanup so state.json reflects reality and we
        # don't keep doing this masking work on every request. Best-
        # effort — a write failure only means we'll re-mask next time.
        try:
            def _drop_default_model(state: dict) -> None:
                prefs = state.get("preferences")
                if isinstance(prefs, dict):
                    prefs.pop("defaultModel", None)
            update_state(_drop_default_model)
            logger.info(
                "Dropped stale Minds sentinel %r from cowork preferences "
                "(active planning_provider=%r).",
                saved_default, env_planning_provider,
            )
        except Exception as e:
            logger.debug("Could not persist defaultModel cleanup: %s", e)
        saved_default = ""

    if not saved_default:
        saved_default = env_planning_model
    merged["defaultModel"] = saved_default

    return merged


@router.get("")
async def get_settings():
    status = get_config_status()
    ui = _ui_settings()
    return {
        "planningProvider": _get_env("ANTON_PLANNING_PROVIDER", "anthropic"),
        "planningModel":    _get_env("ANTON_PLANNING_MODEL", "claude-sonnet-4-6"),
        "codingProvider":   _get_env("ANTON_CODING_PROVIDER", "anthropic"),
        "codingModel":      _get_env("ANTON_CODING_MODEL", "claude-haiku-4-5-20251001"),
        "openaiBaseUrl":    _get_env("ANTON_OPENAI_BASE_URL", ""),
        "anthropicApiKey":  _masked("ANTON_ANTHROPIC_API_KEY"),
        "openaiApiKey":     _masked("ANTON_OPENAI_API_KEY"),
        "mindsApiKey":      _masked("ANTON_MINDS_API_KEY"),
        "mindsUrl":         _get_env("ANTON_MINDS_URL", "https://mdb.ai"),
        "mindsMindName":    _get_env("ANTON_MINDS_MIND_NAME", ""),
        "mindsDatasource":  _get_env("ANTON_MINDS_DATASOURCE", ""),
        "mindsDatasourceEngine": _get_env("ANTON_MINDS_DATASOURCE_ENGINE", ""),
        "mindsSslVerify":   _bool_env("ANTON_MINDS_SSL_VERIFY", True),
        "publishUrl":       _get_env("ANTON_PUBLISH_URL", "https://4nton.ai"),
        "memoryEnabled":    _bool_env("ANTON_MEMORY_ENABLED", True),
        "memoryMode":       _get_env("ANTON_MEMORY_MODE", "autopilot"),
        "episodicMemory":   _bool_env("ANTON_EPISODIC_MEMORY", True),
        "proactiveDashboards": _bool_env("ANTON_PROACTIVE_DASHBOARDS", False),
        "configReady": status["config_ready"],
        "configError": status["config_error"],
        "providerLabel": status["provider_label"],
        "greeting":      ui["greeting"],
        "tone":          ui["tone"],
        "defaultModel":  ui["defaultModel"] or status["model"],
        "autoPin":       ui["autoPin"],
        "showDots":      ui["showDots"],
        "accentVariant": ui["accentVariant"],
        "uiUpdateMode":  _get_env("UI_UPDATE_MODE", "manual"),
    }


class SettingsPatch(BaseModel):
    greeting:             Optional[str] = None
    tone:                 Optional[str] = None
    defaultModel:         Optional[str] = None
    autoPin:              Optional[bool] = None
    showDots:             Optional[bool] = None
    accentVariant:        Optional[str] = None
    planningProvider:     Optional[str] = None
    planningModel:        Optional[str] = None
    codingProvider:       Optional[str] = None
    codingModel:          Optional[str] = None
    openaiBaseUrl:        Optional[str] = None
    anthropicApiKey:      Optional[str] = None
    openaiApiKey:         Optional[str] = None
    mindsApiKey:          Optional[str] = None
    mindsUrl:             Optional[str] = None
    mindsMindName:        Optional[str] = None
    mindsDatasource:      Optional[str] = None
    mindsDatasourceEngine: Optional[str] = None
    mindsSslVerify:       Optional[bool] = None
    publishUrl:           Optional[str] = None
    memoryEnabled:        Optional[bool] = None
    memoryMode:           Optional[str] = None
    episodicMemory:       Optional[bool] = None
    proactiveDashboards:  Optional[bool] = None
    uiUpdateMode:         Optional[str] = None


@router.put("")
async def update_settings(patch: SettingsPatch):
    writes: dict[str, str] = {}
    delete_keys: list[str] = []
    pref_writes: dict[str, Any] = {}

    if patch.greeting is not None:
        pref_writes["greeting"] = patch.greeting
    if patch.tone is not None:
        pref_writes["tone"] = patch.tone
    if patch.defaultModel is not None:
        pref_writes["defaultModel"] = patch.defaultModel
    if patch.autoPin is not None:
        pref_writes["autoPin"] = patch.autoPin
    if patch.showDots is not None:
        pref_writes["showDots"] = patch.showDots
    if patch.accentVariant is not None:
        pref_writes["accentVariant"] = patch.accentVariant

    _stage_string_env(patch.planningProvider, "ANTON_PLANNING_PROVIDER", writes, delete_keys)
    _stage_string_env(patch.planningModel, "ANTON_PLANNING_MODEL", writes, delete_keys)
    if patch.planningModel is not None and patch.planningModel.strip() and patch.defaultModel is None:
        pref_writes["defaultModel"] = patch.planningModel
    _stage_string_env(patch.codingProvider, "ANTON_CODING_PROVIDER", writes, delete_keys)
    _stage_string_env(patch.codingModel, "ANTON_CODING_MODEL", writes, delete_keys)
    _stage_string_env(patch.openaiBaseUrl, "ANTON_OPENAI_BASE_URL", writes, delete_keys)
    _stage_string_env(patch.memoryMode, "ANTON_MEMORY_MODE", writes, delete_keys)
    _stage_string_env(patch.mindsUrl, "ANTON_MINDS_URL", writes, delete_keys)
    _stage_string_env(patch.mindsMindName, "ANTON_MINDS_MIND_NAME", writes, delete_keys)
    _stage_string_env(patch.mindsDatasource, "ANTON_MINDS_DATASOURCE", writes, delete_keys)
    _stage_string_env(patch.mindsDatasourceEngine, "ANTON_MINDS_DATASOURCE_ENGINE", writes, delete_keys)
    _stage_string_env(patch.publishUrl, "ANTON_PUBLISH_URL", writes, delete_keys)
    if patch.memoryEnabled is not None:
        writes["ANTON_MEMORY_ENABLED"] = str(patch.memoryEnabled).lower()
    if patch.episodicMemory is not None:
        writes["ANTON_EPISODIC_MEMORY"] = str(patch.episodicMemory).lower()
    if patch.proactiveDashboards is not None:
        writes["ANTON_PROACTIVE_DASHBOARDS"] = str(patch.proactiveDashboards).lower()
    if patch.mindsSslVerify is not None:
        writes["ANTON_MINDS_SSL_VERIFY"] = str(patch.mindsSslVerify).lower()
    _stage_string_env(patch.uiUpdateMode, "UI_UPDATE_MODE", writes, delete_keys)

    # API keys — write if provided and not the masked sentinel
    if patch.anthropicApiKey is not None and patch.anthropicApiKey != "***":
        if patch.anthropicApiKey.strip():
            writes["ANTON_ANTHROPIC_API_KEY"] = patch.anthropicApiKey.strip()
        else:
            delete_keys.append("ANTON_ANTHROPIC_API_KEY")
    if patch.openaiApiKey is not None and patch.openaiApiKey != "***":
        if patch.openaiApiKey.strip():
            writes["ANTON_OPENAI_API_KEY"] = patch.openaiApiKey.strip()
        else:
            delete_keys.append("ANTON_OPENAI_API_KEY")
    if patch.mindsApiKey is not None and patch.mindsApiKey != "***":
        if patch.mindsApiKey.strip():
            writes["ANTON_MINDS_API_KEY"] = patch.mindsApiKey.strip()
        else:
            delete_keys.append("ANTON_MINDS_API_KEY")

    if writes or delete_keys:
        try:
            _write_dotenv(GLOBAL_ENV_PATH, writes, tuple(delete_keys))
            # Also update current process environment
            for k, v in writes.items():
                os.environ[k] = v
            for key in delete_keys:
                os.environ.pop(key, None)
        except Exception as e:
            logger.warning("Failed to write settings: %s", e)
            raise HTTPException(status_code=500, detail="Settings could not be saved.") from e

    if pref_writes:
        try:
            update_state(lambda state: state.setdefault("preferences", {}).update(pref_writes))
        except Exception as e:
            logger.warning("Failed to write local preferences: %s", e)
            raise HTTPException(status_code=500, detail="Local preferences could not be saved.") from e

    status = get_config_status()
    return {
        "status": "ok",
        "updated": list(writes.keys()) + [f"deleted:{key}" for key in delete_keys] + [f"ui:{key}" for key in pref_writes],
        "configReady": status["config_ready"],
        "configError": status["config_error"],
    }


@router.post("/validate")
async def validate_settings():
    status = get_config_status()
    return {
        "status": "ok" if status["config_ready"] else "needs_config",
        "configReady": status["config_ready"],
        "configError": status["config_error"],
        "provider": status["provider"],
        "model": status["model"],
    }
