"""Passthrough alias resolution.

Callers write ``_xxx_`` in the request ``model`` field; we resolve that to a
concrete upstream provider + model + credentials at request time. Adding a
new alias means: (1) add a model-name setting on the relevant provider block
in :mod:`minds.common.settings.app_settings`, and (2) add an entry to
``_ALIAS_PRIORITY`` below. No new factory functions per alias — the
provider-level builders read the model name from settings.

Routing layout for v1:

    _reason_   — Anthropic claude-sonnet-4-6, falls back to OpenAI gpt-5.2
    _code_     — Anthropic claude-haiku-4-5-20251001, falls back to OpenAI gpt-5.1-codex
    _kimi_     — Fireworks Kimi K2.6 (Anthropic-shape API at fireworks.ai)
    _gemini_   — Google gemini-3.1-pro-preview (native generateContent API)

When a single-provider alias (``_kimi_``, ``_gemini_``) is requested but its
key isn't configured we raise HTTP 400 rather than silently falling back —
otherwise callers get strictly different model behavior than they asked for
with no signal.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from fastapi import HTTPException

from minds.common.settings.app_settings import AppSettings, get_app_settings

_PASSTHROUGH_PATTERN = re.compile(r"^_([a-zA-Z0-9]+)_$")


class ApiKind(StrEnum):
    """Which upstream transport + translation path the agent uses."""

    OPENAI_RESPONSES = "openai_responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    GEMINI_NATIVE = "gemini_native"


class WebSearchMode(StrEnum):
    """How generic ``web_search`` / ``fetch`` tools should be translated."""

    OPENAI_NATIVE = "openai_native"  # Responses API: tools=[{"type":"web_search"}]
    ANTHROPIC_NATIVE = "anthropic_native"  # Messages API: web_search_20250305 / web_fetch_20250910
    GEMINI_GOOGLE_SEARCH = "gemini_google_search"  # Gemini: Tool(google_search=GoogleSearch())
    DROP = "drop"  # silently omit the generic web_search/fetch tools


@dataclass(frozen=True)
class PassthroughModelConfig:
    """Concrete routing for a single resolved alias.

    ``api_kind`` chooses the SDK and translation path; ``base_url`` is None
    for OpenAI/Anthropic-direct (SDK default) and set for Anthropic-shape
    proxies like Fireworks. ``label`` survives the old ``provider`` string
    for log-grep continuity.
    """

    api_kind: ApiKind
    model_name: str
    api_key: str
    base_url: str | None = None
    web_search_mode: WebSearchMode = WebSearchMode.DROP
    label: str = ""


def is_passthrough_model(model: str) -> bool:
    """Return True if *model* matches the ``_xxx_`` passthrough pattern."""
    return _PASSTHROUGH_PATTERN.match(model) is not None


# ---------------------------------------------------------------------------
# Provider availability predicates
# ---------------------------------------------------------------------------


def _openai_available(settings: AppSettings) -> bool:
    key = settings.openai.api_key
    return bool(key) and key != "not set"


def _anthropic_available(settings: AppSettings) -> bool:
    return bool(settings.anthropic.api_key)


def _fireworks_available(settings: AppSettings) -> bool:
    return bool(settings.fireworks.api_key)


def _gemini_available(settings: AppSettings) -> bool:
    return bool(settings.gemini.api_key)


# ---------------------------------------------------------------------------
# Per-provider config builders (one per upstream, not one per alias)
# ---------------------------------------------------------------------------
#
# Each builder takes the resolved model name as an argument rather than
# hardcoding it, so adding a new alias served by an existing provider is a
# settings entry + a one-line ``_ALIAS_PRIORITY`` change. No new code path
# per alias.


def _config_for_openai(settings: AppSettings, model_name: str) -> PassthroughModelConfig:
    return PassthroughModelConfig(
        api_kind=ApiKind.OPENAI_RESPONSES,
        model_name=model_name,
        api_key=settings.openai.api_key,
        base_url=settings.openai.api_url,
        web_search_mode=WebSearchMode.OPENAI_NATIVE,
        label="openai",
    )


def _config_for_anthropic(settings: AppSettings, model_name: str) -> PassthroughModelConfig:
    return PassthroughModelConfig(
        api_kind=ApiKind.ANTHROPIC_MESSAGES,
        model_name=model_name,
        api_key=settings.anthropic.api_key,
        web_search_mode=WebSearchMode.ANTHROPIC_NATIVE,
        label="anthropic",
    )


def _config_for_fireworks(settings: AppSettings, model_name: str) -> PassthroughModelConfig:
    # Fireworks exposes an Anthropic-compatible Messages API at this base
    # URL; the Anthropic SDK appends /v1/messages. Web search is unavailable
    # — Fireworks doesn't host a search index — so generic web_search/fetch
    # tools are dropped before forwarding upstream.
    return PassthroughModelConfig(
        api_kind=ApiKind.ANTHROPIC_MESSAGES,
        model_name=model_name,
        api_key=settings.fireworks.api_key,
        base_url=settings.fireworks.anthropic_base_url,
        web_search_mode=WebSearchMode.DROP,
        label="fireworks",
    )


def _config_for_gemini(settings: AppSettings, model_name: str) -> PassthroughModelConfig:
    return PassthroughModelConfig(
        api_kind=ApiKind.GEMINI_NATIVE,
        model_name=model_name,
        api_key=settings.gemini.api_key,
        web_search_mode=WebSearchMode.GEMINI_GOOGLE_SEARCH,
        label="gemini",
    )


# ---------------------------------------------------------------------------
# Alias → priority-list of (predicate, config builder) pairs
# ---------------------------------------------------------------------------
#
# Each entry's builder is a ``Callable[[AppSettings], PassthroughModelConfig]``
# that reads the model name from settings; the first predicate that returns
# True wins. Single-provider aliases (``kimi``, ``gemini``) deliberately
# have only one entry — missing key → 400, not silent fallback.

AliasBuilder = Callable[[AppSettings], PassthroughModelConfig]
AliasPredicate = Callable[[AppSettings], bool]


_ALIAS_PRIORITY: dict[str, list[tuple[AliasPredicate, AliasBuilder]]] = {
    "reason": [
        (
            _anthropic_available,
            lambda s: _config_for_anthropic(s, s.anthropic.passthrough_reason_model),
        ),
        (
            _openai_available,
            lambda s: _config_for_openai(s, s.openai.passthrough_reason_model),
        ),
    ],
    "code": [
        (
            _anthropic_available,
            lambda s: _config_for_anthropic(s, s.anthropic.passthrough_code_model),
        ),
        (
            _openai_available,
            lambda s: _config_for_openai(s, s.openai.passthrough_code_model),
        ),
    ],
    "kimi": [
        (
            _fireworks_available,
            lambda s: _config_for_fireworks(s, s.fireworks.passthrough_kimi_model),
        ),
    ],
    "gemini": [
        (
            _gemini_available,
            lambda s: _config_for_gemini(s, s.gemini.passthrough_model),
        ),
    ],
}


def resolve_passthrough_model(model: str) -> PassthroughModelConfig:
    """Resolve a ``_xxx_`` model name to a fully-populated config.

    Unknown aliases fall back to ``_reason_``'s table (matches prior behavior).
    Raises ``HTTPException(400)`` when no entry in the alias's priority list
    has its key configured — keeps the passthrough contract honest instead of
    silently routing ``_kimi_`` to Claude.
    """
    m = _PASSTHROUGH_PATTERN.match(model)
    if not m:
        raise ValueError(f"{model!r} is not a valid passthrough model name")

    alias = m.group(1).lower()
    settings = get_app_settings()

    candidates = _ALIAS_PRIORITY.get(alias, _ALIAS_PRIORITY["reason"])
    for predicate, builder in candidates:
        if predicate(settings):
            return builder(settings)

    raise HTTPException(
        status_code=400,
        detail=(
            f"No provider configured for passthrough alias '_{alias}_'. "
            "Set the API key for at least one supported provider on this "
            "alias (see service docs for the alias → provider mapping)."
        ),
    )
