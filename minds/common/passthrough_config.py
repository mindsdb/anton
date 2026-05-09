"""Passthrough alias resolution.

Callers write ``_xxx_`` in the request ``model`` field; we resolve that to a
concrete upstream provider + model + credentials at request time. Adding a new
provider means: (1) adding a settings block, (2) adding a factory below, and
(3) adding the alias to ``_ALIAS_PRIORITY``.

Routing layout for v1:

    _reason_   — Anthropic claude-sonnet-4-6, falls back to OpenAI gpt-5.2
    _code_     — Anthropic claude-haiku-4-5-20251001, falls back to OpenAI gpt-5.1-codex
    _kimi_     — Fireworks Kimi K2.6 (Anthropic-shape API at fireworks.ai)
    _gemini_   — Google Gemini 2.5 Pro (native generateContent API)

When a single-provider alias (``_kimi_``, ``_gemini_``) is requested but its
key isn't configured we raise HTTP 400 rather than silently falling back —
otherwise callers get strictly different model behavior than they asked for
with no signal.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from fastapi import HTTPException

from minds.common.settings.app_settings import get_app_settings

_PASSTHROUGH_PATTERN = re.compile(r"^_([a-zA-Z0-9]+)_$")


ApiKind = Literal["openai_responses", "anthropic_messages", "gemini_native"]
WebSearchMode = Literal[
    "openai_native",  # Responses API: tools=[{"type":"web_search"}]
    "anthropic_native",  # Messages API: web_search_20250305 / web_fetch_20250910
    "gemini_google_search",  # Gemini: Tool(google_search=GoogleSearch())
    "drop",  # silently omit the generic web_search/fetch tools
]


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
    web_search_mode: WebSearchMode = "drop"
    label: str = ""


def is_passthrough_model(model: str) -> bool:
    """Return True if *model* matches the ``_xxx_`` passthrough pattern."""
    return _PASSTHROUGH_PATTERN.match(model) is not None


# ---------------------------------------------------------------------------
# Per-provider availability + config factories
# ---------------------------------------------------------------------------
#
# Each factory is only called *after* its availability predicate has returned
# True, so factories can dereference settings unconditionally. Predicates are
# closures over the settings object the resolver fetches per call so tests can
# clear the lru_cache and re-resolve with different env vars.


def _openai_available(settings) -> bool:  # noqa: ANN001
    key = settings.openai.api_key
    return bool(key) and key != "not set"


def _anthropic_available(settings) -> bool:  # noqa: ANN001
    return bool(settings.anthropic.api_key)


def _fireworks_available(settings) -> bool:  # noqa: ANN001
    return bool(settings.fireworks.api_key)


def _gemini_available(settings) -> bool:  # noqa: ANN001
    return bool(settings.gemini.api_key)


def _openai_reason(settings) -> PassthroughModelConfig:  # noqa: ANN001
    return PassthroughModelConfig(
        api_kind="openai_responses",
        model_name="gpt-5.2",
        api_key=settings.openai.api_key,
        base_url=settings.openai.api_url,
        web_search_mode="openai_native",
        label="openai",
    )


def _openai_code(settings) -> PassthroughModelConfig:  # noqa: ANN001
    return PassthroughModelConfig(
        api_kind="openai_responses",
        model_name="gpt-5.1-codex",
        api_key=settings.openai.api_key,
        base_url=settings.openai.api_url,
        web_search_mode="openai_native",
        label="openai",
    )


def _anthropic_reason(settings) -> PassthroughModelConfig:  # noqa: ANN001
    return PassthroughModelConfig(
        api_kind="anthropic_messages",
        model_name="claude-sonnet-4-6",
        api_key=settings.anthropic.api_key,
        web_search_mode="anthropic_native",
        label="anthropic",
    )


def _anthropic_code(settings) -> PassthroughModelConfig:  # noqa: ANN001
    return PassthroughModelConfig(
        api_kind="anthropic_messages",
        model_name="claude-haiku-4-5-20251001",
        api_key=settings.anthropic.api_key,
        web_search_mode="anthropic_native",
        label="anthropic",
    )


def _fireworks_kimi(settings) -> PassthroughModelConfig:  # noqa: ANN001
    # Fireworks exposes an Anthropic-compatible Messages API at this base
    # URL; the Anthropic SDK appends /v1/messages. Web search is unavailable
    # — Fireworks doesn't host a search index — so generic web_search/fetch
    # tools are dropped before forwarding upstream.
    return PassthroughModelConfig(
        api_kind="anthropic_messages",
        model_name="accounts/fireworks/models/kimi-k2p6",
        api_key=settings.fireworks.api_key,
        base_url=settings.fireworks.anthropic_base_url,
        web_search_mode="drop",
        label="fireworks",
    )


def _gemini_pro(settings) -> PassthroughModelConfig:  # noqa: ANN001
    return PassthroughModelConfig(
        api_kind="gemini_native",
        model_name="gemini-3.1-pro-preview",
        api_key=settings.gemini.api_key,
        web_search_mode="gemini_google_search",
        label="gemini",
    )


# Per-alias priority list: the first ``(predicate, factory)`` pair whose
# predicate returns True wins. Order = preference order.
#
# Single-provider aliases (``kimi``, ``gemini``) deliberately have only one
# entry — if the matching key isn't configured we 400 rather than falling
# back to a different model family.
_ALIAS_PRIORITY: dict[str, list[tuple[Callable, Callable[..., PassthroughModelConfig]]]] = {
    "reason": [
        (_anthropic_available, _anthropic_reason),
        (_openai_available, _openai_reason),
    ],
    "code": [
        (_anthropic_available, _anthropic_code),
        (_openai_available, _openai_code),
    ],
    "kimi": [
        (_fireworks_available, _fireworks_kimi),
    ],
    "gemini": [
        (_gemini_available, _gemini_pro),
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
    for predicate, factory in candidates:
        if predicate(settings):
            return factory(settings)

    raise HTTPException(
        status_code=400,
        detail=(
            f"No provider configured for passthrough alias '_{alias}_'. "
            "Set the API key for at least one supported provider on this "
            "alias (see service docs for the alias → provider mapping)."
        ),
    )
