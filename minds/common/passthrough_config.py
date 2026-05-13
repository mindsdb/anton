"""Passthrough alias resolution.

Callers write ``_xxx_`` in the request ``model`` field; we resolve that to a
concrete upstream provider + model + credentials at request time. Adding a
new alias means: (1) add a model-name setting on the relevant provider block
in :mod:`minds.common.settings.app_settings`, and (2) add an entry to
``_ALIAS_PRIORITY`` below. No new factory functions per alias — the
provider-level builders read the model name from settings.

Routing layout:

    _reason_           — Anthropic ``passthrough_reason_model``, falls back to
                          OpenAI ``passthrough_reason_model`` (capability tier)
    _code_             — Anthropic ``passthrough_code_model``, falls back to
                          OpenAI ``passthrough_code_model`` (capability tier)
    _sonnet_           — Anthropic ``passthrough_sonnet_model``
    _opus_             — Anthropic ``passthrough_opus_model``
    _haiku_            — Anthropic ``passthrough_haiku_model``
    _gpt-5.5-low_      — OpenAI ``passthrough_gpt55_model`` + reasoning.effort=low
    _gpt-5.5-medium_   — OpenAI ``passthrough_gpt55_model`` + reasoning.effort=medium
    _gpt-5.5-high_     — OpenAI ``passthrough_gpt55_model`` + reasoning.effort=high
    _gemini-3.1_       — Google ``passthrough_gemini_31_model`` (native generateContent API)
    _kimi-k2.6_        — Fireworks ``passthrough_kimi_k26_model`` (Anthropic-shape API)
    _deepseek-v4_      — Fireworks ``passthrough_deepseek_v4_model`` (Anthropic-shape API)

When a single-provider alias is requested but its key isn't configured we
raise HTTP 400 rather than silently falling back — otherwise callers get
strictly different model behavior than they asked for with no signal.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from fastapi import HTTPException
from pydantic import BaseModel

from minds.common.settings.app_settings import AppSettings, get_app_settings

_PASSTHROUGH_PATTERN = re.compile(r"^_([a-zA-Z0-9.\-]+)_$")


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
    for log-grep continuity. ``alias`` is the bare alias string (no
    underscores) the resolver matched — preserved here so observability can
    record it as metadata without re-parsing the request model. For OpenAI
    reasoning-capable aliases like ``_gpt-5.5-low_``, ``reasoning_effort``
    holds the effort level baked in by the alias; clients never set it.
    """

    api_kind: ApiKind
    model_name: str
    api_key: str
    base_url: str | None = None
    web_search_mode: WebSearchMode = WebSearchMode.DROP
    label: str = ""
    alias: str = ""
    reasoning_effort: str | None = None

    def to_observability_metadata(self) -> "PassthroughObservabilityMetadata":
        """Project the config into the metadata blob recorded on Langfuse.

        Centralized here (rather than constructed inline at the handler)
        so adding a new metadata field is a one-place change and a typo in
        a field name surfaces at this construction site, not downstream.
        """
        return PassthroughObservabilityMetadata(
            passthrough_alias=self.alias,
            provider=self.label,
            api_kind=self.api_kind.value,
            reasoning_effort=self.reasoning_effort,
        )


class PassthroughObservabilityMetadata(BaseModel):
    """Metadata attached to the Langfuse generation for a passthrough request.

    Mirrors :class:`minds.requests.context.LangfuseContextMetadata`'s pattern
    of typing observability payloads rather than passing free-form dicts —
    a typo in a key name now fails at construction instead of producing an
    unfilterable trace. ``to_metadata()`` returns the dict the Langfuse SDK
    accepts, dropping fields whose value is None so the dashboard doesn't
    render spurious ``null`` rows for aliases (Anthropic, Gemini, Fireworks)
    that have no reasoning-level concept.
    """

    passthrough_alias: str
    provider: str
    api_kind: str
    reasoning_effort: str | None = None

    def to_metadata(self) -> dict:
        return self.model_dump(exclude_none=True)


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


def _config_for_openai(
    settings: AppSettings,
    model_name: str,
    *,
    alias: str,
    reasoning_effort: str | None = None,
) -> PassthroughModelConfig:
    return PassthroughModelConfig(
        api_kind=ApiKind.OPENAI_RESPONSES,
        model_name=model_name,
        api_key=settings.openai.api_key,
        base_url=settings.openai.api_url,
        web_search_mode=WebSearchMode.OPENAI_NATIVE,
        label="openai",
        alias=alias,
        reasoning_effort=reasoning_effort,
    )


def _config_for_anthropic(settings: AppSettings, model_name: str, *, alias: str) -> PassthroughModelConfig:
    return PassthroughModelConfig(
        api_kind=ApiKind.ANTHROPIC_MESSAGES,
        model_name=model_name,
        api_key=settings.anthropic.api_key,
        web_search_mode=WebSearchMode.ANTHROPIC_NATIVE,
        label="anthropic",
        alias=alias,
    )


def _config_for_fireworks(settings: AppSettings, model_name: str, *, alias: str) -> PassthroughModelConfig:
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
        alias=alias,
    )


def _config_for_gemini(settings: AppSettings, model_name: str, *, alias: str) -> PassthroughModelConfig:
    return PassthroughModelConfig(
        api_kind=ApiKind.GEMINI_NATIVE,
        model_name=model_name,
        api_key=settings.gemini.api_key,
        web_search_mode=WebSearchMode.GEMINI_GOOGLE_SEARCH,
        label="gemini",
        alias=alias,
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
    # Semantic capability-tier aliases — multi-provider fallback chain.
    "reason": [
        (
            _anthropic_available,
            lambda s: _config_for_anthropic(s, s.anthropic.passthrough_reason_model, alias="reason"),
        ),
        (
            _openai_available,
            lambda s: _config_for_openai(s, s.openai.passthrough_reason_model, alias="reason"),
        ),
    ],
    "code": [
        (
            _anthropic_available,
            lambda s: _config_for_anthropic(s, s.anthropic.passthrough_code_model, alias="code"),
        ),
        (
            _openai_available,
            lambda s: _config_for_openai(s, s.openai.passthrough_code_model, alias="code"),
        ),
    ],
    # Anthropic explicit-model aliases.
    "sonnet": [
        (
            _anthropic_available,
            lambda s: _config_for_anthropic(s, s.anthropic.passthrough_sonnet_model, alias="sonnet"),
        ),
    ],
    "opus": [
        (
            _anthropic_available,
            lambda s: _config_for_anthropic(s, s.anthropic.passthrough_opus_model, alias="opus"),
        ),
    ],
    "haiku": [
        (
            _anthropic_available,
            lambda s: _config_for_anthropic(s, s.anthropic.passthrough_haiku_model, alias="haiku"),
        ),
    ],
    # OpenAI gpt-5.5 with three reasoning levels — same upstream model name,
    # reasoning_effort baked into config so clients don't have to know.
    "gpt-5.5-low": [
        (
            _openai_available,
            lambda s: _config_for_openai(
                s, s.openai.passthrough_gpt55_model, alias="gpt-5.5-low", reasoning_effort="low"
            ),
        ),
    ],
    "gpt-5.5-medium": [
        (
            _openai_available,
            lambda s: _config_for_openai(
                s, s.openai.passthrough_gpt55_model, alias="gpt-5.5-medium", reasoning_effort="medium"
            ),
        ),
    ],
    "gpt-5.5-high": [
        (
            _openai_available,
            lambda s: _config_for_openai(
                s, s.openai.passthrough_gpt55_model, alias="gpt-5.5-high", reasoning_effort="high"
            ),
        ),
    ],
    # Single-provider explicit-model aliases.
    "gemini-3.1": [
        (
            _gemini_available,
            lambda s: _config_for_gemini(s, s.gemini.passthrough_gemini_31_model, alias="gemini-3.1"),
        ),
    ],
    "kimi-k2.6": [
        (
            _fireworks_available,
            lambda s: _config_for_fireworks(s, s.fireworks.passthrough_kimi_k26_model, alias="kimi-k2.6"),
        ),
    ],
    "deepseek-v4": [
        (
            _fireworks_available,
            lambda s: _config_for_fireworks(
                s, s.fireworks.passthrough_deepseek_v4_model, alias="deepseek-v4"
            ),
        ),
    ],
}


def resolve_passthrough_model(model: str) -> PassthroughModelConfig:
    """Resolve a ``_xxx_`` model name to a fully-populated config.

    Unknown aliases raise ``HTTPException(400)``. Likewise, when an alias's
    priority list has no entry whose key is configured we raise 400 —
    keeps the passthrough contract honest instead of silently routing
    ``_kimi-k2.6_`` to Claude.
    """
    m = _PASSTHROUGH_PATTERN.match(model)
    if not m:
        raise ValueError(f"{model!r} is not a valid passthrough model name")

    alias = m.group(1).lower()
    settings = get_app_settings()

    candidates = _ALIAS_PRIORITY.get(alias)
    if candidates is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown passthrough alias '_{alias}_'. "
                "See service docs for the supported alias → provider mapping."
            ),
        )
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
