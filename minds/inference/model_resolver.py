"""LLM model resolution and routing logic.

The ModelResolver maps passthrough alias names (e.g., latest:sonnet) to concrete
provider configs (API key, model name, transport layer). Each alias routes to
exactly one provider; if that provider isn't configured we raise 400 rather than
falling back to another provider.

Routing layout:

    latest:sonnet      — Anthropic ``passthrough_sonnet_model``
    latest:opus        — Anthropic ``passthrough_opus_model``
    latest:haiku       — Anthropic ``passthrough_haiku_model``
    latest:gpt         — OpenAI  ``passthrough_gpt_model``       + reasoning.effort=low (default)
    latest:gpt-low     — OpenAI  ``passthrough_gpt_model``       + reasoning.effort=low
    latest:gpt-medium  — OpenAI  ``passthrough_gpt_model``       + reasoning.effort=medium
    latest:gpt-high    — OpenAI  ``passthrough_gpt_model``       + reasoning.effort=high
    latest:gpt-codex   — OpenAI  ``passthrough_gpt_codex_model``
    latest:gpt-mini    — OpenAI  ``passthrough_gpt_mini_model``
    latest:gpt-nano    — OpenAI  ``passthrough_gpt_nano_model``
    latest:gemini      — Google  ``passthrough_gemini_model``       (native generateContent API)
    latest:gemini-flash — Google ``passthrough_gemini_flash_model`` (Flash sibling, lower latency/cost)
    latest:kimi        — Fireworks ``passthrough_kimi_model``    (Anthropic-shape API)
    latest:deepseek    — Fireworks ``passthrough_deepseek_model`` (Anthropic-shape API)
    latest:qwen        — Fireworks ``passthrough_qwen_model``    (Anthropic-shape API)
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from fastapi import HTTPException

if TYPE_CHECKING:
    from minds.common.passthrough_config import PassthroughModelConfig
    from minds.common.settings.app_settings import AppSettings

_PASSTHROUGH_PATTERN = re.compile(r"^latest:([a-zA-Z0-9\-]+)$")

# Pre-`latest:*` model names that some callers still send. Each maps to the
# canonical alias body it should resolve through; the resolver preserves the
# deprecated spelling on the returned config so observability can track usage.
_DEPRECATED_ALIASES: dict[str, str] = {
    "_reason_": "sonnet",
    "_code_": "haiku",
}


# Provider availability predicates
def _openai_available(settings: AppSettings) -> bool:
    key = settings.openai.api_key
    return bool(key) and key != "not set"


def _anthropic_available(settings: AppSettings) -> bool:
    return bool(settings.anthropic.api_key)


def _fireworks_available(settings: AppSettings) -> bool:
    return bool(settings.fireworks.api_key)


def _gemini_available(settings: AppSettings) -> bool:
    return bool(settings.gemini.api_key)


# Per-provider config builders
def _config_for_openai(
    settings: AppSettings,
    model_name: str,
    *,
    alias: str,
    reasoning_effort: str | None = None,
) -> PassthroughModelConfig:
    from minds.common.passthrough_config import ApiKind, PassthroughModelConfig, WebSearchMode

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
    from minds.common.passthrough_config import ApiKind, PassthroughModelConfig, WebSearchMode

    return PassthroughModelConfig(
        api_kind=ApiKind.ANTHROPIC_MESSAGES,
        model_name=model_name,
        api_key=settings.anthropic.api_key,
        web_search_mode=WebSearchMode.ANTHROPIC_NATIVE,
        label="anthropic",
        alias=alias,
    )


def _config_for_fireworks(settings: AppSettings, model_name: str, *, alias: str) -> PassthroughModelConfig:
    from minds.common.passthrough_config import ApiKind, PassthroughModelConfig, WebSearchMode

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
    from minds.common.passthrough_config import ApiKind, PassthroughModelConfig, WebSearchMode

    return PassthroughModelConfig(
        api_kind=ApiKind.GEMINI_NATIVE,
        model_name=model_name,
        api_key=settings.gemini.api_key,
        web_search_mode=WebSearchMode.GEMINI_GOOGLE_SEARCH,
        label="gemini",
        alias=alias,
    )


AliasBuilder = Callable[["AppSettings"], "PassthroughModelConfig"]
AliasPredicate = Callable[["AppSettings"], bool]

_ALIAS_PRIORITY: dict[str, list[tuple[AliasPredicate, AliasBuilder]]] = {
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
    # OpenAI flagship reasoning model with three effort levels. ``latest:gpt``
    # is a deliberate alias for ``latest:gpt-low`` (the cheapest, fastest
    # variant of the flagship) — gives callers a one-word default without
    # having to know about reasoning_effort.
    "gpt": [
        (
            _openai_available,
            lambda s: _config_for_openai(s, s.openai.passthrough_gpt_model, alias="gpt", reasoning_effort="low"),
        ),
    ],
    "gpt-low": [
        (
            _openai_available,
            lambda s: _config_for_openai(s, s.openai.passthrough_gpt_model, alias="gpt-low", reasoning_effort="low"),
        ),
    ],
    "gpt-medium": [
        (
            _openai_available,
            lambda s: _config_for_openai(
                s, s.openai.passthrough_gpt_model, alias="gpt-medium", reasoning_effort="medium"
            ),
        ),
    ],
    "gpt-high": [
        (
            _openai_available,
            lambda s: _config_for_openai(s, s.openai.passthrough_gpt_model, alias="gpt-high", reasoning_effort="high"),
        ),
    ],
    # OpenAI specialized variants — distinct upstream models, no reasoning
    # level (those are configured server-side per model family).
    "gpt-codex": [
        (
            _openai_available,
            lambda s: _config_for_openai(s, s.openai.passthrough_gpt_codex_model, alias="gpt-codex"),
        ),
    ],
    "gpt-mini": [
        (
            _openai_available,
            lambda s: _config_for_openai(s, s.openai.passthrough_gpt_mini_model, alias="gpt-mini"),
        ),
    ],
    "gpt-nano": [
        (
            _openai_available,
            lambda s: _config_for_openai(s, s.openai.passthrough_gpt_nano_model, alias="gpt-nano"),
        ),
    ],
    # Single-provider explicit-model aliases.
    "gemini": [
        (
            _gemini_available,
            lambda s: _config_for_gemini(s, s.gemini.passthrough_gemini_model, alias="gemini"),
        ),
    ],
    "gemini-flash": [
        (
            _gemini_available,
            lambda s: _config_for_gemini(s, s.gemini.passthrough_gemini_flash_model, alias="gemini-flash"),
        ),
    ],
    "kimi": [
        (
            _fireworks_available,
            lambda s: _config_for_fireworks(s, s.fireworks.passthrough_kimi_model, alias="kimi"),
        ),
    ],
    "deepseek": [
        (
            _fireworks_available,
            lambda s: _config_for_fireworks(s, s.fireworks.passthrough_deepseek_model, alias="deepseek"),
        ),
    ],
    "qwen": [
        (
            _fireworks_available,
            lambda s: _config_for_fireworks(s, s.fireworks.passthrough_qwen_model, alias="qwen"),
        ),
    ],
}


class ModelResolver:
    """Resolves passthrough aliases to provider configurations."""

    def __init__(self, settings: AppSettings) -> None:
        """Initialize the resolver with application settings."""
        self.settings = settings

    def resolve(self, model_name: str) -> PassthroughModelConfig:
        """Resolve a passthrough model name to a fully-populated config.

        Accepts either the canonical ``latest:<alias>`` form or one of the
        deprecated bare spellings in :data:`_DEPRECATED_ALIASES`. Unknown
        aliases raise ``HTTPException(400)``. Likewise, when an alias's
        priority list has no entry whose key is configured we raise 400 —
        keeps the passthrough contract honest instead of silently routing
        ``latest:kimi`` to Claude.
        """
        if model_name in _DEPRECATED_ALIASES:
            config = self._resolve_alias(_DEPRECATED_ALIASES[model_name])
            # Preserve the deprecated spelling on the returned config so
            # observability records that the legacy name was used; lets us
            # measure adoption before actually deleting these.
            return dataclasses.replace(config, alias=model_name)

        m = _PASSTHROUGH_PATTERN.match(model_name)
        if not m:
            raise ValueError(f"{model_name!r} is not a valid passthrough model name")

        return self._resolve_alias(m.group(1).lower())

    def is_passthrough_model(self, model_name: str) -> bool:
        """Return True if *model_name* is a passthrough name.

        Matches the canonical ``latest:<alias>`` pattern or one of the deprecated
        bare spellings in :data:`_DEPRECATED_ALIASES`.
        """
        if model_name in _DEPRECATED_ALIASES:
            return True
        return _PASSTHROUGH_PATTERN.match(model_name) is not None

    def list_available(self) -> list[PassthroughModelConfig]:
        """Resolve every alias whose provider is configured.

        Used by ``GET /v1/models`` to advertise currently-callable models.
        Deprecated aliases (``_reason_``, ``_code_``) are not in
        :data:`_ALIAS_PRIORITY` so they're excluded by construction.
        """
        configs: list[PassthroughModelConfig] = []
        for candidates in _ALIAS_PRIORITY.values():
            for predicate, builder in candidates:
                if predicate(self.settings):
                    configs.append(builder(self.settings))
                    break
        return configs

    def _resolve_alias(self, alias: str) -> PassthroughModelConfig:
        """Resolve an alias body (e.g. ``"sonnet"``) to a populated config."""
        candidates = _ALIAS_PRIORITY.get(alias)
        if candidates is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown passthrough alias 'latest:{alias}'. "
                    "See service docs for the supported alias → provider mapping."
                ),
            )
        for predicate, builder in candidates:
            if predicate(self.settings):
                return builder(self.settings)

        raise HTTPException(
            status_code=400,
            detail=(
                f"No provider configured for passthrough alias 'latest:{alias}'. "
                "Set the API key for at least one supported provider on this "
                "alias (see service docs for the alias → provider mapping)."
            ),
        )
