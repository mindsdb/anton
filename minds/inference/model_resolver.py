"""LLM model resolution and routing logic.

Maps passthrough alias names (e.g., latest:sonnet) to provider configs.
Each alias routes to exactly one provider; missing configs raise 400.
"""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING

from fastapi import HTTPException

from minds.inference.types import (
    AliasMapping,
    ApiKind,
    PassthroughAlias,
    PassthroughModelConfig,
    ProviderConfig,
    WebSearchMode,
)
from minds.schemas.passthrough import PassthroughModelStatsigConfig

if TYPE_CHECKING:
    from minds.common.settings.app_settings import AppSettings

_PASSTHROUGH_PATTERN = re.compile(r"^latest:([a-zA-Z0-9\-]+)$")

_DEPRECATED_ALIASES: dict[str, str] = {
    "_reason_": "sonnet",
    "_code_": "haiku",
}


_ALIASES: dict[str, AliasMapping] = {
    PassthroughAlias.SONNET: AliasMapping("anthropic", "passthrough_sonnet_model"),
    PassthroughAlias.OPUS: AliasMapping("anthropic", "passthrough_opus_model"),
    PassthroughAlias.HAIKU: AliasMapping("anthropic", "passthrough_haiku_model"),
    PassthroughAlias.FABLE: AliasMapping("anthropic", "passthrough_fable_model"),
    PassthroughAlias.GPT: AliasMapping("openai", "passthrough_gpt_model", reasoning_effort="low"),
    PassthroughAlias.GPT_LOW: AliasMapping("openai", "passthrough_gpt_model", reasoning_effort="low"),
    PassthroughAlias.GPT_MEDIUM: AliasMapping("openai", "passthrough_gpt_model", reasoning_effort="medium"),
    PassthroughAlias.GPT_HIGH: AliasMapping("openai", "passthrough_gpt_model", reasoning_effort="high"),
    PassthroughAlias.GPT_CODEX: AliasMapping("openai", "passthrough_gpt_codex_model"),
    PassthroughAlias.GPT_MINI: AliasMapping("openai", "passthrough_gpt_mini_model"),
    PassthroughAlias.GPT_NANO: AliasMapping("openai", "passthrough_gpt_nano_model"),
    PassthroughAlias.GEMINI: AliasMapping("gemini", "passthrough_gemini_model"),
    PassthroughAlias.GEMINI_FLASH: AliasMapping("gemini", "passthrough_gemini_flash_model"),
    PassthroughAlias.KIMI: AliasMapping("fireworks", "passthrough_kimi_model"),
    PassthroughAlias.DEEPSEEK: AliasMapping("fireworks", "passthrough_deepseek_model"),
    PassthroughAlias.QWEN: AliasMapping("fireworks", "passthrough_qwen_model"),
}

_PROVIDER_CONFIG: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        api_kind="OPENAI_RESPONSES",
        web_search_mode="OPENAI_NATIVE",
        label="openai",
        api_key_attr="api_key",
        base_url_attr="api_url",
    ),
    "anthropic": ProviderConfig(
        api_kind="ANTHROPIC_MESSAGES",
        web_search_mode="ANTHROPIC_NATIVE",
        label="anthropic",
        api_key_attr="api_key",
    ),
    "gemini": ProviderConfig(
        api_kind="GEMINI_NATIVE",
        web_search_mode="GEMINI_GOOGLE_SEARCH",
        label="gemini",
        api_key_attr="api_key",
    ),
    # Fireworks shares the Anthropic transport shape but has no hosted search
    # index, so it routes through its own adapter (ApiKind.FIREWORKS) that drives
    # a server-side external-search loop (WebSearchMode.EXTERNAL_TOOL).
    "fireworks": ProviderConfig(
        api_kind="FIREWORKS",
        web_search_mode="EXTERNAL_TOOL",
        label="fireworks",
        api_key_attr="api_key",
        base_url_attr="anthropic_base_url",
    ),
}


class ModelResolver:
    """Resolves passthrough aliases to provider configurations."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def resolve(
        self,
        model_name: str,
        *,
        policy: PassthroughModelStatsigConfig | None = None,
    ) -> PassthroughModelConfig:
        """Resolve passthrough model name to config.

        Accepts canonical ``latest:<alias>`` or deprecated bare spellings.
        Raises ValueError for invalid patterns, HTTPException(400) if unconfigured.

        ``policy`` carries the per-user Statsig routing policy (overrides,
        allow-list, search settings — see
        :mod:`minds.common.statsig.dynamic_config.model_config`). It is a plain
        Pydantic data model, so the resolver stays free of any Statsig/Context
        import; ``None`` (the empty policy) means "no policy" and every
        context-free caller is unchanged.
        """
        policy = policy or PassthroughModelStatsigConfig()
        if model_name in _DEPRECATED_ALIASES:
            config = self._resolve_alias(_DEPRECATED_ALIASES[model_name], policy=policy)
            return dataclasses.replace(config, alias=model_name)

        m = _PASSTHROUGH_PATTERN.match(model_name)
        if not m:
            raise ValueError(f"{model_name!r} is not a valid passthrough model name")

        return self._resolve_alias(m.group(1).lower(), policy=policy)

    def is_passthrough_model(self, model_name: str) -> bool:
        """Return True if model_name matches passthrough pattern."""
        if model_name in _DEPRECATED_ALIASES:
            return True
        return _PASSTHROUGH_PATTERN.match(model_name) is not None

    def list_available(
        self,
        *,
        policy: PassthroughModelStatsigConfig | None = None,
    ) -> list[PassthroughModelConfig]:
        """List all configured provider models.

        Used by ``GET /v1/models`` to advertise currently-callable models.

        ``policy.allowed_aliases`` (None = all) filters the listing to the
        aliases a user may call, and ``policy.alias_overrides`` repoints the
        advertised ``model_name`` — so the listing matches what :meth:`resolve`
        would return for that user.
        """
        from contextlib import suppress

        policy = policy or PassthroughModelStatsigConfig()
        configs: list[PassthroughModelConfig] = []
        for alias in _ALIASES:
            if policy.allowed_aliases is not None and alias not in policy.allowed_aliases:
                continue
            with suppress(HTTPException):
                configs.append(self._resolve_alias(alias, policy=policy))
        return configs

    def _resolve_alias(
        self,
        alias: str,
        *,
        policy: PassthroughModelStatsigConfig,
    ) -> PassthroughModelConfig:
        """Resolve alias to config, checking provider availability.

        Order of checks matters: an *unknown* alias is a 400 (a typo, regardless
        of who asked), a *known but disallowed* alias is a 403 (real alias, this
        user can't use it), and a known/allowed alias with no configured
        provider is a 400 (unchanged contract). An ``overrides`` entry only
        repoints the model id within an already-configured provider — it cannot
        enable a provider whose key is missing, since the availability check
        runs first. The per-user search policy is stamped onto the returned
        config for the Fireworks adapter to read.
        """
        mapping = _ALIASES.get(alias)
        if not mapping:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown passthrough alias 'latest:{alias}'",
            )

        if policy.allowed_aliases is not None and alias not in policy.allowed_aliases:
            raise HTTPException(
                status_code=403,
                detail=f"Passthrough alias 'latest:{alias}' is not available for this user.",
            )

        provider_settings = getattr(self.settings, mapping.provider)
        model_name_value = getattr(provider_settings, mapping.model_name_setting)
        api_key = getattr(provider_settings, "api_key", None)

        # Check if provider is configured: key must exist and not be "not set" sentinel
        is_configured = bool(api_key) and api_key != "not set"
        if not model_name_value or not is_configured:
            raise HTTPException(
                status_code=400,
                detail=f"No provider configured for passthrough alias 'latest:{alias}'",
            )

        # An override only repoints the model id within the already-configured
        # provider; it runs after the availability check so it can't enable a
        # provider whose key is missing.
        if alias in policy.alias_overrides:
            model_name_value = policy.alias_overrides[alias]

        provider_config = _PROVIDER_CONFIG[mapping.provider]

        api_kind = getattr(ApiKind, provider_config.api_kind)
        web_search_mode = getattr(WebSearchMode, provider_config.web_search_mode)

        base_url = None
        if provider_config.base_url_attr:
            base_url = getattr(provider_settings, provider_config.base_url_attr, None)

        return PassthroughModelConfig(
            api_kind=api_kind,
            model_name=model_name_value,
            api_key=api_key,
            base_url=base_url,
            web_search_mode=web_search_mode,
            label=provider_config.label,
            alias=alias,
            reasoning_effort=mapping.reasoning_effort,
            search_enabled=policy.search_enabled,
            search_provider_name=policy.search_provider,
        )
