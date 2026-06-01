"""Passthrough alias resolution and backward-compatibility shims.

For documentation and implementation details, see :mod:`minds.inference.model_resolver`.

This module provides backward-compatible exports of model resolution functions that
delegate to :class:`minds.inference.model_resolver.ModelResolver`.
"""

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel

from minds.common.settings.app_settings import get_app_settings


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
    for log-grep continuity. ``alias`` is the bare alias body the resolver
    captured (e.g. ``"sonnet"`` for ``"latest:sonnet"``) — preserved here so
    observability can record it as metadata without re-parsing the request
    model. For OpenAI reasoning-tier aliases like ``latest:gpt-medium``,
    ``reasoning_effort`` holds the effort level baked in by the alias;
    clients never set it.
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


# Backward-compatibility shims that delegate to ModelResolver
def is_passthrough_model(model: str) -> bool:
    """Return True if *model* is a passthrough name.

    Matches the canonical ``latest:<alias>`` pattern or one of the deprecated
    bare spellings.

    Delegates to :meth:`minds.inference.model_resolver.ModelResolver.is_passthrough_model`.
    """
    from minds.inference.model_resolver import ModelResolver

    resolver = ModelResolver(get_app_settings())
    return resolver.is_passthrough_model(model)


def resolve_passthrough_model(model: str) -> PassthroughModelConfig:
    """Resolve a passthrough model name to a fully-populated config.

    Accepts either the canonical ``latest:<alias>`` form or one of the
    deprecated bare spellings. Unknown aliases raise ``HTTPException(400)``.
    Likewise, when an alias's priority list has no entry whose key is
    configured we raise 400.

    Delegates to :meth:`minds.inference.model_resolver.ModelResolver.resolve`.
    """
    from minds.inference.model_resolver import ModelResolver

    resolver = ModelResolver(get_app_settings())
    return resolver.resolve(model)


def list_available_passthrough_models() -> list[PassthroughModelConfig]:
    """Resolve every alias whose provider is configured.

    Used by ``GET /v1/models`` to advertise currently-callable models.
    Deprecated aliases are excluded by construction.

    Delegates to :meth:`minds.inference.model_resolver.ModelResolver.list_available`.
    """
    from minds.inference.model_resolver import ModelResolver

    resolver = ModelResolver(get_app_settings())
    return resolver.list_available()
