"""
Pydantic model for the passthrough routing policy (Statsig dynamic config).

Represents the shape of the ``passthrough-model-config`` dynamic config that
controls, per user, how ``latest:*`` aliases resolve and which web-search
backend (if any) Fireworks-hosted models use. Kept here in ``schemas/`` rather
than in ``minds.common.passthrough_config`` so the resolver stays free of any
Statsig/Context import (see that module's docstring). ``schemas/limits.py`` is
the precedent for a Statsig-shaped config model.
"""

from pydantic import BaseModel, ConfigDict, Field

from minds.inference.effort import EffortCapability
from minds.inference.types import PassthroughAlias


class EffortOverride(BaseModel):
    """Statsig-shaped override of one effort-catalog entry.

    ``levels`` is an ordered list of opaque level strings (forwarded verbatim
    upstream), so a brand-new provider level needs only a config edit. An
    empty list disables effort for the matched models (kill switch).
    """

    model_config = ConfigDict(extra="ignore")

    levels: list[str] = Field(default_factory=list)
    default: str | None = None

    def to_capability(self) -> EffortCapability:
        return EffortCapability(tuple(self.levels), self.default)


class PassthroughModelStatsigConfig(BaseModel):
    """Per-user passthrough routing policy.

    All fields default to the "no policy" value so an empty config (self-hosted
    or Statsig fail-open) reproduces today's env-backed behavior exactly:
    no overrides, every alias allowed, the env-configured search provider, and
    search enabled.

    Alias-keyed fields are typed with :class:`PassthroughAlias` rather than bare
    strings so a typo in the Statsig console fails closed (Pydantic raises and
    ``get_passthrough_model_config`` falls back to the empty policy) instead of
    silently routing to a non-existent alias.
    """

    # ``extra="ignore"`` so unknown keys an admin adds in the Statsig console
    # never break parsing — keeps the layer fail-open friendly.
    model_config = ConfigDict(extra="ignore")

    alias_overrides: dict[PassthroughAlias, str] = Field(
        default_factory=dict,
        description="Map of alias (e.g. PassthroughAlias.OPUS) → upstream model id, overriding the env default.",
    )
    allowed_aliases: list[PassthroughAlias] | None = Field(
        default=None,
        description="Allow-list of callable aliases. None/absent = all allowed; a list restricts to those.",
    )
    search_provider: str | None = Field(
        default=None,
        description="Search provider name override. None/absent = use the env-configured SEARCH__PROVIDER.",
    )
    search_enabled: bool = Field(
        default=True,
        description="Per-user web-search kill switch. False disables external search entirely for the user.",
    )
    effort_overrides: dict[str, EffortOverride] = Field(
        default_factory=dict,
        description=(
            "Map of concrete model-id prefix → effort override. Merged over the in-code effort catalog "
            "(longest prefix wins, override beats catalog on ties) so new models / new effort levels ship "
            "via this config without a deploy."
        ),
    )

    def effort_capabilities(self) -> dict[str, EffortCapability]:
        """Project ``effort_overrides`` into the shape the effort catalog consumes."""
        return {prefix: override.to_capability() for prefix, override in self.effort_overrides.items()}
