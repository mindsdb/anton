"""Per-model reasoning-effort capability catalog.

Maps *concrete* upstream model ids (not aliases — Statsig ``alias_overrides``
can repoint an alias to any model id) to the discrete effort levels the model
accepts and the level its provider applies by default. Lookup is
longest-prefix match, so exact entries win over family fallbacks
(``claude-opus-4-8`` beats ``claude-opus-``), and family fallbacks give a
brand-new model version a safe baseline before anyone updates Statsig.

Levels are deliberately plain strings end to end: validation is
membership-only and the provider proxies forward the value verbatim, so a new
level name a provider ships (e.g. ``max`` on a future GPT) needs only a
Statsig ``effort_overrides`` entry — never a code change. The same applies to
new model versions; a PR is only required for a new provider/transport.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import HTTPException


@dataclass(frozen=True)
class EffortCapability:
    """Discrete effort levels a model accepts, in display order.

    An empty ``levels`` list means the model is explicitly known *not* to
    support effort (distinct from "unknown model", which has no entry at all
    — both reject requests, but explicit entries also shadow family
    fallbacks in the longest-prefix match).
    """

    levels: tuple[str, ...] = field(default=())
    default: str | None = None

    @property
    def supported(self) -> bool:
        return bool(self.levels)


_UNSUPPORTED = EffortCapability()

# Keyed by concrete-model-id prefix; longest match wins. Ordering of levels is
# preserved into the /v1/models listing so UIs can render pickers as-is.
_EFFORT_CATALOG: dict[str, EffortCapability] = {
    # --- Anthropic (output_config={"effort": ...}) ---
    "claude-fable-5": EffortCapability(("low", "medium", "high", "xhigh", "max"), "high"),
    "claude-opus-4-8": EffortCapability(("low", "medium", "high", "xhigh", "max"), "high"),
    "claude-opus-4-7": EffortCapability(("low", "medium", "high", "xhigh", "max"), "high"),
    "claude-opus-4-6": EffortCapability(("low", "medium", "high", "max"), "high"),
    "claude-opus-4-5": EffortCapability(("low", "medium", "high"), "high"),
    "claude-sonnet-4-6": EffortCapability(("low", "medium", "high", "max"), "high"),
    # Explicitly unsupported — the provider 400s on effort for these. These
    # entries shadow the family fallbacks below.
    "claude-sonnet-4-5": _UNSUPPORTED,
    "claude-haiku-": _UNSUPPORTED,
    # Family fallbacks: a safe baseline for model versions newer than this
    # table (e.g. claude-opus-4-9 on launch day, before Statsig is updated).
    "claude-fable-": EffortCapability(("low", "medium", "high", "xhigh", "max"), "high"),
    "claude-opus-": EffortCapability(("low", "medium", "high"), "high"),
    "claude-sonnet-": EffortCapability(("low", "medium", "high"), "high"),
    # --- OpenAI (reasoning={"effort": ...} on the Responses API) ---
    "gpt-5.5": EffortCapability(("none", "low", "medium", "high", "xhigh"), "medium"),
    "gpt-5.4": EffortCapability(("none", "low", "medium", "high", "xhigh"), "none"),
    "gpt-5.3-codex": EffortCapability(("low", "medium", "high", "xhigh"), "medium"),
    "gpt-5.2": EffortCapability(("none", "low", "medium", "high", "xhigh"), "none"),
    "gpt-5.1": EffortCapability(("none", "low", "medium", "high"), "none"),
    "o3": EffortCapability(("low", "medium", "high"), "medium"),
    "o4": EffortCapability(("low", "medium", "high"), "medium"),
    # Family fallback for GPT versions newer than this table.
    "gpt-5.": EffortCapability(("low", "medium", "high"), "medium"),
    # --- Fireworks (reasoning_effort via the Anthropic-compatible endpoint) ---
    "accounts/fireworks/models/deepseek-v4": EffortCapability(
        ("none", "low", "medium", "high", "xhigh", "max"), "high"
    ),
    "accounts/fireworks/models/gpt-oss": EffortCapability(("low", "medium", "high"), "medium"),
    # kimi / qwen: no documented discrete effort ladder on Fireworks for the
    # pinned models — no entry. Enable later via Statsig effort_overrides.
}


def get_effort_capability(
    model_name: str,
    overrides: dict[str, EffortCapability] | None = None,
) -> EffortCapability | None:
    """Return the effort capability for a concrete model id, or None if unknown.

    ``overrides`` (per-user Statsig ``effort_overrides``, already coerced to
    :class:`EffortCapability`) participates in the same longest-prefix match
    and wins ties against the in-code catalog, so an override can refine or
    shadow any built-in entry without a deploy.
    """
    best_prefix = ""
    best: EffortCapability | None = None
    for source in (_EFFORT_CATALOG, overrides or {}):
        for prefix, capability in source.items():
            # ``>=`` so an override with the same prefix wins (overrides are
            # iterated second).
            if model_name.startswith(prefix) and len(prefix) >= len(best_prefix):
                best_prefix = prefix
                best = capability
    return best


def validate_effort(
    effort: str,
    model_name: str,
    alias: str,
    overrides: dict[str, EffortCapability] | None = None,
) -> None:
    """Raise HTTPException(400) unless ``effort`` is valid for ``model_name``."""
    capability = get_effort_capability(model_name, overrides)
    if capability is None or not capability.supported:
        raise HTTPException(
            status_code=400,
            detail=f"Model 'latest:{alias}' does not support reasoning effort.",
        )
    if effort not in capability.levels:
        allowed = ", ".join(capability.levels)
        raise HTTPException(
            status_code=400,
            detail=(f"Invalid reasoning effort {effort!r} for model 'latest:{alias}'. Allowed values: {allowed}."),
        )
