"""Chimera's knobs, read once per session from the settings or set by the host."""

from __future__ import annotations

from dataclasses import dataclass

# The cache lifetimes MindsHub accepts in `X-MindsHub-Cache-TTL`.
CACHE_TTLS = ("5m", "1h")
# What the main agent asks for under Chimera when `main_cache_ttl` is unset: it
# idles through whole delegations, which outlast the gateway's 5 minutes.
CHIMERA_MAIN_CACHE_TTL = "1h"


@dataclass(frozen=True)
class ChimeraConfig:
    """Whether the main agent may delegate, and how a delegation is bounded.

    ``max_rounds`` is a delegation's rounds when the agent names none, and
    ``max_rounds_cap`` the most it may ask for. ``jev_every`` asks Jev about
    the sidekick every that many rounds (0 or less: only after tool errors and
    on `finish`), and ``jev_threshold`` is the probability an answer needs to
    be acted on. ``upgrade_ladder`` lists sidekick models from cheapest to
    strongest; a sidekick that keeps getting stuck moves one step up it when
    it compacts (`SubAgent`).
    """

    enabled: bool
    max_rounds: int = 20
    max_rounds_cap: int = 40
    jev_every: int = 4
    jev_threshold: float = 0.8
    upgrade_ladder: tuple[str, ...] = ()

    @classmethod
    def from_settings(cls, settings: object) -> "ChimeraConfig":
        """The config the settings describe. Reads with getattr and checks
        types, so an older settings object or a MagicMock reads as disabled
        with the defaults."""
        default = cls(enabled=False)
        enabled = getattr(settings, "chimera_enabled", False) is True
        rounds = _int_or(getattr(settings, "chimera_max_rounds", None), default.max_rounds)
        every = _int_or(getattr(settings, "chimera_jev_every", None), default.jev_every)
        threshold = getattr(settings, "chimera_jev_threshold", None)
        if isinstance(threshold, bool) or not isinstance(threshold, int | float) or not 0 < threshold <= 1:
            threshold = default.jev_threshold
        return cls(
            enabled=enabled,
            max_rounds=min(max(1, rounds), default.max_rounds_cap),
            jev_every=every,
            jev_threshold=float(threshold),
        )

    def rounds_for(self, asked: object) -> int:
        """The rounds a delegation gets: what the agent asked for, or the
        default, never below 1 or above the cap."""
        rounds = _int_or(asked, self.max_rounds)
        return min(max(1, rounds), self.max_rounds_cap)


def _int_or(value: object, default: int) -> int:
    # bool is an int subclass; `true` is not a round count.
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def resolve_main_cache_ttl(settings: object, chimera: ChimeraConfig | None = None) -> str | None:
    """The cache lifetime the planning role asks MindsHub for: `main_cache_ttl`
    when set, else "1h" under Chimera, else None (the gateway's default).

    ``chimera`` is the session's config when the host passed one; without it
    the settings' own `chimera_enabled` decides.
    """
    ttl = getattr(settings, "main_cache_ttl", None)
    if isinstance(ttl, str) and ttl in CACHE_TTLS:
        return ttl
    enabled = chimera.enabled if chimera is not None else getattr(settings, "chimera_enabled", False) is True
    return CHIMERA_MAIN_CACHE_TTL if enabled else None
