from __future__ import annotations

from collections import defaultdict


class TimeEstimator:
    """Tracks historical skill durations and provides ETA estimates.

    In-memory only — Phase 5 adds persistence via SQLite.
    """

    def __init__(self) -> None:
        self._history: dict[str, list[float]] = defaultdict(list)

    def record(self, skill_name: str, duration_seconds: float) -> None:
        """Record an observed execution duration for a skill."""
        self._history[skill_name].append(duration_seconds)

    def estimate(self, skill_name: str) -> float | None:
        """Return the average duration for a skill, or None if no history."""
        durations = self._history.get(skill_name)
        if not durations:
            return None
        return sum(durations) / len(durations)

    def estimate_plan(self, skill_names: list[str]) -> float | None:
        """Estimate total duration for a list of skill names.

        Returns None if no skills have history. Only includes skills
        with known durations in the estimate.
        """
        total = 0.0
        has_any = False
        for name in skill_names:
            est = self.estimate(name)
            if est is not None:
                total += est
                has_any = True
        return total if has_any else None
