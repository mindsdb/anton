"""When Jev looks at the sidekick, and what its answers make the loop do.

Jev is asked every `jev_every` rounds, after `ERROR_STREAK` tool errors in a
row, and whenever the sidekick calls `finish`; at most once a round. Its
answers count only at or above `jev_threshold`:

  - `take_back` ends the delegation as `needs_main`;
  - `stuck` nudges the sidekick once, and ends it as `needs_main` the second
    time in one delegation;
  - `off_scope` points the sidekick back at the brief;
  - `done`, asked on `finish`, refuses the first `finish` Jev is confident
    does not meet the done-when condition (p(done) at or below 1 - threshold).

`take_back` and `stuck` each count one signal on the sidekick's state. At two,
the sidekick moves one step up `upgrade_ladder` the next time it compacts.

Every failure fails open: no answer, no action.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .jev_questions import DONE, OFF_SCOPE, STUCK, TAKE_BACK, sidekick_questions, sidekick_state

if TYPE_CHECKING:
    from anton.core.llm.jev import JevDecision

    from .config import ChimeraConfig

logger = logging.getLogger(__name__)

# (state, questions, trace metadata) -> the decision, or None when Jev is
# unavailable here (off, or not on MindsHub). `ChatSession._jev_decide`.
Decide = Callable[[dict, dict, dict[str, str]], Awaitable["JevDecision | None"]]

TRIGGER_EVERY = "every"
TRIGGER_ERRORS = "errors"
TRIGGER_FINISH = "finish"
# Tool errors in a row that make the next round a checked one.
ERROR_STREAK = 2
# Signals that make the next compaction move the sidekick one step up the ladder.
UPGRADE_SIGNALS = 2


@dataclass(frozen=True)
class Verdict:
    """What one check found, already held against the threshold. All False
    when there was no check or it failed."""

    take_back: bool = False
    stuck: bool = False
    off_scope: bool = False
    not_done: bool = False


class ChimeraPolicy:
    """One session's Chimera rules. ``checks`` keeps every check this session
    made, content-free, for the delegation's telemetry."""

    def __init__(self, config: "ChimeraConfig", *, decide: Decide | None = None) -> None:
        self.config = config
        self._decide = decide
        self.checks: list[dict] = []

    def trigger(self, *, round_idx: int, error_streak: int, finishing: bool) -> str | None:
        """Why this round is checked, or None when it is not."""
        if finishing:
            return TRIGGER_FINISH
        if error_streak >= ERROR_STREAK:
            return TRIGGER_ERRORS
        every = self.config.jev_every
        if every > 0 and (round_idx + 1) % every == 0:
            return TRIGGER_EVERY
        return None

    async def check(
        self,
        *,
        trigger: str,
        brief: str,
        done_when: str,
        transcript: str,
        metadata: dict[str, str] | None = None,
    ) -> Verdict:
        """Ask Jev about the sidekick's transcript. Never raises."""
        if self._decide is None:
            return Verdict()
        try:
            decision = await self._decide(
                sidekick_state(brief=brief, done_when=done_when, transcript=transcript),
                sidekick_questions(),
                {**(metadata or {}), "jev_trigger": trigger},
            )
        except Exception:  # a check must never break the delegation it watches
            logger.warning("chimera: the sidekick check raised; carrying on without it", exc_info=True)
            return Verdict()
        if decision is None:
            return Verdict()
        self.checks.append({
            "trigger": trigger,
            "error": decision.error,
            "ms": decision.ms,
            **{name: decision.p(name) for name in (TAKE_BACK, STUCK, OFF_SCOPE, DONE)},
        })
        if decision.error:
            return Verdict()
        threshold = self.config.jev_threshold

        def confident(name: str) -> bool:
            p = decision.p(name)
            return p is not None and p >= threshold

        done = decision.p(DONE)
        return Verdict(
            take_back=confident(TAKE_BACK),
            stuck=confident(STUCK),
            off_scope=confident(OFF_SCOPE),
            not_done=trigger == TRIGGER_FINISH and done is not None and done <= 1 - threshold,
        )

    def next_model(self, current: str, signals: int) -> str | None:
        """The model one step up `upgrade_ladder` from ``current`` once the
        sidekick has earned it, else None. A model not on the ladder has no
        known step up, so it stays where it is."""
        ladder = self.config.upgrade_ladder
        if signals < UPGRADE_SIGNALS or current not in ladder:
            return None
        index = ladder.index(current)
        return ladder[index + 1] if index + 1 < len(ladder) else None

    @staticmethod
    def on_main_compaction(session: object) -> None:
        """The main agent just summarized its history. v1 only records it:
        the sidekick keeps its own messages, which the summary does not touch."""
        state = getattr(session, "_sidekick_state", None)
        logger.info(
            "chimera: main agent compacted its history (sidekick=%s status=%s)",
            getattr(state, "sidekick_id", None) or "-",
            getattr(state, "status", None) or "-",
        )
