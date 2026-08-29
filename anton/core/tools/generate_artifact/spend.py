"""Turn-budget guard for the generation pipeline (I-20).

The pipeline is ONE tool-use round from the outer agent's point of view, and
the spend ceiling is only checked BETWEEN those rounds (`session.py`'s
`_spend_ceiling_stops_the_tool_loop`). A measured run crossed the ceiling in
the middle of a single `generate_artifact` call, spent ~1.25M tokens over 35
LLM calls with nothing observing it, and only tripped the ceiling after the
money was gone.

No separate accounting lives here: `TurnCost` already sees every internal
call, because they all go through `session._llm` and the turn installs its
`usage_listener` at turn start. What was missing was a place to LOOK, and
that is all this class is.

Two levels, deliberately asymmetric:

  - wind-down is STICKY, and that state is shared: `_spend_ceiling_reached()`
    stays true forever once it flips, so re-reading it cannot distinguish
    "just crossed" from "still over". The transition is recorded here, once,
    for the whole run.
  - the stop is a COUNTER, and that counter is NOT here. It belongs to each
    write loop, because the fullstack path runs two of them concurrently and
    a shared budget would give each about one round — not enough to both emit
    a closing chunk and call `finish`.
"""

from __future__ import annotations

from dataclasses import dataclass

# Rounds EACH write loop gets to close its file after wind-down is announced.
# Two, not one: the model needs a round to emit the closing `write_file` and
# a round to call `finish`. Read by `_run_loop`, which counts down its own
# copy — see the class docstring for why this is not a field on the guard.
WIND_DOWN_ROUNDS: int = 2


@dataclass
class SpendGuard:
    """Reads the turn's spend ceiling on behalf of the generation pipeline.

    Holds exactly one piece of state — the sticky latch — because everything
    else about wind-down is per-loop.
    """

    session: object
    winding_down: bool = False

    def should_wind_down(self) -> bool:
        """True once this turn has reached its spend ceiling.

        Sticky: the first true answer latches. A session with no public probe
        (bench harness, unit-test stubs) never winds down, and a probe that
        raises is treated as "below ceiling" — a budget check must never be
        the reason a generation dies.
        """
        if self.winding_down:
            return True
        probe = getattr(self.session, "spend_ceiling_reached", None)
        if not callable(probe):
            return False
        try:
            reached = bool(probe())
        except Exception:  # noqa: BLE001 - see docstring
            return False
        if reached:
            self.winding_down = True
        return self.winding_down
