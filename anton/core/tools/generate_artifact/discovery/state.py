"""Backwards-compatible names for the discovery phases' state.

There is one state object for the whole pipeline (`GenState`). This module
keeps the old names pointing at it so the phase modules and their tests did
not all have to be rewritten in the same commit as the state merge itself.
"""

from __future__ import annotations

from ..state import (  # noqa: F401
    PHASE2_RESERVED_QUESTIONS,
    GenState as PrdState,
    gathering_question_budget,
)
