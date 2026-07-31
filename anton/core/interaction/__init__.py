"""Mid-turn human interaction primitives for the agent loop.

The agent depends only on the abstract :class:`Elicitor` strategy and the
shared ``elicit()`` entry point; each host (CLI, cowork-server harness, …)
supplies a concrete implementation, so the core never learns how a question
is surfaced.

``elicit`` itself is deliberately NOT re-exported here: it would shadow the
same-named submodule, so ``import anton.core.interaction.elicit as e`` would
bind the function instead of the module. Import it from
``anton.core.interaction.elicit``.
"""

from anton.core.interaction.elicit import (
    AskAnswer,
    AskOption,
    AskRequest,
    Elicitor,
    validate_request,
)
from anton.core.interaction.emitter import TurnEmitter

__all__ = [
    "AskAnswer",
    "AskOption",
    "AskRequest",
    "Elicitor",
    "TurnEmitter",
    "validate_request",
]
