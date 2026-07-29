"""Mid-turn human interaction primitives for the agent loop.

The agent depends only on the abstract :class:`Elicitor` strategy and the
shared :func:`elicit` entry point; each host (CLI, cowork-server harness, …)
supplies a concrete implementation, so the core never learns how a question
is surfaced.
"""

from anton.core.interaction.elicit import (
    AskAnswer,
    AskOption,
    AskRequest,
    Elicitor,
    elicit,
    validate_request,
)
from anton.core.interaction.emitter import TurnEmitter

__all__ = [
    "AskAnswer",
    "AskOption",
    "AskRequest",
    "Elicitor",
    "TurnEmitter",
    "elicit",
    "validate_request",
]
