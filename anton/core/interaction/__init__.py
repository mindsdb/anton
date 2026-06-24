"""Mid-turn human interaction primitives for the agent loop.

Currently this package hosts file/folder disambiguation (the ``select_path``
tool). The agent depends only on the abstract :class:`SelectionElicitor`
strategy; each host (CLI, cowork-server harness, …) supplies a concrete
implementation, so the core never learns how the prompt is surfaced.
"""

from anton.core.interaction.selection import (
    SelectionElicitor,
    SelectionOption,
    SelectionRequest,
)

__all__ = ["SelectionElicitor", "SelectionOption", "SelectionRequest"]
