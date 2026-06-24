"""File/folder disambiguation: value types and the elicitor strategy.

When the agent cannot tell which file or folder the user means, the
``select_path`` tool offers the candidates and asks the user to pick one
*within the current turn* — the choice comes back as the tool result, so the
agent simply keeps going (no separate "I picked X" user message).

How the prompt is surfaced is a host concern, isolated behind
:class:`SelectionElicitor`:

* standalone CLI injects a terminal picker;
* the cowork-server harness injects a streaming picker that emits a GUI event
  and awaits the reply.

The tool and the agent loop depend only on the protocol — never on a concrete
transport — so the same disambiguation logic serves every host.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

__all__ = ["SelectionOption", "SelectionRequest", "SelectionElicitor"]


@dataclass(frozen=True, slots=True)
class SelectionOption:
    """One selectable candidate offered to the user.

    ``value`` is the absolute path handed back to the model on selection;
    ``label`` is what the user sees (typically the path relative to the
    project root).
    """

    value: str
    label: str
    kind: str  # "file" | "folder"
    detail: str = ""


@dataclass(frozen=True, slots=True)
class SelectionRequest:
    """A request for the user to choose a file or folder.

    Two modes:

    * ``"pick"`` — disambiguate between the concrete ``options`` (e.g. several
      files share a name). This is the default.
    * ``"browse"`` — the target is unknown/unspecified; the host shows a
      navigable browser rooted at ``root`` and the user locates the path
      themselves. ``options`` is empty in this mode.
    """

    prompt: str
    options: tuple[SelectionOption, ...] = ()
    kind: str = "any"  # "file" | "folder" | "any" — what is being chosen
    mode: str = "pick"  # "pick" | "browse"
    root: str = ""  # browse-mode start directory (absolute)


class SelectionElicitor(Protocol):
    """Strategy for surfacing a :class:`SelectionRequest` and awaiting a choice.

    A Protocol (not an ABC) on purpose: hosts satisfy it by shape, so the
    cowork-server streaming picker implements ``elicit`` without importing or
    inheriting this class. The tool depends only on the shape.
    """

    async def elicit(self, request: SelectionRequest) -> str | None:
        """Present *request*, block until the user responds.

        Returns the chosen option's ``value``, or ``None`` if the user
        cancelled / dismissed the picker without choosing.
        """
        ...
