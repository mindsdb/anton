"""What a yolo run asks for and what it produces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel, Field

__all__ = ["Change", "Outcome", "Progress", "ReadRequest"]


class Change(BaseModel):
    """One attempt at the whole job.

    Plan and patch are deliberately one object, and one request. Splitting
    them cost a round trip and, worse, resent every file: the contents are
    already in the conversation from the read calls, so a separate patch
    request shipped them again to learn nothing new. Measured on
    yolocoder, merging the two cut input bytes by roughly 60%.

    `summary` and `files` come before `diff` in the schema so the model
    still states its intent before writing the edit — the ordering is the
    only thing left doing the work the separate planning call used to do.
    """

    summary: str = Field(description="One line saying what the change does.")
    files: list[str] = Field(
        default_factory=list,
        description="Every file the diff modifies or creates, workspace-relative.",
    )
    diff: str = Field(
        description=(
            "The patch. Either a unified diff or the "
            "'*** Begin Patch / *** Update File:' format."
        )
    )
    need_files: list[str] = Field(
        default_factory=list,
        description=(
            "Files you must see before you can write this change. Leave empty "
            "if you have everything. Asking is always better than guessing at "
            "the contents of a file you were not shown."
        ),
    )
    need_search: list[str] = Field(
        default_factory=list,
        description=(
            "Regular expressions to find, when you do not know which file holds "
            "what you need to change. Leave empty if you know where to look."
        ),
    )


class ReadRequest(BaseModel):
    """The model asking to see files before it commits to a change."""

    paths: list[str] = Field(
        default_factory=list,
        description="Workspace-relative paths to read, from the map.",
    )
    search: list[str] = Field(
        default_factory=list,
        description=(
            "Regular expressions to find, when the file names alone do not tell "
            "you where something lives. Case-insensitive, matched per line, at "
            "most 5 patterns. Escape anything you mean literally. Avoid nested "
            "quantifiers like (x+)+ — they are cut off as runaway."
        ),
    )
    reason: str = Field(default="", description="Why these files.")


@dataclass
class Outcome:
    """What the run amounted to."""

    applied: bool
    summary: str = ""
    files: list[str] = field(default_factory=list)
    # Why it ended where it did. On failure this is the evidence to hand
    # to whatever picks the job up next, which is the whole point of
    # keeping it rather than collapsing to a bool.
    detail: str = ""
    attempts: int = 0
    # Which rung it stopped on, so a caller can tell "the model would not
    # produce a usable diff" from "the folder is not writable".
    # Spelled `status` and not `reason` on purpose. anton reserves the
    # `reason="..."` kwarg for handler failure sentinels, which are tiered
    # in core/root_cause.py as self-fixable or environment wall, and a
    # tree-wide test asserts every one of them is classified. These are
    # not that — "applied" is a success — so registering them would put
    # unrelated values into a telemetry taxonomy and skew the wall counts
    # it exists to measure.
    status: Literal["applied", "patch_failed", "no_diff", "error", ""] = ""


class Progress(Protocol):
    """Where a run says what it is doing.

    A protocol rather than a logger so the caller decides: anton streams
    these into the chat UI, tests collect them into a list, a CLI prints
    them.
    """

    def status(self, message: str) -> None: ...

    def log(self, message: str) -> None: ...
