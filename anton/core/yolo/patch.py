"""Applying a model-written diff by finding its content.

The one thing worth knowing about LLMs and diffs, learned the hard way
building yolocoder: **models get the content right and the arithmetic
wrong.** They reproduce the lines of a file faithfully and then miscount
`@@ -14,7 +14,9 @@`, start a hunk one line off, or end a hunk on its
last change with no trailing context — which `git apply` rejects outright,
because a hunk with no trailing context asserts the file ends there.

None of that bookkeeping needs the model. The file is right here. So this
module throws away every line number and count in the patch and places
each hunk by locating its text.

That single change took patch application from roughly half of attempts to
almost all of them, and it is the reason this module exists as pure
functions over strings with no dependencies: it is the valuable part, and
it should be trivially testable and liftable into any other project.

The other half of the lesson is where it *refuses*. A hunk whose context
cannot be found, or a short hunk matching in several places, is not
something to guess at — guessing means editing the wrong part of someone's
file and reporting success. Both are errors, and their messages are
written to be fed straight back to the model as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "Hunk",
    "FilePatch",
    "PatchError",
    "apply_hunks",
    "is_apply_patch_format",
    "locate",
    "parse_patch",
]


class PatchError(Exception):
    """A patch that could not be read or could not be placed.

    The message is written for the model, not only for the log: it is fed
    back as the evidence for the next attempt, so it says what was looked
    for and what was wrong rather than just that something failed.
    """


@dataclass
class Hunk:
    """One contiguous edit.

    `before` is what must be found in the file (context + removed lines,
    in order); `after` is what replaces it (context + added lines). Line
    numbers are deliberately absent — there is nowhere to put them.
    """

    before: list[str] = field(default_factory=list)
    after: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.before and not self.after


@dataclass
class FilePatch:
    path: str
    hunks: list[Hunk] = field(default_factory=list)


# Headers of OpenAI's apply_patch (V4A) format. Models trained on it reach
# for it whatever you ask for, and it happens to suit content placement
# perfectly because it carries no line numbers at all — so it is accepted
# rather than fought.
_BEGIN = "*** Begin Patch"
_END = "*** End Patch"
_UPDATE = "*** Update File:"
_ADD = "*** Add File:"
_DELETE = "*** Delete File:"


def is_apply_patch_format(patch: str) -> bool:
    """Whether the patch uses apply_patch headers rather than a unified diff."""
    for line in patch.split("\n"):
        line = line.strip()
        if line.startswith((_BEGIN, _UPDATE, _ADD, _DELETE)):
            return True
        if line.startswith(("diff --git ", "--- ")):
            return False
    return False


def parse_patch(patch: str) -> list[FilePatch]:
    """Pull the per-file hunks out of a diff, in either dialect.

    Accepting both is not indulgence. Which one a model reaches for
    depends on what it was trained on, and rejecting the other dialect
    throws away a perfectly good edit over its packaging.
    """
    patches: list[FilePatch] = []
    current: FilePatch | None = None
    active: Hunk | None = None

    # Trailing blank lines come from the patch text ending in a newline,
    # not from empty context lines. Counting them as context makes every
    # final hunk unmatchable.
    for line in patch.rstrip("\n").split("\n"):
        if line.startswith(("diff --git ", "--- ")):
            active = None
        elif line.startswith("+++ "):
            active = None
            path = _patch_path(line[len("+++ ") :])
            if not path:
                current = None
                continue
            current = FilePatch(path=path)
            patches.append(current)

        elif line.startswith((_BEGIN, _END)):
            active = None
        elif line.startswith((_UPDATE, _ADD)):
            _, _, raw = line.partition(":")
            path = _patch_path(raw)
            if not path:
                current, active = None, None
                continue
            current = FilePatch(path=path)
            patches.append(current)
            # "*** Add File:" is followed straight by its "+" lines with no
            # "@@" of its own. Start collecting immediately or the whole
            # file's contents are read as noise and nothing gets created.
            # "*** Update File:" normally does have a "@@"; the empty hunk
            # that leaves behind is dropped at the end.
            active = Hunk()
            current.hunks.append(active)
        elif line.startswith(_DELETE):
            _, _, raw = line.partition(":")
            raise PatchError(
                f"the patch deletes {raw.strip()}, which this does not do"
            )

        elif line.startswith("@@"):
            if current is None:
                raise PatchError("hunk before any file header")
            active = Hunk()
            current.hunks.append(active)
        elif active is None:
            pass  # preamble, index lines, or trailing noise
        elif line.startswith("-"):
            active.before.append(line[1:])
        elif line.startswith("+"):
            active.after.append(line[1:])
        elif line.startswith(" "):
            active.before.append(line[1:])
            active.after.append(line[1:])
        elif line == "":
            # Empty inside a hunk is an empty context line; empty at the
            # end of the patch is not. It only counts while collecting.
            active.before.append("")
            active.after.append("")
        elif line.startswith("\\"):
            pass  # "\ No newline at end of file"
        else:
            active = None

    if not patches:
        raise PatchError("no file headers in patch")

    # A hunk that collected nothing carries no instruction, and leaving one
    # in would read as "replace this file with nothing".
    for file_patch in patches:
        file_patch.hunks = [hunk for hunk in file_patch.hunks if not hunk.empty]
    return patches


def _patch_path(raw: str) -> str:
    """Turn a diff header path into a workspace-relative one."""
    raw = raw.strip()
    tab = raw.find("\t")
    if tab != -1:
        raw = raw[:tab]
    if raw == "/dev/null":
        return ""
    for prefix in ("a/", "b/"):
        if raw.startswith(prefix):
            return raw[len(prefix) :]
    return raw


def apply_hunks(content: str, hunks: list[Hunk]) -> str:
    """Apply hunks to content, placing each by its text.

    Hunks are applied in order against the running result, so two edits to
    the same file compose instead of the second being measured against a
    file that no longer looks like that.
    """
    lines = content.split("\n")
    for hunk in hunks:
        if not hunk.before:
            # A pure insertion has no context, so there is nowhere it
            # provably belongs — unless the file is empty, in which case
            # there is only one place it can go.
            if not content.strip():
                lines = list(hunk.after)
                continue
            raise PatchError("a hunk has no context to place it by")
        index = locate(lines, hunk.before)
        lines = lines[:index] + list(hunk.after) + lines[index + len(hunk.before) :]
    return "\n".join(lines)


def locate(lines: list[str], block: list[str]) -> int:
    """Find the one place block occurs in lines.

    Two refusals are deliberate. A block that cannot be found is not
    quietly skipped, and a short block found in several places is not
    resolved by taking the first — that is how an edit lands in the wrong
    function and reports success. Both raise, and both messages are
    written to be handed back to the model.
    """
    matches = _find_all(lines, block, lambda a, b: a == b)
    if not matches:
        # Fall back to ignoring indentation drift, which a model
        # reproducing a file by eye often gets slightly wrong while
        # getting every actual character right.
        matches = _find_all(lines, block, lambda a, b: a.strip() == b.strip())

    if not matches:
        raise PatchError(
            "could not find this hunk's lines in the file:\n" + _preview(block)
        )
    if len(matches) > 1 and len(block) < _AMBIGUOUS_BELOW:
        raise PatchError(
            f"this hunk's lines appear {len(matches)} times, "
            f"too ambiguous to place:\n{_preview(block)}"
        )
    return matches[0]


# Below this many lines, a block matching more than once is treated as
# ambiguous rather than resolved by position. Three is where a match stops
# being a coincidence in practice.
_AMBIGUOUS_BELOW = 3


def _find_all(lines: list[str], block: list[str], equal) -> list[int]:
    matches = []
    for start in range(len(lines) - len(block) + 1):
        if all(equal(lines[start + offset], want) for offset, want in enumerate(block)):
            matches.append(start)
    return matches


def _preview(block: list[str]) -> str:
    return "\n".join(block[:4])
