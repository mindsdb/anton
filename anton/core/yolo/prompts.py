"""The instructions, and why each paragraph is there.

Every rule below was added because something specific went wrong. They are
kept in one place, with the reason attached, so that nobody trims one for
being wordy without knowing which failure it is holding back.
"""

from __future__ import annotations

__all__ = ["CHANGE_INSTRUCTIONS", "READ_INSTRUCTIONS", "render_task"]


# Read selection. Models are genuinely good at this — shown thirty paths
# they ask for the right two or three — so the instruction is short and
# mostly about not over-reading.
READ_INSTRUCTIONS = """\
You are about to make one change to a folder, and you are looking at its file listing.

Name the files you need to see to make the change correctly. Ask for the ones you will edit
and the ones you must not break: a file that imports what you are changing, a config that
names it, a test that covers it.

If the file names do not tell you where something lives, put a regular expression in search
instead of guessing at paths. Searching costs nothing — it does not call a model — so a query is
always cheaper than reading a file on the off-chance. Patterns are case-insensitive and matched
a line at a time; escape anything you mean literally, and keep them simple, because a pattern
that backtracks badly is cut off rather than waited for.

Do not ask for everything. Reading a file you do not need costs the room you will want for
the edit itself. Three or four files is usually right; more than eight almost never is.
Return an empty list only if the listing alone is genuinely enough."""


# The change itself. Every paragraph after the first is scar tissue.
CHANGE_INSTRUCTIONS = """\
You are making one change to a folder. You have been shown its file listing and the full
contents of the files you asked for.

Answer with the change: a one-line summary, every file it touches, and the diff that makes it.
Make the smallest complete change that does the job.

The diff may be a unified diff or the "*** Begin Patch / *** Update File:" format. Either is
read by matching its text against the file, so line numbers, @@ headers and hunk counts are
IGNORED and do not need to be correct. Do not spend effort on them.

What must be exact is the text. Copy every context line and every removed line from the file
character for character: the same indentation, the same quotes, the same HTML entities
(&amp; stays &amp;), the same trailing spaces. A line that differs by one character cannot be
found, and the hunk it belongs to will not apply.

Surround each change with a few unchanged lines above and below, so there is exactly one place
in the file it could go. A hunk with one line of context that appears in twenty places will be
refused as ambiguous rather than applied to the wrong one.

A file that does not exist yet is created by the same diff: use "*** Add File: <path>" followed
by every line of its contents, each prefixed with "+", or a unified diff whose header is
"--- /dev/null". Every file you list must actually appear in the diff with its full contents.
Naming a file you meant to add, or describing it instead of including it, leaves it uncreated
and the change half-done.

Some folders carry generated data as "<name>.data.js" with a "<name>.schema.json" sidecar
beside it. The schemas are shown to you in full; the data files never are, because they are far
too large. Never write a diff against a .data.js file — it is produced by something else and
editing it by hand is not a change anyone can review. To use the data, read its schema and write
code against the global variable the schema names.

If you find you cannot write the change because you have not seen the right file, say so
instead of guessing: put the paths in need_files, or — if you do not know which file it is —
put a regular expression in need_search, and leave diff empty. You will be shown what
was found and asked again. A guessed diff against a file you were never shown cannot apply.

Do not delete files. Do not touch anything outside the folder."""


def render_task(
    task: str,
    file_map: str,
    contents: str = "",
    background: str = "",
    evidence: str = "",
    found: str = "",
) -> str:
    """Lay out one request.

    The order is not cosmetic. It runs from the part that never changes to
    the part that changes every attempt — background, then the map, then
    the file contents, then the task, and the failure evidence last. A
    provider's prefix cache can only reuse an unchanged head, so anything
    stable that sits behind something volatile is paid for again on every
    retry. On a three-attempt repair loop that is most of the bill.
    """
    blocks = []
    if background.strip():
        blocks.append(f"BACKGROUND:\n{background.strip()}")
    blocks.append(f"THE FOLDER:\n{file_map}")
    if found.strip():
        blocks.append(f"SEARCH RESULTS:\n{found.strip()}")
    if contents.strip():
        blocks.append(f"FILE CONTENTS:\n{contents.rstrip()}")
    blocks.append(f"TASK:\n{task.strip()}")
    if evidence.strip():
        blocks.append(f"WHAT WENT WRONG LAST TIME:\n{evidence.strip()}")
    return "\n\n".join(blocks)
