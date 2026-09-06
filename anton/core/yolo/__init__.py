"""Yolo mode: edit files with a diff instead of a program that writes one.

Anton's existing way to change an artifact is the scratchpad — the model
writes Python that writes the file. That works, and for anything
generative it is the right tool. For *modifying* a file that already
exists it is a long way round: the model reasons about string surgery,
writes a program to perform it, and the program is a second thing that can
be wrong. A one-line title change becomes a script.

Yolo mode is the short way. The model is shown the folder, asks for the
files it needs, and returns a diff. The diff is applied here, by finding
each hunk's text in the file — no line numbers involved.

What was learned building yolocoder, and what this preserves:

* **Models get diff content right and diff arithmetic wrong.** Line
  numbers, hunk counts and trailing context are the failure, not the
  edit. So all of it is ignored and hunks are placed by their text.
  This is the single highest-value idea here.
* **Models pick files well.** Given a listing of thirty paths they ask
  for the right two or three. That judgement is worth one cheap call and
  is not worth second-guessing.
* **Plan and patch belong in one call.** Split, the second call resends
  every file to learn nothing.
* **Failure evidence must include the model's own output.** Told only
  "it failed", a model reproduces the same diff.
* **Refusing beats guessing.** An unfindable hunk, or a short one
  matching in three places, is an error — not a reason to pick the
  first match and edit the wrong function.

`patch.py` and `workspace.py` import nothing at all — they are pure
functions over strings, which is what makes the engine provable in
milliseconds. `agent.py` uses anton's own `LLMClient` directly, so a yolo
run gets the configured provider, the coding-model split, forced
`tool_choice`, pydantic validation and turn tracing for free, and the
handler passes the same client the rest of the agent already holds.

    from anton.core.yolo import Workspace, YoloEditor

    editor = YoloEditor(workspace=Workspace(folder), llm_client=llm_client)
    outcome = await editor.edit("rename the title to TicTacTris")
    if not outcome.applied:
        ...  # outcome.detail says why; fall back to the scratchpad
"""

from anton.core.yolo.agent import MAX_ATTEMPTS, YoloEditor, apply_patch_text
from anton.core.yolo.models import Change, Outcome, Progress, ReadRequest
from anton.core.yolo.patch import (
    FilePatch,
    Hunk,
    PatchError,
    apply_hunks,
    is_apply_patch_format,
    locate,
    parse_patch,
)
from anton.core.yolo.workspace import Workspace, WorkspaceError

__all__ = [
    "MAX_ATTEMPTS",
    "Change",
    "FilePatch",
    "Hunk",
    "Outcome",
    "PatchError",
    "Progress",
    "ReadRequest",
    "Workspace",
    "WorkspaceError",
    "YoloEditor",
    "apply_hunks",
    "apply_patch_text",
    "is_apply_patch_format",
    "locate",
    "parse_patch",
]
