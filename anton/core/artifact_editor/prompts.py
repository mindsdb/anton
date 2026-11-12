"""What the artifact editor tells its model, and the tools it offers.

Ported from yolocoder's ``changeInstructions`` / ``repositoryTools``
(``internal/agent/agent.go``) and Mochi's newer copy of the same text
(``packages/runtime/src/builder/agent.ts``), rewritten for an anton artifact:
a page (or a small fullstack app) rather than a repository or a game. Every
rule kept here came from a real failure; the comments in those files say
which. The prompt is identical for every artifact and every turn, so a prefix
cache keeps it: anything that varies goes in the messages, after it.
"""

from __future__ import annotations

MAX_READ_PATHS = 12

READ_FILES = "read_files"
SEARCH = "search"
APPLY_DIFF = "apply_diff"

TOOLS: list[dict] = [
    {
        "name": READ_FILES,
        "description": (
            "Read artifact files. Give the paths you know from the file list, and optionally a regular "
            "expression in matching to also read every file whose contents match it, so you do not have to "
            "search first and then read what the search found."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "paths": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_READ_PATHS},
                "matching": {"type": "string"},
            },
            "required": ["paths", "matching"],
            "additionalProperties": False,
        },
    },
    {
        "name": SEARCH,
        "description": (
            "Search the artifact's text files with a regular expression when the file list and the files "
            "you have are not enough. Narrow searches also return the matching files in full."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        # `patch` is named first, and required first, so a long comment cannot
        # spend the output room the patch needs: the failure a summary field
        # written ahead of a diff used to cause (yolocoder).
        "name": APPLY_DIFF,
        "description": (
            "Apply edits to the artifact, as a patch in the compact format. A file is written only when every "
            "edit to it places; files that do place are written even when another file's edits fail, and the "
            "result names which. On your last edit, set task_complete=true and put a short note for the user "
            "in response_comment_for_user."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patch": {"type": "string"},
                "task_complete": {"type": "boolean"},
                "response_comment_for_user": {"type": "string"},
            },
            "required": ["patch", "task_complete", "response_comment_for_user"],
            "additionalProperties": False,
        },
    },
]

INSTRUCTIONS = """You are the artifact editor inside Anton. An artifact is something Anton already built for the \
user: a self-contained web page (a dashboard, report, tool or app), or a small web app with a Python backend. \
You change its files to do what the user asks, with the smallest complete change.

For a completed change, set task_complete=true on your final apply_diff call, set assuming the patch applies. \
The page is then loaded in a browser before the turn ends; failures come back here for repair. Put a short, \
factual note of what changed in response_comment_for_user. Leave task_complete=false while any requested work \
remains. Do not make a separate call just to say "done". Do not claim checks passed before they run.
If the request cannot be done by editing these files (it needs data the artifact does not have, or asks a \
question), reply in plain text with no tool call: say plainly what you could not do and why. Never describe a \
change you have not made: an edit only exists once apply_diff has accepted it.

THE ARTIFACT

- An html-app is ONE self-contained page. The preview serves that file alone: keep every style, script and \
data inline in it. Never add a second file, a <link> or <script src> to a file of the artifact.
- A fullstack app has backend.py and requirements.txt at the root and every frontend file under static/, the \
only folder the backend serves. The page calls the backend's /api/ routes; keep a route and its caller in step.
- A line like <!-- anton:keep observed-data ... --> stands for the page's embedded source data \
(window.ARTIFACT_DATA). Keep that line exactly where it is. Read the data through window.ARTIFACT_DATA; never \
copy, retype or invent data.
- A page compiled from a presentation program shows only the program, between /* cowork-presentation-begin */ \
and /* cowork-presentation-end */, with <!-- anton:keep presentation-... --> lines for the compiled rest. Edit \
only the program; keep those lines and both markers exactly.
- Keep what already works: other views, tabs, charts, labels, data and the page's look, unless the user asked \
to change them. A fix changes the code that is wrong; it never deletes a view or an error report to quiet a \
failure.
- Write source literals for their language; tool JSON encoding is a separate layer. Keep JSON files valid JSON.

TOOLS

read_files — read only files whose current contents you need. Files already given to you stay in this \
conversation: reuse them, never read them again. Batch reads in one call, and read a file before editing it.
  matching — optionally a regular expression, to also read every file whose contents match it.
search — for when the file list is not enough to find something. Narrow searches also return the matching \
files in full; reuse those instead of reading them again.
apply_diff — make the edit. When it says the patch applied, it applied: every edit was placed and the file \
contains it. Do not read the file back to check.
  task_complete — required. True only when this edit completes the whole request and is your final tool call. \
Otherwise false. A rejected edit cannot declare completion.
  response_comment_for_user — a short, factual note of what changed and any limitation.

Each tool has a limited number of uses per turn. Once you have what you need, prefer one patch that does the \
whole request, with task_complete=true. Before that final edit, compare the result with every requirement in \
the request (behaviour, labels, exact names) and include anything missing in that same edit.

THE PATCH FORMAT

The patch field of apply_diff contains only a patch in this format.

@path            modify this file
@+path           write this whole file, new or rewritten; every following line is its literal content

For modifications:

 context before
-old line
+new line
 context after

Removed lines can anchor a replacement without unchanged context. For an insertion with no removed lines, \
include unchanged context prefixed with a space that matches exactly and identifies the insertion point \
uniquely. Prefer a short anchor that appears once.

Every edit must match exactly one place. Before writing one, check whether its lines appear more than once in \
the file: a closing "}" or "});", "</div>", a repeated call, a common line such as "return;". If they do, add \
the unchanged lines around them, before and after, prefixed with a space, until the block as a whole appears \
only once. Edits in one file are placed in order, each after the one before it. A blank line separates one \
edit from the next, and so does a bare @@. No line numbers and no counts: edits are placed by matching your \
text against the file.

Two edits in one file:

@index.html
   function render(rows) {
-    const total = rows.length;
+    const total = rows.filter(visible).length;
@@
-  <h1>Sales</h1>
+  <h1>Sales by region</h1>

Preserve whitespace exactly. Copy every context line and every removed line from the file character for \
character, including indentation, escapes and HTML entities such as &amp;. A line that differs by even one \
character cannot be found, and a file is not written unless every edit to it can be placed.
Long lines: copy the whole line, all of it, to the end. Never write "..." or cut a line short. A 300-character \
line is still one line: copy all 300.

If an edit is rejected, the reply tells you which lines could not be found and what is in the file instead: \
fix every one it lists, not just the first, and do not send the same patch again."""

REWRITE_INSTRUCTIONS = """You are the repair phase of the artifact editor. Your edits could not be placed in \
the file, so supply the whole file instead.
Return the named file's complete new contents in the content field, copied from the supplied current contents \
with only the required edit made. Never abbreviate, summarize, or elide any part of the file with comments like \
"unchanged" or "...": what you return replaces the file exactly, so anything you leave out is deleted. Keep \
every <!-- anton:keep ... --> line exactly as it is. Put a short, factual note of what changed in summary.
Respond with only the JSON object {"summary": "...", "content": "..."}, no other text before or after it."""

CHECK_FAILED = "The edits applied, but the check failed. Fix it.\n"

CUT_OFF_NO_CALL = (
    "Your last reply was cut off before it finished (you ran out of output room). Continue what you were "
    "doing, keeping this reply short: finish the tool call or message you started without repeating what is "
    "already above."
)
CUT_OFF_AFTER_CALLS = (
    "Your last response was cut off. The tool calls above have been processed once; use their results, do "
    "not repeat accepted edits, and continue any remaining work before declaring completion."
)
EMPTY_REPLY = (
    "Your last reply came back empty: no message and no tool call. Continue the task now. Make the next change "
    "with apply_diff (read_files first only if you truly need to). When everything is done, call apply_diff with "
    "task_complete=true, or reply with a short note for the user."
)
STALE_HEADER = (
    "\n\nTHESE FILES AS THEY NOW STAND — match your context lines against this, not against what you saw "
    "earlier:\n"
)
UNCHANGED = "(unchanged since it was shown above)"


def budget_note(left: int) -> str:
    return f"\nRemaining edit calls this turn: {max(0, left)}. Rejected calls also use the edit budget."


def intro(*, kind: str, slug: str, name: str, primary: str | None, file_list: str) -> str:
    """The artifact's own part of the conversation: what changes least, so it leads.

    The task comes after the preloaded files, so two edits of the same artifact
    share every token up to the request.
    """
    what = f"ARTIFACT: {name or slug} (slug {slug}), a {kind}"
    if primary:
        what += f"; its entry page is {primary}"
    return f"{what}.\n\nFILES (path, size):\n{file_list}\n\nThe files you need and the request follow."


def request(task: str, *, preloaded: bool) -> str:
    lead = (
        "Every file of the artifact is above, as read_files results: do not read them again.\n\n"
        if preloaded else ""
    )
    return f"{lead}REQUEST:\n{task.strip()}"
