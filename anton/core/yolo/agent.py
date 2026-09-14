"""The yolo loop: pick files, write one diff, place it, repair, give up well.

Shape of a run, and the reasoning behind each step:

    map ──▶ pick files ──▶ read ──▶ plan + diff ──▶ apply ─┬─▶ done
             + search                  (1 call)            │
             (1 call)                     ▲                │
                                          └── evidence ◀───┘
                                             widen / search
                                              (up to MAX_ATTEMPTS)

Two LLM calls for a change that lands, and the second one is the only one
that repeats. Everything else — searching, locating hunks, checking that
promised files exist, deciding whether a match is ambiguous — is
arithmetic done here, because it is arithmetic and models are bad at it.

Searching in particular costs no model call at all, which is why the
model may ask for it freely: at the pick step when file names give
nothing away, and again on a failed attempt when it turns out to have
been looking in the wrong file.

Giving up is a designed step, not a fallthrough. When the diff will not
apply after MAX_ATTEMPTS, the outcome carries the last failure verbatim so
the caller can hand the job to something else with the diagnosis attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from anton.core.llm.client import LLMClient
from anton.core.yolo.models import Change, Outcome, Progress, ReadRequest
from anton.core.yolo.patch import PatchError, apply_hunks, parse_patch
from anton.core.yolo.prompts import (
    CHANGE_INSTRUCTIONS,
    READ_INSTRUCTIONS,
    render_task,
)
from anton.core.yolo.workspace import (
    SCHEMA_SUFFIX,
    Workspace,
    WorkspaceError,
    is_generated_data,
)

__all__ = ["MAX_ATTEMPTS", "YoloEditor", "apply_patch_text"]


# How many diffs to accept before handing the job on. Three is not
# arbitrary: in yolocoder, a diff that fails three times with the file
# contents already in front of the model is almost never fixed by a
# fourth — the model is wrong about the file, not careless, and more
# attempts spend tokens to arrive at the same place.
MAX_ATTEMPTS = 3

# A read budget. Models pick files well but will happily ask for the whole
# folder if the folder is small enough to seem free.
MAX_READS = 12


class _Quiet:
    def status(self, message: str) -> None: ...
    def log(self, message: str) -> None: ...


@dataclass
class YoloEditor:
    """Makes one change to one folder."""

    workspace: Workspace
    llm_client: LLMClient
    progress: Progress = field(default_factory=_Quiet)
    max_attempts: int = MAX_ATTEMPTS

    async def edit(self, task: str, background: str = "") -> Outcome:
        """Make the change, or come back saying exactly why not."""
        try:
            return await self._edit(task, background)
        except WorkspaceError as error:
            return Outcome(applied=False, detail=str(error), status="error")

    async def _edit(self, task: str, background: str) -> Outcome:
        file_map = self.workspace.map()
        self.progress.status("Looking at the folder...")
        self.progress.log(f"  {len(self.workspace.files())} files")

        wanted, found = await self._pick_files(task, file_map, background)
        contents = ""
        if wanted:
            self.progress.log("  reading " + ", ".join(wanted))
            contents = self.workspace.read_many(wanted)

        evidence = ""
        change: Change | None = None

        for attempt in range(1, self.max_attempts + 1):
            self.progress.status(
                "Working out the change..." if attempt == 1 else "Trying again..."
            )
            change = await self.llm_client.generate_object_code(
                Change,
                system=CHANGE_INSTRUCTIONS,
                messages=[
                    {
                        "role": "user",
                        "content": render_task(
                            task, file_map, contents, background, evidence, found
                        ),
                    }
                ],
            )
            if attempt == 1:
                self.progress.log(f"  plan: {change.summary}")

            if not change.diff.strip():
                # "I need to see more first" is a legitimate answer, not a
                # dead end. Honour it rather than treating an empty diff as
                # a failure to produce one.
                widened, extra = self._widen(wanted, change)
                if (widened != wanted or extra) and attempt < self.max_attempts:
                    wanted, contents = widened, self.workspace.read_many(widened)
                    found = _join(found, extra)
                    evidence = _asked_note(change.need_files, change.need_search)
                    self.progress.log(
                        "  asked for "
                        + ", ".join(change.need_files + change.need_search)
                    )
                    continue
                # Otherwise there is nothing to apply and nothing to repair
                # from; another attempt asks the same question again.
                return Outcome(
                    applied=False,
                    summary=change.summary,
                    detail="the model returned no diff",
                    attempts=attempt,
                    status="no_diff",
                )

            self.progress.status("Applying the patch...")
            try:
                written = apply_patch_text(self.workspace, change.diff)
            except (PatchError, WorkspaceError) as error:
                self.progress.log(f"  patch did not apply: {error}")
                # A wrong first pick is the one failure the repair note
                # cannot fix. Telling a model to copy the lines more
                # carefully is useless advice about a file it was never
                # shown, and without widening the read set it will rewrite
                # the same doomed diff until the attempts run out. So when
                # the patch named a file we did not read, that — not the
                # careless-copying lecture — is the evidence to send back.
                widened, extra = self._widen(wanted, change)
                fresh = [path for path in widened if path not in wanted]
                if fresh or extra:
                    if fresh:
                        self.progress.log("  reading " + ", ".join(fresh) + " and retrying")
                    wanted, contents = widened, self.workspace.read_many(widened)
                    found = _join(found, extra)
                    evidence = _unseen_note(fresh, str(error))
                else:
                    evidence = _repair_note(str(error), change.diff)
                continue

            # The model said which files it would touch. That claim is
            # worth checking, because a patch that quietly omits the new
            # file it promised applies perfectly and looks like success.
            missing = [
                path
                for path in change.files
                if not self.workspace.exists(path) and path not in written
            ]
            if missing:
                self.progress.log(f"  but {', '.join(missing)} was not created")
                evidence = _missing_note(missing)
                # This patch did apply, so the files on disk are no longer
                # the ones quoted in the prompt. Re-read them, or the next
                # attempt writes a diff against a version that is gone and
                # fails for a reason that has nothing to do with the task.
                wanted = _merge(wanted, sorted(written))
                contents = self.workspace.read_many(wanted)
                continue

            self.progress.log(f"  wrote {', '.join(sorted(written))}")
            return Outcome(
                applied=True,
                summary=change.summary,
                files=sorted(written),
                attempts=attempt,
                status="applied",
            )

        return Outcome(
            applied=False,
            summary=change.summary if change else "",
            files=list(change.files) if change else [],
            detail=evidence,
            attempts=self.max_attempts,
            status="patch_failed",
        )

    def _widen(self, wanted: list[str], change: Change) -> tuple[list[str], str]:
        """Widen the read set from what the model named or asked to find.

        This is the bounded stand-in for a read/search tool loop. The loop
        in yolocoder was not really about the first pick — models are good
        at that — it was about recovering when the first pick was wrong.
        Removing it left nothing to recover with.

        Rather than reopen an unbounded loop, the read set widens only in
        reaction to a failure, only to files the model itself named or
        found, and only within the attempts already budgeted. Searching is
        free — it calls no model — so it costs nothing to answer.

        Returns the new read set and any search results to show.
        """
        widened = list(wanted)
        rendered, hits = "", []
        if change.need_search:
            rendered, hits = self.workspace.search_many(change.need_search)
        for path in list(change.need_files) + hits + list(change.files):
            path = path.strip().lstrip("./")
            if path and path not in widened and self.workspace.exists(path):
                widened.append(path)
        return widened[:MAX_READS], rendered

    async def _pick_files(
        self, task: str, file_map: str, background: str
    ) -> tuple[list[str], str]:
        """Ask which files the change needs, and search for what it cannot name.

        This is the step models are reliably good at, which is why it gets
        its own cheap call instead of a tool loop: one request, a list of
        names, done. Anything they name that is not in the folder is
        dropped here rather than becoming a confusing read error.

        Where names are not enough — forty files and no clue which one
        sets the title — the same call can ask for a search instead. The
        search itself is deterministic and costs no model call, so the
        whole discovery step stays at one request. Files that match are
        added to the read set, and the matching lines are shown so the
        model can see why they are there.
        """
        self.progress.status("Choosing what to read...")
        request = await self.llm_client.generate_object_code(
            ReadRequest,
            system=READ_INSTRUCTIONS,
            messages=[
                {
                    "role": "user",
                    "content": render_task(task, file_map, background=background),
                }
            ],
        )
        rendered, hits = "", []
        if request.search:
            self.progress.log("  searching for " + ", ".join(request.search))
            rendered, hits = self.workspace.search_many(request.search)

        known = {info.path for info in self.workspace.files()}
        wanted, seen = [], set()
        for path in list(request.paths) + hits:
            path = path.strip().lstrip("./")
            if path in known and path not in seen:
                seen.add(path)
                wanted.append(path)
        return wanted[:MAX_READS], rendered


def apply_patch_text(workspace: Workspace, diff: str) -> set[str]:
    """Apply a whole patch, all files or none.

    Every file's new contents are worked out before anything is written,
    so a hunk that will not place in the third file does not leave the
    first two edited and the change half-done. Returns the paths written.
    """
    patches = parse_patch(diff)
    updated: dict[str, str] = {}

    for file_patch in patches:
        if not file_patch.hunks:
            continue
        # A patch may touch one file in several blocks — models happily
        # emit two "*** Begin Patch" sections for the same path. Each has
        # to build on the last, or the final block silently discards
        # everything before it.
        if file_patch.path in updated:
            current = updated[file_patch.path]
        elif workspace.exists(file_patch.path):
            current = workspace.read(file_patch.path)
        else:
            current = ""  # a new file: hunks are pure insertions
        try:
            updated[file_patch.path] = apply_hunks(current, file_patch.hunks)
        except PatchError as error:
            raise PatchError(f"{file_patch.path}: {error}") from error

    if not updated:
        raise PatchError("the patch changed nothing")

    # Generated data belongs to whatever produced it. A diff against two
    # megabytes of rows is not an edit anyone reviewed, and regenerating
    # the file is both correct and cheaper. Enforced rather than merely
    # instructed, for the same reason an ambiguous hunk is refused: the
    # rule is checkable here, so it should not depend on the model
    # remembering it.
    generated = sorted(path for path in updated if is_generated_data(path))
    if generated:
        raise PatchError(
            f"{', '.join(generated)} is generated data and is not edited by hand. "
            f"Change whatever produces it and let it be written again. To use it, "
            f"read its {SCHEMA_SUFFIX} sidecar and write code against the global it names."
        )

    for path, content in updated.items():
        workspace.write(path, content)
    return set(updated)


def _merge(existing: list[str], extra: list[str]) -> list[str]:
    """Add paths to the read set, keeping order and dropping duplicates."""
    merged = list(existing)
    for path in extra:
        if path not in merged:
            merged.append(path)
    return merged[:MAX_READS]


def _repair_note(error: str, diff: str) -> str:
    """What to tell the model after a diff would not apply.

    It gets both the complaint and the diff back. Without seeing its own
    output a model has no way to tell what was wrong with it, and
    reproduces it almost verbatim — the single most common way a repair
    loop burns three attempts achieving nothing.
    """
    return (
        "Your diff did not apply. Hunks are placed by finding their text in the file, so "
        "line numbers were not the problem — the context or removed lines did not match "
        "the file exactly. Compare the lines below against the file contents you were "
        "shown and copy them character for character, including indentation, quoting and "
        "HTML entities. If a hunk was called ambiguous, give it more surrounding context.\n\n"
        f"WHAT FAILED:\n{error}\n\nTHE DIFF THAT FAILED:\n{diff}"
    )


def _unseen_note(fresh: list[str], error: str) -> str:
    """What to say when the diff failed against a file we had not read."""
    return (
        f"Your diff did not apply, and it touched {', '.join(fresh)} — which you had not "
        f"been shown when you wrote it. The contents are included above now. Write the "
        f"change again against what is actually there.\n\nWHAT FAILED:\n{error}"
    )


def _asked_note(files: list[str], queries: list[str]) -> str:
    asked = ", ".join(files + [f'"{query}"' for query in queries])
    return (
        f"You asked about {asked} before writing the change. What was found is above. "
        f"Now write it."
    )


def _join(existing: str, extra: str) -> str:
    """Keep earlier search results alongside later ones."""
    return "\n".join(block for block in (existing, extra) if block.strip())


def _missing_note(missing: list[str]) -> str:
    return (
        f"The patch applied but did not create {', '.join(missing)}, which you listed as "
        "files it touches. A file that does not exist yet must be created by the patch "
        'itself: use "*** Add File: <path>" followed by every line of its contents each '
        'prefixed with "+", or a unified diff whose header is "--- /dev/null". Include the '
        "complete contents, not a description of them."
    )
