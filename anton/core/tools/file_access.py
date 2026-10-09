"""Which files `read_text_file` may read.

Not a security boundary: the scratchpad can read any file the process can.
The policy keeps this tool to the files the agent is meant to work with and
keeps secrets (`.anton/.env`, a nested `.published.json`) out of one call.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

REFUSED_OUTSIDE = "outside the project, the artifacts and the conversation's attachments"
REFUSED_HIDDEN = "a dot-file or inside a dot-directory"


def read_refusal(
    resolved: Path, *, workspace: Path, owned_roots: Sequence[Path] = ()
) -> str | None:
    """Why `resolved` may not be read, or None when it may.

    Readable: what a conversation may attach (`refusal_reason`), and files
    inside a folder anton manages (`owned_roots`: the artifacts root, skill
    drafts) with no dot-component below that folder. Roots are compared after
    `resolve()`, like the file, so a symlinked project path still matches.
    """
    # Imported here: importing the generate_artifact package loads its engine.
    from anton.core.tools.generate_artifact.attachments import (
        REFUSED_HIDDEN as ATTACHMENT_HIDDEN,
        refusal_reason,
    )

    workspace_reason = refusal_reason(resolved, workspace)
    if workspace_reason is None:
        return None
    hidden = workspace_reason == ATTACHMENT_HIDDEN
    for root in owned_roots:
        try:
            rel = resolved.relative_to(Path(root).resolve())
        except (ValueError, OSError, RuntimeError):
            continue
        if not any(part.startswith(".") for part in rel.parts):
            return None
        hidden = True
    return REFUSED_HIDDEN if hidden else REFUSED_OUTSIDE
