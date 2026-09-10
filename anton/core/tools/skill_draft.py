"""The `create_skill_draft` tool — claim a staging folder for an agent-built skill.

A skill the agent builds is never written into the live store: it is staged in
a drafts folder the host drains at end of turn and shows to the user, who saves
it explicitly. This tool claims that folder and, when a skill of the same name
is already saved, pre-fills it so an edit starts from the current version
rather than a blank file.

The LLM-facing contract (description + schema) is deliberately identical to the
one cowork-server's harnesses register, so the tool reads the same to the model
whichever host is driving.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from anton.core.tools.skill_format import SKILL_FILE, normalize_name
from anton.core.tools.registry import ToolOutcome
from anton.core.tools.tool_defs import ToolDef

if TYPE_CHECKING:
    from anton.core.session import ChatSession

logger = logging.getLogger(__name__)

#: Per-skill counters live with the store, not with the skill's content — a
#: draft carrying them back would re-import another store's usage history.
_NOT_CONTENT = {"stats.json"}


_DESCRIPTION = (
    "Claim a staging folder for a skill you are building or improving for the "
    "user (e.g. while running the skill-creator skill). Call this BEFORE writing "
    "the skill; it returns {slug, path, skill_file} — write your SKILL.md to "
    "`skill_file` (and any sibling files into `path`). If a skill with this name "
    "is already saved, the folder is pre-filled with its current contents so you "
    "edit from the saved version. The skill is staged, NOT saved: it surfaces as "
    "a card the user explicitly saves or downloads. NEVER write a skill into the "
    "project `skills/` directory and NEVER use `create_artifact` for a skill."
)

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "The skill's name (becomes its slug, e.g. 'competitive-analysis').",
        },
        "description": {
            "type": "string",
            "description": "Optional one-line trigger description shown on the card.",
        },
    },
    "required": ["name"],
}


def _seed_from_store(folder: Path, slug: str, store) -> None:
    """Copy the saved skill's files into `folder` so an edit starts from it.

    Best-effort: an unseeded draft is a worse starting point, not a failed turn.
    Symlinks and out-of-tree children are skipped — a shared store could link
    into files this turn has no business reading.
    """
    if store is None:
        return
    try:
        src = store.dir_for(slug)
    except Exception:
        logger.warning("skill draft %r: could not locate the saved skill", slug, exc_info=True)
        return
    if src is None or src.is_symlink() or not (src / SKILL_FILE).is_file():
        return
    src_resolved = src.resolve()
    try:
        for child in src.iterdir():
            if child.is_symlink() or not child.is_file() or child.name in _NOT_CONTENT:
                continue
            if child.resolve().parent != src_resolved:
                logger.warning("skill draft %r: skipping out-of-tree file %r", slug, child.name)
                continue
            shutil.copy2(child, folder / child.name)
    except OSError:
        logger.warning("skill draft %r: could not seed from the store", slug, exc_info=True)


async def handle_create_skill_draft(
    session: "ChatSession", tc_input: dict
) -> "str | ToolOutcome":
    """Claim `<skill_drafts_root>/<slug>`; return `{slug, path, skill_file}`."""
    root = getattr(session, "_skill_drafts_root", None)
    if root is None:
        # Tier 2 (ENG-2248): the host wired no draft store; no input can work.
        return ToolOutcome(
            content=json.dumps({"error": "Skill drafts are unavailable in this session."}),
            ok=False, reason="store_unavailable",
        )

    name = str(tc_input.get("name") or "").strip()
    if not name:
        # Tier 2: malformed call.
        return ToolOutcome(
            content=json.dumps({"error": "`name` is required."}),
            ok=False, reason="missing_name",
        )
    slug = normalize_name(name)
    if not slug:
        # Tier 2: malformed call.
        return ToolOutcome(
            content=json.dumps({"error": "`name` must contain at least one letter or digit."}),
            ok=False, reason="invalid_type",
        )

    folder = Path(root) / slug
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("skill draft %r: could not claim a folder", slug, exc_info=True)
        # Tier 2: the filesystem refused; the draft does not exist.
        return ToolOutcome(
            content=json.dumps({"error": f"Could not claim a folder for {slug!r}: {exc}"}),
            ok=False, reason="store_unavailable",
        )

    # Only seed an unclaimed folder: a second call in the same turn must not
    # overwrite what the agent has already written into it.
    if not (folder / SKILL_FILE).is_file():
        _seed_from_store(folder, slug, getattr(session, "_skill_store", None))

    # Tier 1: a draft folder was claimed and its descriptor returned.
    return ToolOutcome(content=json.dumps({
        "slug": slug,
        "path": str(folder),
        "skill_file": str(folder / SKILL_FILE),
    }), ok=True)


CREATE_SKILL_DRAFT_TOOL = ToolDef(
    name="create_skill_draft",
    description=_DESCRIPTION,
    input_schema=_INPUT_SCHEMA,
    handler=handle_create_skill_draft,
)
