"""Names of the files the generation pipeline leaves in an artifact folder
that are inputs to generation rather than artifact content.

`generate_artifact` writes `prd.md`, `discovery.json`, `spec.md` and
`openapi.json`, and reads the PRD back from the same folder. Every one of them
physically sits next to `index.html` or `backend.py`, and every one of them
would otherwise be reported to the user as part of what was built.

One definition each, because both ends of every name matter and are far apart:
the tool that writes the file and the store that must leave it out of `files[]`.
A literal on each side would drift silently — a renamed PRD would simply stop
being found, and generation would fall back to building from a brief the user
never confirmed.

Lives here rather than in either tool package because these files are a
property of the artifact folder, not of whichever tool touches them.

This module also owns the folder's other reserved names (housekeeping files
and directories). `NON_CONTENT_NAMES` is every top-level name that is never
artifact content: match it against the first component of an artifact-relative
path, or walk with `store.iter_content_files`. cowork-server imports it from
here. Consumers that match differently say so where they do it.
"""

from __future__ import annotations

PRD_FILENAME = "prd.md"
TECH_SPEC_FILENAME = "spec.md"
API_SPEC_FILENAME = "openapi.json"
# Machine-readable state of the discovery phases (gathering -> brief -> PRD):
# fingerprints, pipeline stage, declared data sources, the brief, and the
# rendered data/web notes. `prd.md` is the human-readable record of the same
# phases; this is what the pipeline itself reads back on a cold start.
DISCOVERY_FILENAME = "discovery.json"

# Reported by `generate_artifact` as `internal_files` and excluded from an
# artifact's `files[]`: generation inputs, not deliverables.
GENERATION_INPUT_FILES = frozenset({
    PRD_FILENAME,
    TECH_SPEC_FILENAME,
    API_SPEC_FILENAME,
    DISCOVERY_FILENAME,
})


METADATA_FILENAME = "metadata.json"
README_FILENAME = "README.md"
PUBLISHED_FILENAME = ".published.json"
BACKEND_LOG_FILENAME = "backend.log"
# The local STATE driver's SQLite database; -wal/-shm carry the freshest writes.
STATE_DB_FILENAME = ".anton_state.db"
STATE_SNAPSHOT_FILENAME = ".state_manifest.published.json"
REVISIONS_DIRNAME = ".revisions"

# Store-owned, publish-state or running-backend files, not authored content.
# `state_manifest.json` is deliberately absent: it is a deliverable the
# publisher bundles.
HOUSEKEEPING_FILES = frozenset({
    METADATA_FILENAME,
    README_FILENAME,
    PUBLISHED_FILENAME,
    BACKEND_LOG_FILENAME,
    STATE_DB_FILENAME,
    f"{STATE_DB_FILENAME}-wal",
    f"{STATE_DB_FILENAME}-shm",
    STATE_SNAPSHOT_FILENAME,
})

# Separate from the file set because `store._reconcile` matches the two
# differently (see there).
HOUSEKEEPING_DIRS = frozenset({REVISIONS_DIRNAME})

NON_CONTENT_NAMES = HOUSEKEEPING_FILES | GENERATION_INPUT_FILES | HOUSEKEEPING_DIRS
