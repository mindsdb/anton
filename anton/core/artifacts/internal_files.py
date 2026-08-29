"""Names of the files the generation pipeline leaves in an artifact folder
that are inputs to generation rather than artifact content.

`generate_prd` writes `prd.md`; `generate_artifact` writes `spec.md` and
`openapi.json`, and reads the PRD back from the same folder (ENG-969 → ENG-968
handoff). Every one of them physically sits next to `dashboard.html` or
`backend.py`, and every one of them would otherwise be reported to the user as
part of what was built.

One definition each, because both ends of every name matter and are far apart:
the tool that writes the file and the store that must leave it out of `files[]`.
A literal on each side would drift silently — a renamed PRD would simply stop
being found, and generation would fall back to building from a brief the user
never confirmed.

Lives here rather than in either tool package because these files are a
property of the artifact folder, not of whichever tool touches them.
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
