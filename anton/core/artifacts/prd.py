"""The PRD file's name, shared by the tool that writes it and the one that
reads it.

`generate_prd` writes `<artifact folder>/prd.md`; `generate_artifact` picks it
up from there as its requirements source (ENG-969 → ENG-968 handoff). A
literal in both packages would let the two drift apart silently — the reader
would simply find nothing and fall back to `context`, producing an artifact
built from a brief the user never confirmed. One name, one definition.

Lives here rather than in either tool package because the file is a property
of the artifact folder, not of whichever tool touches it.
"""

from __future__ import annotations

PRD_FILENAME = "prd.md"
