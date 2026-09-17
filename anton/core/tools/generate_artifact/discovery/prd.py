"""Phase C: expand the confirmed brief into `prd.md` and save it.

The document stays on disk as the human-readable record of what the user
agreed to, and `prd_section` hands it verbatim to the spec and generation
nodes as the authoritative requirements source. What shrank is its mandate,
not its reach: connection code and source material travel through
`data_notes` / `web_notes`, and the tech spec is told not to restate it.
"""

from __future__ import annotations

from anton.core.artifacts.internal_files import PRD_FILENAME

from . import sub_tools
from .state import PrdState


_WRITE_PRD_INSTRUCTION = (
    "## Your task\n"
    "Write the full PRD: the requirements the user agreed to, as a document "
    "a person will read. Do not call any tool. Reply with the PRD only, as "
    "markdown, no other text.\n\n"
    "The latest brief in this conversation is the source. Earlier briefs and "
    "the feedback that led to them are history: the latest brief already "
    "reflects them.\n"
    "- Its Requirements stay requirements.\n"
    "- Its Proposals are accepted: write them as plain requirements, without "
    "marking them as proposals.\n"
    "- For each Question, the stated default is the decision. Record it as a "
    "requirement.\n"
    "Do not add requirements the brief does not contain. Where the brief is "
    "silent, the build step decides, not this document.\n\n"
    "## Sections\n"
    "1. Goal: what the artifact is and does for the user, 1-3 sentences.\n"
    "2. Artifact type: the type and one line on why.\n"
    "3. Data model: the external sources the artifact reads, with their "
    "fields and a few sample rows each. Sample rows must not carry data that "
    "looks sensitive (personal names, emails, phone numbers, addresses, "
    "credentials, account or card numbers, salaries, health records): "
    "replace such values with made-up ones of the same shape, and keep the "
    "real ones out of this document. For a fullstack-stateful-app, also the "
    "app-owned durable state: each kind of stored item as a named collection "
    "with its fields and what identifies one item. If there is no external "
    "data and nothing stored, one line saying so. Do not describe in-memory "
    "variables or invent content lists.\n"
    "4. Functional requirements: one numbered line per behavior. Omit for "
    "static artifacts.\n"
    "5. UI/UX requirements: the layout and components that follow from the "
    "functional requirements, the user's stated preferences, and what the "
    "brief said about the look. Do not invent a visual style, animations or "
    "effects here; those are decided at the build step.\n\n"
    "## Rules\n"
    "- Describe only what the artifact IS and does: no mention of this "
    "generation tool, the PRD workflow, prior attempts, or that this is a "
    "redo/regeneration of something.\n"
    "- State only what IS used. Do not list data sources, connections, "
    "storage, protocols or technologies that the artifact does NOT use: a "
    "connected database the artifact never reads is not part of its data "
    "model and is not mentioned.\n"
    "- This document is not the only thing the build step reads: the working "
    "data-access code and the source material reach it through their own "
    "channels. Do not restate connection code, environment variable names, "
    "or long-form source content (article text, document bodies, fetched "
    "pages). Describe STRUCTURE where the content is long: for a "
    "presentation, a slide outline with one line per slide, not the slide "
    "texts. Short samples (a few rows, a title list) are fine.\n"
    "- Write in the language of the user request.\n"
    "- Keep the PRD as short as the requirements allow. A small request gets "
    "a small PRD: no section is padded to look complete, and nothing is "
    "repeated across sections."
)



async def write_prd(state: PrdState) -> str:
    """Phase 2 step 5 (or the best-effort path from an unconfirmed budget):
    expand the brief into the full PRD, save it, and update the artifact's
    `type` in metadata.json if it changed. The in-memory state follows the
    same change (`GenState.settle_artifact_type`), so the rest of this call
    builds the type that was agreed, not the one that was registered.
    Returns the full PRD markdown."""
    # The type is settled BEFORE the step runs, not after: the step's own
    # progress line is the first one that carries `step N of M`, and `M`
    # depends on the type; the metadata write below stays where it was.
    registered_type = state.artifact_type
    final_type = state.final_artifact_type or registered_type
    state.settle_artifact_type(final_type)

    # An empty reply raises inside `plan_step`: writing an empty prd.md and
    # reporting `prd_written` would be a silent lie about what happened.
    full_prd, _ = await sub_tools.plan_step(
        state, sub_tools.STEP_WRITE_PRD, doing="writing the PRD"
    )
    state.messages.append({"role": "assistant", "content": full_prd})

    # Same constant the cold-start path reads back — see
    # anton/core/artifacts/internal_files.py.
    (state.artifact_path / PRD_FILENAME).write_text(full_prd, encoding="utf-8")
    # `prd_section` renders `state.prd`, and it is declared to the spec and
    # generation nodes as the authoritative requirements source. Leaving it
    # holding the version read at entry means a PRD rewritten during THIS
    # call — which is what every user correction produces — never reaches
    # them, and the correction is lost inside one run.
    state.prd = full_prd

    if final_type != registered_type:
        # Reuse the exact same store-construction helper the handler used
        # (`tool_handlers.resolve_artifact_store`, keyed off `session._workspace`)
        # instead of guessing the artifacts root back out of `artifact_path`
        # — that guess (`artifact_path.parent`) only holds while
        # `artifact_path == <artifacts_root>/<slug>`, which is true today
        # but is exactly the kind of assumption that breaks silently later.
        from anton.core.tools.tool_handlers import resolve_artifact_store

        store = resolve_artifact_store(state.session)
        if store is not None:
            store.update(state.slug, type=final_type)

    state.trace_log.node("write_prd", "done", detail=str(state.artifact_path / PRD_FILENAME))
    return full_prd

