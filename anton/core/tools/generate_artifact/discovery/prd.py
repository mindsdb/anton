"""Phase C: expand the confirmed brief into `prd.md` and save it.

The document stays on disk as the human-readable record of what the user
agreed to. It is no longer the channel the generator reads its requirements
from — that is `spec.md` on the hot path and `discovery.json` on a cold
start — which is why its mandate can shrink.
"""

from __future__ import annotations

from anton.core.artifacts.internal_files import PRD_FILENAME

from . import sub_tools
from . import prompts
from .state import PrdState


_WRITE_PRD_INSTRUCTION = (
    "Write the FULL PRD now, expanding the same five sections with full "
    "detail. Describe only what the artifact IS and does — do NOT mention "
    "this generation tool, the PRD workflow, prior attempts, or that this "
    "is a redo/regeneration of something.\n\n"
    "Goal, Artifact type (with justification for the choice), "
    "Data model (the sources, their fields, and a few sample rows; for a "
    "fullstack-stateful-app additionally list the app-owned durable state: "
    "each kind of stored item as a named collection with its fields and what "
    "identifies one item), Functional requirements (omit for static "
    "artifacts), UI/UX requirements (layout, components, style, and any "
    "known user preferences).\n\n"
    "This document records WHAT THE USER AGREED TO. It is not the only thing "
    "the build step reads: the working data-access code and the source "
    "material reach it through their own channels. So do not restate "
    "connection code, environment variable names, or long-form source "
    "content (article text, document bodies, fetched pages) here — describe "
    "the requirements, and keep the document short enough that a person will "
    "actually read it. Describe STRUCTURE where the content is long: for a "
    "presentation, a slide outline with one line per slide, not the slide "
    "texts. Short samples (a few rows, a title list) are fine.\n\n"
    "Reply with the PRD document only, as markdown, no other text."
)



async def write_prd(state: PrdState) -> str:
    """Phase 2 step 5 (or the best-effort path from an unconfirmed budget):
    expand the brief into the full PRD, save it, and update the artifact's
    `type` in metadata.json if it changed. Returns the full PRD markdown."""
    state.step_started(sub_tools.STEP_WRITE_PRD)
    state.messages.append({
        "role": "user",
        "content": prompts.step_message(sub_tools.STEP_WRITE_PRD, state),
    })
    await sub_tools.signal_thinking(state.session)
    system = state.pipeline_system
    response = await state.session._llm.plan(
        system=system,
        messages=state.messages,
        tools=state.pipeline_tools,
    )
    state.trace_log.llm_call(
        node="write_prd", method="plan", system=system,
        messages=state.messages, response=response,
    )
    full_prd = (response.content or "").strip()
    if not full_prd:
        state.trace_log.node("write_prd", "fail", detail="model replied with no text")
        # Same failure shape as draft_brief's guard, and the same reason:
        # writing an empty prd.md and reporting `prd_written` would be a
        # silent lie about what happened. Raising here surfaces it as a
        # generator crash, wrapped by `_prd_generation_failed` — never an
        # empty file on disk with a success status.
        raise RuntimeError(
            "write_prd: the model replied with no text — it may have "
            "called a tool instead of writing the PRD."
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

    final_type = state.final_artifact_type or state.artifact_type
    if final_type != state.artifact_type:
        # Reuse the exact same store-construction helper the handler used
        # (`tool_handlers._artifact_store`, keyed off `session._workspace`)
        # instead of guessing the artifacts root back out of `artifact_path`
        # — that guess (`artifact_path.parent`) only holds while
        # `artifact_path == <artifacts_root>/<slug>`, which is true today
        # but is exactly the kind of assumption that breaks silently later.
        from anton.core.tools.tool_handlers import _artifact_store

        store = _artifact_store(state.session)
        if store is not None:
            store.update(state.slug, type=final_type)

    state.trace_log.node("write_prd", "done", detail=str(state.artifact_path / "prd.md"))
    return full_prd

