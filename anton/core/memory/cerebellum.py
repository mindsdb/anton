"""Cerebellum — supervised error learning over scratchpad execution.

Brain analogue
==============

The cerebellum's classical role is *forward modeling and error correction*:
when a motor command is issued, the cerebellum predicts the expected
sensory consequences. When actual feedback arrives, it computes the
prediction error and uses it to refine future commands. The learning
rule is supervised — there's an explicit "teacher" (the actual outcome)
against which the prediction is compared.

Anton's analogue
================

For Anton, the "motor command" is a scratchpad cell. Before the cell
runs, the LLM declares its intent via the `one_line_description` field
on the scratchpad tool. That description IS the forward model — the
prediction of what the cell should do. After the cell runs, we have
its actual outcome (stdout, stderr, error). The cerebellum compares
the two and, when they diverge meaningfully, encodes a correction
note that future code-generating LLM calls will see.

Storage and retrieval
=====================

The cerebellum is a *producer* only. It does not own its own storage.
Corrections are encoded as `Engram` objects with `kind="lesson"` and
`topic="scratchpad"` via `Cortex.encode()`, the same path that manual
lessons and the consolidator already use. They flow back into future
prompts via the existing `Cortex.get_scratchpad_context()` →
`recall_scratchpad_wisdom()` → scratchpad tool description injection
chain. We add nothing to the storage layer — the cerebellum just
generates new entries for the existing pipe.

Decoupling
==========

This module knows nothing about scratchpad runtimes. It exposes two
async hook methods (`on_pre_execute`, `on_post_execute`) that take a
`Cell` object. Whoever orchestrates execution (today: the
`handle_scratchpad` dispatcher) is responsible for calling them at the
right moments. The runtime backends — local, future remote, future
Docker — are completely hook-agnostic. See discussion in v2.2 design.

Diff frequency
==============

The cerebellum batches its observations *per turn*, not per cell. Each
post-execute hook only buffers the cell; the actual diff/encoding work
runs at end-of-turn (triggered explicitly by the session, or by the
next pre-execute call from a different turn — whichever comes first).
This keeps cost predictable: at most one extra LLM call per
scratchpad-using turn, regardless of how many cells the LLM ran.

Cheap path
==========

Cells that complete cleanly (no error, empty stderr) contribute zero
LLM-call cost — they're never sent to the diff function. Only cells
that errored or warned trigger the post-turn diff.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from anton.core.backends.base import Cell
from anton.core.memory.hippocampus import Engram

if TYPE_CHECKING:
    from anton.core.llm.client import LLMClient
    from anton.core.memory.cortex import Cortex


_log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LLM-facing schema (Pydantic) — used by LLMClient.generate_object
# ─────────────────────────────────────────────────────────────────────────────


class _LessonDraft(BaseModel):
    """One generalizable lesson the diff pass extracted from a cell."""

    text: str = Field(
        ...,
        description=(
            "One sentence (ideally < 25 words) the next code-writing agent "
            "should know to avoid the same trap. Specific and actionable. "
            "Generalizable across files/projects, not session-specific."
        ),
    )
    topic: str = Field(
        default="scratchpad",
        description="Topic tag for retrieval. Default 'scratchpad'.",
    )


class _DiffPassResult(BaseModel):
    """Wrapper schema returned by the cerebellum's diff LLM call.

    The LLM is *forced* to call a tool whose input matches this schema,
    so we never have to parse text JSON or strip markdown fences.
    """

    lessons: list[_LessonDraft] = Field(
        default_factory=list,
        description=(
            "Generalizable lessons extracted from the batch. Empty list if "
            "the cells either succeeded cleanly or didn't reveal any "
            "reusable insight. Prefer fewer, broader lessons over many "
            "narrow ones."
        ),
    )


_DIFF_SYSTEM_PROMPT = (
    "You are a code-execution post-mortem analyst. You receive a batch of "
    "scratchpad cells the agent just ran in one turn — for each cell, the "
    "agent's stated intent and what actually happened. Your job is to "
    "extract any *generalizable* lessons that future code-writing agents "
    "should know to avoid the same trap."
)


_DIFF_USER_TEMPLATE = """\
The agent just executed the following scratchpad cells in a single turn. For each cell, you see:
- The agent's stated intent (the `one_line_description` field on the tool call)
- The actual result (stdout, stderr, error)

Identify any *generalizable* lessons from these executions that a future code-writing agent should know to avoid the same trap. Be specific and actionable. Skip cells where everything went fine — only report lessons from genuine divergences.

Cells:

{cells_block}

Rules:
- Only report a lesson if the cell genuinely failed or diverged from its stated intent. If everything went fine, return an empty list.
- Lessons should be generalizable, not session-specific. "Use low_memory=False with mixed-dtype CSVs" is good. "The file sales_q3.csv had a problem on line 12000" is bad.
- Prefer fewer, more reusable lessons over many narrow ones.
- Do NOT invent lessons that aren't supported by the cell evidence.
- Each lesson should be one sentence, ideally under 25 words.
"""


def _format_cell_for_diff(cell: Cell, index: int) -> str:
    """Render a single cell as a compact section for the diff prompt."""
    lines: list[str] = [f"### Cell {index}"]
    intent = (cell.description or "").strip() or "(no description provided)"
    lines.append(f"**Intent:** {intent}")

    code = (cell.code or "").strip()
    if code:
        snippet = code if len(code) <= 800 else code[:800] + "\n... [truncated]"
        lines.append("**Code:**")
        lines.append("```python")
        lines.append(snippet)
        lines.append("```")

    stdout = (cell.stdout or "").strip()
    if stdout:
        excerpt = stdout if len(stdout) <= 500 else stdout[:500] + "\n... [truncated]"
        lines.append(f"**stdout:** {excerpt}")

    stderr = (cell.stderr or "").strip()
    if stderr:
        excerpt = stderr if len(stderr) <= 400 else stderr[:400] + "\n... [truncated]"
        lines.append(f"**stderr:** {excerpt}")

    error = (cell.error or "").strip() if cell.error else ""
    if error:
        excerpt = error if len(error) <= 400 else error[:400] + "\n... [truncated]"
        lines.append(f"**error:** {excerpt}")

    if not stdout and not stderr and not error:
        lines.append("(no output produced)")

    return "\n".join(lines)


@dataclass
class CerebellumLesson:
    """A single correction extracted by the diff pass.

    The dataclass form is the public type the cerebellum exposes to the
    rest of Anton (Cell, Engram, etc. all use dataclasses). Internally,
    the diff LLM call uses the Pydantic `_LessonDraft` schema for forced
    structured output; we convert at the boundary.
    """

    text: str
    topic: str = "scratchpad"


class Cerebellum:
    """Forward-model + error-driven learner over scratchpad cells.

    Usage:

        cb = Cerebellum(cortex=session._cortex, llm=session._llm)
        # Wired into the dispatcher as a scratchpad observer:
        await cb.on_pre_execute(prelim_cell)
        await cb.on_post_execute(final_cell)
        # At end of turn (or before the next pre-execute from a different turn):
        await cb.flush()

    Cells are buffered until `flush()` is called. The diff pass runs only
    on the buffered cells that errored or warned — clean cells contribute
    nothing. Lessons are encoded via `cortex.encode()` and flow into
    future prompts through the existing wisdom-injection pipeline.
    """

    def __init__(
        self,
        *,
        cortex: "Cortex | None",
        llm: "LLMClient | None",
        max_lessons_per_flush: int = 3,
    ) -> None:
        self._cortex = cortex
        self._llm = llm
        self._max_lessons = max_lessons_per_flush
        # Cells observed since the last flush. Indexed by insertion order.
        self._buffered: list[Cell] = []
        # Optional intent capture from pre-execute. Today we just trust the
        # cell that arrives at post_execute, but pre_execute is also a
        # natural place to clear stale buffers from prior cancelled turns.
        self._pre_count: int = 0

    # ── observer hooks ─────────────────────────────────────────────

    async def on_pre_execute(self, cell: Cell) -> None:
        """Called by the dispatcher right before a cell is sent to the runtime.

        The cell here is preliminary — code + description are populated,
        but stdout/stderr/error are empty. We don't currently use the
        prelim cell directly (the LLM's intent is already captured in
        cell.description, which post_execute will see again), but the
        hook exists so future expansions can do real forward-model work.
        """
        self._pre_count += 1

    async def on_post_execute(self, cell: Cell) -> None:
        """Called by the dispatcher right after a cell finishes executing.

        We buffer the cell here. The actual diff/encode work runs in
        flush() — see class docstring for the batching rationale.
        """
        # Cheap path: clean cells never need diffing
        if self._is_clean(cell):
            return
        self._buffered.append(cell)

    async def flush(self) -> list[CerebellumLesson]:
        """Run the batched diff pass and encode any extracted lessons.

        Should be called at end-of-turn. Safe to call when the buffer is
        empty (returns []). Safe to call multiple times (idempotent — the
        buffer is cleared each time).

        Returns the lessons that were encoded, mostly for testing /
        observability. Production code typically ignores the return value.
        """
        if not self._buffered:
            return []
        if self._cortex is None or self._llm is None:
            # Best-effort: silently no-op if memory infrastructure is missing
            self._buffered.clear()
            return []

        cells = self._buffered[:]
        self._buffered.clear()

        try:
            lessons = await self._run_diff(cells)
        except Exception as exc:
            _log.warning("cerebellum diff pass failed: %s", exc)
            return []

        if not lessons:
            return []

        try:
            await self._encode_lessons(lessons)
        except Exception as exc:
            _log.warning("cerebellum failed to encode lessons: %s", exc)

        return lessons

    def reset(self) -> None:
        """Drop the current buffer without encoding anything.

        Used when a turn is cancelled mid-flight — we don't want to
        encode lessons from a turn the user backed out of.
        """
        self._buffered.clear()
        self._pre_count = 0

    @property
    def buffered_count(self) -> int:
        """Number of cells currently waiting for the next flush."""
        return len(self._buffered)

    # ── internals ──────────────────────────────────────────────────

    @staticmethod
    def _is_clean(cell: Cell) -> bool:
        """A cell is 'clean' if it produced no error and no stderr text.

        Clean cells contribute zero diff cost — they're skipped before
        the LLM call ever happens. This is the cheap path that keeps
        the cerebellum's overhead near zero on a happy-path turn.
        """
        if cell.error:
            return False
        stderr = (cell.stderr or "").strip()
        if stderr:
            return False
        return True

    async def _run_diff(self, cells: list[Cell]) -> list[CerebellumLesson]:
        """Send the buffered cells to the LLM and return validated lessons.

        Uses `LLMClient.generate_object_code` (the cheap/fast coding
        provider) to force structured output via a forced tool call
        whose schema is the `_DiffPassResult` Pydantic model. There's
        no manual JSON parsing, no markdown fence stripping, and no
        try/except around `json.loads` — Pydantic and the forced
        tool_choice eliminate those failure modes entirely. If the
        LLM round-trip itself fails (network, validation), the
        caller's try/except in `flush()` swallows it.

        We use the *coding* provider (not planning) because this is a
        fast post-mortem on cell output — exactly the kind of cheap
        structured task the coding model is sized for.
        """
        cells_block = "\n\n".join(
            _format_cell_for_diff(c, i + 1) for i, c in enumerate(cells)
        )
        prompt = _DIFF_USER_TEMPLATE.format(cells_block=cells_block)

        result: _DiffPassResult = await self._llm.generate_object_code(
            _DiffPassResult,
            system=_DIFF_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
        )

        # Convert the validated Pydantic drafts to the public dataclass
        # form. Cap at max_lessons_per_flush to bound the encoding cost
        # of any single flush.
        out: list[CerebellumLesson] = []
        for draft in result.lessons[: self._max_lessons]:
            text = (draft.text or "").strip()
            if not text:
                continue
            topic = (draft.topic or "scratchpad").strip() or "scratchpad"
            out.append(CerebellumLesson(text=text, topic=topic))
        return out

    async def _encode_lessons(self, lessons: list[CerebellumLesson]) -> None:
        """Hand the lessons to Cortex for storage in the existing pipeline."""
        engrams = [
            Engram(
                text=lesson.text,
                kind="lesson",
                scope="project",
                confidence="medium",
                topic=lesson.topic,
                source="consolidation",
            )
            for lesson in lessons
        ]
        if not engrams:
            return
        await self._cortex.encode(engrams)


__all__ = ["Cerebellum", "CerebellumLesson"]
