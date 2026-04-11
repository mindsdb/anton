"""Consolidator — Anton's sleep-like memory consolidation process.

Named for hippocampal-cortical replay during Slow-Wave Sleep (SWS).

During sleep, the hippocampus "replays" recent experiences to the neocortex
in compressed, accelerated bursts (sharp-wave ripples). This offline process:
  - Reviews what happened during waking hours
  - Extracts statistical regularities and important lessons
  - Transfers knowledge from episodic (hippocampal) to semantic (cortical) storage
  - Is selective — emotionally tagged and goal-relevant experiences get priority

The Consolidator mirrors this exactly: after a scratchpad session ends, it
replays the cell history, asks "what would I tell myself to do differently?",
and encodes the resulting lessons into long-term memory via the Cortex.

Like sleep, consolidation is:
  - Offline (runs after the task, not during)
  - Compressed (summarizes cells, doesn't replay in full)
  - Selective (only triggers when there were errors, long sessions, or cancellations)
  - Background (doesn't block the user's next interaction)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from anton.core.llm.prompts import CONSOLIDATION_PROMPT
from anton.core.memory.hippocampus import Engram

if TYPE_CHECKING:
    from anton.core.llm.client import LLMClient
    from anton.core.backends.base import Cell


# ─────────────────────────────────────────────────────────────────────────────
# LLM-facing schema (Pydantic) — used by LLMClient.generate_object_code
# ─────────────────────────────────────────────────────────────────────────────


class _ConsolidatedLesson(BaseModel):
    """One engram extracted from a scratchpad replay."""

    text: str = Field(
        ...,
        description=(
            "The lesson itself — what a future agent should know to do "
            "this kind of task better. Concrete and actionable."
        ),
    )
    kind: Literal["always", "never", "when", "lesson"] = Field(
        default="lesson",
        description=(
            "Engram type. 'always'/'never' = behavioral rules, "
            "'when' = conditional rule, 'lesson' = semantic fact."
        ),
    )
    scope: Literal["global", "project"] = Field(
        default="project",
        description=(
            "'global' = applies across all projects, 'project' = "
            "specific to this codebase. Default project."
        ),
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description=(
            "How confident you are this lesson generalizes. 'high' "
            "auto-encodes; 'medium'/'low' may require user confirmation."
        ),
    )
    topic: str = Field(
        default="",
        description="Optional topic tag for retrieval grouping.",
    )


class _ConsolidatedLessons(BaseModel):
    """Wrapper for the list of lessons returned by the consolidator."""

    items: list[_ConsolidatedLesson] = Field(
        default_factory=list,
        description=(
            "Lessons extracted from the scratchpad replay. Empty list "
            "if nothing worth remembering. Cap at ~5 — be selective."
        ),
    )


class Consolidator:
    """Extracts durable lessons from scratchpad sessions via offline replay.

    Brain analog: hippocampal sharp-wave ripples during SWS that replay
    compressed versions of waking experiences to the neocortex for
    long-term storage.
    """

    def should_replay(self, cells: list[Cell]) -> bool:
        """Heuristic gate — determines if this session warrants consolidation.

        Like the amygdala tagging experiences for priority replay:
        emotionally significant events (errors, long sessions, cancellations)
        are preferentially consolidated. No LLM call needed.

        Triggers when:
          - Any cell had an error (negative emotional valence)
          - Session was long (>=5 cells — rich experience to mine)
          - Any cell was cancelled (interrupted action — what went wrong?)
        """
        if len(cells) < 2:
            return False

        # Long sessions are worth reviewing
        if len(cells) >= 5:
            return True

        # Errors are high-signal learning opportunities
        for cell in cells:
            if cell.error:
                return True

        # Check for cancellation markers in stderr
        for cell in cells:
            if cell.stderr and (
                "cancelled" in cell.stderr.lower() or "killed" in cell.stderr.lower()
            ):
                return True

        return False

    async def replay_and_extract(
        self, cells: list[Cell], llm_client: LLMClient
    ) -> list[Engram]:
        """Replay the scratchpad session and extract lessons.

        Like SWS replay: compresses the full session into a compact summary,
        then runs a fast LLM pass asking:
          "If you were to do this task again, what would you tell yourself?"

        Returns structured Engram objects ready for encoding via the Cortex.
        """
        # Build compact cell summary
        summary_lines: list[str] = []
        for i, cell in enumerate(cells, 1):
            desc = cell.description or "(no description)"
            status = "error" if cell.error else "ok"
            output_preview = ""
            if cell.stdout:
                first_line = cell.stdout.strip().split("\n")[0][:200]
                output_preview = f" → {first_line}"
            elif cell.error:
                first_line = cell.error.strip().split("\n")[-1][:200]
                output_preview = f" → ERROR: {first_line}"

            summary_lines.append(f"Cell {i} [{status}]: {desc}{output_preview}")

            # Include code snippet for error cells (helpful context)
            if cell.error and cell.code:
                code_preview = cell.code[:300]
                if len(cell.code) > 300:
                    code_preview += "..."
                summary_lines.append(f"  Code: {code_preview}")

        session_summary = "\n".join(summary_lines)

        try:
            result: _ConsolidatedLessons = await llm_client.generate_object_code(
                _ConsolidatedLessons,
                system=CONSOLIDATION_PROMPT,
                messages=[{"role": "user", "content": session_summary}],
                max_tokens=2048,
            )
        except Exception:
            return []

        engrams: list[Engram] = []
        for item in result.items:
            text = (item.text or "").strip()
            if not text:
                continue
            engrams.append(
                Engram(
                    text=text,
                    kind=item.kind,
                    scope=item.scope,
                    confidence=item.confidence,
                    topic=item.topic,
                    source="consolidation",
                )
            )

        # Cap extraction to prevent memory bloat from single sessions
        return engrams[:5]
