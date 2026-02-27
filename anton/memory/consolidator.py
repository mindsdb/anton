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

import json
from typing import TYPE_CHECKING

from anton.memory.hippocampus import Engram

if TYPE_CHECKING:
    from anton.llm.client import LLMClient
    from anton.scratchpad import Cell


_CONSOLIDATION_PROMPT = """\
You are a memory consolidation system for an AI coding assistant.

Review this scratchpad session (sequence of code cells with their results) and
extract durable, reusable lessons. Focus on:

1. **Rules** — patterns to always/never follow:
   - "Always call progress() before long API calls in scratchpad"
   - "Never use time.sleep() in scratchpad cells"
   - Conditional rules: "If fetching paginated data → use async + progress()"

2. **Lessons** — factual knowledge discovered:
   - API behaviors: "CoinGecko free tier rate-limits at ~50 req/min"
   - Library quirks: "pandas read_csv needs encoding='utf-8-sig' for BOM files"
   - Data facts: "Bitcoin price data via /coins/bitcoin/market_chart/range"

Return a JSON array of objects:
[
  {
    "text": "the memory to encode",
    "kind": "always" | "never" | "when" | "lesson",
    "scope": "global" | "project",
    "topic": "optional-topic-slug",
    "confidence": "high" | "medium"
  }
]

Rules for scope:
- "global": universal knowledge useful across any project
- "project": specific to this workspace (file paths, project-specific APIs)

Rules for confidence:
- "high": clearly correct, verified by the session results
- "medium": probably correct but worth confirming

If no meaningful lessons exist, return [].
Do NOT extract trivial observations. Only encode genuinely reusable knowledge.
"""


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
            if cell.stderr and ("cancelled" in cell.stderr.lower() or "killed" in cell.stderr.lower()):
                return True

        return False

    async def replay_and_extract(self, cells: list[Cell], llm_client: LLMClient) -> list[Engram]:
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
            response = await llm_client.code(
                system=_CONSOLIDATION_PROMPT,
                messages=[{"role": "user", "content": session_summary}],
                max_tokens=2048,
            )

            raw = response.content.strip()
            # Handle markdown code fences
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            items = json.loads(raw)
            if not isinstance(items, list):
                return []

        except Exception:
            return []

        engrams: list[Engram] = []
        for item in items:
            if not isinstance(item, dict) or "text" not in item:
                continue

            kind = item.get("kind", "lesson")
            if kind not in ("always", "never", "when", "lesson"):
                kind = "lesson"

            scope = item.get("scope", "project")
            if scope not in ("global", "project"):
                scope = "project"

            confidence = item.get("confidence", "medium")
            if confidence not in ("high", "medium", "low"):
                confidence = "medium"

            engrams.append(Engram(
                text=item["text"],
                kind=kind,
                scope=scope,
                confidence=confidence,
                topic=item.get("topic", ""),
                source="consolidation",
            ))

        return engrams
