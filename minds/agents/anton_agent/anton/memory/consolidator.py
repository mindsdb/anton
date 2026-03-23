"""Consolidator — Anton's sleep-like memory consolidation process.

Ported from anton/memory/consolidator.py with local import paths.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .hippocampus import Engram
from .prompts import _CONSOLIDATION_PROMPT

if TYPE_CHECKING:
    from ..backends.base import Cell
    from ..llm.client import LLMClient


class Consolidator:
    """Extracts durable lessons from scratchpad sessions via offline replay."""

    def should_replay(self, cells: list[Cell]) -> bool:
        if len(cells) < 2:
            return False

        if len(cells) >= 5:
            return True

        for cell in cells:
            if cell.error:
                return True

        for cell in cells:
            if cell.stderr and ("cancelled" in cell.stderr.lower() or "killed" in cell.stderr.lower()):
                return True

        return False

    async def replay_and_extract(self, cells: list[Cell], llm_client: LLMClient) -> list[Engram]:
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

            engrams.append(
                Engram(
                    text=item["text"],
                    kind=kind,
                    scope=scope,
                    confidence=confidence,
                    topic=item.get("topic", ""),
                    source="consolidation",
                )
            )

        return engrams
