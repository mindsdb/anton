"""Cortex — Anton's executive memory coordinator.

Ported from anton/memory/cortex.py with local import paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from .hippocampus import Engram, Hippocampus
from .prompts import _COMPACTION_PROMPT, _IDENTITY_EXTRACT_PROMPT

if TYPE_CHECKING:
    from ..llm.client import LLMClient


class Cortex:
    """Executive coordinator for Anton's memory systems."""

    def __init__(
        self,
        global_dir: Path,
        project_dir: Path,
        mode: str = "autopilot",
        llm_client: LLMClient | None = None,
    ) -> None:
        self.global_hc = Hippocampus(global_dir)
        self.project_hc = Hippocampus(project_dir)
        self.mode = mode
        self._llm = llm_client
        self._turn_count = 0

    def build_memory_context(self) -> str:
        sections: list[str] = []

        identity = self.global_hc.recall_identity()
        if identity:
            sections.append(f"## Your Memory — Identity\n{identity}")

        global_rules = self.global_hc.recall_rules()
        if global_rules:
            sections.append(f"## Your Memory — Global Rules\n{global_rules}")

        project_rules = self.project_hc.recall_rules()
        if project_rules:
            sections.append(f"## Your Memory — Project Rules\n{project_rules}")

        global_lessons = self.global_hc.recall_lessons(token_budget=1000)
        if global_lessons:
            sections.append(f"## Your Memory — Global Lessons\n{global_lessons}")

        project_lessons = self.project_hc.recall_lessons(token_budget=1000)
        if project_lessons:
            sections.append(f"## Your Memory — Project Lessons\n{project_lessons}")

        if not sections:
            return ""

        return "\n\n" + "\n\n".join(sections)

    def get_scratchpad_context(self) -> str:
        parts: list[str] = []

        global_wisdom = self.global_hc.recall_scratchpad_wisdom()
        if global_wisdom:
            parts.append(global_wisdom)

        project_wisdom = self.project_hc.recall_scratchpad_wisdom()
        if project_wisdom:
            parts.append(project_wisdom)

        return "\n".join(parts)

    async def encode(self, engrams: list[Engram]) -> list[str]:
        if self.mode == "off":
            return ["Memory encoding is disabled."]

        actions: list[str] = []
        for engram in engrams:
            hc = self.global_hc if engram.scope == "global" else self.project_hc

            if engram.kind == "profile":
                existing = hc.recall_identity()
                entries = []
                if existing:
                    for line in existing.splitlines():
                        stripped = line.strip()
                        if stripped.startswith("- "):
                            entries.append(stripped[2:])
                        elif stripped and not stripped.startswith("#"):
                            entries.append(stripped)
                entries.append(engram.text)
                hc.rewrite_identity(entries)
                actions.append(f"Updated identity: {engram.text}")

            elif engram.kind in ("always", "never", "when"):
                hc.encode_rule(
                    engram.text,
                    kind=engram.kind,
                    confidence=engram.confidence,
                    source=engram.source,
                )
                actions.append(f"Encoded {engram.kind} rule: {engram.text}")

            elif engram.kind == "lesson":
                hc.encode_lesson(
                    engram.text,
                    topic=engram.topic,
                    source=engram.source,
                )
                actions.append(f"Encoded lesson: {engram.text}")

        return actions

    def encoding_gate(self, engram: Engram) -> bool:
        if self.mode == "autopilot":
            return False
        if self.mode == "off":
            return False
        return engram.confidence != "high"

    _COMPACTION_THRESHOLD = 50

    def needs_compaction(self) -> bool:
        return (
            self.global_hc.entry_count() > self._COMPACTION_THRESHOLD
            or self.project_hc.entry_count() > self._COMPACTION_THRESHOLD
        )

    async def compact_all(self) -> None:
        if self._llm is None:
            return

        for hc in (self.global_hc, self.project_hc):
            if hc.entry_count() > self._COMPACTION_THRESHOLD:
                await self._compact_file(hc, hc._lessons_path, "lesson")
                await self._compact_file(hc, hc._rules_path, "rules")

    async def _compact_file(self, hc: Hippocampus, path: Path, kind: str) -> None:
        if not path.is_file():
            return

        content = path.read_text(encoding="utf-8")
        entries = [ln.strip() for ln in content.splitlines() if ln.strip().startswith("- ")]

        if len(entries) < 20:
            return

        try:
            response = await self._llm.code(
                system=_COMPACTION_PROMPT,
                messages=[{"role": "user", "content": "\n".join(entries)}],
                max_tokens=4096,
            )
            result = json.loads(response.content)
            kept = result.get("kept", entries)
        except Exception:
            return

        if not kept:
            return

        if kind == "rules":
            always = [
                e for e in kept if "always" in e.lower() or not any(k in e.lower() for k in ("never", "when", "if "))
            ]
            never = [e for e in kept if "never" in e.lower()]
            when_rules = [e for e in kept if "when" in e.lower() or "if " in e.lower()]

            lines = ["# Rules\n", "## Always"]
            lines.extend(f"- {e}" if not e.startswith("- ") else e for e in always)
            lines.extend(["", "## Never"])
            lines.extend(f"- {e}" if not e.startswith("- ") else e for e in never)
            lines.extend(["", "## When"])
            lines.extend(f"- {e}" if not e.startswith("- ") else e for e in when_rules)
            new_content = "\n".join(lines) + "\n"
        else:
            lines = ["# Lessons"]
            lines.extend(f"- {e}" if not e.startswith("- ") else e for e in kept)
            new_content = "\n".join(lines) + "\n"

        hc._encode_with_lock(path, new_content, mode="write")

    async def maybe_update_identity(self, user_message: str) -> None:
        if self._llm is None or self.mode == "off":
            return

        try:
            response = await self._llm.code(
                system=_IDENTITY_EXTRACT_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                max_tokens=512,
            )
            facts = json.loads(response.content)
            if not isinstance(facts, list) or not facts:
                return
        except Exception:
            return

        existing = self.global_hc.recall_identity()
        existing_entries: list[str] = []
        if existing:
            for line in existing.splitlines():
                stripped = line.strip()
                if stripped.startswith("- "):
                    existing_entries.append(stripped[2:])
                elif stripped and not stripped.startswith("#"):
                    existing_entries.append(stripped)

        for fact in facts:
            if isinstance(fact, str) and fact not in existing_entries:
                key = fact.split(":")[0].strip().lower() if ":" in fact else ""
                if key:
                    existing_entries = [e for e in existing_entries if not e.lower().startswith(key + ":")]
                existing_entries.append(fact)

        if existing_entries:
            self.global_hc.rewrite_identity(existing_entries)
