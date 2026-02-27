"""Cortex — Anton's executive memory coordinator.

Named for the Prefrontal Cortex (PFC), the brain's executive center that
orchestrates memory retrieval by sending top-down signals to the hippocampus
and other memory systems.

The dorsolateral PFC handles strategic retrieval — selecting which memories
to pull into working memory. The ventromedial PFC integrates across memory
systems to provide coherent context. The Cortex class mirrors both:

  - build_memory_context() → dlPFC: strategic retrieval for the system prompt
  - get_scratchpad_context() → vmPFC: integrating relevant knowledge for tools
  - encode() → executive decision to encode (directing the hippocampus)
  - encoding_gate() → encoding gate modulated by the memory mode

The Cortex coordinates two Hippocampus instances (global + project scope),
like how the PFC coordinates retrieval from multiple brain memory systems.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from anton.memory.hippocampus import Engram, Hippocampus

if TYPE_CHECKING:
    from anton.llm.client import LLMClient


_IDENTITY_EXTRACT_PROMPT = """\
Extract identity facts from this user message. Return a JSON array of strings,
each a concise fact about the user (name, timezone, expertise, preferences, tools).

If no identity-relevant information is found, return [].

Examples of identity facts:
- "Name: Jorge"
- "Timezone: PST"
- "Prefers dark mode"
- "Uses uv over pip"

Only extract facts that are clearly about the user's identity, preferences,
or working style. Ignore transient conversation details.
"""

_COMPACTION_PROMPT = """\
You are a memory compaction system. Review these memory entries and:
1. Remove exact duplicates
2. Merge entries that say the same thing differently — keep the clearest version
3. Remove entries that are superseded by newer, more specific entries
4. Keep all unique, useful entries

Return a JSON object with:
- "kept": array of entry strings to keep (cleaned up, no metadata comments)
- "merged": array of strings describing what was merged
- "pruned": array of strings describing what was removed and why

Be conservative — when in doubt, keep the entry.
"""


class Cortex:
    """Executive coordinator for Anton's memory systems.

    Manages two Hippocampus instances (global + project scope), decides what
    memories to load into working memory (the context window), and gates
    encoding based on the current memory mode (the neuromodulatory setting).
    """

    def __init__(
        self,
        global_dir: Path,
        project_dir: Path,
        mode: str = "copilot",
        llm_client: LLMClient | None = None,
    ) -> None:
        """Initialize the executive with two hippocampal stores.

        Args:
            global_dir: Path to ~/.anton/memory/ (cross-project memories)
            project_dir: Path to <project>/.anton/memory/ (project-specific)
            mode: Memory mode — autopilot|copilot|manual|off (encoding gate)
            llm_client: For LLM-assisted operations (profile extraction, compaction)
        """
        self.global_hc = Hippocampus(global_dir)
        self.project_hc = Hippocampus(project_dir)
        self.mode = mode
        self._llm = llm_client
        self._turn_count = 0

    def build_memory_context(self) -> str:
        """Assemble memories for the system prompt — the 'working memory' load.

        Like the dlPFC performing strategic retrieval: selects what enters
        the context window based on relevance and budget.

        Retrieval order (mirrors neural priority):
          1. Identity (profile) — always loaded, ~300 tokens
          2. Global rules — behavioral constraints, ~1500 tokens
          3. Project rules — scope-specific constraints, ~1500 tokens
          4. Global lessons — semantic knowledge, ~1000 tokens
          5. Project lessons — scope-specific facts, ~1000 tokens

        Total budget: ~5800 tokens (~3% of 200K context).
        """
        sections: list[str] = []

        # 1. Identity (global only — identity is singular)
        identity = self.global_hc.recall_identity()
        if identity:
            sections.append(f"## Your Memory — Identity\n{identity}")

        # 2. Global rules
        global_rules = self.global_hc.recall_rules()
        if global_rules:
            sections.append(f"## Your Memory — Global Rules\n{global_rules}")

        # 3. Project rules
        project_rules = self.project_hc.recall_rules()
        if project_rules:
            sections.append(f"## Your Memory — Project Rules\n{project_rules}")

        # 4. Global lessons
        global_lessons = self.global_hc.recall_lessons(token_budget=1000)
        if global_lessons:
            sections.append(f"## Your Memory — Global Lessons\n{global_lessons}")

        # 5. Project lessons
        project_lessons = self.project_hc.recall_lessons(token_budget=1000)
        if project_lessons:
            sections.append(f"## Your Memory — Project Lessons\n{project_lessons}")

        if not sections:
            return ""

        return "\n\n" + "\n\n".join(sections)

    def get_scratchpad_context(self) -> str:
        """Retrieve procedural knowledge for scratchpad tool injection.

        Like the vmPFC integrating memories for action planning — combines
        global + project scratchpad wisdom into a coherent set of guidelines.
        """
        parts: list[str] = []

        global_wisdom = self.global_hc.recall_scratchpad_wisdom()
        if global_wisdom:
            parts.append(global_wisdom)

        project_wisdom = self.project_hc.recall_scratchpad_wisdom()
        if project_wisdom:
            parts.append(project_wisdom)

        return "\n".join(parts)

    async def encode(self, engrams: list[Engram]) -> list[str]:
        """Direct the hippocampus to encode new memories.

        Routes each engram to the appropriate hippocampal store based on scope.
        Returns list of actions taken for logging.
        """
        if self.mode == "off":
            return ["Memory encoding is disabled."]

        actions: list[str] = []
        for engram in engrams:
            hc = self.global_hc if engram.scope == "global" else self.project_hc

            if engram.kind == "profile":
                # Profile entries accumulate, then rewrite
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
        """Whether this engram needs user confirmation before encoding.

        Brain analog: the Locus Coeruleus-NE system modulating encoding gain.
        - autopilot (high NE): encode everything → never confirm
        - copilot (moderate NE): auto-encode high-confidence, confirm ambiguous
        - manual (low NE): always confirm
        - off (suppressed ACh): never encode (but also never writes)
        """
        if self.mode == "autopilot":
            return False
        if self.mode == "manual":
            return True
        if self.mode == "off":
            return False  # Won't reach encoding anyway
        # copilot: auto-encode high confidence user-sourced, confirm rest
        return engram.confidence != "high"

    # --- Compaction: Systems Consolidation + Synaptic Homeostasis ---

    _COMPACTION_THRESHOLD = 50  # entries before compaction triggers

    def needs_compaction(self) -> bool:
        """Check if memory files have grown beyond the compaction threshold.

        Brain analog: synaptic saturation — during waking hours, synapses
        strengthen indiscriminately. When the load exceeds a threshold,
        consolidation/pruning is triggered.
        """
        return (
            self.global_hc.entry_count() > self._COMPACTION_THRESHOLD
            or self.project_hc.entry_count() > self._COMPACTION_THRESHOLD
        )

    async def compact_all(self) -> None:
        """Run systems consolidation on all memory files.

        Brain analog: the Synaptic Homeostasis Hypothesis (Tononi-Cirelli).
        Uses the coding model for fast, cheap deduplication.
        """
        if self._llm is None:
            return

        for hc in (self.global_hc, self.project_hc):
            if hc.entry_count() > self._COMPACTION_THRESHOLD:
                await self._compact_file(hc, hc._lessons_path, "lesson")
                await self._compact_file(hc, hc._rules_path, "rules")

    async def _compact_file(self, hc: Hippocampus, path: Path, kind: str) -> None:
        """Compact a single memory file using LLM-assisted deduplication."""
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
            return  # Don't corrupt memory on failure

        if not kept:
            return

        # Rebuild the file
        if kind == "rules":
            # Preserve section structure
            always = [e for e in kept if "always" in e.lower() or not any(
                k in e.lower() for k in ("never", "when", "if ")
            )]
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

    # --- Profile Extraction ---

    async def maybe_update_identity(self, user_message: str) -> None:
        """Check if conversation reveals identity facts worth profiling.

        Brain analog: the Default Mode Network passively monitoring for
        self-relevant information. Runs infrequently (every ~5 turns)
        to avoid overhead. Uses fast coding model for classification.
        """
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

        # Merge with existing identity
        existing = self.global_hc.recall_identity()
        existing_entries: list[str] = []
        if existing:
            for line in existing.splitlines():
                stripped = line.strip()
                if stripped.startswith("- "):
                    existing_entries.append(stripped[2:])
                elif stripped and not stripped.startswith("#"):
                    existing_entries.append(stripped)

        # Add new facts, avoiding duplicates
        for fact in facts:
            if isinstance(fact, str) and fact not in existing_entries:
                # Check if this updates an existing fact (same key prefix)
                key = fact.split(":")[0].strip().lower() if ":" in fact else ""
                if key:
                    existing_entries = [
                        e for e in existing_entries
                        if not e.lower().startswith(key + ":")
                    ]
                existing_entries.append(fact)

        if existing_entries:
            self.global_hc.rewrite_identity(existing_entries)
