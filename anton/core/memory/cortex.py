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

The Cortex coordinates two HippocampusProtocol instances (global + project scope),
like how the PFC coordinates retrieval from multiple brain memory systems.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from anton.core.memory.base import HippocampusProtocol
from anton.core.memory.hippocampus import Engram, Hippocampus

if TYPE_CHECKING:
    from anton.core.llm.client import LLMClient


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas — used by LLMClient.generate_object
# ─────────────────────────────────────────────────────────────────────────────


class _IdentityFacts(BaseModel):
    """Result of the identity-extraction LLM call.

    Each fact is a concise statement about the user (name, timezone,
    expertise, preferences, tools). Empty list when nothing relevant
    is found in the message.
    """

    facts: list[str] = Field(
        default_factory=list,
        description=(
            "Identity facts extracted from the user message. Each fact "
            "is a concise statement about the user. Examples: "
            "'Name: Jorge', 'Timezone: PST', 'Prefers dark mode', "
            "'Uses uv over pip'. Only extract facts that are clearly "
            "about the user's identity, preferences, or working style. "
            "Ignore transient conversation details. Return an empty list "
            "if nothing identity-relevant is found."
        ),
    )


class _CompactionResult(BaseModel):
    """Result of the memory-compaction LLM call.

    Returns the deduplicated entries to keep, plus optional metadata
    about what was merged and pruned (purely for logging — the cortex
    only acts on `kept`).
    """

    kept: list[str] = Field(
        ...,
        description=(
            "Entry strings to keep after compaction. Preserve the "
            "trailing `<!-- ... -->` metadata comment on each entry "
            "exactly as it appears in the input."
        ),
    )
    merged: list[str] = Field(
        default_factory=list,
        description="Strings describing what was merged (for logging).",
    )
    pruned: list[str] = Field(
        default_factory=list,
        description="Strings describing what was removed and why (for logging).",
    )


_IDENTITY_EXTRACT_PROMPT = """\
Extract identity facts from this user message — concise statements about the user (name, timezone, expertise, preferences, tools). Only extract facts that are clearly about the user's identity, preferences, or working style. Ignore transient conversation details. Return an empty list if nothing identity-relevant is found.
"""

_COMPACTION_PROMPT = """\
You are a memory compaction system. Review these memory entries and:
1. Remove exact duplicates
2. Merge entries that say the same thing differently — keep the clearest version
3. Remove entries that are superseded by newer, more specific entries
4. Keep all unique, useful entries

Be conservative — when in doubt, keep the entry. Preserve the trailing `<!-- ... -->` metadata comment on each kept entry exactly as it appears.
"""


class Cortex:
    """Executive coordinator for Anton's memory systems.

    Manages two HippocampusProtocol instances (global + project scope), decides what
    memories to load into working memory (the context window), and gates
    encoding based on the current memory mode (the neuromodulatory setting).
    """

    def __init__(
        self,
        global_hc: HippocampusProtocol,
        project_hc: HippocampusProtocol,
        mode: str = "autopilot",
        llm_client: LLMClient | None = None,
    ) -> None:
        """Initialize the executive with two hippocampal stores.

        Args:
            global_hc: Memory store for cross-project memories (global scope)
            project_hc: Memory store for project-specific memories
            mode: Memory mode — autopilot|copilot|off (encoding gate)
            llm_client: For LLM-assisted operations (profile extraction, compaction)
        """
        self.global_hc = global_hc
        self.project_hc = project_hc
        self.mode = mode
        self._llm = llm_client
        self._turn_count = 0

    # ~6000 chars ≈ ~1500 tokens — above this, use LLM to filter rules
    _RULES_BUDGET_CHARS = 6000

    _RULES_RETRIEVAL_PROMPT = """\
Given the user's current message, select only the conditional (When/If) rules that are \
relevant. Return the selected rules exactly as they appear, one per line (keep the "- " prefix).
If all rules are relevant, return them all. If none are relevant, return "NONE".
Do NOT add, modify, or summarize rules — return them verbatim.
"""

    async def build_memory_context(self, user_message: str = "") -> str:
        """Assemble memories for the system prompt — the 'working memory' load.

        Like the dlPFC performing strategic retrieval: selects what enters
        the context window based on relevance and budget.

        Args:
            user_message: Current user message for cue-dependent retrieval.
                When rules exceed the token budget, only relevant rules are loaded.
        """
        sections: list[str] = []

        # 1. Identity (global only — identity is singular)
        identity = self.global_hc.recall_identity()
        if identity:
            sections.append(f"## Your Memory — Identity\n{identity}")

        # 2. Global rules (with smart retrieval)
        global_rules = self.global_hc.recall_rules()
        if global_rules:
            global_rules = await self._retrieve_relevant_rules(
                global_rules, user_message
            )
            if global_rules:
                sections.append(f"## Your Memory — Global Rules\n{global_rules}")

        # 3. Project rules (with smart retrieval)
        project_rules = self.project_hc.recall_rules()
        if project_rules:
            project_rules = await self._retrieve_relevant_rules(
                project_rules, user_message
            )
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

        # 6. Minds datasource context (auto-loaded if present)
        minds_topic = self.project_hc.recall_topic("minds-datasource")
        if minds_topic:
            sections.append(f"## Minds — Datasource Context\n{minds_topic}")

        if not sections:
            return ""

        return "\n\n" + "\n\n".join(sections)

    async def _retrieve_relevant_rules(self, all_rules: str, user_message: str) -> str:
        """Filter rules to only those relevant to the current user message.

        Brain analog: dlPFC cue-dependent recall — the prefrontal cortex
        selects which memories to activate based on current goals, rather
        than loading everything into working memory.

        Always/Never rules are behavioral constraints — always loaded in full.
        Only conditional (When/If) rules are filtered by relevance.
        If rules are under budget or no LLM is available, returns as-is.
        """
        if not user_message or self._llm is None:
            return all_rules
        if len(all_rules) <= self._RULES_BUDGET_CHARS:
            return all_rules

        # Split rules into mandatory (Always/Never) and filterable (When)
        lines = all_rules.splitlines()
        mandatory_lines: list[str] = []
        when_lines: list[str] = []
        current_section = ""

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("## Always"):
                current_section = "always"
                mandatory_lines.append(line)
            elif stripped.startswith("## Never"):
                current_section = "never"
                mandatory_lines.append(line)
            elif stripped.startswith("## When"):
                current_section = "when"
                mandatory_lines.append(line)  # keep the header
            elif stripped.startswith("## ") or stripped.startswith("# "):
                current_section = ""
                mandatory_lines.append(line)
            elif current_section == "when":
                when_lines.append(line)
            else:
                mandatory_lines.append(line)

        # If When section is small, no need to filter
        when_text = "\n".join(when_lines).strip()
        if not when_text or len(when_text) < 1000:
            return all_rules

        # Filter only the When rules
        try:
            response = await self._llm.code(
                system=self._RULES_RETRIEVAL_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": f"User message: {user_message}\n\nRules:\n{when_text}",
                    }
                ],
                max_tokens=4096,
            )
            result = response.content.strip()
            if result == "NONE":
                filtered_when = ""
            elif result:
                filtered_when = result
            else:
                filtered_when = when_text
        except Exception:
            filtered_when = when_text

        # Reassemble: mandatory sections + filtered When rules
        output = "\n".join(mandatory_lines)
        if filtered_when:
            output += "\n" + filtered_when
        return output

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
        - off (suppressed ACh): never encode (but also never writes)

        Confirmations are always deferred until after the user has received
        their answer — never shown during scratchpad execution or mid-turn.
        """
        if self.mode == "autopilot":
            return False
        if self.mode == "off":
            return False  # Won't reach encoding anyway
        # copilot: auto-encode high confidence user-sourced, confirm rest
        return engram.confidence != "high"

    # --- Compaction: Systems Consolidation + Synaptic Homeostasis ---

    _COMPACTION_THRESHOLD = 20  # entries before compaction triggers
    _VACUUM_INTERVAL = 10  # check compaction every N turns

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
            if not isinstance(hc, Hippocampus):
                continue  # compaction is file-specific; non-file backends skip
            if hc.entry_count() > self._COMPACTION_THRESHOLD:
                await self._compact_file(hc, hc._lessons_path, "lesson")
                await self._compact_file(hc, hc._rules_path, "rules")

    async def vacuum(self) -> None:
        """Run compaction unconditionally on all memory files.

        Public entry point for on-demand cleanup (e.g. after /connect).
        Unlike compact_all(), skips the threshold check — always runs.
        """
        if self._llm is None:
            return
        for hc in (self.global_hc, self.project_hc):
            if not isinstance(hc, Hippocampus):
                continue  # compaction is file-specific; non-file backends skip
            await self._compact_file(hc, hc._lessons_path, "lesson")
            await self._compact_file(hc, hc._rules_path, "rules")

    def maybe_vacuum(self) -> None:
        """Periodic vacuum check — call after each assistant turn.

        Every _VACUUM_INTERVAL turns, checks if compaction is needed and
        fires it in the background if so.
        """
        import asyncio

        self._turn_count += 1
        if self._turn_count % self._VACUUM_INTERVAL != 0:
            return
        if not self.needs_compaction():
            return
        asyncio.create_task(self.compact_all())

    async def _compact_file(self, hc: Hippocampus, path: Path, kind: str) -> None:
        """Compact a single memory file using LLM-assisted deduplication."""
        if not path.is_file():
            return

        content = path.read_text(encoding="utf-8")
        entries = [
            ln.strip() for ln in content.splitlines() if ln.strip().startswith("- ")
        ]

        if len(entries) < 8:
            return

        try:
            result: _CompactionResult = await self._llm.generate_object_code(
                _CompactionResult,
                system=_COMPACTION_PROMPT,
                messages=[{"role": "user", "content": "\n".join(entries)}],
                max_tokens=4096,
            )
            kept = result.kept or entries
        except Exception:
            return  # Don't corrupt memory on failure

        if not kept:
            return

        # Rebuild the file
        if kind == "rules":
            # Preserve section structure
            always = [
                e
                for e in kept
                if "always" in e.lower()
                or not any(k in e.lower() for k in ("never", "when", "if "))
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
        """Check if conversation reveals identity facts worth profiling.

        Brain analog: the Default Mode Network passively monitoring for
        self-relevant information. Runs infrequently (every ~5 turns)
        to avoid overhead. Uses fast coding model for classification.
        """
        if self._llm is None or self.mode == "off":
            return

        try:
            result: _IdentityFacts = await self._llm.generate_object_code(
                _IdentityFacts,
                system=_IDENTITY_EXTRACT_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                max_tokens=512,
            )
            facts = result.facts
            if not facts:
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
                        e
                        for e in existing_entries
                        if not e.lower().startswith(key + ":")
                    ]
                existing_entries.append(fact)

        if existing_entries:
            self.global_hc.rewrite_identity(existing_entries)
