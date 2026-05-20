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
from anton.core.memory.base import Engram
from anton.core.memory.hippocampus import Hippocampus
from anton.core.memory.ranker import Ranker
from anton.core.memory.rule_stats import RuleStats, rule_id

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

        # Layer 3 — retrieval-scored rule ranking.
        # Stateless BM25 ranker (no LLM call, no API key). The
        # cortex re-uses it for both global and project rule paths.
        self._ranker = Ranker()
        # Per-rule retrieval / outcome counters. Sidecar JSON lives
        # alongside global rules.md (one file across project switches —
        # rules can fire either scope). Best-effort: when the
        # hippocampus is a remote / protocol-only backend without a
        # local `_dir`, RuleStats stays None and the cortex skips the
        # counter bumps without losing the ranker behaviour itself.
        global_dir = getattr(global_hc, "_dir", None)
        self._rule_stats: RuleStats | None = (
            RuleStats(Path(global_dir) / "rules.stats.json")
            if isinstance(global_dir, (str, Path))
            else None
        )
        # Phase C — outcome bridge. Cumulative set of rule IDs that
        # landed in this turn's system prompt across however many
        # `build_memory_context` calls happened (a turn may rebuild
        # the prompt mid-flight on retries/compaction). The ACC's
        # end-of-turn flush drains this via `consume_retrieved_this_turn`
        # and, for each lesson whose rule_id is in the set, bumps the
        # corresponding rule's `ignored` counter — the LLM saw the
        # rule and the pattern still fired.
        self._retrieved_this_turn: set[str] = set()

        # One-time migration: identity is singular and global. Any entries that
        # landed in project scope from the old encode() bug are merged upward.
        # Global wins on key conflicts — orphaned entries are likely stale
        # (the bug wrote them; the user may have since corrected to global),
        # so we only import keys that don't already exist globally.
        orphaned = [e.text for e in self.project_hc.get_identities()]
        if orphaned:
            existing_global_keys = {
                e.text.split(":", 1)[0].strip().lower()
                for e in self.global_hc.get_identities()
                if ":" in e.text
            }
            to_migrate = [
                fact
                for fact in orphaned
                if not (
                    ":" in fact
                    and fact.split(":", 1)[0].strip().lower() in existing_global_keys
                )
            ]
            if to_migrate:
                self.global_hc.rewrite_identity(to_migrate)
            self.project_hc.clear_identity()

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
        identity = self.global_hc.recall_identities()
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

        # Layer 3 — flush the buffered retrieval counters once per
        # build (one disk write per turn, not one per rule). Best-
        # effort: if there's no stats backing store (remote
        # hippocampus, missing dir), this is a no-op.
        if self._rule_stats is not None:
            try:
                self._rule_stats.flush()
            except OSError:
                # Stats are telemetry, not gating data — a failed write
                # must not break system-prompt assembly.
                pass

        if not sections:
            return ""

        return "\n\n" + "\n\n".join(sections)

    # Regex to strip <!-- ... --> metadata from a rule line. Module-
    # scoped at the class for readability; cheap to (re)compile.
    import re as _re
    _METADATA_RE = _re.compile(r"<!--.*?-->", _re.DOTALL)

    def _extract_rule_body(self, line: str) -> str:
        """Pull the human-readable rule text out of a bullet line.

        ``- Use httpx instead of requests <!-- confidence:high ts:... -->``
        →
        ``Use httpx instead of requests``

        Used as both the BM25 document AND the stable-hash input for
        `RuleStats`, so the metadata comments (which carry per-rule
        timestamps that change on every write) don't corrupt either.
        """
        s = (line or "").strip()
        if s.startswith("- "):
            s = s[2:].strip()
        s = self._METADATA_RE.sub("", s).strip()
        return s

    def _record_retrievals_for_lines(self, lines: list[str]) -> None:
        """Bump retrieval counters for every rule-bullet line that
        actually carries content. Section headers / blank lines are
        skipped — they aren't rules. No-op when ``_rule_stats`` is
        unavailable (remote backend, etc.).

        Also populates ``self._retrieved_this_turn`` (rule-ID set)
        so the Phase C outcome bridge can correlate fired lessons
        against rules-that-were-actually-loaded."""
        if self._rule_stats is None:
            return
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith("- "):
                continue
            body = self._extract_rule_body(line)
            if body:
                self._rule_stats.record_retrieval(body)
                self._retrieved_this_turn.add(rule_id(body))

    def consume_retrieved_this_turn(self) -> set[str]:
        """Return the set of rule IDs retrieved into the system prompt
        since the last call, AND clear the set.

        Take-and-clear: the consumer (typically the ACC end-of-turn
        flush) reads the snapshot once per turn. Multiple consumers
        would each see a different filtered view, which is rarely
        what callers want — if more than one consumer needs the
        signal, build a fan-out at the wiring layer instead.

        Empty set is a valid answer (cold start, no rules in memory,
        or remote hippocampus where stats tracking is disabled)."""
        out = self._retrieved_this_turn
        self._retrieved_this_turn = set()
        return out

    async def _retrieve_relevant_rules(self, all_rules: str, user_message: str) -> str:
        """Select the rules that go into the system prompt.

        Layer 3 — retrieval-scored rule ranking:

          - ``## Always`` / ``## Never`` rules are unconditional and
            always loaded in full. They're not ranked because ranking
            unconditional rules is a category error.
          - ``## When`` rules are ranked by BM25 relevance against the
            current ``user_message``. The top-K within budget land in
            the prompt; the rest are dropped for this turn.
          - Every rule that lands in the prompt bumps its retrieval
            counter via ``RuleStats``. Phase C (outcome bridge) will
            later use these counters to compute an "ignored" signal
            when ACC detects the corresponding pattern despite the
            rule having been loaded.

        Cold-start behaviour: when the corpus fits in the char budget
        OR the user message has no scorable terms, all rules are
        loaded and their retrievals recorded. The ranker is a
        budget-pressure tool, not a permanent filter.

        Brain analog: dlPFC cue-dependent recall. The PFC scores
        relevance against current goals and activates the top
        candidates rather than loading everything into working memory.
        """
        # No query → unfiltered. Still record retrievals so the
        # telemetry is honest about what's in the prompt.
        if not user_message:
            self._record_retrievals_for_lines(all_rules.splitlines())
            return all_rules

        # Under budget → no point ranking; load all + record.
        if len(all_rules) <= self._RULES_BUDGET_CHARS:
            self._record_retrievals_for_lines(all_rules.splitlines())
            return all_rules

        # Split into mandatory (Always / Never / non-section lines) vs.
        # rankable (When bullets). Section headers stay with mandatory
        # so the output keeps its markdown structure.
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
                mandatory_lines.append(line)
            elif stripped.startswith("## ") or stripped.startswith("# "):
                current_section = ""
                mandatory_lines.append(line)
            elif current_section == "when":
                when_lines.append(line)
            else:
                mandatory_lines.append(line)

        # Tiny When section → no ranking work to do.
        when_text = "\n".join(when_lines).strip()
        if not when_text or len(when_text) < 1000:
            self._record_retrievals_for_lines(lines)
            return all_rules

        # Build (body, original_line) pairs so we can rank on bodies
        # but emit the original markdown lines (preserving metadata
        # comments the consumer / consolidator might still read).
        candidates: list[tuple[str, str]] = []
        for line in when_lines:
            body = self._extract_rule_body(line)
            if body:
                candidates.append((body, line))

        if not candidates:
            self._record_retrievals_for_lines(lines)
            return all_rules

        bodies = [b for b, _ in candidates]
        body_to_line = {b: l for b, l in candidates}

        ranked = self._ranker.rank(bodies, user_message)

        # Remaining char budget = total budget minus what mandatory
        # lines already consume. Convert to a rough token budget
        # (~4 chars/token English heuristic) for the ranker's selector.
        mandatory_chars = sum(len(l) + 1 for l in mandatory_lines)
        remaining_chars = max(0, self._RULES_BUDGET_CHARS - mandatory_chars)
        remaining_tokens = max(100, remaining_chars // 4)
        selected = self._ranker.select_within_budget(
            ranked, budget_tokens=remaining_tokens
        )

        selected_lines: list[str] = []
        for r in selected:
            line = body_to_line.get(r.text)
            if line is not None:
                selected_lines.append(line)

        # Record retrievals for everything that lands in the prompt —
        # mandatory rules AND the selected When rules. (Section
        # headers and blanks are filtered inside the helper.)
        self._record_retrievals_for_lines(mandatory_lines + selected_lines)

        output = "\n".join(mandatory_lines)
        if selected_lines:
            output += "\n" + "\n".join(selected_lines)
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
            if engram.kind == "profile":
                hc = self.global_hc
            else:
                hc = self.global_hc if engram.scope == "global" else self.project_hc

            if engram.kind == "profile":
                hc.rewrite_identity([engram.text])

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

        self.global_hc.rewrite_identity(facts)
