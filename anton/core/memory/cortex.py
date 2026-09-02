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

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from anton.core.llm.tracing import tagged_trace
from anton.core.llm.structured import (
    generate_with_truncation_retry,
    no_preamble_instruction,
)
from anton.core.memory.base import HippocampusProtocol
from anton.core.memory.base import Engram
from anton.core.memory.safety import assess_automatic_memory, is_safe_for_prompt
from anton.core.memory.hippocampus import Hippocampus

if TYPE_CHECKING:
    from anton.core.llm.client import LLMClient
    from anton.core.memory.episodes import EpisodicMemory

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Rule-retrieval observability (ENG-1390)
# ─────────────────────────────────────────────────────────────────────────────
# `_retrieve_relevant_rules` makes an LLM call during PROMPT ASSEMBLY, through
# the same `llm.code` entry point as the completion verifier and every other
# coding call — so it was indistinguishable from them in a trace, and its
# frequency was unknown. Measured retroactively before this shipped (ENG-1390):
# 1.77% of all LLM calls, 4 users, median 4.3k input tokens, p90 latency 23.9s
# added BEFORE the turn starts.
#
# Two sinks, because neither alone is sufficient:
#   * the trace tag isolates the call in Langfuse, but only for MindsHub-routed
#     traffic with a turn in flight;
#   * the analytics event works for every user and every provider, and is the
#     only way to observe the EXCEPTION outcome at all (a failed call leaves no
#     observation to count).
_RULE_RETRIEVAL_TAG = "rule-retrieval"
_RULE_RETRIEVAL_MAX_TOKENS = 4096

# Built once: `AntonSettings()` reads the environment, and while this call site
# is rare it sits on the latency-critical prompt-assembly path.
_analytics_settings = None


def _emit_rule_retrieval(**shape) -> None:
    """Report one rule-retrieval call's SHAPE to analytics.

    Counts, sizes and enums ONLY — never rule text and never the user message,
    both of which are user content (ENG-1390's security note, and the standing
    rule for this sink: numbers, names and IDs only).

    Every `extra` kwarg becomes a queryable PostHog property, so these parameter
    names are a published schema — renaming one breaks whatever reads it.
    Silent on failure: observability must never break prompt assembly.
    """
    global _analytics_settings
    try:
        if _analytics_settings is None:
            from anton.config.settings import AntonSettings

            _analytics_settings = AntonSettings()
        from anton.analytics import send_event

        send_event(
            _analytics_settings,
            "rule_retrieval",
            **{key: str(value) for key, value in shape.items()},
        )
    except Exception:
        logger.debug("rule_retrieval analytics event failed", exc_info=True)


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

    Survivors are named by INDEX, never echoed back as text. Two reasons, and
    the first is why compaction was broken:

    - Echoing entries made the response scale with the whole file, so a file
      big enough to need compaction was also big enough to blow the output
      budget. Because a failed compaction leaves the file untouched, it then
      grew and failed again — once over the line, never compacted again.
    - Model text never reaches the file, so compaction can drop an entry but
      can never reword, truncate or corrupt one.

    The cost is that compaction selects, it does not rewrite: two entries that
    each hold a different detail can only both be kept, not combined.
    """

    keep: list[int] = Field(
        ...,
        description=(
            "Indices of the numbered entries to keep. Any index left out is "
            "deleted, so name every entry that should survive."
        ),
    )


_IDENTITY_EXTRACT_PROMPT = """\
Extract identity facts from this user message — concise statements about the user (name, timezone, expertise, preferences, tools). Only extract facts that are clearly about the user's identity, preferences, or working style. Ignore transient conversation details. Return an empty list if nothing identity-relevant is found.
"""

_COMPACTION_PROMPT = """\
You are a memory compaction system. The user message is a numbered list of memory entries. Report the indices of the entries to keep:
1. Of exact duplicates, keep one
2. Where several entries say the same thing differently, keep only the clearest of them — you cannot reword an entry, only choose between the ones you were given
3. Drop entries superseded by newer, more specific ones
4. Keep every unique, useful entry

An index you leave out is deleted. Be conservative — when in doubt, keep the entry: two overlapping entries cost less than losing what only one of them said.
"""

# The `## ` headings of `rules.md`, in file order. `save_rules` writes exactly
# these three and `get_rules` reads each entry's `kind` off them, so compaction
# has to round-trip the same set. First one doubles as the fallback heading.
_RULE_SECTIONS = ("always", "never", "when")


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
        episodic: EpisodicMemory | None = None,
    ) -> None:
        """Initialize the executive with two hippocampal stores.

        Args:
            global_hc: Memory store for cross-project memories (global scope)
            project_hc: Memory store for project-specific memories
            mode: Memory mode — autopilot|copilot|off (encoding gate)
            llm_client: For LLM-assisted operations (profile extraction, compaction)
            episodic: For logging memory_read/memory_write events per session
        """
        self.global_hc = global_hc
        self.project_hc = project_hc
        self.mode = mode
        self._llm = llm_client
        self._episodic = episodic
        self._seen_texts: set[str] = set()
        self._turn_count = 0

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

        # 2. Global rules (with smart retrieval). get_rules(exclude_scratchpad_when=True)
        # drops scratchpad-related "when" rules — those are already injected
        # into the scratchpad tool description by get_scratchpad_context(),
        # and showing them here too would double their token cost.
        global_engrams = [
            engram for engram in self.global_hc.get_rules(exclude_scratchpad_when=True)
            if is_safe_for_prompt(engram)
        ]
        if global_engrams:
            global_engrams = await self._retrieve_relevant_rules(global_engrams, user_message)
            if global_engrams:
                sections.append(
                    f"## Your Memory — Global Rules\n{self._format_rules_engrams(global_engrams)}"
                )

        # 3. Project rules (with smart retrieval) — same scratchpad exclusion.
        project_engrams = [
            engram for engram in self.project_hc.get_rules(exclude_scratchpad_when=True)
            if is_safe_for_prompt(engram)
        ]
        if project_engrams:
            project_engrams = await self._retrieve_relevant_rules(project_engrams, user_message)
            if project_engrams:
                sections.append(
                    f"## Your Memory — Project Rules\n{self._format_rules_engrams(project_engrams)}"
                )
                for engram in project_engrams:
                    self._log_read_engram(engram)

        # 4. Global lessons. recall_lessons() excludes scratchpad-related
        # entries internally — same reasoning as the rules exclusion above.
        global_lesson_engrams = [
            engram for engram in self.global_hc.get_lessons(token_budget=1000, exclude_scratchpad=True)
            if is_safe_for_prompt(engram)
        ]
        if global_lesson_engrams:
            global_lessons = "\n".join(f"- {engram.text}" for engram in global_lesson_engrams)
            sections.append(f"## Your Memory — Global Lessons\n{global_lessons}")

        # 5. Project lessons — same scratchpad exclusion.
        project_lesson_engrams = [
            engram for engram in self.project_hc.get_lessons(token_budget=1000, exclude_scratchpad=True)
            if is_safe_for_prompt(engram)
        ]
        if project_lesson_engrams:
            project_lessons = "\n".join(f"- {engram.text}" for engram in project_lesson_engrams)
            sections.append(f"## Your Memory — Project Lessons\n{project_lessons}")
            if self._episodic is not None:
                for engram in project_lesson_engrams:
                    self._log_read_engram(engram)

        # 6. Minds datasource context (auto-loaded if present)
        minds_topic = self.project_hc.recall_topic("minds-datasource")
        if minds_topic:
            sections.append(f"## Minds — Datasource Context\n{minds_topic}")

        if not sections:
            return ""

        return "\n\n" + "\n\n".join(sections)

    @staticmethod
    def _format_rules_engrams(engrams: list[Engram]) -> str:
        """Format rule engrams to section display format (## Always / Never / When)."""
        by_kind: dict[str, list[Engram]] = {}
        for e in engrams:
            by_kind.setdefault((e.kind or "always").lower(), []).append(e)
        parts: list[str] = []
        for section in ("always", "never", "when"):
            items = by_kind.get(section, [])
            if items:
                parts.append(f"## {section.capitalize()}")
                parts.extend(f"- {e.text}" for e in items)
        return "\n".join(parts)

    async def _retrieve_relevant_rules(
        self, engrams: list[Engram], user_message: str
    ) -> list[Engram]:
        """Filter rule engrams to those relevant to the current user message.

        Brain analog: dlPFC cue-dependent recall — the prefrontal cortex
        selects which memories to activate based on current goals, rather
        than loading everything into working memory.

        Always/Never rules are behavioral constraints — always loaded in full.
        Only conditional (When/If) rules are filtered by relevance.
        If rules are under budget or no LLM is available, returns as-is.
        """
        if not user_message or self._llm is None:
            return engrams

        if len(self._format_rules_engrams(engrams)) <= self._RULES_BUDGET_CHARS:
            return engrams

        mandatory = [e for e in engrams if (e.kind or "").lower() in ("always", "never")]
        when_engrams = [e for e in engrams if (e.kind or "").lower() == "when"]

        if not when_engrams:
            return engrams

        when_text = "\n".join(f"- {e.text}" for e in when_engrams)
        if len(when_text) < 1000:
            return engrams

        # ENG-1390 instrumentation. Purely observational — every branch below
        # produces exactly the value it produced before. What is new is that the
        # branches are now DISTINGUISHABLE: `filtered_when = when_engrams` is
        # reached three different ways (no exact match, empty response, exception)
        # and all three are byte-identical in their effect on the prompt, so a
        # count of invocations could never tell a working filter from a broken one.
        # Measured shares before this shipped: 55.7% filtered, 30.2% dropped_all,
        # 14.2% kept_all_no_match, plus an exception rate that was unobservable.
        outcome = "error"
        kept = len(when_engrams)
        stop_reason = ""
        usage = None
        early: list[Engram] | None = None
        started = time.monotonic()
        try:
            # Isolate this call from every other `llm.code` caller in the trace.
            with tagged_trace(_RULE_RETRIEVAL_TAG):
                response = await self._llm.code(
                    system=self._RULES_RETRIEVAL_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": f"User message: {user_message}\n\nRules:\n{when_text}",
                        }
                    ],
                    max_tokens=_RULE_RETRIEVAL_MAX_TOKENS,
                )
            usage = getattr(response, "usage", None)
            stop_reason = getattr(response, "stop_reason", None) or ""
            result = response.content.strip()
            if result.upper() == "NONE":
                # A fourth outcome the ticket did not name: EVERY conditional rule
                # is discarded. The opposite of the permissive fallback, and ~30%
                # of calls.
                outcome, kept, early = "dropped_all", 0, mandatory
            elif result:
                returned = {line.lstrip("- ").strip() for line in result.splitlines() if line.strip()}
                filtered_when = [e for e in when_engrams if e.text in returned]
                if not filtered_when:
                    # Exact string equality against the model's echo. Any rewrap,
                    # added punctuation, or a response truncated at the ceiling
                    # yields zero matches — and then we silently keep everything.
                    outcome, filtered_when = "kept_all_no_match", when_engrams
                else:
                    outcome, kept = "filtered", len(filtered_when)
            else:
                outcome, filtered_when = "kept_all_empty", when_engrams
        except asyncio.CancelledError:
            # A user pressing STOP, or an abandoned SSE stream, cancels the turn
            # mid-call. `CancelledError` is a BaseException, so `except Exception`
            # below does NOT catch it — but the `finally` still fires, and without
            # this branch every user abort was reported as `error`, inflating the
            # filter-failure rate in the one metric this instrumentation exists to
            # produce. Reported distinctly and re-raised: cancellation is not ours
            # to swallow. (`kept_rules` is meaningless here — no decision was
            # reached — which is why consumers must key on `outcome` first.)
            outcome = "cancelled"
            raise
        except Exception:
            filtered_when = when_engrams
            outcome = "error"
        finally:
            _emit_rule_retrieval(
                outcome=outcome,
                when_rules=len(when_engrams),
                kept_rules=kept,
                rules_chars=len(when_text),
                # `stop_reason`, not output_tokens == max_tokens: a truncated
                # verbatim echo is silently accepted as "the relevant rules", so
                # this is the difference between a real filter and one that just
                # ran out of room. The provider reason is authoritative where the
                # exact-cap heuristic gives false positives.
                stop_reason=stop_reason,
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                duration_ms=int((time.monotonic() - started) * 1000),
            )

        if early is not None:
            return early
        return mandatory + filtered_when

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
            decision = assess_automatic_memory(engram)
            if not decision.allowed:
                # Candidate text may contain tool output or a credential. Keep it
                # out of files, event logs, and user-facing action messages.
                actions.append(f"Rejected unsafe automatic memory ({decision.reason}).")
                continue

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
                if engram.scope != "global":
                    self._log_write_engram(engram)
                actions.append(f"Encoded {engram.kind} rule: {engram.text}")

            elif engram.kind == "lesson":
                hc.encode_lesson(
                    engram.text,
                    topic=engram.topic,
                    source=engram.source,
                )
                if engram.scope != "global":
                    self._log_write_engram(engram)
                actions.append(f"Encoded lesson: {engram.text}")

        return actions

    def _log_write_engram(self, engram: Engram) -> None:
        if self._episodic is None:
            return
        self._seen_texts.add(engram.text)
        self._episodic.log_turn(
            0, "memory_write", engram.text,
            kind=engram.kind or "lesson",
            topic=engram.topic or "",
        )

    def _log_read_engram(self, engram: Engram) -> None:
        if self._episodic is None or engram.text in self._seen_texts:
            return
        self._seen_texts.add(engram.text)
        self._episodic.log_turn(
            0, "memory_read", engram.text,
            kind=engram.kind or "lesson",
            topic=engram.topic or "",
        )

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

        # Read with sections
        sections: list[str] = []
        entries: list[str] = []
        # Entries above the first heading (a hand-edited file) are kept rather
        # than dropped, under the same heading the keyword pass defaulted to.
        section = _RULE_SECTIONS[0]
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                heading = stripped[3:].lower()
                section = heading if heading in _RULE_SECTIONS else _RULE_SECTIONS[0]
            elif stripped.startswith("- "):
                sections.append(section)
                entries.append(stripped)

        if len(entries) < 8:
            return

        # Numbers replace the `- ` marker so each line carries one, and the
        # `<!-- ... -->` comment stays: its `ts` is how the model can tell which
        # of two overlapping entries supersedes the other.
        numbered = "\n".join(
            f"{i}. {entry.removeprefix('- ')}" for i, entry in enumerate(entries, 1)
        )

        try:
            # The response is a list of integers, so it no longer scales with
            # the file. The ladder stays because the narration some models emit
            # before the forced tool call does not shrink with it (ENG-1084).
            result: _CompactionResult = await generate_with_truncation_retry(
                self._llm.generate_object_code,
                _CompactionResult,
                system=_COMPACTION_PROMPT + no_preamble_instruction(_CompactionResult),
                messages=[{"role": "user", "content": numbered}],
                budgets=(4096, 8192),
                log=logger,
                subsystem="memory-compaction",
            )
            # Entries are copied from `entries`, never from the response, so a
            # bad index costs at most the entry it failed to name and can never
            # alter one. Walking the file also bounds the indices for free:
            # duplicates collapse into the set, invented ones never come up.
            keep = set(result.keep)
            kept = [
                pair
                for i, pair in enumerate(zip(sections, entries), 1)
                if i in keep
            ]
        except Exception as exc:
            # Don't corrupt memory on failure — but never silently: an
            # unlogged swallow made a dead memory subsystem indistinguishable
            # from a working one (ENG-1084). Type name only; the message can
            # quote conversation-derived content.
            logger.warning(
                "memory-compaction failed (%s) — keeping the file as-is",
                type(exc).__name__,
            )
            return

        # Nothing usable came back — a model that named no valid index gets to
        # leave memory alone, not to empty it.
        if not kept:
            return

        # Rebuild the file. Every entry still carries the `- ` prefix it was
        # selected by, so nothing here has to re-add one.
        if kind == "rules":
            # Each entry goes back under its own heading and no other, so the
            # rewrite is idempotent — a rule cannot be filed twice.
            lines = ["# Rules\n"]
            for name in _RULE_SECTIONS:
                lines.append(f"## {name.capitalize()}")
                lines.extend(e for s, e in kept if s == name)
                lines.append("")
            new_content = "\n".join(lines[:-1]) + "\n"
        else:
            new_content = "\n".join(["# Lessons", *(e for _, e in kept)]) + "\n"

        hc._encode_with_lock(path, new_content, mode="write")

        # The only record that the file changed size, and the only pressure
        # against over-pruning: naming survivors by index makes dropping most
        # of the file a *shorter* answer than keeping it, so the call itself no
        # longer penalises it. Counts only — the entries are user content.
        logger.info(
            "memory-compaction: %s kept %d of %d entries",
            path.name,
            len(kept),
            len(entries),
        )

    async def maybe_update_identity(self, user_message: str) -> None:
        """Check if conversation reveals identity facts worth profiling.

        Brain analog: the Default Mode Network passively monitoring for
        self-relevant information. Runs infrequently (every ~5 turns)
        to avoid overhead. Uses fast coding model for classification.
        """
        if self._llm is None or self.mode == "off":
            return

        try:
            # 512 sat inside the measured narration range (245–1,654+), so
            # narrating models truncated on essentially every pass and the
            # silent except below hid it — confirmed live in prod on
            # `mindshub_air` (ENG-1084).
            result: _IdentityFacts = await generate_with_truncation_retry(
                self._llm.generate_object_code,
                _IdentityFacts,
                system=_IDENTITY_EXTRACT_PROMPT + no_preamble_instruction(_IdentityFacts),
                messages=[{"role": "user", "content": user_message}],
                log=logger,
                subsystem="identity-extraction",
            )
            facts = result.facts
            if not facts:
                return
        except Exception as exc:
            logger.warning(
                "identity-extraction failed (%s) — no facts stored this pass",
                type(exc).__name__,
            )
            return

        self.global_hc.rewrite_identity(facts)
