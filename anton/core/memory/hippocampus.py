"""Hippocampus — Anton's memory encoding and retrieval engine.

Named for the brain's hippocampus (CA3/CA1 subfields), which handles the
fundamental operations of memory: encoding new traces (writing) and
pattern-completing partial cues into full memories (reading).

The hippocampus doesn't decide *what* to remember — that's the cortex's job.
It simply executes storage and retrieval at a single scope (global or project),
like how the brain's hippocampus encodes at the level of individual memory traces
without executive judgment about importance.

Each Hippocampus instance manages one scope's files:
  - profile.md  → identity (mPFC / Default Mode Network analogy)
  - rules.md    → behavioral gates (Basal Ganglia / OFC analogy)
  - lessons.md  → semantic facts (Anterior Temporal Lobe analogy)
  - topics/*.md → domain expertise (Cortical Association Areas analogy)
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class Engram:
    """A single memory trace — the fundamental unit of memory.

    Named for Karl Lashley's 'engram' — the physical substrate of a memory.
    Each engram carries its content plus metadata about confidence, origin,
    and topic for later retrieval and consolidation.
    """

    text: str
    kind: Literal["always", "never", "when", "lesson", "profile"]
    scope: Literal["global", "project"]
    confidence: Literal["high", "medium", "low"] = "medium"
    topic: str = ""
    source: Literal["user", "consolidation", "llm"] = "llm"


class Hippocampus:
    """Reads and writes memory traces at a single scope (global OR project).

    Like the hippocampal CA3 region (pattern completion for reads) and CA1
    region (pattern separation for writes), this class handles the low-level
    mechanics of memory storage without higher-order decisions about relevance
    or importance.
    """

    def __init__(self, base_dir: Path) -> None:
        """Initialize for a single scope.

        Args:
            base_dir: ~/.anton/memory/ (global) or <project>/.anton/memory/ (project)
        """
        self._dir = base_dir
        self._profile_path = base_dir / "profile.md"
        self._rules_path = base_dir / "rules.md"
        self._lessons_path = base_dir / "lessons.md"
        self._topics_dir = base_dir / "topics"

    def recall_identity(self) -> str:
        """Load the always-on self-model (profile.md).

        Brain analog: medial Prefrontal Cortex / Default Mode Network.
        This is the identity substrate — always active, never "looked up",
        it contextualizes all other processing. Global scope only.
        """
        if not self._profile_path.is_file():
            return ""
        try:
            return self._profile_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return ""

    def recall_rules(self) -> str:
        """Load behavioral gates (rules.md) as formatted always/never/when.

        Brain analog: Basal Ganglia (Go/No-Go pathways) + Orbitofrontal Cortex
        (conditional behavioral rules). These aren't memories to recall —
        they're constraints that shape action selection.
        """
        if not self._rules_path.is_file():
            return ""
        try:
            return self._rules_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return ""

    def recall_lessons(self, token_budget: int = 1000) -> str:
        """Load semantic knowledge (lessons.md), most recent first, within budget.

        Brain analog: Anterior Temporal Lobe — the convergence hub for semantic
        facts distilled from many episodes. Budget enforced at ~4 chars/token.
        """
        if not self._lessons_path.is_file():
            return ""
        try:
            content = self._lessons_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return ""

        if not content:
            return ""

        # Extract individual entries (lines starting with "- ")
        lines = [ln for ln in content.splitlines() if ln.strip()]
        # Keep header, then entries in reverse order (most recent last → first)
        header_lines = []
        entry_lines = []
        for ln in lines:
            if ln.startswith("- ") or ln.startswith("  "):
                entry_lines.append(ln)
            else:
                header_lines.append(ln)

        # Reverse entries so most recent are first
        entry_lines.reverse()

        # Budget: ~4 chars per token
        char_budget = token_budget * 4
        result_lines = list(header_lines)
        used = sum(len(ln) for ln in result_lines)

        for ln in entry_lines:
            if used + len(ln) + 1 > char_budget:
                break
            result_lines.append(ln)
            used += len(ln) + 1

        return "\n".join(result_lines)

    def recall_topic(self, slug: str) -> str:
        """Load deep domain expertise on demand (topics/{slug}.md).

        Brain analog: Cortical Association Areas — specialized regions activated
        associatively when contextual cues indicate relevance.
        """
        safe_slug = self._sanitize_slug(slug)
        path = self._topics_dir / f"{safe_slug}.md"
        if not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return ""

    def recall_scratchpad_wisdom(self) -> str:
        """Retrieve procedural knowledge relevant to scratchpad execution.

        Returns all "when" rules + lessons with topic starting with "scratchpad-".
        Injected into tool descriptions so the LLM sees them when composing code.
        """
        parts: list[str] = []

        # Extract "when" rules
        rules = self.recall_rules()
        if rules:
            in_when = False
            for line in rules.splitlines():
                if line.strip().startswith("## When"):
                    in_when = True
                    continue
                elif line.strip().startswith("## "):
                    in_when = False
                    continue
                if in_when and line.strip().startswith("- "):
                    parts.append(line.strip())

        # Extract scratchpad-related lessons
        lessons = self._read_full_lessons()
        for line in lessons.splitlines():
            if line.strip().startswith("- ") and "scratchpad" in line.lower():
                stripped = line.strip()
                if stripped not in parts:
                    parts.append(stripped)

        # Check topics/scratchpad-*.md files
        if self._topics_dir.is_dir():
            for path in sorted(self._topics_dir.iterdir()):
                if path.name.startswith("scratchpad-") and path.suffix == ".md":
                    try:
                        content = path.read_text(encoding="utf-8").strip()
                        if content:
                            parts.append(content)
                    except (OSError, UnicodeDecodeError):
                        continue

        return "\n".join(parts)

    def _read_full_lessons(self) -> str:
        """Read lessons.md without budget constraint (for internal use)."""
        if not self._lessons_path.is_file():
            return ""
        try:
            return self._lessons_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return ""

    def encode_rule(
        self,
        text: str,
        kind: Literal["always", "never", "when"],
        confidence: str = "medium",
        source: str = "llm",
    ) -> None:
        """Write a new behavioral gate to rules.md.

        Appends under the correct section (Always/Never/When).
        Uses file locking for safety — like how the hippocampus
        prevents interference between overlapping encoding events.
        """
        self._dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y-%m-%d")
        metadata = f"<!-- confidence:{confidence} source:{source} ts:{ts} -->"
        entry = f"- {text} {metadata}\n"

        section_header = f"## {kind.capitalize()}"

        # Read existing content or create skeleton
        if self._rules_path.is_file():
            content = self._rules_path.read_text(encoding="utf-8")
        else:
            content = "# Rules\n\n## Always\n\n## Never\n\n## When\n"

        # Check for duplicate (exact entry match, ignoring metadata)
        if text in self._extract_entry_texts(content):
            return

        # Find the section and append
        lines = content.splitlines(keepends=True)
        new_lines: list[str] = []
        inserted = False

        i = 0
        while i < len(lines):
            new_lines.append(lines[i])
            if lines[i].strip() == section_header and not inserted:
                # Skip to end of section (next ## or end of file)
                i += 1
                section_entries: list[str] = []
                while i < len(lines) and not (
                    lines[i].strip().startswith("## ")
                    and lines[i].strip() != section_header
                ):
                    section_entries.append(lines[i])
                    i += 1
                # Add existing entries
                new_lines.extend(section_entries)
                # Ensure we have a blank line before the entry if needed
                if section_entries and section_entries[-1].strip():
                    new_lines.append("\n")
                elif not section_entries:
                    pass  # Section was empty, entry follows header
                new_lines.append(entry)
                inserted = True
                continue
            i += 1

        if not inserted:
            # Section didn't exist — add it
            new_lines.append(f"\n{section_header}\n{entry}")

        self._encode_with_lock(self._rules_path, "".join(new_lines), mode="write")

    def encode_lesson(
        self,
        text: str,
        topic: str = "",
        source: str = "llm",
    ) -> None:
        """Write a semantic fact to lessons.md.

        If a topic is provided, also creates/appends to topics/{slug}.md.
        """
        self._dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y-%m-%d")
        topic_tag = f" topic:{topic}" if topic else ""
        entry = f"- {text} <!--{topic_tag} ts:{ts} -->\n"

        # Append to lessons.md
        if not self._lessons_path.is_file():
            self._encode_with_lock(
                self._lessons_path,
                f"# Lessons\n{entry}",
                mode="write",
            )
        else:
            # Check for duplicate (exact entry match, ignoring metadata)
            existing = self._lessons_path.read_text(encoding="utf-8")
            if text in self._extract_entry_texts(existing):
                return
            self._encode_with_lock(self._lessons_path, entry, mode="append")

        # Also write to topic file if topic is substantial
        if topic:
            self._topics_dir.mkdir(parents=True, exist_ok=True)
            slug = self._sanitize_slug(topic)
            topic_path = self._topics_dir / f"{slug}.md"
            if not topic_path.is_file():
                self._encode_with_lock(
                    topic_path,
                    f"# {topic}\n{entry}",
                    mode="write",
                )
            else:
                existing = topic_path.read_text(encoding="utf-8")
                if text not in self._extract_entry_texts(existing):
                    self._encode_with_lock(topic_path, entry, mode="append")

    def rewrite_identity(self, entries: list[str]) -> None:
        """Replace the identity snapshot (profile.md) — full rewrite, not append.

        Unlike other memory operations, identity is a coherent snapshot, not
        an append log. Like how your self-concept updates as a whole, not
        by appending new facts to old ones.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        content = "# Profile\n" + "\n".join(f"- {e}" for e in entries) + "\n"
        self._encode_with_lock(self._profile_path, content, mode="write")

    def entry_count(self) -> int:
        """Count total entries across rules.md and lessons.md."""
        count = 0
        for path in (self._rules_path, self._lessons_path):
            if path.is_file():
                try:
                    content = path.read_text(encoding="utf-8")
                    count += sum(
                        1 for ln in content.splitlines() if ln.strip().startswith("- ")
                    )
                except (OSError, UnicodeDecodeError):
                    continue
        return count

    def _encode_with_lock(self, path: Path, text: str, mode: str = "append") -> None:
        """Write with file locking (fcntl.flock on Unix, no-op on Windows).

        Prevents corruption from concurrent Anton sessions writing to
        global memory — like synaptic tagging ensuring encoding fidelity
        despite concurrent neural activity.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        if mode == "write":
            # Atomic write via temp file + rename
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                if sys.platform != "win32":
                    import fcntl

                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(text)
                    f.flush()
                finally:
                    if sys.platform != "win32":
                        import fcntl

                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            tmp_path.replace(path)
        else:
            # Append mode
            with open(path, "a", encoding="utf-8") as f:
                if sys.platform != "win32":
                    import fcntl

                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(text)
                    f.flush()
                finally:
                    if sys.platform != "win32":
                        import fcntl

                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _extract_entry_texts(content: str) -> set[str]:
        """Extract the set of normalized entry texts from a markdown memory file.

        Strips the leading ``- ``, trailing metadata comments, and whitespace
        so that dedup comparisons are exact-match on the *meaning* line only.
        """
        texts: set[str] = set()
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped.startswith("- "):
                continue
            # Remove leading "- "
            entry = stripped[2:]
            # Remove trailing <!-- ... --> metadata
            entry = re.sub(r"\s*<!--[\s\S]*?-->\s*$", "", entry)
            entry = entry.strip()
            if entry:
                texts.add(entry)
        return texts

    @staticmethod
    def _sanitize_slug(name: str) -> str:
        """Sanitize a topic name into a safe file slug."""
        text = name.lower().strip()
        text = re.sub(r"[^a-z0-9\s_-]", "", text)
        text = re.sub(r"[\s]+", "-", text)
        text = re.sub(r"-+", "-", text)
        return text.strip("-") or "general"
