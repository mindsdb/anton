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

import datetime as dt
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from anton.core.memory.base import Engram


def _extract_metadata(text: str) -> tuple[str, dict]:
    """Find and remove the trailing metadata comment from an entry line.

    Parses <!-- key:value ... --> annotations and returns structured
    metadata as a dict whose keys match Engram fields. 'ts' maps to
    'updated_at' and is parsed into a datetime; all other keys are kept
    as strings.

    Returns:
        (clean_text, metadata_dict)
        metadata_dict is empty when no comment is present.
    """
    match = re.search(r"\s*<!--([^>]*)-->\s*$", text)
    if not match:
        return text.strip(), {}

    raw: dict[str, str] = dict(re.findall(r"(\w+):(\S+)", match.group(1)))
    clean_text = text[: match.start()].strip()

    meta: dict = {}

    ts = raw.pop("ts", None)
    if ts:
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
            try:
                meta["updated_at"] = dt.datetime.strptime(ts, fmt)
                break
            except ValueError:
                pass

    for key in ("topic", "kind", "confidence", "source"):
        if key in raw:
            if raw[key]:
                meta[key] = raw[key]

    return clean_text, meta


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

    def clear(self) -> None:
        for path in (self._profile_path, self._rules_path, self._lessons_path):
            if path.is_file():
                path.unlink()
        topics_dir = self._dir / "topics"
        if topics_dir.is_dir():
            for f in topics_dir.glob("*.md"):
                f.unlink()

    # ---------- identity -------------------

    def recall_identities(self) -> str:
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

    def get_identities(self) -> list[Engram]:
        """Load identity profile as Engrams."""
        content = self.recall_identities()
        engrams = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                text = stripped[2:]
            elif stripped and not stripped.startswith("#"):
                text = stripped
            else:
                continue
            if text:
                engrams.append(Engram(text=text))
        return engrams

    def del_identity(self, id):
        entries = self.get_identities()
        entries_out = []
        for entry in entries:
            if entry.id != id:
                entries_out.append(entry)

        self.save_identities(entries_out)

    def update_identity(self, id, text):
        entries = self.get_identities()

        for entry in entries:
            if entry.id != id:
                continue
            entry.text = text
            entry.updated_at = dt.datetime.now()

            self.save_identities(entries)
            return

    def rewrite_identity(self, entries: list[str]) -> None:
        """Merge new entries into the identity snapshot (profile.md) and rewrite.

        Deduplicates by exact text and by key prefix (e.g. "Name:" replaces
        any existing "Name: ..." entry). Unlike a pure append, identity stays
        a coherent snapshot rather than growing unboundedly.
        """

        existing_entries = [
            item.text
            for item in self.get_identities()
        ]

        for fact in entries:
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

        self.save_identities([
            Engram(text)
            for text in existing_entries
        ])

    def save_identities(self, entries: list[Engram]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        content = "# Profile\n" + "\n".join(f"- {e.text}" for e in entries) + "\n"
        self._encode_with_lock(self._profile_path, content, mode="write")

    def clear_identity(self) -> None:
        if self._profile_path.is_file():
            self._profile_path.unlink()


    # ---------  lessons --------------

    def recall_lessons(self, token_budget: int|None = 1000) -> str:
        """Load semantic knowledge (lessons.md), most recent first, within budget.

        Brain analog: Anterior Temporal Lobe — the convergence hub for semantic
        facts distilled from many episodes. Budget enforced at ~4 chars/token.
        """
        return self._lessons_to_text(self.get_lessons(token_budget))


    def get_lessons(self, token_budget: int = None) -> list[Engram]:
        """Load semantic knowledge (lessons.md) as Engrams.

        When token_budget is None (default) returns all entries in file order.
        When token_budget is set, returns entries newest-first up to that limit
        (~4 chars/token).
        """
        if not self._lessons_path.is_file():
            return []
        try:
            content = self._lessons_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return []

        if not content:
            return []

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

        entries = []
        for text in entry_lines:
            text, meta = _extract_metadata(text)
            entries.append(Engram(text=text.removeprefix("- "), **meta))

        if token_budget is None:
            return entries

        # Reverse entries so most recent are first
        entries.reverse()

        # Budget: ~4 chars per token
        char_budget = token_budget * 4
        result_lines = []
        used = sum(len(ln) for ln in result_lines)

        for ln in entries:
            if used + len(ln.text) + 1 > char_budget:
                break
            result_lines.append(ln)
            used += len(ln.text) + 1

        return result_lines

    def del_lesson(self, id):
        entries = self.get_lessons()
        entries_out = []
        for entry in entries:
            if entry.id != id:
                entries_out.append(entry)

        if len(entries_out) != len(entries):
            self._encode_with_lock(self._lessons_path, self._lessons_to_text(entries_out), mode="write")


    def update_lesson(self, id, text):
        entries = self.get_lessons()

        for entry in entries:
            if entry.id != id:
                continue
            entry.text = text
            entry.updated_at = dt.datetime.now()

            self._encode_with_lock(self._lessons_path, self._lessons_to_text(entries), mode="write")

            break

    @staticmethod
    def _lessons_to_text(entries: list[Engram], header="Lessons") -> str:
        """Serialize a list of Engram objects back to lessons.md line format.

        Writes all non-None/non-empty Engram fields into a trailing
        <!-- key:value --> comment. 'updated_at' is serialised as 'ts'.
        """
        lines = []
        for entry in entries:
            parts: list[str] = []

            for key in ("confidence", "source", "topic"):
                val = getattr(entry, key)
                if val is not None and val != "":
                    parts.append(f"{key}:{val}")

            ts = (
                entry.updated_at.strftime("%Y-%m-%d")
                if entry.updated_at
                else time.strftime("%Y-%m-%d")
            )
            parts.append(f"ts:{ts}")

            lines.append(f"- {entry.text} <!-- {' '.join(parts)} -->\n")

        if len(lines) == 0:
            return ""
        return f"# {header}\n" + "".join(lines)

    def recall_topic(self, slug: str) -> str:
        """Load deep domain expertise on demand by given slug.

        Brain analog: Cortical Association Areas — specialized regions activated
        associatively when contextual cues indicate relevance.
        """
        slug = self._sanitize_slug(slug)

        items = []
        for item in self.get_lessons():
            if item.topic == slug:
                items.append(item)

        return self._lessons_to_text(items, header=slug)

    def recall_scratchpad_wisdom(self) -> str:
        """Retrieve procedural knowledge relevant to scratchpad execution.

        Returns all "when" rules + lessons whose text contains "scratchpad".
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
        for lesson in self.get_lessons():
            if "scratchpad" in lesson.text.lower() or (lesson.topic and "scratchpad" in lesson.topic.lower()):
                entry = f"- {lesson.text}"
                if entry not in parts:
                    parts.append(entry)

        return "\n".join(parts)

    # ---------- rules -------------------

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

    def get_rules(self) -> list[Engram]:
        """Load behavioral rules (rules.md) as Engrams, grouped by kind."""
        if not self._rules_path.is_file():
            return []
        try:
            content = self._rules_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []

        current_kind = None
        entries: list[Engram] = []

        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                current_kind = stripped[3:].lower()
            elif stripped.startswith("- ") and current_kind:
                text, meta = _extract_metadata(stripped[2:])
                if text and current_kind is not None:
                    entries.append(Engram(text=text, kind=current_kind, **meta))

        entries.sort(key=lambda x: x.kind)

        return entries

    def del_rule(self, id):
        entries = self.get_rules()
        entries_out = []
        for entry in entries:
            if entry.id != id:
                entries_out.append(entry)

        self.save_rules(entries_out)

    def update_rule(self, id, text):
        entries = self.get_rules()

        for entry in entries:
            if entry.id != id:
                continue
            entry.text = text
            entry.updated_at = dt.datetime.now()

            self.save_rules(entries)
            return

    def save_rules(self, rules: list[Engram]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        sections: dict[str, list[Engram]] = {"always": [], "never": [], "when": []}
        for rule in rules:
            if rule.kind in sections:
                sections[rule.kind].append(rule)

        lines = ["# Rules\n"]
        for section, entries in sections.items():
            lines.append(f"\n## {section.capitalize()}\n")
            for e in entries:
                ts = e.updated_at.strftime("%Y-%m-%d") if e.updated_at else time.strftime("%Y-%m-%d")
                meta = f"<!-- confidence:{e.confidence} source:{e.source} ts:{ts} -->"
                lines.append(f"- {e.text} {meta}\n")

        self._encode_with_lock(self._rules_path, "".join(lines), mode="write")

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

        rules = self.get_rules()

        new_rule = Engram(
            text=text,
            updated_at=dt.datetime.now(),
            confidence=confidence,
            kind=kind,
            source=source,
        )

        # Check for duplicate (exact entry match, ignoring metadata)
        for rule in rules:
            if rule.text == new_rule.text:
                return

        rules.append(new_rule)

        self.save_rules(rules)

    def encode_lesson(
        self,
        text: str,
        topic: str = "",
        source: str = "llm",
    ) -> None:
        """Append a semantic fact to lessons.md, tagged with optional topic metadata."""
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
