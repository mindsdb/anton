"""Hippocampus — Anton's memory encoding and retrieval engine.

Named for the brain's hippocampus (CA3/CA1 subfields), which handles the
fundamental operations of memory: encoding new traces (writing) and
pattern-completing partial cues into full memories (reading).

The hippocampus doesn't decide *what* to remember — that's the cortex's job.
It simply executes storage and retrieval at a single scope (global or project),
like how the brain's hippocampus encodes at the level of individual memory traces
without executive judgment about importance.

Each Hippocampus instance manages one scope's files:
  - profile.jsonl  → identity (mPFC / Default Mode Network analogy)
  - rules.jsonl    → behavioral gates (Basal Ganglia / OFC analogy)
  - lessons.jsonl  → semantic facts (Anterior Temporal Lobe analogy)
  - topics/*.jsonl → domain expertise (Cortical Association Areas analogy)

Storage format: JSONL (one engram per line). Each line is a JSON object with
fields: id, text, kind, scope, confidence, topic, source, session_id, created_at.

Output contract: recall_* methods return the same markdown strings as before
so cortex.py and all callers are unaffected by the format change.
"""

from __future__ import annotations

import json
import re
import sys
import time
import uuid
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
    session_id: str | None = None
    id: str | None = None
    created_at: str | None = None


# ── JSONL helpers ─────────────────────────────────────────────────────────────

def _new_id() -> str:
    """Generate a short unique engram ID."""
    return "m_" + uuid.uuid4().hex[:8]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _engram_to_record(engram: Engram) -> dict:
    """Serialize an Engram to a JSONL record dict."""
    return {
        "id": engram.id or _new_id(),
        "text": engram.text,
        "kind": engram.kind,
        "scope": engram.scope,
        "confidence": engram.confidence,
        "topic": engram.topic,
        "source": engram.source,
        "session_id": engram.session_id,
        "created_at": engram.created_at or _now_iso(),
    }


def _read_jsonl(path: Path) -> list[dict]:
    """Read all valid records from a JSONL file. Skips malformed lines."""
    if not path.is_file():
        return []
    records = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # skip malformed lines, log could be added here
    except (OSError, UnicodeDecodeError):
        pass
    return records


# ── Migration ─────────────────────────────────────────────────────────────────

def _migrate_md_to_jsonl(md_path: Path, jsonl_path: Path) -> None:
    """One-time migration of a .md memory file to .jsonl format.

    Parses markdown entries (lines starting with '- '), extracts metadata
    from inline HTML comments, and writes JSONL. Renames the .md file to
    .md.bak on success. If the .md file is missing or already migrated,
    this is a no-op.
    """
    if not md_path.is_file():
        return
    if jsonl_path.is_file():
        return  # already migrated

    records = []
    try:
        content = md_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return

    current_kind: str = "lesson"
    for line in content.splitlines():
        stripped = line.strip()

        # Track section headers to infer kind for rules.md
        if stripped.startswith("## Always"):
            current_kind = "always"
            continue
        elif stripped.startswith("## Never"):
            current_kind = "never"
            continue
        elif stripped.startswith("## When"):
            current_kind = "when"
            continue
        elif stripped.startswith("## ") or stripped.startswith("# "):
            current_kind = "lesson"
            continue

        if not stripped.startswith("- "):
            continue

        entry = stripped[2:]

        # Extract metadata from <!-- confidence:X source:Y ts:Z topic:T -->
        meta: dict[str, str] = {}
        meta_match = re.search(r"<!--(.*?)-->", entry)
        if meta_match:
            for part in meta_match.group(1).split():
                if ":" in part:
                    k, v = part.split(":", 1)
                    meta[k.strip()] = v.strip()
            entry = re.sub(r"\s*<!--[\s\S]*?-->\s*$", "", entry).strip()

        if not entry:
            continue

        records.append({
            "id": _new_id(),
            "text": entry,
            "kind": meta.get("kind", current_kind),
            "scope": "project",  # inferred; .md files don't store scope
            "confidence": meta.get("confidence", "medium"),
            "topic": meta.get("topic", ""),
            "source": meta.get("source", "llm"),
            "session_id": None,  # unknown origin for migrated records
            "created_at": meta.get("ts") or _now_iso(),
        })

    if not records:
        return

    # Write JSONL
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    # Rename original to .md.bak
    md_path.rename(md_path.with_suffix(".md.bak"))


class Hippocampus:
    """Reads and writes memory traces at a single scope (global OR project).

    Like the hippocampal CA3 region (pattern completion for reads) and CA1
    region (pattern separation for writes), this class handles the low-level
    mechanics of memory storage without higher-order decisions about relevance
    or importance.

    Storage: JSONL files. Output: same markdown strings as before so
    cortex.py and all callers are unaffected by the format change.
    """

    def __init__(self, base_dir: Path) -> None:
        self._dir = base_dir
        self._profile_path = base_dir / "profile.jsonl"
        self._rules_path = base_dir / "rules.jsonl"
        self._lessons_path = base_dir / "lessons.jsonl"
        self._topics_dir = base_dir / "topics"
        self._migrate()

    def _migrate(self) -> None:
        """Run one-time migration of any legacy .md files to .jsonl."""
        _migrate_md_to_jsonl(self._dir / "profile.md", self._profile_path)
        _migrate_md_to_jsonl(self._dir / "rules.md", self._rules_path)
        _migrate_md_to_jsonl(self._dir / "lessons.md", self._lessons_path)
        if self._topics_dir.is_dir():
            for md_file in self._topics_dir.glob("*.md"):
                if md_file.suffix == ".md":
                    _migrate_md_to_jsonl(
                        md_file,
                        self._topics_dir / (md_file.stem + ".jsonl"),
                    )

    # ── Recall (read) methods ─────────────────────────────────────────────────
    # All return the same markdown strings as before — output contract unchanged.

    def recall_identity(self) -> str:
        """Load the always-on self-model (profile.jsonl → markdown)."""
        records = _read_jsonl(self._profile_path)
        if not records:
            return ""
        lines = ["# Profile"] + [f"- {r['text']}" for r in records]
        return "\n".join(lines)

    def recall_rules(self) -> str:
        """Load behavioral gates (rules.jsonl) as formatted Always/Never/When markdown."""
        records = _read_jsonl(self._rules_path)
        if not records:
            return ""

        sections: dict[str, list[str]] = {"always": [], "never": [], "when": []}
        for r in records:
            kind = r.get("kind", "when")
            if kind in sections:
                ts = r.get("created_at", "")[:10]
                conf = r.get("confidence", "medium")
                src = r.get("source", "llm")
                meta = f"<!-- confidence:{conf} source:{src} ts:{ts} -->"
                sections[kind].append(f"- {r['text']} {meta}")

        parts = ["# Rules"]
        for section, entries in sections.items():
            parts.append(f"\n## {section.capitalize()}")
            if entries:
                parts.extend(entries)
        return "\n".join(parts)

    def recall_lessons(self, token_budget: int = 1000) -> str:
        """Load semantic knowledge (lessons.jsonl), most recent first, within budget."""
        records = _read_jsonl(self._lessons_path)
        if not records:
            return ""

        # Most recent first (reverse file order)
        entries = list(reversed(records))

        char_budget = token_budget * 4
        result_lines = ["# Lessons"]
        used = len(result_lines[0])

        for r in entries:
            ts = r.get("created_at", "")[:10]
            topic = r.get("topic", "")
            topic_tag = f" topic:{topic}" if topic else ""
            line = f"- {r['text']} <!--{topic_tag} ts:{ts} -->"
            if used + len(line) + 1 > char_budget:
                break
            result_lines.append(line)
            used += len(line) + 1

        return "\n".join(result_lines)

    def recall_topic(self, slug: str) -> str:
        """Load deep domain expertise on demand (topics/{slug}.jsonl → markdown)."""
        safe_slug = self._sanitize_slug(slug)
        path = self._topics_dir / f"{safe_slug}.jsonl"
        records = _read_jsonl(path)
        if not records:
            return ""
        lines = [f"# {slug}"] + [f"- {r['text']}" for r in records]
        return "\n".join(lines)

    def recall_scratchpad_wisdom(self) -> str:
        """Retrieve procedural knowledge relevant to scratchpad execution."""
        parts: list[str] = []

        # "when" rules from rules.jsonl
        for r in _read_jsonl(self._rules_path):
            if r.get("kind") == "when":
                parts.append(f"- {r['text']}")

        # lessons with scratchpad topic
        for r in _read_jsonl(self._lessons_path):
            if "scratchpad" in r.get("topic", "").lower() or "scratchpad" in r.get("text", "").lower():
                entry = f"- {r['text']}"
                if entry not in parts:
                    parts.append(entry)

        # topics/scratchpad-*.jsonl files
        if self._topics_dir.is_dir():
            for path in sorted(self._topics_dir.iterdir()):
                if path.name.startswith("scratchpad-") and path.suffix == ".jsonl":
                    for r in _read_jsonl(path):
                        parts.append(r.get("text", ""))

        return "\n".join(p for p in parts if p)

    # ── Encode (write) methods ────────────────────────────────────────────────

    def encode_rule(
        self,
        text: str,
        kind: Literal["always", "never", "when"],
        confidence: str = "medium",
        source: str = "llm",
        session_id: str | None = None,
    ) -> None:
        """Write a new behavioral gate to rules.jsonl."""
        self._dir.mkdir(parents=True, exist_ok=True)

        existing = _read_jsonl(self._rules_path)
        if text in {r["text"] for r in existing}:
            return  # dedup

        record = _engram_to_record(Engram(
            text=text,
            kind=kind,
            scope="project",
            confidence=confidence,
            source=source,
            session_id=session_id,
        ))
        self._append_jsonl(self._rules_path, record)

    def encode_lesson(
        self,
        text: str,
        topic: str = "",
        source: str = "llm",
        session_id: str | None = None,
    ) -> None:
        """Write a semantic fact to lessons.jsonl (and topic file if applicable)."""
        self._dir.mkdir(parents=True, exist_ok=True)

        existing = _read_jsonl(self._lessons_path)
        if text in {r["text"] for r in existing}:
            return  # dedup

        record = _engram_to_record(Engram(
            text=text,
            kind="lesson",
            scope="project",
            confidence="medium",
            topic=topic,
            source=source,
            session_id=session_id,
        ))
        self._append_jsonl(self._lessons_path, record)

        if topic:
            self._topics_dir.mkdir(parents=True, exist_ok=True)
            slug = self._sanitize_slug(topic)
            topic_path = self._topics_dir / f"{slug}.jsonl"
            topic_existing = _read_jsonl(topic_path)
            if text not in {r["text"] for r in topic_existing}:
                self._append_jsonl(topic_path, record)

    def rewrite_identity(self, entries: list[str]) -> None:
        """Replace the identity snapshot (profile.jsonl) — full rewrite."""
        self._dir.mkdir(parents=True, exist_ok=True)
        records = [
            _engram_to_record(Engram(
                text=e,
                kind="profile",
                scope="global",
                confidence="high",
                source="user",
            ))
            for e in entries
        ]
        self._write_jsonl(self._profile_path, records)

    def entry_count(self) -> int:
        """Count total entries across rules.jsonl and lessons.jsonl."""
        return len(_read_jsonl(self._rules_path)) + len(_read_jsonl(self._lessons_path))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _append_jsonl(self, path: Path, record: dict) -> None:
        """Append one record to a JSONL file with file locking."""
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record) + "\n"
        with open(path, "a", encoding="utf-8") as f:
            if sys.platform != "win32":
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
            finally:
                if sys.platform != "win32":
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        """Atomic full-rewrite of a JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".jsonl.tmp")
        content = "\n".join(json.dumps(r) for r in records) + "\n"
        with open(tmp_path, "w", encoding="utf-8") as f:
            if sys.platform != "win32":
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(content)
                f.flush()
            finally:
                if sys.platform != "win32":
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        tmp_path.replace(path)

    @staticmethod
    def _extract_entry_texts(content: str) -> set[str]:
        """Extract plain entry texts from a markdown memory file (used in tests)."""
        texts: set[str] = set()
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped.startswith("- "):
                continue
            entry = stripped[2:]
            entry = re.sub(r"\s*<!--[\s\S]*?-->\s*$", "", entry).strip()
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
