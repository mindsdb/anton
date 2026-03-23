"""Hippocampus — Anton's memory encoding and retrieval engine.

File-based storage with fcntl locking (Linux). Copied from anton/memory/hippocampus.py.
"""

from __future__ import annotations

import fcntl
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class Engram:
    """A single memory trace — the fundamental unit of memory."""

    text: str
    kind: Literal["always", "never", "when", "lesson", "profile"]
    scope: Literal["global", "project"]
    confidence: Literal["high", "medium", "low"] = "medium"
    topic: str = ""
    source: Literal["user", "consolidation", "llm"] = "llm"


class Hippocampus:
    """Reads and writes memory traces at a single scope (global OR project)."""

    def __init__(self, base_dir: Path) -> None:
        self._dir = base_dir
        self._profile_path = base_dir / "profile.md"
        self._rules_path = base_dir / "rules.md"
        self._lessons_path = base_dir / "lessons.md"
        self._topics_dir = base_dir / "topics"

    def recall_identity(self) -> str:
        if not self._profile_path.is_file():
            return ""
        try:
            return self._profile_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return ""

    def recall_rules(self) -> str:
        if not self._rules_path.is_file():
            return ""
        try:
            return self._rules_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return ""

    def recall_lessons(self, token_budget: int = 1000) -> str:
        if not self._lessons_path.is_file():
            return ""
        try:
            content = self._lessons_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return ""

        if not content:
            return ""

        lines = [ln for ln in content.splitlines() if ln.strip()]
        header_lines = []
        entry_lines = []
        for ln in lines:
            if ln.startswith("- ") or ln.startswith("  "):
                entry_lines.append(ln)
            else:
                header_lines.append(ln)

        entry_lines.reverse()

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
        safe_slug = self._sanitize_slug(slug)
        path = self._topics_dir / f"{safe_slug}.md"
        if not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return ""

    def recall_scratchpad_wisdom(self) -> str:
        parts: list[str] = []

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

        lessons = self._read_full_lessons()
        for line in lessons.splitlines():
            if line.strip().startswith("- ") and "scratchpad" in line.lower():
                stripped = line.strip()
                if stripped not in parts:
                    parts.append(stripped)

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
        self._dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y-%m-%d")
        metadata = f"<!-- confidence:{confidence} source:{source} ts:{ts} -->"
        entry = f"- {text} {metadata}\n"

        section_header = f"## {kind.capitalize()}"

        if self._rules_path.is_file():
            content = self._rules_path.read_text(encoding="utf-8")
        else:
            content = "# Rules\n\n## Always\n\n## Never\n\n## When\n"

        if text in content:
            return

        lines = content.splitlines(keepends=True)
        new_lines: list[str] = []
        inserted = False

        i = 0
        while i < len(lines):
            new_lines.append(lines[i])
            if lines[i].strip() == section_header and not inserted:
                i += 1
                section_entries: list[str] = []
                while i < len(lines) and not (
                    lines[i].strip().startswith("## ") and lines[i].strip() != section_header
                ):
                    section_entries.append(lines[i])
                    i += 1
                new_lines.extend(section_entries)
                if section_entries and section_entries[-1].strip():
                    new_lines.append("\n")
                new_lines.append(entry)
                inserted = True
                continue
            i += 1

        if not inserted:
            new_lines.append(f"\n{section_header}\n{entry}")

        self._encode_with_lock(self._rules_path, "".join(new_lines), mode="write")

    def encode_lesson(
        self,
        text: str,
        topic: str = "",
        source: str = "llm",
    ) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y-%m-%d")
        topic_tag = f" topic:{topic}" if topic else ""
        entry = f"- {text} <!--{topic_tag} ts:{ts} -->\n"

        if not self._lessons_path.is_file():
            self._encode_with_lock(
                self._lessons_path,
                f"# Lessons\n{entry}",
                mode="write",
            )
        else:
            existing = self._lessons_path.read_text(encoding="utf-8")
            if text in existing:
                return
            self._encode_with_lock(self._lessons_path, entry, mode="append")

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
                if text not in existing:
                    self._encode_with_lock(topic_path, entry, mode="append")

    def rewrite_identity(self, entries: list[str]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        content = "# Profile\n" + "\n".join(f"- {e}" for e in entries) + "\n"
        self._encode_with_lock(self._profile_path, content, mode="write")

    def entry_count(self) -> int:
        count = 0
        for path in (self._rules_path, self._lessons_path):
            if path.is_file():
                try:
                    content = path.read_text(encoding="utf-8")
                    count += sum(1 for ln in content.splitlines() if ln.strip().startswith("- "))
                except (OSError, UnicodeDecodeError):
                    continue
        return count

    def _encode_with_lock(self, path: Path, text: str, mode: str = "append") -> None:
        """Write with file locking (fcntl.flock on Linux)."""
        path.parent.mkdir(parents=True, exist_ok=True)

        if mode == "write":
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(text)
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            tmp_path.replace(path)
        else:
            with open(path, "a", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(text)
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _sanitize_slug(name: str) -> str:
        text = name.lower().strip()
        text = re.sub(r"[^a-z0-9\s_-]", "", text)
        text = re.sub(r"[\s]+", "-", text)
        text = re.sub(r"-+", "-", text)
        return text.strip("-") or "general"
