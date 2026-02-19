from __future__ import annotations

import json
import re
import time
from pathlib import Path


class LearningStore:
    """Manages learning files and the recall index under ~/.anton/learnings/."""

    def __init__(self, base_dir: Path) -> None:
        self._learnings_dir = base_dir / "learnings"
        self._learnings_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._learnings_dir / "index.json"

    def _read_index(self) -> dict[str, dict]:
        if self._index_path.exists():
            return json.loads(self._index_path.read_text())
        return {}

    def _write_index(self, index: dict[str, dict]) -> None:
        self._index_path.write_text(json.dumps(index, indent=2))

    @staticmethod
    def _slugify(topic: str) -> str:
        text = topic.lower()
        text = re.sub(r"[^a-z0-9\s_]", "", text)
        text = re.sub(r"[\s]+", "_", text.strip())
        text = re.sub(r"_+", "_", text)
        return text.strip("_") or "general"

    async def record(self, topic: str, content: str, summary: str) -> None:
        slug = self._slugify(topic)
        file_path = self._learnings_dir / f"{slug}.md"

        # Append content to the topic's Markdown file
        header = f"\n## {topic}\n\n" if not file_path.exists() else f"\n---\n\n## {topic}\n\n"
        with open(file_path, "a") as f:
            f.write(header + content + "\n")

        # Update index
        index = self._read_index()
        index[slug] = {
            "topic": topic,
            "summary": summary,
            "path": str(file_path),
            "updated_at": time.time(),
        }
        self._write_index(index)

    def find_relevant(self, task: str, limit: int = 3) -> list[dict]:
        index = self._read_index()
        if not index:
            return []

        task_words = set(task.lower().split())

        scored: list[tuple[float, str, dict]] = []
        for slug, entry in index.items():
            summary_words = set(entry.get("summary", "").lower().split())
            topic_words = set(entry.get("topic", "").lower().split())
            all_words = summary_words | topic_words
            overlap = len(task_words & all_words)
            if overlap > 0:
                scored.append((overlap, slug, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for _score, slug, entry in scored[:limit]:
            file_path = self._learnings_dir / f"{slug}.md"
            content = file_path.read_text() if file_path.exists() else ""
            results.append({
                "topic": entry.get("topic", slug),
                "summary": entry.get("summary", ""),
                "content": content,
            })
        return results

    def list_all(self) -> list[dict]:
        index = self._read_index()
        return [
            {"topic": entry.get("topic", slug), "summary": entry.get("summary", "")}
            for slug, entry in index.items()
        ]
