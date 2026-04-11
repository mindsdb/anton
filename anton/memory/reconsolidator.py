"""Reconsolidator — Migrates legacy memory formats to the new schema.

Named for memory reconsolidation (Nader et al., 2000): when a consolidated
memory is reactivated, it enters a labile state where it can be modified
before being re-stored. The memory content is preserved but the "format"
(synaptic organization, schema integration) changes.

The Reconsolidator does exactly this: reactivates old-format memories
(SelfAwarenessContext, LearningStore) and re-encodes them into the new
Hippocampus format. Like biological reconsolidation, this process:
  - Preserves content while updating structure
  - Risks corruption if interrupted (uses atomic writes)
  - Is necessary for long-term system health
  - Only runs once per memory (one-time migration)
"""

from __future__ import annotations

import json
from pathlib import Path

from anton.core.memory.hippocampus import Hippocampus


def needs_reconsolidation(project_dir: Path) -> bool:
    """Check if legacy memory formats exist and need migration.

    Returns True if old-format files exist AND new-format files don't.
    """
    # Legacy directories
    legacy_context = project_dir / "context"
    legacy_learnings = project_dir / "learnings"

    # New memory directory
    new_memory = project_dir / "memory"

    has_legacy = (
        (legacy_context.is_dir() and any(legacy_context.iterdir()))
        or (legacy_learnings.is_dir() and any(legacy_learnings.iterdir()))
    )

    has_new = new_memory.is_dir() and any(
        f for f in new_memory.iterdir()
        if f.name in ("rules.md", "lessons.md", "profile.md")
    )

    return has_legacy and not has_new


def reconsolidate(project_dir: Path) -> list[str]:
    """One-time migration from legacy memory systems.

    Reactivates and re-encodes:
      .anton/context/*.md → project memory/rules.md + memory/lessons.md
      .anton/learnings/*.md → project memory/lessons.md + memory/topics/

    Returns list of actions taken for logging.
    """
    actions: list[str] = []

    memory_dir = project_dir / "memory"
    hc = Hippocampus(memory_dir)

    # --- Migrate .anton/context/*.md (SelfAwarenessContext files) ---
    context_dir = project_dir / "context"
    if context_dir.is_dir():
        for path in sorted(context_dir.iterdir()):
            if path.name.startswith(".") or path.is_dir():
                continue
            if not path.suffix == ".md":
                continue

            try:
                content = path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError):
                continue

            if not content:
                continue

            # Each context file becomes a set of lessons
            # The filename gives us the topic
            topic = path.stem.replace("-", " ").replace("_", " ")
            slug = Hippocampus._sanitize_slug(topic)

            # Split content into individual facts/lines
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Strip bullet prefixes
                if line.startswith("- "):
                    line = line[2:]
                elif line.startswith("* "):
                    line = line[2:]

                if len(line) > 5:  # Skip very short fragments
                    hc.encode_lesson(line, topic=slug, source="user")

            actions.append(f"Migrated context/{path.name} → memory/lessons.md + topics/{slug}.md")

    # --- Migrate .anton/learnings/*.md (LearningStore files) ---
    learnings_dir = project_dir / "learnings"
    if learnings_dir.is_dir():
        # Read index for topic metadata
        index_path = learnings_dir / "index.json"
        index: dict = {}
        if index_path.is_file():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        for path in sorted(learnings_dir.iterdir()):
            if path.name.startswith(".") or path.is_dir():
                continue
            if path.name == "index.json":
                continue
            if not path.suffix == ".md":
                continue

            try:
                content = path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError):
                continue

            if not content:
                continue

            slug = path.stem
            topic = index.get(slug, {}).get("topic", slug.replace("_", " "))

            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line == "---":
                    continue

                if line.startswith("- "):
                    line = line[2:]
                elif line.startswith("* "):
                    line = line[2:]

                if len(line) > 5:
                    hc.encode_lesson(line, topic=slug, source="consolidation")

            actions.append(f"Migrated learnings/{path.name} → memory/lessons.md + topics/{slug}.md")

    if actions:
        actions.insert(0, f"Reconsolidated {len(actions)} legacy memory files into new format")

    return actions
