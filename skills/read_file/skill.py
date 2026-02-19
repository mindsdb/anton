from __future__ import annotations

from pathlib import Path

from anton.skill.base import SkillResult, skill


@skill("read_file", "Read the contents of a file at the given path")
async def read_file(path: str) -> SkillResult:
    p = Path(path)
    if not p.exists():
        return SkillResult(output=None, metadata={"error": f"File not found: {path}"})
    if not p.is_file():
        return SkillResult(output=None, metadata={"error": f"Not a file: {path}"})
    content = p.read_text(encoding="utf-8", errors="replace")
    return SkillResult(output=content, metadata={"size": len(content), "path": str(p.resolve())})
