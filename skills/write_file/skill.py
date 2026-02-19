from __future__ import annotations

from pathlib import Path

from anton.skill.base import SkillResult, skill


@skill("write_file", "Write content to a file at the given path")
async def write_file(path: str, content: str) -> SkillResult:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return SkillResult(
        output=f"Wrote {len(content)} bytes to {path}",
        metadata={"path": str(p.resolve()), "size": len(content)},
    )
