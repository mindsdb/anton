from __future__ import annotations

from pathlib import Path

from anton.skill.base import SkillResult, skill


@skill("list_files", "List files matching a glob pattern in a directory")
async def list_files(pattern: str = "*", directory: str = ".") -> SkillResult:
    p = Path(directory)
    if not p.is_dir():
        return SkillResult(output=None, metadata={"error": f"Not a directory: {directory}"})

    matches = sorted(str(m) for m in p.glob(pattern))
    return SkillResult(
        output="\n".join(matches) if matches else "No files found.",
        metadata={"count": len(matches)},
    )
