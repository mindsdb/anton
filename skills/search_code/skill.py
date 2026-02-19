from __future__ import annotations

import asyncio
import shutil

from anton.skill.base import SkillResult, skill


@skill("search_code", "Search for a pattern in files under a directory")
async def search_code(pattern: str, directory: str = ".") -> SkillResult:
    # Prefer ripgrep if available, fall back to grep
    rg = shutil.which("rg")
    if rg:
        cmd = f"rg --no-heading --line-number {_shell_quote(pattern)} {_shell_quote(directory)}"
    else:
        cmd = f"grep -rn {_shell_quote(pattern)} {_shell_quote(directory)}"

    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

    output = stdout.decode(errors="replace")
    return SkillResult(
        output=output if output else "No matches found.",
        metadata={"match_count": output.count("\n") if output else 0},
    )


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"
