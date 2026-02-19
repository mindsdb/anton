from __future__ import annotations

import asyncio

from anton.skill.base import SkillResult, skill


@skill("run_command", "Execute a shell command and return its output")
async def run_command(command: str, timeout: int = 30) -> SkillResult:
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return SkillResult(output=None, metadata={"error": f"Command timed out after {timeout}s"})

    return SkillResult(
        output=stdout.decode(errors="replace"),
        metadata={
            "returncode": proc.returncode,
            "stderr": stderr.decode(errors="replace"),
        },
    )
