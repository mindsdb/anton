from __future__ import annotations

from pydantic import BaseModel


class SkillSpec(BaseModel):
    """Specification for a skill to be built by the SkillBuilder."""

    name: str
    description: str
    parameters: dict[str, str] = {}  # parameter name -> type (e.g. "path": "str")
    expected_output: str = ""
    test_inputs: dict[str, str] = {}  # parameter name -> test value
