"""The `reading-files` skill is offered only where its libraries are installed.

Its code imports them directly. Where some are missing, as on Desktop today,
the agent imports them without declaring them first and each one costs a
failed cell, so the skill is left out of the prompt there.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

from anton.core.backends.manager import ScratchpadManager
from anton.core.memory import skills as skills_mod
from anton.core.session import ChatSession, ChatSessionConfig
from anton.workspace import Workspace
from tests.conftest import make_mock_llm, run_turn

# What the Web sandbox image installs, as the interpreter reports the names.
_SANDBOX = ["requests", "beautifulsoup4", "lxml", "pypdf", "pdfplumber", "pypdfium2", "python-docx", "pillow"]
# What the Desktop backend has today: no requests, pdfplumber or pypdfium2.
_DESKTOP = ["beautifulsoup4", "lxml", "pypdf", "python-docx", "pillow", "httpx"]


@pytest.fixture
def builtin_skills(monkeypatch):
    """The real built-in skills, which the suite hides."""
    monkeypatch.setattr(skills_mod, "_BUILTIN_SKILLS_ROOT", Path(skills_mod.__file__).parent / "builtin_skills")


async def _system_prompt(tmp_path: Path, monkeypatch, installed: list[str]) -> str:
    monkeypatch.setattr(ScratchpadManager, "probe_packages", staticmethod(lambda: ["anton-agent", *installed]))
    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()
    llm = make_mock_llm()
    session = ChatSession(ChatSessionConfig(llm_client=llm, workspace=workspace))
    try:
        await run_turn(session, "Read the attached scan.")
    finally:
        await session.close()
    return llm.plan.call_args_list[0].kwargs["system"]


async def test_offered_where_every_library_is_installed(tmp_path, monkeypatch, builtin_skills):
    system = await _system_prompt(tmp_path, monkeypatch, _SANDBOX)
    assert "- `reading-files` — " in system


async def test_left_out_where_libraries_are_missing(tmp_path, monkeypatch, builtin_skills):
    system = await _system_prompt(tmp_path, monkeypatch, _DESKTOP)
    assert "reading-files" not in system
    # Only this skill is gated: the other built-ins are still listed.
    assert "- `public-data-sources` — " in system


async def test_names_are_compared_without_case(tmp_path, monkeypatch, builtin_skills):
    system = await _system_prompt(tmp_path, monkeypatch, [name.upper() for name in _SANDBOX])
    assert "- `reading-files` — " in system


# The distribution behind each module the skill's code may import.
_DISTRIBUTION = {
    "requests": "requests", "bs4": "beautifulsoup4", "lxml": "lxml", "pypdf": "pypdf",
    "pdfplumber": "pdfplumber", "pypdfium2": "pypdfium2", "docx": "python-docx", "PIL": "pillow",
}


def _skill_imports() -> set[str]:
    """Top-level modules the skill's code imports, outside the standard library.

    Parsed from the code blocks rather than listed, so an import the skill
    gains is seen here. `BeautifulSoup(..., "lxml")` loads lxml by name, so
    a parser named that way counts as an import too.
    """
    skill = (Path(skills_mod.__file__).parent / "builtin_skills" / "reading-files" / "SKILL.md").read_text()
    code = "\n".join(re.findall(r"```python\n(.*?)```", skill, re.S))
    modules: set[str] = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import):
            modules |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules.add(node.module.split(".")[0])
    modules |= set(re.findall(r'BeautifulSoup\([^)]*"(\w+)"\)', code))
    return modules - set(sys.stdlib_module_names)


def test_gate_lists_every_library_the_skill_imports():
    modules = _skill_imports()
    unknown = modules - set(_DISTRIBUTION)
    # A new import must be added to the gate (ChatSession._READING_FILES_PACKAGES)
    # and to _DISTRIBUTION here, or the skill is offered where that import fails.
    assert not unknown, f"the skill imports {sorted(unknown)}, which the gate does not check"
    assert {_DISTRIBUTION[m] for m in modules} <= ChatSession._READING_FILES_PACKAGES
    # The parse itself works: these are imported by the skill today.
    assert {"requests", "bs4", "lxml", "pdfplumber", "pypdfium2", "docx", "PIL"} <= modules
