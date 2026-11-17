"""The `reading-files` skill is offered only where its libraries are installed.

Its code imports them directly. Where some are missing, as on Desktop today,
the agent imports them without declaring them first and each one costs a
failed cell, so the skill is left out of the prompt there.
"""
from __future__ import annotations

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


def test_gate_lists_every_library_the_skill_imports():
    # A library the skill starts importing must join the gate, or the skill is
    # offered where that import fails.
    skill = (Path(skills_mod.__file__).parent / "builtin_skills" / "reading-files" / "SKILL.md").read_text()
    import_names = {
        "requests": "requests", "bs4": "beautifulsoup4", "pdfplumber": "pdfplumber",
        "pypdfium2": "pypdfium2", "docx": "python-docx", "PIL": "pillow",
    }
    imported = {dist for module, dist in import_names.items() if f"import {module}" in skill or f"from {module} " in skill}
    assert imported <= ChatSession._READING_FILES_PACKAGES
    assert '"lxml"' in skill and "pypdf" in skill
