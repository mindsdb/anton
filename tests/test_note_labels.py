"""No note anton adds to the history opens with "SYSTEM:" or tells the model to hide it (ENG-3478).

Models read such a note as an injection and refuse it, and Azure's filter stops the reply.
Scans every string in the package, so a new note can't bring the old shape back.
"""

from __future__ import annotations

import ast
from pathlib import Path

from anton.core.llm import prompts
from anton.core.llm.prompts import ANTON_NOTE

PACKAGE = Path(__file__).resolve().parents[1] / "anton"


def _strings(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


def test_no_string_in_anton_is_a_system_note_or_hides_itself():
    found = []
    for path in PACKAGE.rglob("*.py"):
        for line, text in _strings(ast.parse(path.read_text(), filename=str(path))):
            # The bare prefix is how anton still recognises notes in older saved histories.
            if text == "SYSTEM:":
                continue
            if text.lstrip().startswith("SYSTEM:") or "\nSYSTEM:" in text or "do not mention" in text.lower():
                found.append(f"{path.relative_to(PACKAGE.parent)}:{line}")
    assert not found, found


def test_tool_result_nudges_are_labelled():
    nudges = [value for name, value in vars(prompts).items() if name.endswith("_NUDGE") and isinstance(value, str)]
    assert len(nudges) == 6
    assert all(nudge.lstrip().startswith(ANTON_NOTE) for nudge in nudges)
