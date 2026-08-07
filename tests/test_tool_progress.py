"""ToolProgress lives in its own leaf module — see design doc S1 in
docs/eng-763/2026-08-03-streaming-tool-progress-design.md for why: tool_defs.py
pulls in tool_handlers.py's heavy dependency chain, and registry.py imports
ToolDef only under TYPE_CHECKING (a deliberate boundary this must not cross).
"""

from __future__ import annotations

from anton.core.tools.progress import ToolProgress


def test_is_a_plain_dataclass_with_text_field():
    p = ToolProgress("step_1 done")
    assert p.text == "step_1 done"


def test_equality_by_value():
    assert ToolProgress("a") == ToolProgress("a")
    assert ToolProgress("a") != ToolProgress("b")


def test_reexported_from_tool_defs_is_the_same_class():
    from anton.core.tools.tool_defs import ToolProgress as ReExported

    assert ReExported is ToolProgress
