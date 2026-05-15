"""Tests for /goal argument parsing and ToolRegistry.unregister_tool."""

from __future__ import annotations

from anton.chat import _parse_goal_args
from anton.core.tools.registry import ToolRegistry
from anton.core.tools.tool_defs import ToolDef


class TestParseGoalArgs:
    def test_objective_only(self):
        obj, turns = _parse_goal_args('"write hello.txt"')
        assert obj == "write hello.txt"
        assert turns == 50

    def test_objective_with_turns(self):
        obj, turns = _parse_goal_args('"write hello.txt" --turns 10')
        assert obj == "write hello.txt"
        assert turns == 10

    def test_newline_splits_turns_flag(self):
        # Terminal line-wrap can split '--turns 20' into '--tur\nns 20'.
        # Without normalisation this would default to 50 and leave the
        # fragment in the objective — the bug seen in manual testing.
        obj, turns = _parse_goal_args('"write test suite" --tur\nns 20')
        assert turns == 20
        assert "tur" not in obj
        assert "ns 20" not in obj

    def test_carriage_return_normalised(self):
        # \r\n within a word (Windows-style terminal wrap artefact).
        obj, turns = _parse_goal_args('"my goal" --tur\r\nns 20')
        assert turns == 20
        assert obj == "my goal"

    def test_unquoted_objective(self):
        obj, turns = _parse_goal_args('do something useful --turns 3')
        assert obj == "do something useful"
        assert turns == 3

    def test_single_quoted_objective(self):
        obj, turns = _parse_goal_args("'run the linter'")
        assert obj == "run the linter"
        assert turns == 50

    def test_empty_string_returns_empty_objective(self):
        obj, turns = _parse_goal_args("")
        assert obj == ""
        assert turns == 50

    def test_only_turns_flag_returns_empty_objective(self):
        obj, turns = _parse_goal_args("--turns 5")
        assert obj == ""
        assert turns == 5


# ---------------------------------------------------------------------------
# ToolRegistry.unregister_tool
# ---------------------------------------------------------------------------

def _make_tool(name: str) -> ToolDef:
    async def _noop(_session, _input):
        return ""

    return ToolDef(
        name=name,
        description=f"tool {name}",
        input_schema={"type": "object", "properties": {}},
        handler=_noop,
    )


class TestUnregisterTool:
    def test_removes_named_tool(self):
        reg = ToolRegistry()
        reg.register_tool(_make_tool("alpha"))
        reg.register_tool(_make_tool("beta"))
        reg.unregister_tool("alpha")
        names = [t.name for t in reg.get_tool_defs()]
        assert "alpha" not in names
        assert "beta" in names

    def test_noop_when_tool_not_found(self):
        reg = ToolRegistry()
        reg.register_tool(_make_tool("alpha"))
        reg.unregister_tool("nonexistent")  # must not raise
        assert len(reg.get_tool_defs()) == 1

    def test_removes_only_matching_tool(self):
        reg = ToolRegistry()
        for name in ("a", "b", "c"):
            reg.register_tool(_make_tool(name))
        reg.unregister_tool("b")
        names = [t.name for t in reg.get_tool_defs()]
        assert names == ["a", "c"]

    def test_registry_empty_after_removing_last_tool(self):
        reg = ToolRegistry()
        reg.register_tool(_make_tool("only"))
        reg.unregister_tool("only")
        assert not reg  # __bool__ returns False when empty

    def test_dump_excludes_unregistered_tool(self):
        reg = ToolRegistry()
        reg.register_tool(_make_tool("target"))
        reg.unregister_tool("target")
        assert reg.dump() == []
