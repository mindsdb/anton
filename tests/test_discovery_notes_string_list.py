"""notes.string_list: the one place a model's JSON list becomes Python.

Used by the gathering loop, `render_gathering_notes` and `redraw_brief`, so
its tolerance decides whether a schema-violating reply loses data in all
three."""
from __future__ import annotations

from anton.core.tools.generate_artifact.discovery.notes import string_list


def test_a_list_is_stringified_and_cleaned():
    assert string_list(["a", None, "", "  b  ", 42]) == ["a", "b", "42"]


def test_a_multi_line_string_becomes_one_item_per_line_without_markers():
    text = "- first\n* second\n• third\n1. fourth\n2) fifth\n\n   \nsixth"
    assert string_list(text) == ["first", "second", "third", "fourth", "fifth", "sixth"]


def test_a_single_line_string_is_one_item():
    assert string_list("one decision") == ["one decision"]


def test_tool_call_markup_is_removed_from_items():
    """Shape seen live (the original question was in the user's language;
    the wording here is a translation)."""
    leaked = '\n<parameter name="open_points">What does «your symbol» mean in a solo game'
    assert string_list(leaked) == ["What does «your symbol» mean in a solo game"]
    assert string_list(["<parameter name=\"x\">a</parameter>", "b"]) == ["a", "b"]


def test_a_dash_inside_a_line_is_not_a_marker():
    assert string_list("X - blue, O - pink") == ["X - blue, O - pink"]


def test_anything_else_is_empty():
    assert string_list(None) == []
    assert string_list(7) == []
    assert string_list({"a": 1}) == []
    assert string_list("") == []
    assert string_list("   \n  ") == []
