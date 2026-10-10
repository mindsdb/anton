"""`safe_load_cached`: one parse per distinct text, a fresh copy per call."""

from __future__ import annotations

import pytest
import yaml

from anton.core.utils import yaml_cache
from anton.core.utils.yaml_cache import safe_load_cached


def test_identical_text_is_parsed_once(yaml_parses):
    assert safe_load_cached("engine: acme\n") == {"engine": "acme"}
    assert safe_load_cached("engine: acme\n") == {"engine": "acme"}
    assert yaml_parses == ["engine: acme\n"]


def test_changed_text_is_parsed_again(yaml_parses):
    safe_load_cached("engine: acme\n")
    assert safe_load_cached("engine: other\n") == {"engine": "other"}
    assert yaml_parses == ["engine: acme\n", "engine: other\n"]


def test_each_call_gets_its_own_copy(yaml_parses):
    text = "name_from: [host, database]\nfields:\n  - {name: host}\n"
    first = safe_load_cached(text)
    first["name_from"].append("port")
    first["fields"][0]["name"] = "changed"

    again = safe_load_cached(text)
    assert again["name_from"] == ["host", "database"]
    assert again["fields"] == [{"name": "host"}]
    assert len(yaml_parses) == 1


def test_malformed_text_raises_on_every_call(yaml_parses):
    for _ in range(2):
        with pytest.raises(yaml.YAMLError):
            safe_load_cached("fields: [unclosed\n")
    assert len(yaml_parses) == 2


def test_text_over_the_cap_is_parsed_every_time_and_not_kept(yaml_parses):
    value = "x" * (yaml_cache._MAX_CACHED_TEXT_CHARS + 1 - len("description: "))
    text = f"description: {value}"
    for _ in range(2):
        assert safe_load_cached(text) == {"description": value}
    assert len(yaml_parses) == 2
    assert yaml_cache._parse.cache_info().currsize == 0


def test_text_at_the_cap_is_kept(yaml_parses):
    value = "x" * (yaml_cache._MAX_CACHED_TEXT_CHARS - len("description: "))
    text = f"description: {value}"
    for _ in range(2):
        assert safe_load_cached(text) == {"description": value}
    assert len(yaml_parses) == 1
    assert yaml_cache._parse.cache_info().currsize == 1


@pytest.mark.parametrize(
    ("directives", "merge_key"),
    [
        ("", "<<"),
        ("", "!!merge x"),
        ("", "!<tag:yaml.org,2002:merge> x"),
        ("%TAG !m! tag:yaml.org,2002:\n---\n", "!m!merge x"),
    ],
)
def test_short_text_with_a_large_parse_is_parsed_every_time_and_not_kept(
    yaml_parses, directives, merge_key
):
    # Each merge copies all 100 entries of `b` into a new mapping.
    anchored = ", ".join(f"k{i}: {i}" for i in range(100))
    merges = ", ".join([f"{{{merge_key}: *b}}"] * 40)
    text = f"{directives}b: &b {{{anchored}}}\nm: [{merges}]\n"
    assert len(text) <= yaml_cache._MAX_CACHED_TEXT_CHARS
    for _ in range(2):
        parsed = safe_load_cached(text)
        assert parsed["m"] == [parsed["b"]] * 40
    assert len(yaml_parses) == 2
    assert yaml_cache._parse.cache_info().currsize == 0


def _loop_entered_inside(aliases: int) -> str:
    """`x` holds `y`, `y` holds `x`, and the last key names `y`.

    The walk takes the last key first, so it reaches `y` before `x`.
    """
    lines = ["x0: &x [&y [*x, yyyyyyyyyy]]"]
    lines += [f"x{i}: *x" for i in range(1, aliases)]
    lines.append("last: *y")
    return "\n".join(lines)


@pytest.mark.parametrize(
    "text",
    ["&a [*a]", "&a {k: *a}", "&a {<<: *a, k: v}", _loop_entered_inside(3)],
)
def test_alias_to_a_node_that_holds_it_raises_on_every_call(yaml_parses, text):
    for _ in range(2):
        with pytest.raises(yaml.YAMLError, match="refers to a node that holds it"):
            safe_load_cached(text)
    assert len(yaml_parses) == 2
    assert yaml_cache._parse.cache_info().currsize == 0


def test_loop_entered_inside_is_refused_however_few_units_it_counts():
    # Each alias of `x` prints all of `y` from its own str(), so the printed
    # text grows with the aliases times len(y), not with the walk's count.
    text = _loop_entered_inside(3)
    printed = sum(len(str(value)) for value in yaml.safe_load(text).values())
    assert printed > 4 * len("yyyyyyyyyy")
    with pytest.raises(yaml.YAMLError, match="refers to a node that holds it"):
        yaml_cache._alias_growth(yaml.compose(text, Loader=yaml.SafeLoader))


def test_shared_node_without_a_loop_is_not_a_loop(yaml_parses):
    text = "a: &a [x]\nb: [*a, *a]\nc: {k: *a}"
    assert safe_load_cached(text) == yaml.safe_load(text)


def _nested_aliases(levels: int) -> str:
    lines = ["l0: &l0 [x, x, x, x, x, x, x, x, x]"]
    for i in range(1, levels):
        lines.append(f"l{i}: &l{i} [" + ", ".join([f"*l{i - 1}"] * 9) + "]")
    return "\n".join(lines)


def _nested_merges(levels: int) -> str:
    lines = ["m0: &m0 {a: x, b: x, c: x}"]
    for i in range(1, levels):
        merges = ", ".join([f"*m{i - 1}"] * 9)
        lines.append(f"m{i}: &m{i} {{<<: [{merges}], k{i}: x}}")
    return "\n".join(lines)


@pytest.mark.parametrize("text", [_nested_aliases(3), _nested_merges(3)])
def test_aliases_within_the_limit_parse_as_safe_load_does(yaml_parses, text):
    assert safe_load_cached(text) == yaml.safe_load(text)


@pytest.mark.parametrize("text", [_nested_aliases(3), _nested_merges(3)])
def test_aliases_past_the_limit_raise_on_every_call_and_are_not_kept(
    yaml_parses, monkeypatch, text
):
    # Three levels stay small; a lower limit stands in for deeper levels.
    monkeypatch.setattr(yaml_cache, "_MAX_ALIAS_GROWTH", 1000)
    for _ in range(2):
        with pytest.raises(yaml.YAMLError, match="aliases expand this YAML"):
            safe_load_cached(text)
    assert len(yaml_parses) == 2
    assert yaml_cache._parse.cache_info().currsize == 0


def test_long_text_with_aliases_past_the_limit_raises(yaml_parses, monkeypatch):
    monkeypatch.setattr(yaml_cache, "_MAX_ALIAS_GROWTH", 1000)
    text = _nested_aliases(3) + "\npad: " + "x" * yaml_cache._MAX_CACHED_TEXT_CHARS
    with pytest.raises(yaml.YAMLError, match="aliases expand this YAML"):
        safe_load_cached(text)


def test_text_without_aliases_never_counts_against_the_limit(yaml_parses, monkeypatch):
    monkeypatch.setattr(yaml_cache, "_MAX_ALIAS_GROWTH", 0)
    text = "fields:\n" + "".join(f"  - {{name: f{i}, type: str}}\n" for i in range(200))
    assert safe_load_cached(text) == yaml.safe_load(text)
