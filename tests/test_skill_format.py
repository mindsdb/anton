"""`parse_skill_dir`: ordinary frontmatter as before, bounded for any other."""

from __future__ import annotations

import logging
import subprocess
import sys
import tracemalloc
from pathlib import Path

import pytest

from anton.core.memory.skills import SkillStore
from anton.core.tools import skill_format
from anton.core.tools.skill_format import parse_skill_dir
from anton.core.utils import yaml_cache


def _write_skill(parent: Path, frontmatter: str, folder: str = "my-skill") -> Path:
    skill_dir = parent / folder
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\n{frontmatter}\n---\nStep one.\n", encoding="utf-8"
    )
    return skill_dir


def _write_skill_md(parent: Path, content: bytes) -> Path:
    skill_dir = parent / "my-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_bytes(content)
    return skill_dir


def _warnings(caplog) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.name == skill_format.logger.name and r.levelno == logging.WARNING
    ]


def _nested_aliases(levels: int) -> str:
    """Frontmatter whose list at level i holds nine aliases of level i - 1."""
    lines = ["name: s", "description: d", "l0: &l0 [x, x, x, x, x, x, x, x, x]"]
    for i in range(1, levels):
        lines.append(f"l{i}: &l{i} [" + ", ".join([f"*l{i - 1}"] * 9) + "]")
    return "\n".join(lines)


def _nested_merges(levels: int) -> str:
    """Frontmatter whose mapping at level i merges level i - 1 nine times."""
    lines = ["name: s", "description: d", "m0: &m0 {a: x, b: x, c: x}"]
    for i in range(1, levels):
        merges = ", ".join([f"*m{i - 1}"] * 9)
        lines.append(f"m{i}: &m{i} {{<<: [{merges}], k{i}: x}}")
    return "\n".join(lines)


# ─── ordinary skills parse as before ─────────────────────────────────────────


def test_spec_fields_and_metadata(tmp_path):
    skill = parse_skill_dir(_write_skill(tmp_path, "\n".join([
        "name: CSV Summary",
        "description: Summarize a CSV.",
        "license: MIT",
        "compatibility: requires network",
        "allowed-tools: read_file write_file",
        "metadata:",
        "  display_name: CSV Summary",
        "  provenance: manual",
        '  created_at: "2026-06-15T15:20:42+00:00"',
        "  revision: 3",
    ])))

    assert skill.name == "csv-summary"
    assert skill.description == "Summarize a CSV."
    assert skill.license == "MIT"
    assert skill.compatibility == "requires network"
    assert skill.allowed_tools == "read_file write_file"
    assert skill.instructions == "Step one.\n"
    assert skill.metadata == {
        "display_name": "CSV Summary",
        "provenance": "manual",
        "created_at": "2026-06-15T15:20:42+00:00",
        "revision": "3",
    }


def test_scalar_extra_keys_fold_into_metadata_as_text(tmp_path):
    skill = parse_skill_dir(_write_skill(tmp_path, "\n".join([
        "name: s",
        "description: d",
        "owner: data-team",
        "version: 2",
        "enabled: false",
        "released: 2026-01-02",
    ])))

    assert skill.metadata == {
        "owner": "data-team",
        "version": "2",
        "enabled": "False",
        "released": "2026-01-02",
    }


def test_list_and_mapping_extra_keys_fold_in_as_their_str(tmp_path):
    skill = parse_skill_dir(_write_skill(tmp_path, "\n".join([
        "name: s",
        "description: d",
        "tags: [csv, stats]",
        "limits: {rows: 100, cols: 5}",
        "base: &base [a, b]",
        "copy: *base",
    ])))

    assert skill.metadata == {
        "tags": "['csv', 'stats']",
        "limits": "{'rows': 100, 'cols': 5}",
        "base": "['a', 'b']",
        "copy": "['a', 'b']",
    }


def test_spec_metadata_wins_over_an_extra_key_of_the_same_name(tmp_path):
    skill = parse_skill_dir(_write_skill(tmp_path, "\n".join([
        "name: s",
        "description: d",
        "display_name: From the top level",
        "metadata: {display_name: From metadata}",
    ])))

    assert skill.metadata == {"display_name": "From metadata"}


def test_a_name_that_is_not_text_is_read_as_its_str(tmp_path):
    # normalize_name used to raise AttributeError on a number.
    skill = parse_skill_dir(_write_skill(tmp_path, "name: 2024\ndescription: d"))
    assert skill.name == "2024"


def test_aliases_within_the_limit_parse(tmp_path):
    skill = parse_skill_dir(_write_skill(tmp_path, _nested_aliases(3)))

    assert skill is not None
    assert skill.metadata["l2"].count("'x'") == 9**3


def test_frontmatter_at_the_length_cap_parses(tmp_path):
    head = "name: s\ndescription: "
    value = "x" * (skill_format._FRONTMATTER_MAX_CHARS - len(head))
    skill = parse_skill_dir(_write_skill(tmp_path, head + value))

    assert skill is not None
    assert skill.description == value


@pytest.mark.parametrize(
    ("content", "description", "body"),
    [
        (b"---\ndescription: d\n---\nStep one.\n", "d", "Step one.\n"),
        (b"--- \t\r\ndescription: d\r\n---  \r\nStep one.", "d", "Step one."),
        (b"---\ndescription: d\n---", "d", ""),
        (b"---\ndescription: d\n---\n\n---\nmore\n", "d", "\n---\nmore\n"),
        (b"---\ndescription: a --- b\n---\nStep one.\n", "a --- b", "Step one.\n"),
    ],
)
def test_delimiter_lines_split_frontmatter_from_body(
    tmp_path, content, description, body
):
    skill = parse_skill_dir(_write_skill_md(tmp_path, content))

    assert skill.description == description
    assert skill.instructions == body


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"---",
        b"---\ndescription: d\n",
        b" ---\ndescription: d\n---\nStep one.\n",
        b"----\ndescription: d\n---\nStep one.\n",
        b"---\ndescription: d\n--- x\nStep one.\n",
    ],
)
def test_no_delimiter_pair_is_not_a_skill(tmp_path, content):
    assert parse_skill_dir(_write_skill_md(tmp_path, content)) is None


def test_a_body_of_many_short_lines_is_read_without_splitting_it(tmp_path):
    body = "\n" * 1_000_000
    skill_dir = _write_skill_md(
        tmp_path, b"---\nname: s\ndescription: d\n---\n" + body.encode()
    )

    tracemalloc.start()
    try:
        skill = parse_skill_dir(skill_dir)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert skill.instructions == body
    # Reading the file and slicing the body take a few copies of it. Splitting
    # it into lines took about 18 bytes per byte.
    assert peak < 4 * len(body)


# ─── anything else is refused like malformed YAML ────────────────────────────


@pytest.mark.parametrize("frontmatter", [_nested_aliases(3), _nested_merges(3)])
def test_aliases_past_the_limit_refuse_the_skill(
    tmp_path, monkeypatch, caplog, yaml_parses, frontmatter
):
    # Three levels stay small; a lower limit stands in for the deeper levels
    # that the subprocess test below runs under a memory cap.
    monkeypatch.setattr(yaml_cache, "_MAX_ALIAS_GROWTH", 1000)

    with caplog.at_level(logging.WARNING, logger=skill_format.logger.name):
        assert parse_skill_dir(_write_skill(tmp_path, frontmatter)) is None

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "error_type=AliasGrowthError" in warnings[0]


def test_alias_to_a_node_that_holds_it_refuses_the_skill(
    tmp_path, caplog, yaml_parses
):
    # Each top-level alias of `x` would print all of `y` from its own str().
    frontmatter = "\n".join([
        "name: s",
        "description: d",
        "x0: &x [&y [*x, yyyyyyyyyy]]",
        "x1: *x",
        "x2: *x",
        "last: *y",
    ])

    with caplog.at_level(logging.WARNING, logger=skill_format.logger.name):
        assert parse_skill_dir(_write_skill(tmp_path, frontmatter)) is None

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "error_type=AliasLoopError" in warnings[0]


@pytest.mark.parametrize(
    ("frontmatter", "message"),
    [
        ("name: s\ndescription: d\ncreated_at: 2026-13-45", "YAML error"),
        ("name: s\ndescription: d\nat: 2026-01-01 10:00:00 +99:00", "YAML error"),
        ("name: s\ndescription: d\nn: " + "9" * 5000, "YAML error"),
        ("name: s\ndescription: d\nn: 0x" + "f" * 3700, "cannot be read as text"),
        ("name: s\ndescription: d\nmetadata: {n: 0x" + "f" * 3700 + "}", "cannot be read as text"),
        ("name: s\ndescription: 0x" + "f" * 3700, "cannot be read as text"),
        ("name: 0x" + "f" * 3700 + "\ndescription: d", "cannot be read as text"),
    ],
    ids=[
        "month-out-of-range",
        "time-zone-out-of-range",
        "integer-too-long-to-read",
        "extra-key-too-long-to-print",
        "metadata-too-long-to-print",
        "description-too-long-to-print",
        "name-too-long-to-print",
    ],
)
def test_a_value_python_cannot_hold_or_print_refuses_the_skill(
    tmp_path, caplog, yaml_parses, frontmatter, message
):
    # PyYAML and str() raise ValueError on these, which escaped parse_skill_dir
    # and failed every listing of a skill store that held one.
    with caplog.at_level(logging.WARNING, logger=skill_format.logger.name):
        assert parse_skill_dir(_write_skill(tmp_path, frontmatter)) is None

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert message in warnings[0]


def test_a_file_that_is_not_utf8_refuses_the_skill(tmp_path, caplog):
    skill_dir = _write_skill_md(
        tmp_path, b"---\nname: s\ndescription: \xff\xfe\n---\nStep one.\n"
    )

    with caplog.at_level(logging.WARNING, logger=skill_format.logger.name):
        assert parse_skill_dir(skill_dir) is None

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "not UTF-8" in warnings[0]


def test_frontmatter_over_the_length_cap_refuses_the_skill(tmp_path, yaml_parses):
    head = "name: s\ndescription: "
    value = "x" * (skill_format._FRONTMATTER_MAX_CHARS - len(head) + 1)

    assert parse_skill_dir(_write_skill(tmp_path, head + value)) is None
    assert yaml_parses == []


def test_store_skips_a_refused_skill_like_a_malformed_one(
    tmp_path, monkeypatch, yaml_parses
):
    monkeypatch.setattr(yaml_cache, "_MAX_ALIAS_GROWTH", 1000)
    root = tmp_path / "skills"
    _write_skill(root, "name: good\ndescription: d", folder="good")
    _write_skill(root, _nested_aliases(3), folder="aliased")
    _write_skill(root, "name: s\ndescription: d\nat: 2026-13-45", folder="bad-date")
    store = SkillStore(root=root, builtin_root=tmp_path / "no-builtins")

    assert [s["label"] for s in store.list_summaries()] == ["good"]
    assert [s.label for s in store.list_all()] == ["good"]
    assert store.load("aliased") is None
    assert store.load("bad-date") is None


# Each child sets its memory cap before it builds its frontmatter, and exits
# with _CANNOT_CAP_MEMORY if the platform refuses the cap (macOS does), so the
# frontmatter never runs uncapped. The child's code before _PARSE_IN_CHILD
# sets `frontmatter`.
_CANNOT_CAP_MEMORY = 77
_CAP_MEMORY_IN_CHILD = f"""
import resource
import sys

limit = 1024 * 1024 * 1024
try:
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
except (ValueError, OSError):
    sys.exit({_CANNOT_CAP_MEMORY})
"""
_PARSE_IN_CHILD = """
import tempfile
from pathlib import Path

from anton.core.tools.skill_format import parse_skill_dir

skill_dir = Path(tempfile.mkdtemp()) / "s"
skill_dir.mkdir()
(skill_dir / "SKILL.md").write_text(
    "---\\n" + frontmatter + "\\n---\\nbody\\n", encoding="utf-8"
)
print("refused" if parse_skill_dir(skill_dir) is None else "parsed")
"""


def _parse_under_a_memory_cap(build_frontmatter: str) -> str:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _CAP_MEMORY_IN_CHILD + build_frontmatter + _PARSE_IN_CHILD,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode == _CANNOT_CAP_MEMORY:
        pytest.skip("this platform does not enforce RLIMIT_AS")

    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_nested_aliases_are_refused_under_a_memory_cap():
    # Without the alias limit this parse fails with MemoryError under the cap.
    assert _parse_under_a_memory_cap("""
lines = ["name: s", "description: d", "l0: &l0 [x, x, x, x, x, x, x, x, x]"]
for i in range(1, 9):
    lines.append(f"l{i}: &l{i} [" + ", ".join([f"*l{i - 1}"] * 9) + "]")
frontmatter = "\\n".join(lines)
""") == "refused"


def test_nesting_deeper_than_the_stack_is_refused_under_a_memory_cap():
    # PyYAML composes nested collections recursively, so this raised
    # RecursionError out of parse_skill_dir.
    assert _parse_under_a_memory_cap("""
frontmatter = "name: s\\ndescription: d\\nn: " + "[" * 2000 + "]" * 2000
""") == "refused"
