"""A datasources.md engine can carry agent-facing usage notes."""
from anton.core.datasources.datasource_registry import _parse_file


def _write(tmp_path, yaml_body: str):
    path = tmp_path / "datasources.md"
    path.write_text(f"## Foo\n\n```yaml\n{yaml_body}```\n", encoding="utf-8")
    return path


def test_usage_notes_are_read_from_the_yaml_block(tmp_path):
    path = _write(
        tmp_path,
        "engine: foo\n"
        "display_name: Foo\n"
        "usage_notes: |\n"
        "  Call /bar for totals.\n",
    )

    assert _parse_file(path)["foo"].usage_notes == "Call /bar for totals.\n"


def test_usage_notes_default_to_empty(tmp_path):
    path = _write(tmp_path, "engine: foo\ndisplay_name: Foo\n")

    assert _parse_file(path)["foo"].usage_notes == ""


def test_non_string_usage_notes_are_ignored(tmp_path):
    path = _write(tmp_path, "engine: foo\nusage_notes: [a, b]\n")

    assert _parse_file(path)["foo"].usage_notes == ""
