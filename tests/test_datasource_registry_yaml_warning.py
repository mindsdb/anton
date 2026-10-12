"""A broken datasources.md block warns where it broke, never the block's text."""

from __future__ import annotations

from pathlib import Path

import pytest

from anton.core.datasources.datasource_registry import _parse_file


TOKEN = "sk-live-private-3304"


def test_yaml_error_prints_its_class_and_file_position_without_the_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "datasources.md"
    # The broken block comes second, so the line is counted from the file's
    # start, not the block's. PyYAML quotes the line it fails on, and this one
    # holds a token.
    path.write_text(
        "# Data sources\n"
        "\n"
        "## Foo\n"
        "\n"
        "```yaml\n"
        "engine: foo\n"
        "display_name: Foo\n"
        "```\n"
        "\n"
        "## Bar\n"
        "\n"
        "```yaml\n"
        "engine: bar\n"
        f"test_snippet: Call the API with key: {TOKEN}\n"
        "```\n",
        encoding="utf-8",
    )

    assert list(_parse_file(path)) == ["foo"]

    err = capsys.readouterr().err
    assert err == (
        f"[anton] Warning: skipping malformed YAML block in {path}: "
        "error_type=ScannerError line=14 column=36\n"
    )
    assert TOKEN not in err
