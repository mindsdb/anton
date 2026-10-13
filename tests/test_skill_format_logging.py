"""A broken SKILL.md frontmatter logs where it broke, never the frontmatter text."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from anton.core.tools.skill_format import parse_skill_dir


TOKEN = "sk-live-private-3304"
SKILL_LOGGER = "anton.core.tools.skill_format"


def test_yaml_error_logs_its_class_and_position_without_the_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    # PyYAML quotes the line it fails on, and this one holds a token.
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: my-skill\ndescription: Call the API with key: {TOKEN}\n---\nBody\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger=SKILL_LOGGER):
        assert parse_skill_dir(skill_dir) is None
    records = [record for record in caplog.records
               if record.name == SKILL_LOGGER and record.levelno == logging.WARNING]
    assert len(records) == 1
    assert records[0].getMessage() == (
        "parse_skill_md: YAML error in SKILL.md: error_type=ScannerError line=3 column=35"
    )
    assert TOKEN not in repr(records[0].args)
