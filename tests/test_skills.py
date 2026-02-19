from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from anton.skill.base import SkillResult

# Import skills directly
from skills.read_file.skill import read_file
from skills.write_file.skill import write_file
from skills.list_files.skill import list_files
from skills.run_command.skill import run_command
from skills.search_code.skill import search_code


class TestReadFile:
    async def test_read_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = await read_file(path=str(f))
        assert result.output == "hello world"
        assert result.metadata["size"] == 11

    async def test_read_nonexistent_file(self):
        result = await read_file(path="/no/such/file.txt")
        assert result.output is None
        assert "error" in result.metadata

    async def test_read_directory_returns_error(self, tmp_path):
        result = await read_file(path=str(tmp_path))
        assert result.output is None
        assert "Not a file" in result.metadata["error"]


class TestWriteFile:
    async def test_write_creates_file(self, tmp_path):
        f = tmp_path / "out.txt"
        result = await write_file(path=str(f), content="hello")
        assert f.read_text() == "hello"
        assert result.metadata["size"] == 5

    async def test_write_creates_parent_dirs(self, tmp_path):
        f = tmp_path / "a" / "b" / "c.txt"
        result = await write_file(path=str(f), content="nested")
        assert f.read_text() == "nested"

    async def test_write_overwrites(self, tmp_path):
        f = tmp_path / "overwrite.txt"
        f.write_text("old")
        await write_file(path=str(f), content="new")
        assert f.read_text() == "new"


class TestListFiles:
    async def test_list_files_default(self, tmp_path):
        (tmp_path / "a.txt").touch()
        (tmp_path / "b.py").touch()
        result = await list_files(directory=str(tmp_path))
        assert "a.txt" in result.output
        assert "b.py" in result.output
        assert result.metadata["count"] == 2

    async def test_list_files_pattern(self, tmp_path):
        (tmp_path / "a.txt").touch()
        (tmp_path / "b.py").touch()
        result = await list_files(pattern="*.py", directory=str(tmp_path))
        assert "b.py" in result.output
        assert "a.txt" not in result.output
        assert result.metadata["count"] == 1

    async def test_list_files_empty_dir(self, tmp_path):
        result = await list_files(directory=str(tmp_path))
        assert result.output == "No files found."
        assert result.metadata["count"] == 0

    async def test_list_files_invalid_dir(self):
        result = await list_files(directory="/no/such/dir")
        assert result.output is None
        assert "error" in result.metadata


class TestRunCommand:
    async def test_echo(self):
        result = await run_command(command="echo hello")
        assert result.output.strip() == "hello"
        assert result.metadata["returncode"] == 0

    async def test_failing_command(self):
        result = await run_command(command="false")
        assert result.metadata["returncode"] != 0

    async def test_timeout(self):
        result = await run_command(command="sleep 10", timeout=1)
        assert "timed out" in result.metadata["error"]


class TestSearchCode:
    async def test_search_finds_pattern(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def hello():\n    pass\n")
        result = await search_code(pattern="hello", directory=str(tmp_path))
        assert "hello" in result.output

    async def test_search_no_matches(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("nothing here\n")
        result = await search_code(pattern="zzzznotfound", directory=str(tmp_path))
        # Either "No matches found." or empty output
        assert result.metadata["match_count"] == 0
