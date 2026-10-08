"""read_text_range: line splitting, ranges, limits and the header."""
from __future__ import annotations

from pathlib import Path

import pytest

from anton.core.tools import text_file
from anton.core.tools.text_file import (
    FileTooLargeError,
    InvalidRangeError,
    NotTextError,
    count_lines,
    read_text_range,
    split_lines,
)


def _write(tmp_path: Path, data: str | bytes, name: str = "f.txt") -> Path:
    path = tmp_path / name
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return path


def _lines(n: int) -> str:
    return "".join(f"line {i}\n" for i in range(1, n + 1))


def _read(path: Path, start=None, end=None, **kw):
    return read_text_range(path, str(path), start, end, **kw)


@pytest.mark.parametrize(
    "text",
    ["", "a", "a\n", "a\nb", "a\r\nb\r\n", "a\rb", "\n", "a\n\nb\n", "x y\fz\vw\n", "a b\x85c\x1cd\n"],
)
def test_count_lines_matches_readlines(tmp_path, text):
    path = _write(tmp_path, text)
    with open(path, encoding="utf-8") as f:
        expected = len(f.readlines())
    assert count_lines(text) == expected


def test_split_lines_drops_line_endings():
    assert split_lines("a\r\nb\rc\nd") == ["a", "b", "c", "d"]


def test_whole_small_file_by_default(tmp_path):
    path = _write(tmp_path, "one\ntwo\nthree\n")
    result = _read(path)
    assert (result.first, result.last, result.total_lines, result.stop) == (1, 3, 3, "done")
    assert result.render() == f"{path} — lines 1-3 of 3\none\ntwo\nthree"


def test_default_page_is_1000_lines(tmp_path):
    path = _write(tmp_path, _lines(1500))
    result = _read(path)
    assert (result.first, result.last, result.stop) == (1, 1000, "page")
    rendered = result.render().splitlines()
    assert rendered[0] == f"{path} — lines 1-1000 of 1500 (partial; continue with start_line=1001)"
    assert rendered[-1] == "line 1000"


def test_explicit_range_is_not_partial(tmp_path):
    path = _write(tmp_path, _lines(1500))
    result = _read(path, 200, 260)
    assert (result.first, result.last, result.stop) == (200, 260, "done")
    assert result.body.splitlines() == [f"line {i}" for i in range(200, 261)]


def test_start_line_without_end_reads_a_page_from_there(tmp_path):
    result = _read(_write(tmp_path, _lines(1500)), 1001)
    assert (result.first, result.last, result.stop) == (1001, 1500, "done")


def test_negative_start_reads_the_tail(tmp_path):
    path = _write(tmp_path, _lines(100))
    result = _read(path, -20)
    assert (result.first, result.last) == (81, 100)


def test_negative_start_beyond_the_file_starts_at_line_one(tmp_path):
    assert _read(_write(tmp_path, _lines(5)), -50).first == 1


def test_end_line_minus_one_reads_to_the_end(tmp_path):
    result = _read(_write(tmp_path, _lines(1500)), 1, -1)
    assert (result.first, result.last, result.stop) == (1, 1500, "done")


def test_negative_end_counts_from_the_end(tmp_path):
    assert _read(_write(tmp_path, _lines(10)), 1, -3).last == 8


def test_end_past_the_file_is_clamped(tmp_path):
    assert _read(_write(tmp_path, _lines(10)), 5, 99).last == 10


@pytest.mark.parametrize(
    "start,end,message",
    [
        (0, None, "start at 1"),
        (1, 0, "start at 1"),
        (11, None, r"past the end of the file \(10 lines\)"),
        (5, 4, "before start_line"),
        (1, -11, "before start_line"),
    ],
)
def test_invalid_ranges(tmp_path, start, end, message):
    with pytest.raises(InvalidRangeError, match=message):
        _read(_write(tmp_path, _lines(10)), start, end)


def test_empty_file(tmp_path):
    path = _write(tmp_path, "")
    result = _read(path, 5, 9)
    assert (result.total_lines, result.first, result.last, result.body) == (0, 0, 0, "")
    assert result.render() == f"{path} — empty file\n"


def test_line_numbers_are_off_by_default_and_optional(tmp_path):
    path = _write(tmp_path, "a\nb\n")
    assert _read(path).body == "a\nb"
    assert _read(path, line_numbers=True).body == "     1\ta\n     2\tb"


def test_a_long_line_keeps_its_start_and_end(tmp_path):
    long_line = "S" * 1000 + "M" * 5000 + "E" * 1000
    body = _read(_write(tmp_path, f"short\n{long_line}\n")).body.splitlines()
    assert body[0] == "short"
    assert body[1] == "S" * 1000 + " … [line truncated: 7000 chars] … " + "E" * 1000


def test_a_line_of_exactly_the_limit_is_kept(tmp_path):
    line = "x" * text_file.MAX_LINE_CHARS
    assert _read(_write(tmp_path, line)).body == line


def test_output_limit_stops_at_a_whole_line(tmp_path, monkeypatch):
    monkeypatch.setattr(text_file, "MAX_OUTPUT_CHARS", 25)
    result = _read(_write(tmp_path, "aaaaaaaaa\nbbbbbbbbb\nccccccccc\n"), 1, -1)
    assert (result.last, result.stop) == (2, "chars")
    assert result.body == "aaaaaaaaa\nbbbbbbbbb"
    assert result.render().splitlines()[0].endswith(
        "(partial: 25-character limit reached; continue with start_line=3)"
    )


def test_output_limit_wins_over_the_default_page(tmp_path, monkeypatch):
    monkeypatch.setattr(text_file, "MAX_OUTPUT_CHARS", 25)
    assert _read(_write(tmp_path, "aaaaaaaaa\n" * 5)).stop == "chars"


@pytest.mark.parametrize("data", [b"\x00abc", b"abc" * 4000 + b"\x00"])
def test_nul_anywhere_is_not_text(tmp_path, data):
    with pytest.raises(NotTextError):
        _read(_write(tmp_path, data))


def test_invalid_utf8_is_not_text(tmp_path):
    with pytest.raises(NotTextError):
        _read(_write(tmp_path, b"caf\xe9 au lait"))


def test_bom_is_dropped(tmp_path):
    assert _read(_write(tmp_path, b"\xef\xbb\xbfhello\n")).body == "hello"


def test_a_character_cut_off_at_the_end_is_dropped(tmp_path):
    data = "ok\ncafé".encode("utf-8")[:-1]   # "é" is two bytes; the second is missing
    assert _read(_write(tmp_path, data)).body == "ok\ncaf"


def test_a_broken_character_in_the_middle_is_not_text(tmp_path):
    with pytest.raises(NotTextError):
        _read(_write(tmp_path, b"caf\xc3\nlait"))


def test_too_large_file_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(text_file, "MAX_FILE_BYTES", 10)
    _read(_write(tmp_path, "x" * 10, "ok.txt"))
    with pytest.raises(FileTooLargeError):
        _read(_write(tmp_path, "x" * 11, "big.txt"))
