"""A text file read as a range of lines, for the `read_text_file` tool.

Knows nothing about sessions or access rules: the caller decides whether the
file may be read and which path to show the agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MAX_FILE_BYTES = 20 * 1024 * 1024
DEFAULT_LINE_COUNT = 1000
# A longer line keeps its first and last half: generated pages put inline data
# on one line, and the end of a file is what a writer needs to see.
MAX_LINE_CHARS = 2000
MAX_OUTPUT_CHARS = 250_000

_UTF8_BOM = b"\xef\xbb\xbf"


class NotTextError(Exception):
    """The file is binary or not UTF-8."""


class FileTooLargeError(Exception):
    """The file is larger than MAX_FILE_BYTES."""


class InvalidRangeError(ValueError):
    """start_line / end_line do not name lines of this file."""


def split_lines(text: str) -> list[str]:
    """Lines as `open(path).readlines()` sees them, without their endings.

    Only \\n, \\r\\n and \\r end a line, and a final line ending adds no empty
    line. `str.splitlines()` would also split on U+2028, \\f and \\v, which
    appear in minified JS and would shift every later line number.
    """
    if not text:
        return []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines


def count_lines(text: str) -> int:
    return len(split_lines(text))


@dataclass(frozen=True)
class TextRange:
    path: str
    total_lines: int
    first: int
    last: int
    body: str
    stop: Literal["done", "page", "chars"]

    def render(self) -> str:
        if self.total_lines == 0:
            return f"{self.path} — empty file\n"
        header = f"{self.path} — lines {self.first}-{self.last} of {self.total_lines}"
        if self.stop == "page":
            header += f" (partial; continue with start_line={self.last + 1})"
        elif self.stop == "chars":
            header += (
                f" (partial: {MAX_OUTPUT_CHARS}-character limit reached; "
                f"continue with start_line={self.last + 1})"
            )
        return f"{header}\n{self.body}"


def _decode(data: bytes) -> str:
    if b"\x00" in data:
        raise NotTextError
    if data.startswith(_UTF8_BOM):
        data = data[len(_UTF8_BOM):]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        # A writer still appending (a running backend's log) can leave the last
        # character half written; that is not a reason to refuse the file.
        if exc.reason == "unexpected end of data" and exc.start >= len(data) - 3:
            try:
                return data[: exc.start].decode("utf-8")
            except UnicodeDecodeError:
                pass
        raise NotTextError from exc


def _shorten(line: str) -> str:
    if len(line) <= MAX_LINE_CHARS:
        return line
    half = MAX_LINE_CHARS // 2
    return f"{line[:half]} … [line truncated: {len(line)} chars] … {line[-half:]}"


def _resolve_range(
    total: int, start_line: int | None, end_line: int | None
) -> tuple[int, int, bool]:
    if start_line == 0 or end_line == 0:
        raise InvalidRangeError(
            "line numbers start at 1; use a negative value to count from the end"
        )
    if start_line is None:
        start = 1
    elif start_line > 0:
        start = start_line
    else:
        start = max(1, total + start_line + 1)
    default_end = end_line is None
    if default_end:
        end = start + DEFAULT_LINE_COUNT - 1
    elif end_line > 0:
        end = end_line
    else:
        end = total + end_line + 1
    if start > total:
        raise InvalidRangeError(
            f"start_line {start} is past the end of the file ({total} lines)"
        )
    if end < start:
        raise InvalidRangeError("end_line is before start_line")
    return start, min(end, total), default_end


def read_text_range(
    path: Path,
    shown_path: str,
    start_line: int | None,
    end_line: int | None,
    *,
    line_numbers: bool = False,
) -> TextRange:
    """Lines `start_line`..`end_line` of a UTF-8 text file.

    The file is read once, so the result is consistent even while something
    keeps appending to it.
    """
    with open(path, "rb") as f:
        data = f.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise FileTooLargeError
    lines = split_lines(_decode(data))
    total = len(lines)
    if total == 0:
        return TextRange(shown_path, 0, 0, 0, "", "done")
    start, end, default_end = _resolve_range(total, start_line, end_line)

    out: list[str] = []
    size = 0
    last = start - 1
    stop: Literal["done", "page", "chars"] = "done"
    for number in range(start, end + 1):
        text = _shorten(lines[number - 1])
        if line_numbers:
            text = f"{number:>6}\t{text}"
        added = len(text) + (1 if out else 0)
        if size + added > MAX_OUTPUT_CHARS:
            stop = "chars"
            break
        out.append(text)
        size += added
        last = number
    if stop == "done" and default_end and last < total:
        stop = "page"
    return TextRange(shown_path, total, start, last, "\n".join(out), stop)
