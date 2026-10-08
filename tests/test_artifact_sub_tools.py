"""write_file behaviour: append mode and the path sandbox."""
from __future__ import annotations

from pathlib import Path

from anton.core.tools.generate_artifact import sub_tools
from anton.core.tools.generate_artifact.sub_tools import WRITE_FILE_SCHEMA, write_file


def test_default_mode_overwrites(tmp_path: Path):
    write_file(tmp_path, "a.html", "first")
    res = write_file(tmp_path, "a.html", "second")
    assert res["ok"] is True
    assert (tmp_path / "a.html").read_text(encoding="utf-8") == "second"


def test_append_mode_appends(tmp_path: Path):
    write_file(tmp_path, "a.html", "<head>", mode="w")
    write_file(tmp_path, "a.html", "<body>", mode="a")
    res = write_file(tmp_path, "a.html", "</body>", mode="a")
    assert res["ok"] is True
    assert (tmp_path / "a.html").read_text(encoding="utf-8") == "<head><body></body>"


def test_append_mode_creates_missing_file(tmp_path: Path):
    """mode="a" on a missing path creates the file instead of failing."""
    res = write_file(tmp_path, "nested/new.html", "chunk", mode="a")
    assert res["ok"] is True
    assert (tmp_path / "nested" / "new.html").read_text(encoding="utf-8") == "chunk"


def test_append_mode_keeps_the_sandbox(tmp_path: Path):
    """The sandbox must not be weakened by the new mode."""
    for bad in ("../escape.html", "a/../../b.html", ""):
        res = write_file(tmp_path, bad, "x", mode="a")
        assert res["ok"] is False
        assert "inside the artifact folder" in res["message"]


def test_absolute_path_is_coerced_into_the_folder(tmp_path: Path):
    """An absolute path is NOT rejected — it is coerced into a relative one.

    `_sandboxed_path` applies `lstrip("/")`, so `/etc/passwd` becomes
    `<artifact>/etc/passwd`. Nothing escapes the folder, so this is safe and,
    judging by the code, deliberate — pinned by a test so nobody mistakes it for a
    sandbox hole and "fixes" it.
    """
    res = write_file(tmp_path, "/etc/passwd", "x", mode="a")
    assert res["ok"] is True
    assert res["written"] == "etc/passwd"
    assert (tmp_path / "etc" / "passwd").is_file()


def test_an_absolute_path_inside_the_folder_is_used_as_is(tmp_path: Path):
    """An absolute path that already names a file in the folder must not be
    re-rooted into a nested copy of the folder's own path."""
    res = write_file(tmp_path, str(tmp_path / "index.html"), "x", mode="w")
    assert res["ok"] is True
    assert res["written"] == "index.html"
    assert (tmp_path / "index.html").read_text(encoding="utf-8") == "x"
    assert not (tmp_path / str(tmp_path).lstrip("/")).exists()


def test_unknown_mode_is_rejected(tmp_path: Path):
    res = write_file(tmp_path, "a.html", "x", mode="x")
    assert res["ok"] is False
    assert "mode" in res["message"]
    assert not (tmp_path / "a.html").exists()


def test_schema_advertises_mode():
    props = WRITE_FILE_SCHEMA["input_schema"]["properties"]
    assert props["mode"]["enum"] == ["w", "a"]
    assert "required" in WRITE_FILE_SCHEMA["input_schema"]
    assert "mode" not in WRITE_FILE_SCHEMA["input_schema"]["required"]


def test_a_successful_write_reminds_where_the_next_part_goes(tmp_path: Path):
    """The reminder lands at the one moment the model gets this wrong.

    Measured twice (2026-09-15): after a successful first part it replied
    "Now I'll append the JavaScript logic:" and called `write_file` with no
    body, costing a round. A continuation round begins by answering a tool
    result, which is where a preamble feels natural, and the system prompt is
    thousands of tokens behind by then — so the correction rides on the result
    itself.
    """
    res = sub_tools.write_file(tmp_path, "d.html", "<head>", mode="w")
    assert res["ok"]
    assert sub_tools.FILE_BEGIN_MARKER in res["message"]
    assert "SAME reply" in res["message"]
    assert "announces the next part" in res["message"]

    # On an append too: the failure was observed after the FIRST part, but the
    # second is no different — every continuation starts the same way.
    res = sub_tools.write_file(tmp_path, "d.html", "<body>", mode="a")
    assert "SAME reply" in res["message"]


def test_a_large_body_is_written_without_a_size_warning(tmp_path: Path):
    """The warning told the model an oversized chunk risked "losing its
    connection" — true while the body rode in the tool argument, false now that
    it arrives as streamed text.

    Leaving it in place would hand the model a stale instruction on every write
    round, and under one-body-per-reply acting on it costs a round each time.
    """
    big = "x" * 16_001
    res = sub_tools.write_file(tmp_path, "d.html", big)
    assert res["ok"]
    assert (tmp_path / "d.html").read_text(encoding="utf-8") == big
    assert "WARNING" not in res["message"]
    assert "chunk limit" not in res["message"]


# ── The line map: what removes the "let me verify" round ────────────────────
#
# Measured 2026-09-14 on the memory-game (`find the pair`) run: after a successful append the
# model spent a whole round on a read whose only new information over the
# write result was the tail, then escalated to a whole-file read — 26 006 characters,
# 31% of the final round's context. The write result reporting a size only is
# what made the first of those rounds look worth spending.

def test_write_file_reports_lines_beside_the_size(tmp_path: Path):
    res = write_file(tmp_path, "a.html", "one\ntwo\nthree", mode="w")
    assert "3 lines" in res["message"]
    assert "file now" in res["message"]


# Both figures in one unit, and that unit the one the model counts in.
#
# Measured 2026-09-15 on the memory-game run: an overwrite of 28 114 Cyrillic
# characters reported "+28114 bytes ... file now 29004 bytes", because the
# delta was len(str) mislabelled while the total came from stat(). For mode="w"
# those two describe the same thing and must agree; the 890 they differed by
# said 890 bytes had been in the file before, which was false. The model
# reasons about REPLY_BODY_CHARS in characters, so characters it is.
def test_an_overwrite_reports_the_same_figure_for_the_chunk_and_the_file(
    tmp_path: Path,
):
    body = "Γειά\nσου\n"  # 9 characters, 17 bytes in UTF-8 (Greek: two bytes a letter)
    res = write_file(tmp_path, "a.html", body, mode="w")
    assert f"+{len(body)} characters" in res["message"], res["message"]
    assert f"file now {len(body)} characters" in res["message"], res["message"]
    assert "bytes" not in res["message"], res["message"]


def test_the_reported_size_is_characters_not_bytes(tmp_path: Path):
    """Locked separately from the agreement above: a delta and a total that
    agree could still both be bytes."""
    body = "Ω" * 10  # 10 characters, 20 bytes
    res = write_file(tmp_path, "a.html", body, mode="w")
    assert "10 characters" in res["message"], res["message"]
    assert "20" not in res["message"], res["message"]


def test_an_append_without_a_line_ending_continues_the_last_line(tmp_path: Path):
    """The part's first character lands on the file's last line, so that is
    where its range starts — counted the way `read_text_file` counts."""
    write_file(tmp_path, "a.html", "1\n2\n3\n4\n5\n6", mode="w")
    res = write_file(tmp_path, "a.html", "\n7\n8", mode="a")
    assert "this part is lines 6-8" in res["message"], res["message"]
    assert "/ 8 lines)" in res["message"], res["message"]


def test_an_append_after_a_line_ending_starts_on_the_next_line(tmp_path: Path):
    write_file(tmp_path, "a.html", "a\nb\n", mode="w")
    res = write_file(tmp_path, "a.html", "c\nd", mode="a")
    assert "this part is lines 3-4" in res["message"], res["message"]
    assert "/ 4 lines)" in res["message"], res["message"]


def test_an_overwrite_starts_at_line_one(tmp_path: Path):
    res = write_file(tmp_path, "a.html", "one\ntwo\nthree\n", mode="w")
    assert "this part is lines 1-3" in res["message"], res["message"]
    assert "/ 3 lines)" in res["message"], res["message"]


def test_an_empty_append_after_text_without_a_line_ending_reports_no_part_lines(
    tmp_path: Path,
):
    """An empty part has no lines, so it must not claim the file's last one."""
    write_file(tmp_path, "a.html", "a\nb", mode="w")
    res = write_file(tmp_path, "a.html", "", mode="a")
    assert res["ok"] is True
    assert "this part is lines" not in res["message"], res["message"]
    assert "file now 3 characters / 2 lines" in res["message"], res["message"]


def test_an_empty_append_after_a_line_ending_reports_no_part_lines(tmp_path: Path):
    write_file(tmp_path, "a.html", "a\nb\n", mode="w")
    res = write_file(tmp_path, "a.html", "", mode="a")
    assert res["ok"] is True
    assert "this part is lines" not in res["message"], res["message"]
    assert "file now 4 characters / 2 lines" in res["message"], res["message"]


def test_an_append_to_a_file_that_is_not_utf8_reports_no_part_lines(tmp_path: Path):
    """The file cannot be read back, so the message gives no line range."""
    (tmp_path / "a.html").write_bytes(b"caf\xe9 au lait")
    res = write_file(tmp_path, "a.html", "x", mode="a")
    assert res["ok"] is True
    assert "this part is lines" not in res["message"], res["message"]


def test_the_write_loop_offers_read_text_file_without_line_numbers():
    from anton.core.tools.tool_defs import READ_TEXT_FILE_TOOL

    schemas = sub_tools.tool_schemas()
    assert [s["name"] for s in schemas] == ["write_file", "read_text_file", "finish", "scratchpad"]
    read = schemas[1]
    props = read["input_schema"]["properties"]
    assert "line_numbers" not in props
    assert props["path"]["description"] == "Path relative to the artifact folder, or absolute."
    assert read["description"] == sub_tools.GEN_READ_TEXT_FILE_DESCRIPTION
    # `tool_schema` hands out the main agent's dict; it must stay intact.
    main = READ_TEXT_FILE_TOOL.input_schema["properties"]
    assert "line_numbers" in main
    assert main["path"]["description"] == "Absolute path, or a path relative to the project root."
