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


def test_oversized_chunk_is_written_but_warned_about(tmp_path: Path):
    big = "x" * (sub_tools.CHUNK_SOFT_LIMIT + 1)
    res = sub_tools.write_file(tmp_path, "d.html", big)
    assert res["ok"]
    assert (tmp_path / "d.html").read_text(encoding="utf-8") == big
    assert "WARNING" in res["message"]
    assert "chunk limit" in res["message"]


def test_chunk_within_limit_gets_no_warning(tmp_path: Path):
    res = sub_tools.write_file(tmp_path, "d.html", "x" * 100)
    assert res["ok"]
    assert "WARNING" not in res["message"]


def test_read_file_returns_size_and_tail_by_default(tmp_path: Path):
    text = "A" * 3000 + "</html>"
    (tmp_path / "d.html").write_text(text, encoding="utf-8")
    res = sub_tools.read_file(tmp_path, "d.html")
    assert res["ok"]
    assert f"{len(text)} characters" in res["message"]
    assert res["message"].rstrip().endswith("(pass `full=true` to read the entire file)")
    assert "</html>" in res["message"]  # the tail is what proves the file is closed
    assert len(res["message"]) < len(text)  # must not ship the whole file


def test_read_file_full_returns_everything(tmp_path: Path):
    text = "A" * 3000
    (tmp_path / "d.html").write_text(text, encoding="utf-8")
    res = sub_tools.read_file(tmp_path, "d.html", full=True)
    assert res["ok"]
    assert res["message"] == text


def test_read_file_small_file_is_returned_whole(tmp_path: Path):
    (tmp_path / "s.txt").write_text("short", encoding="utf-8")
    res = sub_tools.read_file(tmp_path, "s.txt")
    assert res["ok"]
    assert res["message"] == "short"


def test_read_file_schema_advertises_full():
    props = sub_tools.READ_FILE_SCHEMA["input_schema"]["properties"]
    assert "full" in props
