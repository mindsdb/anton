"""ArtifactWorkspace: what the artifact editor's model may see and change."""

from __future__ import annotations

import os
import re
import stat
import sys
import time
from pathlib import Path

import pytest

from anton.core.artifact_editor import workspace as ws_module
from anton.core.artifact_editor.workspace import (
    MAX_READ_BYTES,
    PRELOAD_MAX_BYTES,
    PRELOAD_MAX_FILES,
    SEARCH_MAX_HITS,
    SEARCH_MAX_HITS_PER_FILE,
    ArtifactWorkspace,
    MapEntry,
    WorkspaceError,
)
from anton.core.artifacts.internal_files import (
    GENERATION_INPUT_FILES,
    HOUSEKEEPING_FILES,
)
from anton.core.tools.generate_artifact import component_page, observed_data

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="no-follow and modes are POSIX")

PAGE = "<!doctype html>\n<html>\n<head>\n<title>Sales</title>\n</head>\n<body>\n<h1>Sales</h1>\n</body>\n</html>\n"


def _put(root: Path, rel: str, text: str | bytes) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_bytes(text.encode("utf-8"))
    return path


def _bound(page: str, rows: int = 50) -> str:
    value = {"rows": [{"region": f"R{i}", "amount": i * 1.5} for i in range(rows)]}
    return observed_data.bind(page, {observed_data.identity(value): value})


def _placeholder_lines(view: str) -> list[str]:
    return [line for line in view.split("\n") if "anton:keep" in line]


# ── construction ─────────────────────────────────────────────────────


def test_unknown_kind_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        ArtifactWorkspace(tmp_path, kind="spreadsheet")


def test_html_app_primary_defaults_to_its_only_page(tmp_path):
    _put(tmp_path, "dashboard.html", PAGE)
    assert ArtifactWorkspace(tmp_path, kind="html-app").primary == "dashboard.html"
    assert ArtifactWorkspace(tmp_path, kind="fullstack-stateless-app").primary == "static/index.html"


# ── map hygiene ──────────────────────────────────────────────────────


def test_map_hides_housekeeping_generation_inputs_bytecode_binaries_and_dependency_dirs(tmp_path):
    _put(tmp_path, "index.html", PAGE)
    _put(tmp_path, "notes.md", "# notes\n")
    for name in HOUSEKEEPING_FILES | GENERATION_INPUT_FILES:
        _put(tmp_path, name, "x")
    _put(tmp_path, ".revisions/1/index.html", PAGE)
    _put(tmp_path, "sub/.revisions/old.md", "x")
    _put(tmp_path, "__pycache__/backend.cpython-312.pyc", b"\x00\x01")
    _put(tmp_path, "lib/helper.pyc", b"\x00\x01")
    _put(tmp_path, "node_modules/x/index.js", "x")
    _put(tmp_path, ".git/HEAD", "ref")
    _put(tmp_path, ".venv/bin/python", "x")
    _put(tmp_path, "dist/bundle.js", "x")
    _put(tmp_path, "logo.png", b"\x89PNG\r\n\x1a\n")
    _put(tmp_path, "blob.dat", b"abc\x00def")  # sniffed: NUL byte
    _put(tmp_path, "latin.txt", "caf\xe9".encode("latin-1"))  # sniffed: not UTF-8
    entries = ArtifactWorkspace(tmp_path, kind="html-app").map()
    assert [e.path for e in entries] == ["index.html", "notes.md"]
    assert all(isinstance(e, MapEntry) for e in entries)


@posix_only
def test_map_never_follows_symlinks(tmp_path):
    outside = tmp_path / "outside"
    _put(outside, "secret.txt", "secret")
    art = tmp_path / "art"
    _put(art, "index.html", PAGE)
    os.symlink(outside / "secret.txt", art / "link.txt")
    os.symlink(outside, art / "linkdir")
    assert [e.path for e in ArtifactWorkspace(art, kind="html-app").map()] == ["index.html"]


def test_map_sizes_pages_as_the_model_reads_them(tmp_path):
    page = _bound(PAGE, rows=2000)
    _put(tmp_path, "index.html", page)
    (entry,) = ArtifactWorkspace(tmp_path, kind="html-app").map()
    assert entry.bytes < len(page.encode()) // 10


# ── refuse_reason ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "path, fragment",
    [
        ("metadata.json", "housekeeping"),
        ("README.md", "housekeeping"),
        (".revisions/x.html", "housekeeping folder"),
        ("prd.md", "generated from"),
        ("observed-data.json", "generated from"),
        ("presentation-program.js", "presentation markers"),
        ("__pycache__/a.pyc", "bytecode"),
        ("node_modules/a.js", "node_modules/"),
        (".git/config", ".git/"),
        (".venv/x.py", ".venv/"),
        ("dist/app.js", "dist/"),
        ("a/.revisions/b.md", ".revisions/"),
        ("photo.jpg", "binary"),
        ("../escape.html", "outside the artifact folder"),
        ("/etc/hosts", "absolute"),
        ("a/./b.html", "outside the artifact folder"),
        ("", "empty"),
    ],
)
def test_refuse_reason_names_why(tmp_path, path, fragment):
    reason = ArtifactWorkspace(tmp_path, kind="document").refuse_reason(path, create=True)
    assert reason is not None and fragment in reason


def test_refuse_reason_refuses_existing_binary_by_content(tmp_path):
    _put(tmp_path, "data.bin2", b"\x00\x01\x02")
    reason = ArtifactWorkspace(tmp_path, kind="document").refuse_reason("data.bin2", create=False)
    assert reason and "binary" in reason


def test_html_app_is_one_file(tmp_path):
    _put(tmp_path, "index.html", PAGE)
    _put(tmp_path, "notes.md", "x")
    ws = ArtifactWorkspace(tmp_path, kind="html-app", primary="index.html")
    assert ws.refuse_reason("index.html", create=False) is None
    assert ws.refuse_reason("notes.md", create=False) is None
    for sibling in ("app.js", "style.css", "data.json", "assets/chart.js"):
        reason = ws.refuse_reason(sibling, create=True)
        assert reason and "separate_file" in reason and "inline" in reason
    # A missing primary may be created.
    assert ArtifactWorkspace(tmp_path / "new", kind="html-app", primary="page.html").refuse_reason(
        "page.html", create=True
    ) is None


@pytest.mark.parametrize("kind", ["fullstack-stateless-app", "fullstack-stateful-app"])
def test_fullstack_layout_contract(tmp_path, kind):
    ws = ArtifactWorkspace(tmp_path, kind=kind)
    for ok in ("backend.py", "requirements.txt", "static/index.html", "static/js/app.js"):
        assert ws.refuse_reason(ok, create=True) is None, ok
    for bad in ("index.html", "app.js", "helpers.py", "templates/page.html"):
        reason = ws.refuse_reason(bad, create=True)
        assert reason and "static/" in reason and "backend.py" in reason, bad
    manifest = ws.refuse_reason("state_manifest.json", create=True)
    if kind == "fullstack-stateful-app":
        assert manifest is None
    else:
        assert manifest is not None


@pytest.mark.parametrize("kind", ["document", "dataset", "mixed", "image"])
def test_content_kinds_allow_content_edits(tmp_path, kind):
    ws = ArtifactWorkspace(tmp_path, kind=kind)
    assert ws.refuse_reason("report.md", create=True) is None
    assert ws.refuse_reason("data/table.csv", create=True) is None
    assert ws.refuse_reason("metadata.json", create=False) is not None


@posix_only
def test_refuse_reason_names_a_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    art = tmp_path / "art"
    art.mkdir()
    os.symlink(outside, art / "static")
    reason = ArtifactWorkspace(art, kind="fullstack-stateless-app").refuse_reason("static/index.html", create=True)
    assert reason and "symlink" in reason


# ── observed-data masking ────────────────────────────────────────────


def test_read_masks_observed_data_as_one_line_and_write_restores_it_byte_exact(tmp_path):
    page = _bound(PAGE)
    _put(tmp_path, "index.html", page)
    ws = ArtifactWorkspace(tmp_path, kind="html-app")
    view = ws.read("index.html")
    assert "cowork-observed-json" not in view
    assert len(_placeholder_lines(view)) == 1
    # Removing the placeholder line leaves exactly what strip_binding leaves.
    stripped = "\n".join(line for line in view.split("\n") if "anton:keep" not in line)
    assert stripped.replace("\n", "") == observed_data.strip_binding(page).replace("\n", "")

    ws.write({"index.html": view})  # unchanged round trip
    assert (tmp_path / "index.html").read_bytes() == page.encode()

    ws.write({"index.html": view.replace("<h1>Sales</h1>", "<h1>Revenue</h1>")})
    written = (tmp_path / "index.html").read_text()
    assert written == page.replace("<h1>Sales</h1>", "<h1>Revenue</h1>")
    assert observed_data.previous_data(written) == observed_data.previous_data(page)


def test_masking_round_trips_a_binding_with_no_surrounding_newlines(tmp_path):
    page = _bound("<!doctype html><html><head><title>T</title></head><body>x</body></html>")
    _put(tmp_path, "index.html", page)
    ws = ArtifactWorkspace(tmp_path, kind="html-app")
    view = ws.read("index.html")
    assert any(line.startswith("<!-- anton:keep observed-data") for line in view.split("\n"))
    ws.write({"index.html": view})
    assert (tmp_path / "index.html").read_bytes() == page.encode()


def test_write_refuses_a_removed_placeholder(tmp_path):
    page = _bound(PAGE)
    _put(tmp_path, "index.html", page)
    ws = ArtifactWorkspace(tmp_path, kind="html-app")
    view = ws.read("index.html")
    without = "\n".join(line for line in view.split("\n") if "anton:keep" not in line)
    with pytest.raises(WorkspaceError, match="removed or changed"):
        ws.write({"index.html": without})
    assert (tmp_path / "index.html").read_text() == page


def test_write_refuses_a_duplicated_placeholder(tmp_path):
    page = _bound(PAGE)
    _put(tmp_path, "index.html", page)
    ws = ArtifactWorkspace(tmp_path, kind="html-app")
    view = ws.read("index.html")
    (line,) = _placeholder_lines(view)
    with pytest.raises(WorkspaceError, match="appears 2 times"):
        ws.write({"index.html": view.replace("<h1>", line + "\n<h1>")})
    assert (tmp_path / "index.html").read_text() == page


def test_write_refuses_an_edited_placeholder(tmp_path):
    _put(tmp_path, "index.html", _bound(PAGE))
    ws = ArtifactWorkspace(tmp_path, kind="html-app")
    view = ws.read("index.html")
    with pytest.raises(WorkspaceError, match="removed or changed"):
        ws.write({"index.html": view.replace("restored on save", "restored")})


def test_write_refuses_a_placeholder_copied_into_another_file(tmp_path):
    _put(tmp_path, "index.html", _bound(PAGE))
    _put(tmp_path, "notes.md", "x\n")
    ws = ArtifactWorkspace(tmp_path, kind="document")
    (line,) = _placeholder_lines(ws.read("index.html"))
    with pytest.raises(WorkspaceError, match="does not belong"):
        ws.write({"notes.md": "x\n" + line + "\n"})
    with pytest.raises(WorkspaceError, match="does not belong"):
        ws.write({"new.md": line})


def test_read_cap_is_on_the_masked_size(tmp_path):
    # ~300 KB on disk, almost all of it embedded data.
    page = _bound(PAGE, rows=9000)
    assert len(page.encode()) > MAX_READ_BYTES
    _put(tmp_path, "index.html", page)
    view = ArtifactWorkspace(tmp_path, kind="html-app").read("index.html")
    assert view is not None and len(view.encode()) < 5000

    _put(tmp_path, "big.md", "x" * (MAX_READ_BYTES + 1))
    with pytest.raises(WorkspaceError, match="too large to read whole"):
        ArtifactWorkspace(tmp_path, kind="document").read("big.md")


def test_read_absent_is_none_and_hidden_is_refused(tmp_path):
    _put(tmp_path, "metadata.json", "{}")
    ws = ArtifactWorkspace(tmp_path, kind="document")
    assert ws.read("missing.md") is None
    with pytest.raises(WorkspaceError, match="housekeeping"):
        ws.read("metadata.json")
    with pytest.raises(WorkspaceError, match="outside"):
        ws.read("../x")


# ── component pages ──────────────────────────────────────────────────

PROGRAM = "await CoworkPage({title:'Revenue',lang:'en',sections:[{type:'text',title:'Observed',text:'one'}]});"


def _component(tmp_path, *, with_copy: bool = True) -> tuple[ArtifactWorkspace, str]:
    page = _bound(component_page.document(PROGRAM))
    _put(tmp_path, "index.html", page)
    if with_copy:
        _put(tmp_path, "presentation-program.js", PROGRAM)
    return ArtifactWorkspace(tmp_path, kind="html-app", primary="index.html"), page


def test_component_page_shows_only_its_program(tmp_path):
    ws, page = _component(tmp_path)
    view = ws.read("index.html")
    assert view.split("\n") == [
        _placeholder_lines(view)[0],
        component_page.BEGIN,
        PROGRAM,
        component_page.END,
        _placeholder_lines(view)[1],
    ]
    assert "cowork-observed-json" not in view and "<style>" not in view
    ws.write({"index.html": view})
    assert (tmp_path / "index.html").read_bytes() == page.encode()


def test_component_program_edit_is_written_and_recovery_copy_kept_in_step(tmp_path):
    ws, page = _component(tmp_path)
    edited = PROGRAM.replace("'one'", "'two'")
    ws.write({"index.html": ws.read("index.html").replace(PROGRAM, edited)})
    written = (tmp_path / "index.html").read_text()
    assert component_page.existing_program(written) == edited
    assert written == page.replace(PROGRAM, edited)
    assert (tmp_path / "presentation-program.js").read_text().strip() == edited


def test_component_program_edit_without_recovery_copy_creates_none(tmp_path):
    ws, _ = _component(tmp_path, with_copy=False)
    ws.write({"index.html": ws.read("index.html").replace("'one'", "'two'")})
    assert not (tmp_path / "presentation-program.js").exists()


def test_component_edit_outside_the_program_is_refused(tmp_path):
    ws, page = _component(tmp_path)
    view = ws.read("index.html")
    with pytest.raises(WorkspaceError, match="Only that program can change"):
        ws.write({"index.html": view.replace(component_page.BEGIN, "<p>extra</p>\n" + component_page.BEGIN)})
    with pytest.raises(WorkspaceError, match="markers"):
        ws.write({"index.html": view.replace(component_page.END + "\n", "")})
    assert (tmp_path / "index.html").read_text() == page


def test_component_program_that_would_not_compile_is_refused(tmp_path):
    ws, page = _component(tmp_path)
    view = ws.read("index.html")
    with pytest.raises(WorkspaceError, match="would not compile"):
        ws.write({"index.html": view.replace(PROGRAM, PROGRAM + "</script><script>alert(1)")})
    with pytest.raises(WorkspaceError, match="would not compile"):
        ws.write({"index.html": view.replace(PROGRAM, "console.log(1);")})
    assert (tmp_path / "index.html").read_text() == page
    assert (tmp_path / "presentation-program.js").read_text() == PROGRAM


def test_recovery_copy_is_not_editable_directly(tmp_path):
    ws, _ = _component(tmp_path)
    with pytest.raises(WorkspaceError, match="presentation markers"):
        ws.write({"presentation-program.js": "x"})


# ── preload ──────────────────────────────────────────────────────────


def test_preload_orders_code_before_content_and_masks(tmp_path):
    _put(tmp_path, "a-notes.md", "# notes\n")
    _put(tmp_path, "index.html", _bound(PAGE))
    _put(tmp_path, "z.css", "body{}\n")
    _put(tmp_path, "b.csv", "a,b\n1,2\n")
    _put(tmp_path, "metadata.json", "{}")
    result = ArtifactWorkspace(tmp_path, kind="document").preload()
    assert list(result) == ["index.html", "z.css", "a-notes.md", "b.csv"]
    assert "cowork-observed-json" not in result["index.html"]


def test_preload_skips_lockfiles_and_minified_bundles(tmp_path):
    _put(tmp_path, "index.html", PAGE)
    _put(tmp_path, "package-lock.json", "{}")
    _put(tmp_path, "poetry.lock", "x")
    _put(tmp_path, "lib.min.js", "x")
    assert list(ArtifactWorkspace(tmp_path, kind="document").preload()) == ["index.html"]


def test_preload_is_all_or_nothing(tmp_path):
    _put(tmp_path, "index.html", PAGE)
    ws = ArtifactWorkspace(tmp_path, kind="document")
    assert ws.preload() is not None

    _put(tmp_path, "big.js", "x" * (32 * 1024 + 1))  # one file over the per-file cap
    assert ws.preload() is None
    (tmp_path / "big.js").unlink()

    for i in range(3):  # under the per-file cap, over the total
        _put(tmp_path, f"part{i}.md", "y" * (20 * 1024))
    assert ws.preload() is None
    for i in range(3):
        (tmp_path / f"part{i}.md").unlink()

    for i in range(PRELOAD_MAX_FILES):  # one file too many
        _put(tmp_path, f"f{i:02}.txt", "z")
    assert ws.preload() is None


def test_preload_counts_framing(tmp_path):
    # Content alone fits; with "--- path ---\n" and the trailing newline it does not.
    body = PRELOAD_MAX_BYTES - 10
    _put(tmp_path, "a.md", "q" * (body // 2))
    _put(tmp_path, "b.md", "q" * (body // 2))
    ws = ArtifactWorkspace(tmp_path, kind="document")
    assert ws.preload() is None


def test_preload_empty_folder_is_none(tmp_path):
    assert ArtifactWorkspace(tmp_path, kind="document").preload() is None


# ── search ───────────────────────────────────────────────────────────


def test_search_regex_and_line_numbers_follow_the_masked_view(tmp_path):
    _put(tmp_path, "index.html", _bound(PAGE))
    _put(tmp_path, "notes.md", "first\nfunction drawChart() {}\n")
    ws = ArtifactWorkspace(tmp_path, kind="document")
    hits = ws.search(r"function\s+draw\w*")
    assert [(h.path, h.line) for h in hits] == [("notes.md", 2)]
    (h1,) = ws.search("<h1>")
    view = ws.read("index.html").split("\n")
    assert view[h1.line - 1].startswith("<h1>")
    # Embedded data is never searched.
    assert ws.search("R42") == []
    assert ws.search_note == ""


def test_search_invalid_regex_falls_back_to_literal(tmp_path):
    _put(tmp_path, "a.js", "call(x\n")
    ws = ArtifactWorkspace(tmp_path, kind="document")
    hits = ws.search("call(")
    assert [(h.path, h.line) for h in hits] == [("a.js", 1)]
    assert "literal" in ws.search_note


@pytest.mark.parametrize("pattern", [r"(a+)+b", r"(a|aa)*c", r"(a|ab)+!", r"(x)\1", r".*a.*a.*a.*b", r"(a*)*b"])
def test_search_catastrophic_patterns_run_literally_and_fast(tmp_path, pattern):
    _put(tmp_path, "a.txt", "a" * 5000 + "\n")
    ws = ArtifactWorkspace(tmp_path, kind="document")
    started = time.monotonic()
    assert ws.search(pattern) == []
    assert time.monotonic() - started < 1.0
    assert "backtrack" in ws.search_note


@pytest.mark.parametrize("pattern", [r"(foo|bar)+", r".*TODO.*", r"\bdraw\w*\(", r"[0-9]{4}-[0-9]{2}"])
def test_search_ordinary_patterns_stay_regex(tmp_path, pattern):
    _put(tmp_path, "a.txt", "x\n")
    ws = ArtifactWorkspace(tmp_path, kind="document")
    ws.search(pattern)
    assert ws.search_note == ""


def test_search_is_capped_per_file_and_overall(tmp_path):
    for i in range(10):
        _put(tmp_path, f"f{i}.txt", "hit\n" * 50)
    ws = ArtifactWorkspace(tmp_path, kind="document")
    hits = ws.search("hit")
    assert len(hits) == SEARCH_MAX_HITS
    per_file = {}
    for hit in hits:
        per_file[hit.path] = per_file.get(hit.path, 0) + 1
    assert max(per_file.values()) == SEARCH_MAX_HITS_PER_FILE
    assert "stopped" in ws.search_note


def test_search_finds_matches_deep_in_a_long_line_and_clips_them(tmp_path):
    _put(tmp_path, "min.js", "x" * 5000 + "needle" + "y" * 5000)
    (hit,) = ArtifactWorkspace(tmp_path, kind="document").search("needle")
    assert "needle" in hit.text and len(hit.text) <= 242


def test_search_does_not_use_sigalrm(tmp_path, monkeypatch):
    import signal

    if hasattr(signal, "setitimer"):
        monkeypatch.setattr(signal, "setitimer", lambda *a: pytest.fail("armed an itimer"))
    monkeypatch.setattr(signal, "signal", lambda *a: pytest.fail("installed a signal handler"))
    _put(tmp_path, "a.txt", "hello\n")
    assert len(ArtifactWorkspace(tmp_path, kind="document").search("hel+o")) == 1


# ── write ────────────────────────────────────────────────────────────


def test_write_creates_and_updates_files(tmp_path):
    _put(tmp_path, "report.md", "old\n")
    ws = ArtifactWorkspace(tmp_path, kind="document")
    ws.write({"report.md": "new\n", "data/extra.csv": "a,b\n"})
    assert (tmp_path / "report.md").read_text() == "new\n"
    assert (tmp_path / "data/extra.csv").read_text() == "a,b\n"
    assert not [p for p in tmp_path.rglob("*") if p.name.endswith(".anton-tmp")]


def test_write_validates_every_path_before_writing_any(tmp_path):
    _put(tmp_path, "report.md", "old\n")
    ws = ArtifactWorkspace(tmp_path, kind="document")
    with pytest.raises(WorkspaceError) as caught:
        ws.write({"report.md": "new\n", "metadata.json": "{}", "../x.md": "y"})
    assert "metadata.json" in str(caught.value) and "../x.md" in str(caught.value)
    assert (tmp_path / "report.md").read_text() == "old\n"


@posix_only
def test_write_preserves_file_mode(tmp_path):
    path = _put(tmp_path, "run.sh", "echo 1\n")
    os.chmod(path, 0o751)
    ArtifactWorkspace(tmp_path, kind="document").write({"run.sh": "echo 2\n"})
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o751
    assert path.read_text() == "echo 2\n"


def test_write_rolls_back_when_a_later_rename_fails(tmp_path, monkeypatch):
    _put(tmp_path, "a.md", "a-old\n")
    _put(tmp_path, "b.md", "b-old\n")
    ws = ArtifactWorkspace(tmp_path, kind="document")
    real = ws_module._replace_at
    calls = {"n": 0}

    def flaky(src, dst, dir_fd):
        calls["n"] += 1
        if calls["n"] == 3:  # a.md and b.md land, new/c.md fails
            raise OSError(5, "Input/output error")
        return real(src, dst, dir_fd)

    monkeypatch.setattr(ws_module, "_replace_at", flaky)
    with pytest.raises(WorkspaceError, match="every file was put back"):
        ws.write({"a.md": "a-new\n", "b.md": "b-new\n", "new/c.md": "c\n"})
    assert (tmp_path / "a.md").read_text() == "a-old\n"
    assert (tmp_path / "b.md").read_text() == "b-old\n"
    assert not (tmp_path / "new").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.md", "b.md"]


def test_write_rolls_back_a_created_file(tmp_path, monkeypatch):
    _put(tmp_path, "z.md", "z-old\n")
    ws = ArtifactWorkspace(tmp_path, kind="document")
    real = ws_module._replace_at
    calls = {"n": 0}

    def flaky(src, dst, dir_fd):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(28, "No space left on device")
        return real(src, dst, dir_fd)

    monkeypatch.setattr(ws_module, "_replace_at", flaky)
    with pytest.raises(WorkspaceError):
        ws.write({"a-created.md": "new\n", "z.md": "z-new\n"})
    assert not (tmp_path / "a-created.md").exists()
    assert (tmp_path / "z.md").read_text() == "z-old\n"


@posix_only
def test_write_refuses_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    _put(outside, "victim.md", "keep\n")
    art = tmp_path / "art"
    art.mkdir()
    os.symlink(outside, art / "docs")
    os.symlink(outside / "victim.md", art / "linked.md")
    ws = ArtifactWorkspace(art, kind="document")
    with pytest.raises(WorkspaceError, match="symlink"):
        ws.write({"docs/victim.md": "pwned\n"})
    with pytest.raises(WorkspaceError, match="symlink"):
        ws.write({"linked.md": "pwned\n"})
    with pytest.raises(WorkspaceError, match="symlink"):
        ws.read("docs/victim.md")
    assert (outside / "victim.md").read_text() == "keep\n"


@posix_only
def test_commit_refuses_a_link_swapped_in_after_validation(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    art = tmp_path / "art"
    (art / "sub").mkdir(parents=True)
    ws = ArtifactWorkspace(art, kind="document")
    # Validation passed for a real folder; the folder becomes a link before the commit.
    (art / "sub").rmdir()
    os.symlink(outside, art / "sub")
    with pytest.raises(WorkspaceError):
        ws._commit({"sub/x.md": (b"pwned", -1)})
    assert list(outside.iterdir()) == []


def test_write_unchanged_text_does_not_touch_the_file(tmp_path):
    path = _put(tmp_path, "a.md", "same\n")
    before = os.stat(path).st_ino
    ArtifactWorkspace(tmp_path, kind="document").write({"a.md": "same\n"})
    assert os.stat(path).st_ino == before


class _Policy:
    def __init__(self, readable: Path | None, writable: Path | None):
        self.readable = readable
        self.project_dir = writable
        self.home_dir = None
        self.tmp_dir = None
        self.shared_dir = None

    def allows_read(self, path) -> bool:
        return self.readable is not None and str(os.path.realpath(path)).startswith(str(os.path.realpath(self.readable)))


def test_sandbox_gates_reads_and_writes(tmp_path):
    art = tmp_path / "art"
    _put(art, "a.md", "x\n")
    ws = ArtifactWorkspace(art, kind="document", sandbox=_Policy(readable=None, writable=None))
    assert ws.map() == []
    with pytest.raises(WorkspaceError, match="may read"):
        ws.read("a.md")
    with pytest.raises(WorkspaceError, match="may change"):
        ws.write({"a.md": "y\n"})

    ws = ArtifactWorkspace(art, kind="document", sandbox=_Policy(readable=tmp_path, writable=tmp_path / "other"))
    assert ws.read("a.md") == "x\n"
    with pytest.raises(WorkspaceError, match="may change"):
        ws.write({"a.md": "y\n"})

    ws = ArtifactWorkspace(art, kind="document", sandbox=_Policy(readable=tmp_path, writable=tmp_path))
    ws.write({"a.md": "y\n"})
    assert (art / "a.md").read_text() == "y\n"


def test_sandbox_with_real_cell_sandbox(tmp_path):
    from anton.core.backends.sandbox import CellSandbox

    project = tmp_path / "project"
    art = project / "artifacts" / "a"
    _put(art, "a.md", "x\n")
    policy = CellSandbox(
        project_dir=project, home_dir=tmp_path / "home", tmp_dir=tmp_path / "tmp", user_name="u",
    )
    ws = ArtifactWorkspace(art, kind="document", sandbox=policy)
    assert ws.read("a.md") == "x\n"
    ws.write({"a.md": "y\n"})
    assert (art / "a.md").read_text() == "y\n"

    other = tmp_path / "elsewhere"
    _put(other, "a.md", "x\n")
    ws = ArtifactWorkspace(other, kind="document", sandbox=policy)
    with pytest.raises(WorkspaceError):
        ws.read("a.md")
    with pytest.raises(WorkspaceError):
        ws.write({"a.md": "y\n"})


# ── snapshot ─────────────────────────────────────────────────────────


@posix_only
def test_snapshot_restore_puts_back_bytes_modes_and_absence(tmp_path):
    a = _put(tmp_path, "a.md", "a\r\nold\n")
    os.chmod(a, 0o640)
    ws = ArtifactWorkspace(tmp_path, kind="document")
    snap = ws.snapshot(["a.md", "static/new.md"])
    assert snap.paths == ["a.md", "static/new.md"]
    ws.write({"a.md": "changed\n", "static/new.md": "created\n"})
    snap.restore()
    assert a.read_bytes() == b"a\r\nold\n"
    assert stat.S_IMODE(os.stat(a).st_mode) == 0o640
    assert not (tmp_path / "static/new.md").exists()


def test_snapshot_restore_after_masked_write_is_byte_exact(tmp_path):
    page = _bound(PAGE)
    _put(tmp_path, "index.html", page)
    ws = ArtifactWorkspace(tmp_path, kind="html-app")
    snap = ws.snapshot(["index.html"])
    ws.write({"index.html": ws.read("index.html").replace("Sales", "Other")})
    assert (tmp_path / "index.html").read_text() != page
    snap.restore()
    assert (tmp_path / "index.html").read_bytes() == page.encode()


def test_patch_composition_shape(tmp_path):
    """read() -> edited text -> write() is the composition apply_patch relies on."""
    _put(tmp_path, "index.html", _bound(PAGE))
    ws = ArtifactWorkspace(tmp_path, kind="html-app")
    text = ws.read("index.html")
    new = re.sub(r"<title>.*?</title>", "<title>Q3</title>", text)
    ws.write({"index.html": new})
    assert "<title>Q3</title>" in (tmp_path / "index.html").read_text()
    assert observed_data.previous_data((tmp_path / "index.html").read_text())
