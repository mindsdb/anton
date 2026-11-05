"""render_listing: the text list_artifacts returns."""
from __future__ import annotations

from pathlib import Path

from anton.core.artifacts.listing import (
    FIELDS,
    LEGEND,
    SIZE_LEGEND,
    SUMMARY_FIELDS,
    render_listing,
)
from anton.core.artifacts.models import Artifact, FileEntry

ROOT = Path("/work/project/.anton/artifacts")
WHEN = "2026-10-07T14:02:11+00:00"


def _file(path: str, size: int) -> FileEntry:
    return FileEntry(path=path, bytes=size, modifiedAt=WHEN)


def _artifact(**overrides) -> Artifact:
    data = dict(
        id="3f2c1a9b7d4e4c0f9a51b2e6c8d0f413",
        slug="sales-dashboard-3f2c1a9b",
        createdAt="2026-10-07T10:00:00+00:00",
        updatedAt=WHEN,
        name="Sales dashboard",
        description="Monthly revenue by region",
        type="html-app",
        primary="index.html",
        files=[_file("index.html", 24210)],
    )
    data.update(overrides)
    return Artifact(**data)


def test_summary_fields():
    assert render_listing([(ROOT, [_artifact()])], SUMMARY_FIELDS) == "\n".join([
        LEGEND,
        "",
        f"Artifacts root: {ROOT}",
        "",
        "## sales-dashboard-3f2c1a9b",
        "name: Sales dashboard",
        "type: html-app",
        f"updatedAt: {WHEN}",
        "description: Monthly revenue by region",
        "primary: index.html",
    ])


def test_all_fields_with_service_files(tmp_path):
    folder = tmp_path / "sales-dashboard-3f2c1a9b"
    folder.mkdir()
    (folder / "metadata.json").write_text("{}")
    (folder / "prd.md").write_text("# PRD\n")
    (folder / ".published.json").write_text("{}")

    text = render_listing([(tmp_path, [_artifact()])], FIELDS)

    assert text.splitlines()[:2] == [LEGEND, SIZE_LEGEND]
    assert text.endswith("\n".join([
        "## sales-dashboard-3f2c1a9b",
        "id: 3f2c1a9b7d4e4c0f9a51b2e6c8d0f413",
        "name: Sales dashboard",
        "type: html-app",
        f"updatedAt: {WHEN}",
        "description: Monthly revenue by region",
        "primary: index.html",
        "files:",
        "  24210  index.html",
        "service_files:",
        "  2  metadata.json",
        "  6  prd.md",
    ]))


def test_service_files_skip_symlinks_and_folders(tmp_path):
    from anton.core.artifacts.listing import service_files

    (tmp_path / "prd.md").write_text("# PRD\n")
    (tmp_path / "README.md").symlink_to(tmp_path / "prd.md")
    (tmp_path / "spec.md").mkdir()
    assert service_files(tmp_path) == [("prd.md", 6)]


def test_explicit_fields_keep_the_fixed_order():
    text = render_listing([(ROOT, [_artifact()])], ("primary", "name"))
    assert text.splitlines()[-2:] == ["name: Sales dashboard", "primary: index.html"]
    assert SIZE_LEGEND not in text


def test_name_and_description_are_one_line():
    a = _artifact(name="Sales\ndashboard", description="Line one\n\n## other-slug\tthree")
    lines = render_listing([(ROOT, [a])], ("name", "description")).splitlines()
    assert "name: Sales dashboard" in lines
    assert "description: Line one ## other-slug three" in lines
    assert sum(line.startswith("## ") for line in lines) == 1


def test_control_characters_in_paths_are_escaped():
    a = _artifact(primary="a\nb.html", files=[_file("x\ty.html", 1)])
    lines = render_listing([(ROOT, [a])], ("primary", "files")).splitlines()
    assert "primary: a\\nb.html" in lines
    assert "  1  x\\ty.html" in lines


def test_a_multiline_updated_at_stays_on_one_line():
    a = _artifact(updatedAt="2026-10-07T14:02:11+00:00\n## fake")
    lines = render_listing([(ROOT, [a])], ("updatedAt",)).splitlines()
    assert "updatedAt: 2026-10-07T14:02:11+00:00\\n## fake" in lines
    assert sum(line.startswith("## ") for line in lines) == 1


def test_dot_paths_are_left_out_of_files():
    a = _artifact(files=[_file("static/index.html", 10), _file("static/.published.json", 5)])
    text = render_listing([(ROOT, [a])], ("files",))
    assert "static/index.html" in text
    assert ".published.json" not in text


def test_paths_with_spaces_and_parentheses_stay_whole():
    a = _artifact(files=[_file("reports/Q3 report (final).md", 8150)])
    lines = render_listing([(ROOT, [a])], ("files",)).splitlines()
    assert "  8150  reports/Q3 report (final).md" in lines


def test_empty_values_are_explicit(tmp_path):
    a = _artifact(primary=None, files=[])
    lines = render_listing([(tmp_path, [a])], ("primary", "files", "service_files")).splitlines()
    assert lines[-3:] == ["primary: -", "files: none", "service_files: none"]


def test_unmatched_keys_follow_the_entries():
    text = render_listing([(ROOT, [_artifact()])], ("name",), unmatched=["old-report", "123"])
    assert text.splitlines()[-2:] == ["", "unmatched: old-report, 123"]


def test_nothing_matched():
    assert render_listing([(ROOT, [])], FIELDS, unmatched=["x"]) == "No artifacts matched.\nunmatched: x"


def test_no_artifacts_yet():
    assert render_listing([(ROOT, [])], SUMMARY_FIELDS) == "No artifacts yet."


def test_each_root_is_its_own_block():
    other = Path("/work/other/.anton/artifacts")
    second = _artifact(slug="forecast-77c0e9d1", id="77c0e9d1" + "0" * 24)
    text = render_listing([(ROOT, [_artifact()]), (other, [second])], ("name",))
    positions = [
        text.index(f"Artifacts root: {ROOT}"),
        text.index("## sales-dashboard-3f2c1a9b"),
        text.index(f"Artifacts root: {other}"),
        text.index("## forecast-77c0e9d1"),
    ]
    assert positions == sorted(positions)
