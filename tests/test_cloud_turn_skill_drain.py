"""`drain_pending_skills` — the end-of-turn diff of the staged drafts folder.

The drafts folder outlives the turn (it lives on the workspace PVC), so the
interesting cases are all about the baseline: what counts as changed, what must
not re-report, and what a hostile folder can smuggle onto the wire.
"""

from pathlib import Path
from types import SimpleNamespace

from anton.cloud_turn.session import (
    _DRAFT_FILE_MAX,
    _MAX_DRAFTS_PER_TURN,
    _snapshot_skill_drafts,
    drain_pending_skills,
)

SKILL_MD = "---\nname: my-skill\ndescription: d\n---\nbody"


def _session(root: Path):
    """A session whose baseline is the folder as it stands now — i.e. a turn
    starting against existing drafts."""
    root.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        _skill_drafts_root=root,
        _skill_drafts_before=_snapshot_skill_drafts(root),
    )


def _write(root: Path, slug: str, skill_md: str = SKILL_MD, **siblings: str) -> Path:
    folder = root / slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(skill_md)
    for name, text in siblings.items():
        (folder / name).write_text(text)
    return folder


def test_a_new_draft_is_reported(tmp_path):
    session = _session(tmp_path)
    _write(tmp_path, "my-skill")

    entries = drain_pending_skills(session)
    assert [e["slug"] for e in entries] == ["my-skill"]
    assert entries[0]["files"]["SKILL.md"] == SKILL_MD


def test_siblings_travel_with_the_draft(tmp_path):
    session = _session(tmp_path)
    _write(tmp_path, "my-skill", **{"recipe.md": "steps"})

    files = drain_pending_skills(session)[0]["files"]
    assert files["recipe.md"] == "steps"


def test_a_draft_from_an_earlier_turn_is_not_reported_again(tmp_path):
    _write(tmp_path, "my-skill")
    # A later turn starts with the draft already on disk.
    assert drain_pending_skills(_session(tmp_path)) == []


def test_refining_an_existing_draft_reports_it(tmp_path):
    _write(tmp_path, "my-skill")
    session = _session(tmp_path)
    _write(tmp_path, "my-skill", skill_md=SKILL_MD + "\nrefined")

    assert [e["slug"] for e in drain_pending_skills(session)] == ["my-skill"]


def test_a_sibling_only_edit_counts_as_a_change(tmp_path):
    _write(tmp_path, "my-skill", **{"recipe.md": "v1"})
    session = _session(tmp_path)
    (tmp_path / "my-skill" / "recipe.md").write_text("v2")

    assert [e["slug"] for e in drain_pending_skills(session)] == ["my-skill"]


def test_draining_twice_reports_once(tmp_path):
    session = _session(tmp_path)
    _write(tmp_path, "my-skill")

    assert len(drain_pending_skills(session)) == 1
    # A dismissed card must not come back on the next drain.
    assert drain_pending_skills(session) == []


def test_a_folder_without_skill_md_is_not_a_draft(tmp_path):
    session = _session(tmp_path)
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "notes.txt").write_text("x")

    assert drain_pending_skills(session) == []


def test_a_symlinked_file_never_reaches_the_wire(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("tenant secret")
    session = _session(tmp_path / "drafts")
    folder = _write(tmp_path / "drafts", "my-skill")
    (folder / "leak.txt").symlink_to(secret)

    files = drain_pending_skills(session)[0]["files"]
    assert "leak.txt" not in files
    assert "tenant secret" not in str(files)


def test_a_symlinked_draft_folder_is_dropped(tmp_path):
    outside = tmp_path / "outside"
    _write(outside.parent, "outside")
    drafts = tmp_path / "drafts"
    session = _session(drafts)
    (drafts / "sneaky").symlink_to(tmp_path / "outside", target_is_directory=True)

    assert drain_pending_skills(session) == []


def test_an_oversized_skill_md_drops_the_whole_draft(tmp_path):
    session = _session(tmp_path)
    # With a readable sibling present: dropping only the oversized SKILL.md
    # would otherwise report a draft carrying no procedure at all.
    _write(tmp_path, "my-skill", skill_md="x" * (_DRAFT_FILE_MAX + 1),
           **{"recipe.md": "steps"})

    assert drain_pending_skills(session) == []


def test_an_oversized_sibling_is_skipped_not_truncated(tmp_path):
    session = _session(tmp_path)
    _write(tmp_path, "my-skill", **{"blob.bin": "x" * (_DRAFT_FILE_MAX + 1)})

    files = drain_pending_skills(session)[0]["files"]
    assert "blob.bin" not in files
    assert files["SKILL.md"] == SKILL_MD


def test_the_per_turn_cap_bounds_what_goes_on_the_wire(tmp_path):
    session = _session(tmp_path)
    for i in range(_MAX_DRAFTS_PER_TURN + 3):
        _write(tmp_path, f"skill-{i:02d}")

    assert len(drain_pending_skills(session)) == _MAX_DRAFTS_PER_TURN


def test_no_drafts_root_is_not_an_error(tmp_path):
    assert drain_pending_skills(SimpleNamespace(_skill_drafts_root=None)) == []
