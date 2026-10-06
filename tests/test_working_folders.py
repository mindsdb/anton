"""Working folders a host hands a session, beside the workspace."""

from __future__ import annotations

from types import SimpleNamespace

from anton.config.settings import AntonSettings
from anton.core.settings import CoreSettings
from anton.core.tools.working_folders import normalize_working_folders, working_folder_roots


def test_no_folders_is_an_empty_tuple():
    assert normalize_working_folders(None) == ()
    assert normalize_working_folders([]) == ()


def test_usable_folders_are_resolved_in_order_without_repeats(tmp_path):
    first = tmp_path / "docs"
    second = tmp_path / "reports"
    first.mkdir()
    second.mkdir()

    folders = normalize_working_folders([first, str(second), first / ".." / "docs"])

    assert folders == (first.resolve(), second.resolve())


def test_relative_missing_and_file_entries_are_dropped(tmp_path):
    kept = tmp_path / "kept"
    kept.mkdir()
    a_file = tmp_path / "notes.txt"
    a_file.write_text("x")

    folders = normalize_working_folders(["kept", tmp_path / "missing", a_file, kept])

    assert folders == (kept.resolve(),)


def test_a_session_without_the_list_has_no_working_folders():
    assert working_folder_roots(SimpleNamespace()) == ()
    assert working_folder_roots(SimpleNamespace(_working_folders=None)) == ()


def test_the_session_keeps_the_normalised_list(make_session, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()

    session = make_session(working_folders=(docs, tmp_path / "missing"))

    assert working_folder_roots(session) == (docs.resolve(),)


def test_no_settings_class_can_carry_working_folders():
    """Only a host building the session names these. A settings field would be
    filled from `ANTON_*` env vars and `.env` files, widening what the tools
    may reach without any host asking."""
    assert "working_folders" not in AntonSettings.model_fields
    assert "working_folders" not in CoreSettings.model_fields
