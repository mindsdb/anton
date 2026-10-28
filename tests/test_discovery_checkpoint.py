"""discovery.json: the phase boundary that survives the process."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from anton.core.artifacts.internal_files import DATA_DIRNAME, DISCOVERY_FILENAME
from anton.core.tools.generate_artifact import engine as gen_engine
from anton.core.tools.generate_artifact.data_files import (
    DATA_NOT_SAVED_WARNING, DataFile, clear_data_not_saved,
)
from anton.core.tools.generate_artifact.discovery import checkpoint as cp
from anton.core.tools.generate_artifact.state import GenState


def _cp(**over) -> cp.DiscoveryCheckpoint:
    base = dict(
        request_fingerprint=cp.request_fingerprint("build me a dashboard"),
        call_fingerprint=cp.call_fingerprint("a dashboard", "", ""),
        pipeline_stage=cp.STAGE_PRD_WRITTEN,
        artifact_type="html-app",
        gathering_complete=True,
        declared_sources=["the article"],
        unverified_sources=[],
        brief_markdown="## Goal\nA dashboard.",
        data_notes="Scratchpad `dash`:\n```python\nx = 1\n```",
        web_notes="- https://example.com/x — Title",
        assumptions=["dark theme"],
        open_points=["count cancelled orders?"],
        known_data="site,lat\nToronto,43.7",
    )
    base.update(over)
    return cp.DiscoveryCheckpoint(**base)


# ── normalization and fingerprints ──────────────────────────────────────────


def test_whitespace_noise_does_not_change_the_request_fingerprint():
    a = cp.request_fingerprint("  build   me a\ndashboard ")
    b = cp.request_fingerprint("build me a dashboard")
    assert a == b


def test_different_requests_get_different_fingerprints():
    assert cp.request_fingerprint("a dashboard") != cp.request_fingerprint("a game")


def test_call_fingerprint_covers_all_three_soft_fields():
    base = cp.call_fingerprint("understanding", "data", "prefs")
    assert base != cp.call_fingerprint("other", "data", "prefs")
    assert base != cp.call_fingerprint("understanding", "other", "prefs")
    assert base != cp.call_fingerprint("understanding", "data", "other")


def test_call_fingerprint_covers_the_user_language():
    base = cp.call_fingerprint("understanding", "data", "prefs", "English")
    assert base != cp.call_fingerprint("understanding", "data", "prefs", "Russian")
    assert base == cp.call_fingerprint("understanding", "data", "prefs", "english")


def test_call_fingerprint_keys_on_the_language_the_steps_write_in():
    """No language means the English default, so the two calls write the
    same brief and must not redraw it."""
    assert cp.call_fingerprint("u", "d", "p", "") == cp.call_fingerprint("u", "d", "p", "English")
    assert cp.call_fingerprint("u", "d", "p") == cp.call_fingerprint("u", "d", "p", "English")


def test_an_english_call_matches_a_checkpoint_saved_before_the_language_existed():
    """A checkpoint saved before `user_language` hashed only three fields.
    An English call must still match it, or every existing artifact redraws
    its brief once after the upgrade and asks to confirm it again."""
    before = cp.fingerprint("u", "d", "p")
    assert cp.call_fingerprint("u", "d", "p", "English") == before
    assert cp.call_fingerprint("u", "d", "p", "Russian") != before


def test_fields_are_not_concatenation_ambiguous():
    # "ab" + "" must not fingerprint the same as "a" + "b".
    assert cp.call_fingerprint("ab", "", "") != cp.call_fingerprint("a", "b", "")


# ── round trip ──────────────────────────────────────────────────────────────


def test_save_then_load_round_trips_every_field(tmp_path: Path):
    original = _cp()
    cp.save(tmp_path, original)
    assert (tmp_path / DISCOVERY_FILENAME).is_file()
    loaded = cp.load(tmp_path)
    assert loaded == original


def test_load_returns_none_when_there_is_no_file(tmp_path: Path):
    assert cp.load(tmp_path) is None


def test_load_returns_none_on_unreadable_json(tmp_path: Path):
    (tmp_path / DISCOVERY_FILENAME).write_text("{not json", encoding="utf-8")
    assert cp.load(tmp_path) is None


def test_load_ignores_unknown_keys_from_a_future_version(tmp_path: Path):
    payload = {"pipeline_stage": cp.STAGE_GENERATED, "something_new": 42}
    (tmp_path / DISCOVERY_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    loaded = cp.load(tmp_path)
    assert loaded is not None
    assert loaded.pipeline_stage == cp.STAGE_GENERATED


def test_load_defaults_the_lists_a_pre_feature_file_lacks(tmp_path: Path):
    """`assumptions` / `open_points` were added later than the checkpoint
    format; a file written before them must still load, with the lists empty."""
    payload = {"pipeline_stage": cp.STAGE_PRD_WRITTEN, "brief_markdown": "## Goal"}
    (tmp_path / DISCOVERY_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    loaded = cp.load(tmp_path)
    assert loaded is not None
    assert loaded.assumptions == [] and loaded.open_points == []


def test_load_defaults_known_data_for_a_file_written_before_it(tmp_path: Path):
    payload = {"pipeline_stage": cp.STAGE_PRD_WRITTEN, "brief_markdown": "## Goal"}
    (tmp_path / DISCOVERY_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    loaded = cp.load(tmp_path)
    assert loaded is not None
    assert loaded.known_data == ""
    assert loaded.known_data_verified == []


# ── entry decision ──────────────────────────────────────────────────────────


def test_no_checkpoint_means_the_full_path():
    fp = cp.request_fingerprint("anything")
    assert cp.decide_entry(None, request_fp=fp) == cp.ENTRY_FULL


def test_a_different_request_means_the_full_path():
    stored = _cp()
    other = cp.request_fingerprint("something else entirely")
    assert cp.decide_entry(stored, request_fp=other) == cp.ENTRY_FULL


def test_awaiting_confirmation_resumes_at_the_confirmation_step():
    stored = _cp(pipeline_stage=cp.STAGE_AWAITING_CONFIRMATION)
    assert cp.decide_entry(stored, request_fp=stored.request_fingerprint) == cp.ENTRY_CONFIRM


def test_prd_written_resumes_at_the_spec_phase():
    stored = _cp(pipeline_stage=cp.STAGE_PRD_WRITTEN)
    assert cp.decide_entry(stored, request_fp=stored.request_fingerprint) == cp.ENTRY_SPEC


def test_spec_written_resumes_at_the_generation_phase():
    stored = _cp(pipeline_stage=cp.STAGE_SPEC_WRITTEN)
    assert cp.decide_entry(stored, request_fp=stored.request_fingerprint) == cp.ENTRY_GENERATE


def test_generated_starts_a_new_iteration():
    stored = _cp(pipeline_stage=cp.STAGE_GENERATED)
    assert cp.decide_entry(stored, request_fp=stored.request_fingerprint) == cp.ENTRY_NEW_ITERATION


def test_an_unknown_stage_falls_back_to_the_full_path():
    stored = _cp(pipeline_stage="something_we_never_wrote")
    assert cp.decide_entry(stored, request_fp=stored.request_fingerprint) == cp.ENTRY_FULL


# ── data files ───────────────────────────────────────────────────────────────


def _gen_state(tmp_path, **kw):
    base = dict(session=SimpleNamespace(), artifact_type="html-app", artifact_path=tmp_path, slug="s")
    base.update(kw)
    return GenState(**base)


def _saved(tmp_path, known_data="", known_data_verified=()):
    (tmp_path / DATA_DIRNAME).mkdir(exist_ok=True)
    (tmp_path / DATA_DIRNAME / "1.json").write_text(json.dumps([{"v": 1}]))
    return cp.DiscoveryCheckpoint(
        pipeline_stage=cp.STAGE_GENERATED,
        declared_sources=["orders"],
        known_data=known_data,
        known_data_verified=list(known_data_verified),
        data_files=[DataFile("1", ["orders"], "rows").to_dict()],
    )


def test_data_files_round_trip_through_the_checkpoint(tmp_path):
    stored = _saved(tmp_path)
    cp.save(tmp_path, stored)
    assert cp.load(tmp_path).data_files == [{"id": "1", "sources": ["orders"], "description": "rows"}]


def test_continuing_the_same_run_restores_data_files(tmp_path):
    state = _gen_state(tmp_path)
    assert gen_engine._restore(state, _saved(tmp_path), entry=cp.ENTRY_GENERATE) is False
    assert state.data_files == [DataFile("1", ["orders"], "rows")]
    assert state.unverified_sources == []


@pytest.mark.parametrize("damage", ["delete", "corrupt"])
def test_a_data_file_lost_between_calls_sends_its_sources_back_to_the_data_loop(tmp_path, damage):
    stored = _saved(tmp_path)
    path = tmp_path / DATA_DIRNAME / "1.json"
    if damage == "delete":
        path.unlink()
    else:
        path.write_text("[{")
    state = _gen_state(tmp_path)
    assert gen_engine._restore(state, stored, entry=cp.ENTRY_GENERATE) is True
    assert state.data_files == []
    assert state.unverified_sources == ["orders"]


@pytest.mark.parametrize(
    "known_data, unverified",
    [
        # No data passed: what only the old data verified is read again, except
        # what the saved file covers.
        ("", ["regions"]),
        ("scratchpad `pad`, variable `df` (DataFrame)", []),
    ],
)
def test_new_iteration_without_new_data_reuses_the_files(tmp_path, known_data, unverified):
    stored = _saved(
        tmp_path,
        known_data="scratchpad `pad`, variable `df` (DataFrame)",
        known_data_verified=["orders", "regions"],
    )
    state = _gen_state(tmp_path, known_data=known_data)
    gen_engine._restore(state, stored, entry=cp.ENTRY_NEW_ITERATION)
    assert [f.id for f in state.data_files] == ["1"]
    assert state.unverified_sources == unverified


def test_new_iteration_with_other_data_fetches_again(tmp_path):
    stored = _saved(tmp_path, known_data="scratchpad `pad`, variable `df` (DataFrame)")
    state = _gen_state(tmp_path, known_data="scratchpad `pad`, variable `other` (DataFrame)")
    gen_engine._restore(state, stored, entry=cp.ENTRY_NEW_ITERATION)
    assert state.data_files == []
    assert state.unverified_sources == ["orders"]


def test_a_data_not_saved_warning_survives_the_confirmation_call(tmp_path):
    """Ask-first: gathering ran in the call that ended with needs_confirmation;
    the warning has to reach the call that builds the page."""
    stored = _saved(tmp_path)
    stored.data_not_saved = True
    state = _gen_state(tmp_path)
    gen_engine._restore(state, stored, entry=cp.ENTRY_CONFIRM)
    assert state.data_not_saved is True
    assert DATA_NOT_SAVED_WARNING in state.verify_warnings


def test_the_mark_clears_once_files_cover_every_declared_source(tmp_path):
    state = _gen_state(tmp_path, declared_sources=["orders", "regions"], data_not_saved=True)
    state.verify_warnings.append(DATA_NOT_SAVED_WARNING)
    state.data_files = [DataFile("1", ["orders"])]
    clear_data_not_saved(state)
    assert state.data_not_saved is True  # "regions" is still not covered

    state.data_files.append(DataFile("2", ["regions"]))
    clear_data_not_saved(state)
    assert state.data_not_saved is False
    assert DATA_NOT_SAVED_WARNING not in state.verify_warnings
