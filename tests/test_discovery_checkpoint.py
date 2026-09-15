"""discovery.json: the phase boundary that survives the process."""
from __future__ import annotations

import json
from pathlib import Path

from anton.core.artifacts.internal_files import DISCOVERY_FILENAME
from anton.core.tools.generate_artifact.discovery import checkpoint as cp


def _cp(**over) -> cp.DiscoveryCheckpoint:
    base = dict(
        request_fingerprint=cp.request_fingerprint("build me a dashboard"),
        call_fingerprint=cp.call_fingerprint("a dashboard", "", ""),
        pipeline_stage=cp.STAGE_PRD_WRITTEN,
        artifact_type="html-app",
        gathering_complete=True,
        declared_sources=["habr article"],
        unverified_sources=[],
        brief_markdown="## Goal\nA dashboard.",
        data_notes="Scratchpad `dash`:\n```python\nx = 1\n```",
        web_notes="- https://habr.com/x — Title",
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
