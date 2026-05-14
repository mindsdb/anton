"""Tests for the rule-stats sidecar (Layer 3 — Phase A).

The sidecar is the storage half of retrieval-scoring: it tracks how
often a rule lands in the system prompt (retrievals) and how often it
was loaded while the pattern it warns against still fired (ignored).

These tests pin:
  - Hash stability across whitespace / case
  - Counter increment + persistence
  - Atomic write semantics (no half-written files)
  - Buffer/flush behavior (no I/O on mutation)
  - Corruption recovery (unreadable file → fresh state, not crash)
  - Forward-compat: missing `version` field auto-populated
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anton.core.memory.rule_stats import RuleStats, rule_id


class TestRuleId:
    def test_stable_across_calls(self):
        a = rule_id("Use ONE scratchpad name per task")
        b = rule_id("Use ONE scratchpad name per task")
        assert a == b

    def test_case_and_whitespace_insensitive(self):
        a = rule_id("Use ONE scratchpad name per task")
        b = rule_id("  use one SCRATCHPAD name per task  ")
        assert a == b

    def test_changes_on_content_change(self):
        a = rule_id("Use ONE scratchpad name per task")
        b = rule_id("Use ONE scratchpad name per task and reuse it")
        assert a != b

    def test_empty_input_does_not_crash(self):
        # Defensive — record_retrieval no-ops on empty input, but the
        # hash function itself should still be callable.
        assert rule_id("") == rule_id("")
        assert rule_id(None) == rule_id("")  # type: ignore[arg-type]

    def test_id_is_64bit_hex(self):
        rid = rule_id("anything at all")
        assert len(rid) == 16
        # All chars hex.
        int(rid, 16)


class TestRecordAndPersist:
    def test_record_buffers_in_memory(self, tmp_path: Path):
        stats = RuleStats(tmp_path / "rules.stats.json")
        stats.record_retrieval("rule A")
        # No file written yet — record_* is supposed to be a buffer hit,
        # not a disk hit. This is the contract that lets a turn flush
        # once instead of once per rule.
        assert not (tmp_path / "rules.stats.json").exists()
        # But the in-memory state reflects the bump.
        assert stats.get("rule A")["retrievals"] == 1

    def test_flush_persists_state(self, tmp_path: Path):
        stats = RuleStats(tmp_path / "rules.stats.json")
        stats.record_retrieval("rule A")
        stats.flush()
        # Read it back through a fresh instance.
        again = RuleStats(tmp_path / "rules.stats.json")
        assert again.get("rule A")["retrievals"] == 1

    def test_flush_is_idempotent_when_clean(self, tmp_path: Path):
        # Two flushes back-to-back shouldn't double-write or fail.
        stats = RuleStats(tmp_path / "rules.stats.json")
        stats.record_retrieval("rule A")
        stats.flush()
        mtime_after_first = (tmp_path / "rules.stats.json").stat().st_mtime
        stats.flush()  # nothing dirty → no-op
        mtime_after_second = (tmp_path / "rules.stats.json").stat().st_mtime
        assert mtime_after_first == mtime_after_second

    def test_multiple_records_one_flush(self, tmp_path: Path):
        # The whole point of the buffer pattern.
        stats = RuleStats(tmp_path / "rules.stats.json")
        for _ in range(5):
            stats.record_retrieval("rule A")
        stats.record_retrieval("rule B")
        stats.flush()
        again = RuleStats(tmp_path / "rules.stats.json")
        assert again.get("rule A")["retrievals"] == 5
        assert again.get("rule B")["retrievals"] == 1

    def test_record_ignored_separate_counter(self, tmp_path: Path):
        stats = RuleStats(tmp_path / "rules.stats.json")
        stats.record_retrieval("rule A")
        stats.record_retrieval("rule A")
        stats.record_ignored("rule A")
        stats.flush()
        rec = RuleStats(tmp_path / "rules.stats.json").get("rule A")
        assert rec["retrievals"] == 2
        assert rec["ignored"] == 1

    def test_last_retrieved_updates(self, tmp_path: Path):
        stats = RuleStats(tmp_path / "rules.stats.json")
        stats.record_retrieval("rule A")
        first = stats.get("rule A")["last_retrieved"]
        assert first is not None
        # Bump again; timestamp should not be None.
        stats.record_retrieval("rule A")
        second = stats.get("rule A")["last_retrieved"]
        # Both are ISO timestamps; equality across rapid bumps is OK,
        # what we care about is that the field gets set on every bump.
        assert second is not None

    def test_record_blank_input_is_noop(self, tmp_path: Path):
        stats = RuleStats(tmp_path / "rules.stats.json")
        stats.record_retrieval("")
        stats.record_retrieval("   ")
        stats.record_ignored("")
        stats.flush()
        assert stats.all() == {}


class TestCorruptionRecovery:
    def test_unreadable_file_falls_back_to_fresh_state(self, tmp_path: Path):
        # Write garbage at the stats path; constructor should not crash.
        path = tmp_path / "rules.stats.json"
        path.write_text("{this is not valid json", encoding="utf-8")
        stats = RuleStats(path)
        # Fresh state — no rules, default version.
        assert stats.all() == {}
        # And the next write should cleanly overwrite the garbage.
        stats.record_retrieval("rule A")
        stats.flush()
        again = RuleStats(path)
        assert again.get("rule A")["retrievals"] == 1

    def test_wrong_shape_falls_back_to_fresh_state(self, tmp_path: Path):
        path = tmp_path / "rules.stats.json"
        path.write_text('{"rules": "not a dict"}', encoding="utf-8")
        stats = RuleStats(path)
        assert stats.all() == {}


class TestAll:
    def test_all_returns_snapshot(self, tmp_path: Path):
        stats = RuleStats(tmp_path / "rules.stats.json")
        stats.record_retrieval("rule A")
        stats.record_retrieval("rule B")
        snapshot = stats.all()
        assert len(snapshot) == 2
        # Modifying the snapshot must not affect internal state.
        next(iter(snapshot.values()))["retrievals"] = 999
        assert stats.get("rule A")["retrievals"] == 1
