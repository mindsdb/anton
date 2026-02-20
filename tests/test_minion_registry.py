from __future__ import annotations

import json
from datetime import datetime, timedelta

from anton.minion.registry import MinionInfo, MinionRegistry, MinionStatus


class TestMinionStatus:
    def test_status_values(self):
        assert MinionStatus.PENDING.value == "pending"
        assert MinionStatus.RUNNING.value == "running"
        assert MinionStatus.COMPLETED.value == "completed"
        assert MinionStatus.FAILED.value == "failed"
        assert MinionStatus.KILLED.value == "killed"


class TestMinionInfo:
    def test_make_id(self):
        id1 = MinionInfo.make_id()
        id2 = MinionInfo.make_id()
        assert len(id1) == 12
        assert id1 != id2

    def test_default_status(self):
        m = MinionInfo(id="test", task="do stuff", folder="/tmp")
        assert m.status == MinionStatus.PENDING
        assert m.pid is None
        assert m.error is None
        assert m.completed_at is None
        assert m.cron_expr is None
        assert m.every is None
        assert m.start_at is None
        assert m.end_at is None
        assert m.max_runs is None
        assert m.run_count == 0

    def test_with_cron(self):
        m = MinionInfo(
            id="test", task="periodic check", folder="/tmp",
            cron_expr="*/5 * * * *"
        )
        assert m.cron_expr == "*/5 * * * *"

    def test_with_scheduling(self):
        now = datetime.now()
        later = now + timedelta(hours=2)
        m = MinionInfo(
            id="test", task="scheduled", folder="/tmp",
            every="5m", start_at=now, end_at=later, max_runs=10,
        )
        assert m.every == "5m"
        assert m.start_at == now
        assert m.end_at == later
        assert m.max_runs == 10

    def test_has_runs_remaining_unlimited(self):
        m = MinionInfo(id="t", task="t", folder="/tmp")
        assert m.has_runs_remaining() is True

    def test_has_runs_remaining_with_max(self):
        m = MinionInfo(id="t", task="t", folder="/tmp", max_runs=3)
        assert m.has_runs_remaining() is True
        m.run_count = 2
        assert m.has_runs_remaining() is True
        m.run_count = 3
        assert m.has_runs_remaining() is False

    def test_is_within_schedule_no_bounds(self):
        m = MinionInfo(id="t", task="t", folder="/tmp")
        assert m.is_within_schedule() is True

    def test_is_within_schedule_before_start(self):
        future = datetime.now() + timedelta(hours=1)
        m = MinionInfo(id="t", task="t", folder="/tmp", start_at=future)
        assert m.is_within_schedule() is False

    def test_is_within_schedule_after_end(self):
        past = datetime.now() - timedelta(hours=1)
        m = MinionInfo(id="t", task="t", folder="/tmp", end_at=past)
        assert m.is_within_schedule() is False

    def test_is_within_schedule_in_window(self):
        past = datetime.now() - timedelta(hours=1)
        future = datetime.now() + timedelta(hours=1)
        m = MinionInfo(id="t", task="t", folder="/tmp", start_at=past, end_at=future)
        assert m.is_within_schedule() is True

    def test_record_run(self):
        m = MinionInfo(id="t", task="t", folder="/tmp")
        assert m.run_count == 0
        m.record_run()
        assert m.run_count == 1
        m.record_run()
        assert m.run_count == 2

    def test_minion_dir(self):
        m = MinionInfo(id="abc123", task="t", folder="/workspace")
        assert str(m.minion_dir) == "/workspace/.anton/minions/abc123"

    def test_ensure_dir(self, tmp_path):
        m = MinionInfo(id="abc123", task="t", folder=str(tmp_path))
        d = m.ensure_dir()
        assert d.is_dir()
        assert d == tmp_path / ".anton" / "minions" / "abc123"

    def test_save_status(self, tmp_path):
        m = MinionInfo(id="abc123", task="check email", folder=str(tmp_path), every="1h")
        m.save_status()
        status_file = tmp_path / ".anton" / "minions" / "abc123" / "status.json"
        assert status_file.is_file()
        data = json.loads(status_file.read_text())
        assert data["id"] == "abc123"
        assert data["task"] == "check email"
        assert data["every"] == "1h"
        assert data["status"] == "pending"


class TestMinionRegistry:
    def test_register_and_get(self):
        registry = MinionRegistry()
        m = MinionInfo(id="m1", task="task1", folder="/tmp")
        registry.register(m)

        assert registry.get("m1") is m
        assert registry.get("nonexistent") is None

    def test_list_all(self):
        registry = MinionRegistry()
        m1 = MinionInfo(id="m1", task="task1", folder="/tmp")
        m2 = MinionInfo(id="m2", task="task2", folder="/tmp")
        registry.register(m1)
        registry.register(m2)

        assert len(registry.list_all()) == 2

    def test_list_running(self):
        registry = MinionRegistry()
        m1 = MinionInfo(id="m1", task="t1", folder="/tmp", status=MinionStatus.RUNNING)
        m2 = MinionInfo(id="m2", task="t2", folder="/tmp", status=MinionStatus.PENDING)
        registry.register(m1)
        registry.register(m2)

        running = registry.list_running()
        assert len(running) == 1
        assert running[0].id == "m1"

    def test_list_scheduled_cron(self):
        registry = MinionRegistry()
        m1 = MinionInfo(id="m1", task="t1", folder="/tmp", cron_expr="* * * * *")
        m2 = MinionInfo(id="m2", task="t2", folder="/tmp")
        registry.register(m1)
        registry.register(m2)

        scheduled = registry.list_scheduled()
        assert len(scheduled) == 1
        assert scheduled[0].id == "m1"

    def test_list_scheduled_every(self):
        registry = MinionRegistry()
        m1 = MinionInfo(id="m1", task="t1", folder="/tmp", every="5m")
        m2 = MinionInfo(id="m2", task="t2", folder="/tmp")
        registry.register(m1)
        registry.register(m2)

        scheduled = registry.list_scheduled()
        assert len(scheduled) == 1

    def test_update_status(self):
        registry = MinionRegistry()
        m = MinionInfo(id="m1", task="task1", folder="/tmp")
        registry.register(m)

        result = registry.update_status("m1", MinionStatus.RUNNING)
        assert result is True
        assert registry.get("m1").status == MinionStatus.RUNNING

    def test_update_status_nonexistent(self):
        registry = MinionRegistry()
        assert registry.update_status("nope", MinionStatus.RUNNING) is False

    def test_update_status_completed_sets_timestamp(self):
        registry = MinionRegistry()
        m = MinionInfo(id="m1", task="t1", folder="/tmp")
        registry.register(m)
        assert m.completed_at is None

        registry.update_status("m1", MinionStatus.COMPLETED)
        assert m.completed_at is not None

    def test_update_status_failed_with_error(self):
        registry = MinionRegistry()
        m = MinionInfo(id="m1", task="t1", folder="/tmp")
        registry.register(m)

        registry.update_status("m1", MinionStatus.FAILED, error="boom")
        assert m.status == MinionStatus.FAILED
        assert m.error == "boom"
        assert m.completed_at is not None

    def test_killed_clears_cron(self):
        registry = MinionRegistry()
        m = MinionInfo(id="m1", task="t1", folder="/tmp", cron_expr="*/5 * * * *")
        registry.register(m)

        registry.update_status("m1", MinionStatus.KILLED)
        assert m.status == MinionStatus.KILLED
        assert m.cron_expr is None  # Cron cleared when killed
        assert m.completed_at is not None

    def test_killed_clears_every(self):
        registry = MinionRegistry()
        m = MinionInfo(id="m1", task="t1", folder="/tmp", every="5m")
        registry.register(m)

        registry.update_status("m1", MinionStatus.KILLED)
        assert m.every is None  # Every cleared when killed

    def test_remove(self):
        registry = MinionRegistry()
        m = MinionInfo(id="m1", task="t1", folder="/tmp")
        registry.register(m)

        assert registry.remove("m1") is True
        assert registry.get("m1") is None
        assert registry.remove("m1") is False

    def test_remove_nonexistent(self):
        registry = MinionRegistry()
        assert registry.remove("nope") is False
