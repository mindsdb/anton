from __future__ import annotations

import json


def test_episodic_memory_log_and_recall(tmp_path):
    from minds.agents.anton_agent.anton.memory.episodes import EpisodicMemory

    mem = EpisodicMemory(tmp_path, enabled=True)
    sid = mem.start_session()
    assert sid

    mem.log_turn(1, "user", "hello world")
    mem.log_turn(1, "tool_call", "x" * 9999)  # should be truncated

    assert mem.session_count() == 1
    found = mem.recall("hello", max_results=10)
    assert found
    assert "hello world" in found[0].content
    formatted = mem.recall_formatted("hello")
    assert "hello world" in formatted


def test_episodes_log_disabled_noop(tmp_path):
    from minds.agents.anton_agent.anton.memory.episodes import EpisodicMemory

    mem = EpisodicMemory(tmp_path, enabled=False)
    assert mem.enabled is False
    mem.log_turn(1, "user", "x")
    assert mem.session_count() == 0


def test_episodic_memory_days_back_filters(tmp_path):
    from minds.agents.anton_agent.anton.memory.episodes import EpisodicMemory

    mem = EpisodicMemory(tmp_path, enabled=True)
    old = tmp_path / "20000101_000000.jsonl"
    old.write_text(
        json.dumps({"ts": "2000-01-01T00:00:00Z", "session": "s", "turn": 1, "role": "user", "content": "old"}) + "\n"
    )
    _ = mem.start_session()
    mem.log_turn(1, "user", "new content")

    out = mem.recall("old", days_back=1)
    assert out == []


def test_episodic_memory_recall_handles_missing_dir_and_bad_lines(tmp_path):
    from minds.agents.anton_agent.anton.memory.episodes import EpisodicMemory

    mem = EpisodicMemory(tmp_path / "missing", enabled=True)
    assert mem.recall("x") == []

    d = tmp_path / "episodes"
    d.mkdir()
    (d / "20260101_000000.jsonl").write_text("not-json\n\n{}\n", encoding="utf-8")
    mem2 = EpisodicMemory(d, enabled=True)
    assert mem2.recall("not-json") == []
