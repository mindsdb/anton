from __future__ import annotations


def test_hippocampus_encode_and_recall(tmp_path):
    from minds.agents.anton_agent.anton.memory.hippocampus import Hippocampus

    hc = Hippocampus(tmp_path)
    assert hc.recall_identity() == ""

    hc.encode_rule("Use httpx", kind="always")
    hc.encode_rule("Use httpx", kind="always")  # no duplicate
    rules = hc.recall_rules()
    assert "Use httpx" in rules

    hc.encode_lesson("CoinGecko rate-limits", topic="api-coingecko")
    lessons = hc.recall_lessons(token_budget=1000)
    assert "CoinGecko rate-limits" in lessons
    assert hc.recall_topic("api-coingecko")

    hc.rewrite_identity(["Name: Jorge"])
    assert "Jorge" in hc.recall_identity()
    assert hc.entry_count() >= 1


def test_hippocampus_recall_lessons_budget_and_scratchpad_wisdom(tmp_path):
    from minds.agents.anton_agent.anton.memory.hippocampus import Hippocampus

    hc = Hippocampus(tmp_path)
    hc.encode_rule("If long job -> use progress() in scratchpad", kind="when")
    hc.encode_lesson("Scratchpad cells time out after inactivity", topic="scratchpad-timeouts")
    wisdom = hc.recall_scratchpad_wisdom()
    assert "progress()" in wisdom.lower() or "scratchpad" in wisdom.lower()

    for i in range(200):
        hc.encode_lesson(f"lesson {i}")
    trimmed = hc.recall_lessons(token_budget=10)
    assert trimmed.startswith("# Lessons")


def test_hippocampus_recall_topic_missing_and_slug_sanitization(tmp_path):
    from minds.agents.anton_agent.anton.memory.hippocampus import Hippocampus

    hc = Hippocampus(tmp_path)
    assert hc.recall_topic("does-not-exist") == ""
    hc.encode_lesson("L", topic="Weird Topic!!!")
    assert hc.recall_topic("Weird Topic!!!")
