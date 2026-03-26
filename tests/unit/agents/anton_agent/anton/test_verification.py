from __future__ import annotations

from unittest.mock import Mock

import pytest


def test_task_verification_default_remaining_work_is_not_shared():
    from minds.agents.anton_agent.anton.verification import TaskVerification

    a = TaskVerification(status="incomplete", reason="a")
    b = TaskVerification(status="incomplete", reason="b")
    a.remaining_work.append("x")

    assert b.remaining_work == []


@pytest.mark.asyncio
async def test_verify_task_calls_generate_object_and_builds_expected_message(monkeypatch):
    import minds.agents.anton_agent.anton.verification as verification_mod

    captured: dict = {}

    async def _fake_generate_object(model_cls, *, llm_provider, model, system, messages, max_tokens):
        captured["model_cls"] = model_cls
        captured["llm_provider"] = llm_provider
        captured["model"] = model
        captured["system"] = system
        captured["messages"] = messages
        captured["max_tokens"] = max_tokens
        return model_cls(status="complete", reason="ok")

    monkeypatch.setattr(verification_mod, "generate_object", _fake_generate_object)

    classification_context = "CLASSIFICATION"
    old_0 = "too old 0"
    old_1 = "too old 1"

    history = [
        {"role": "user", "content": old_0},
        {"role": "assistant", "content": old_1},
        {"role": "user", "content": "a" * 2000},  # truncates to 1500
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "b" * 900},  # truncates to 800
                {"type": "tool_result", "content": "c" * 900},  # truncates to 800
                {"type": "ignored", "text": "SHOULD_NOT_APPEAR"},
                "also ignored",
            ],
        },
        {"content": "no role provided"},
        {"role": "assistant", "content": {"type": "text", "text": "dict content ignored"}},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "ok2"},
        {"role": "assistant", "content": [{"type": "tool_result", "content": 123}]},
        {"role": "user", "content": "final user"},
        {"role": "assistant", "content": "final assistant"},
    ]

    llm_provider = Mock()
    result = await verification_mod.verify_task(
        llm_provider=llm_provider,
        model="m",
        classification_context=classification_context,
        history=history,
    )

    assert result.status == "complete"
    assert captured["model_cls"] is verification_mod.TaskVerification
    assert captured["llm_provider"] is llm_provider
    assert captured["model"] == "m"
    assert captured["system"] == verification_mod.VERIFICATION_SYSTEM_PROMPT
    assert captured["max_tokens"] == 512

    assert captured["messages"] and captured["messages"][0]["role"] == "user"
    content = captured["messages"][0]["content"]
    parts = content.split("\n\n")

    # Starts with classification context
    assert parts[0] == classification_context

    # Only last 10 messages from history appear (old_0 and old_1 are dropped)
    assert old_0 not in content
    assert old_1 not in content

    # String content truncation to 1500 chars
    a_line = next(p for p in parts if p.startswith("[user]: ") and set(p[len("[user]: ") :]) == {"a"})
    assert len(a_line[len("[user]: ") :]) == 1500

    # List block text truncation to 800 chars with role prefix
    b_line = next(p for p in parts if p.startswith("[assistant]: ") and set(p[len("[assistant]: ") :]) == {"b"})
    assert len(b_line[len("[assistant]: ") :]) == 800

    # tool_result block truncation to 800 chars
    c_line = next(p for p in parts if p.startswith("[tool_result]: ") and set(p[len("[tool_result]: ") :]) == {"c"})
    assert len(c_line[len("[tool_result]: ") :]) == 800

    # Unknown role fallback
    assert any(p == "[unknown]: no role provided" for p in parts)

    # tool_result block stringification
    assert any(p == "[tool_result]: 123" for p in parts)
