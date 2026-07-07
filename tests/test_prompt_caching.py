"""Prompt caching (ENG-644): stable/volatile system split, Anthropic
cache_control placement, and cache-aware usage accounting.

Invariants that matter:
  - build_parts() splits the prompt WITHOUT changing a byte of it
  - a plain-str system keeps the exact pre-caching provider behavior
  - the session's live history list is never mutated by cache marking
  - context_pressure reflects the TOTAL context (cached + fresh) so
    compaction still fires at the right time
"""

from __future__ import annotations

from unittest.mock import MagicMock

from anton.core.llm.anthropic import (
    _mark_history_for_cache,
    _system_param,
    _usage_from,
)
from anton.core.llm.prompt_builder import ChatSystemPromptBuilder, SystemPromptContext
from anton.core.llm.provider import SystemPrompt
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.settings import CoreSettings


def _builder_kwargs(**overrides):
    kwargs = dict(
        conversation_started="Monday, July 06, 2026",
        current_datetime="Wednesday, July 08, 2026 at 01:00 AM",
        system_prompt_context=SystemPromptContext(
            runtime_context="runtime", prefix="prefix text", suffix="suffix text"
        ),
        proactive_dashboards=False,
        output_dir="/tmp/out",
        memory_context="\n\nMEMORY SNAPSHOT",
        project_context="\n\nproject ctx",
        datasource_context="\n\nds ctx",
    )
    kwargs.update(overrides)
    return kwargs


class TestBuildParts:
    def test_stable_plus_volatile_is_byte_identical_to_build(self):
        builder = ChatSystemPromptBuilder()
        stable, volatile = builder.build_parts(**_builder_kwargs())
        assert stable + volatile == builder.build(**_builder_kwargs())

    def test_split_point_is_the_volatile_tail(self):
        stable, volatile = ChatSystemPromptBuilder().build_parts(**_builder_kwargs())
        assert volatile.startswith("\n\nCurrent date and time:")
        assert volatile.endswith("MEMORY SNAPSHOT")
        # Nothing volatile leaks into the stable prefix.
        assert "Current date and time" not in stable
        assert "MEMORY SNAPSHOT" not in stable
        assert stable.rstrip().endswith("suffix text")


class TestSystemParam:
    def test_plain_str_passes_through(self):
        assert _system_param("plain prompt") == "plain prompt"

    def test_system_prompt_becomes_marked_blocks(self):
        blocks = _system_param(SystemPrompt(stable="STABLE", volatile="VOLATILE"))
        assert blocks == [
            {"type": "text", "text": "STABLE", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "VOLATILE"},
        ]

    def test_empty_volatile_yields_single_block(self):
        blocks = _system_param(SystemPrompt(stable="STABLE"))
        assert len(blocks) == 1 and "cache_control" in blocks[0]


class TestMarkHistoryForCache:
    def test_str_content_wrapped_and_marked(self):
        messages = [{"role": "user", "content": "hi"}]
        marked = _mark_history_for_cache(messages)
        assert marked[-1]["content"] == [
            {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
        ]

    def test_block_content_marks_last_block_only(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                {"type": "tool_result", "tool_use_id": "t2", "content": "ok"},
            ]},
        ]
        marked = _mark_history_for_cache(messages)
        assert "cache_control" not in marked[-1]["content"][0]
        assert marked[-1]["content"][1]["cache_control"] == {"type": "ephemeral"}
        # Earlier messages untouched (same object — no needless copying).
        assert marked[0] is messages[0]

    def test_original_list_never_mutated(self):
        original_content = [{"type": "text", "text": "x"}]
        messages = [{"role": "user", "content": original_content}]
        _mark_history_for_cache(messages)
        assert "cache_control" not in original_content[-1]
        assert messages[-1]["content"] is original_content

    def test_uncacheable_or_empty_shapes_pass_through(self):
        assert _mark_history_for_cache([]) == []
        thinking = [{"role": "assistant", "content": [{"type": "thinking", "thinking": "…"}]}]
        assert _mark_history_for_cache(thinking) is thinking
        empty = [{"role": "user", "content": ""}]
        assert _mark_history_for_cache(empty) is empty


class TestCacheAwareUsage:
    def test_pressure_uses_total_context_not_just_fresh_input(self):
        api_usage = MagicMock(
            cache_read_input_tokens=139_000, cache_creation_input_tokens=0
        )
        # claude-sonnet-4-6 window = 200k → (1000 + 139000) / 200000 = 0.7
        usage = _usage_from("claude-sonnet-4-6", api_usage, 1_000, 50)
        assert usage.input_tokens == 1_000
        assert usage.cache_read_input_tokens == 139_000
        assert abs(usage.context_pressure - 0.7) < 1e-9

    def test_missing_cache_fields_default_to_zero(self):
        usage = _usage_from("claude-sonnet-4-6", object(), 2_000, 10)
        assert usage.cache_read_input_tokens == 0
        assert usage.cache_creation_input_tokens == 0
        assert abs(usage.context_pressure - 0.01) < 1e-9


class TestSessionSystemPrompt:
    async def test_returns_system_prompt_by_default(self):
        session = ChatSession(ChatSessionConfig(
            llm_client=MagicMock(), cortex=None, self_awareness=None,
        ))
        system = await session._build_system_prompt("hello")
        assert isinstance(system, SystemPrompt)
        assert system.stable and "Current date and time" in system.volatile
        # A SystemPrompt IS the full prompt string — legacy consumers that
        # measure or substring-check the system prompt stay working.
        assert system == system.stable + system.volatile

    async def test_kill_switch_returns_plain_str(self):
        session = ChatSession(ChatSessionConfig(
            llm_client=MagicMock(), cortex=None, self_awareness=None,
            settings=CoreSettings(prompt_caching=False),
        ))
        system = await session._build_system_prompt("hello")
        assert not isinstance(system, SystemPrompt)
        assert isinstance(system, str)
        assert "Current date and time" in system
