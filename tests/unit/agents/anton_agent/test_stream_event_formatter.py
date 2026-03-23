from __future__ import annotations

from unittest.mock import patch


def test_stream_event_formatter_emits_initial_thinking_and_text_delta():
    from minds.agents.anton_agent.anton.llm.provider import StreamTextDelta
    from minds.agents.anton_agent.stream_event_formatter import AntonStreamEventFormatter
    from minds.schemas.chat import Role

    with patch("minds.agents.anton_agent.stream_event_formatter.random.choice", return_value="thinking..."):
        fmt = AntonStreamEventFormatter(progress_throttle_seconds=0)
        chunks = fmt.on_event(StreamTextDelta(text="hi"))

    assert chunks[0] == (Role.system, "thinking...")
    assert chunks[1] == (Role.assistant, "hi")


def test_stream_event_formatter_tool_use_and_progress_and_truncation():
    from minds.agents.anton_agent.anton.llm.provider import (
        StreamTaskProgress,
        StreamToolResult,
        StreamToolUseDelta,
        StreamToolUseEnd,
        StreamToolUseStart,
    )
    from minds.agents.anton_agent.stream_event_formatter import AntonStreamEventFormatter
    from minds.schemas.chat import Role

    with patch("minds.agents.anton_agent.stream_event_formatter.random.choice", return_value="thinking..."):
        fmt = AntonStreamEventFormatter(tool_result_max_chars=5, progress_throttle_seconds=0)

        chunks: list[tuple[str, str]] = []
        chunks += fmt.on_event(StreamToolUseStart(id="t1", name="scratchpad"))
        chunks += fmt.on_event(
            StreamToolUseDelta(id="t1", json_delta='{"action":"exec","one_line_description":"Do X"}')
        )
        chunks += fmt.on_event(StreamToolUseEnd(id="t1"))

        chunks += fmt.on_event(StreamTaskProgress(phase="scratchpad", message="running", eta_seconds=3))
        chunks += fmt.on_event(StreamToolResult(content="123456789"))

    assert chunks[0] == (Role.system, "thinking...")
    assert (Role.thought_scratchpad_start, '{"action":"exec","one_line_description":"Do X"}') in chunks
    assert (Role.system, "Scratchpad(Do X): running") in chunks
    assert (Role.thought_scratchpad_end, "1234…") in chunks


def test_stream_event_formatter_analyzing_and_context_compacted_and_throttle():
    from minds.agents.anton_agent.anton.llm.provider import StreamContextCompacted, StreamTaskProgress
    from minds.agents.anton_agent.stream_event_formatter import AntonStreamEventFormatter
    from minds.schemas.chat import Role

    with patch("minds.agents.anton_agent.stream_event_formatter.random.choice", return_value="Analyzing results..."):
        fmt = AntonStreamEventFormatter(progress_throttle_seconds=9999)
        chunks = fmt.on_event(StreamTaskProgress(phase="analyzing", message="x"))
        assert (Role.system, "Analyzing results...") in chunks

        p1 = fmt.on_event(StreamTaskProgress(phase="planning", message="m"))
        p2 = fmt.on_event(StreamTaskProgress(phase="planning", message="m"))
        assert p1 and p2 == []

        cc = fmt.on_event(StreamContextCompacted(message="compacted"))
        assert (Role.thought_context_compacted, "compacted") in cc
