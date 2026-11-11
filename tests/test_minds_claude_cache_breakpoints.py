from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from anton.core.llm.openai import OpenAIProvider, _with_claude_cache_breakpoints
from anton.core.llm.prompt_builder import SESSION_CONTEXT_MARKER

EPHEMERAL = {"type": "ephemeral"}
SYSTEM = "shared rules" + SESSION_CONTEXT_MARKER + "this project"


def test_claude_splits_the_system_prompt_and_marks_the_latest_message():
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
    ]

    out = _with_claude_cache_breakpoints(messages, "sonnet-5-5")

    assert out[0]["content"] == [
        {"type": "text", "text": "shared rules", "cache_control": EPHEMERAL},
        {"type": "text", "text": SESSION_CONTEXT_MARKER + "this project", "cache_control": EPHEMERAL},
    ]
    assert out[-1]["cache_control"] == EPHEMERAL
    # Where the previous request ended: just before the latest assistant turn.
    assert out[1]["content"] == [{"type": "text", "text": "hi", "cache_control": EPHEMERAL}]
    assert out[2] == messages[2]
    assert messages[0]["content"] == SYSTEM


def test_a_user_message_is_marked_on_its_text():
    out = _with_claude_cache_breakpoints(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}], "claude-sonnet-5-5")

    assert out[-1]["content"] == [{"type": "text", "text": "hi", "cache_control": EPHEMERAL}]


def test_other_models_are_left_alone():
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hi"}]

    assert _with_claude_cache_breakpoints(messages, "gpt-luna-6") is messages


def test_the_previous_request_end_and_the_last_text_part_are_marked():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "draw it"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "saved"},
        {"role": "user", "content": [{"type": "text", "text": "screenshot"}, image]},
    ]

    out = _with_claude_cache_breakpoints(messages, "sonnet-5-5")

    assert out[1]["content"][0]["cache_control"] == EPHEMERAL
    assert out[-1]["content"] == [{"type": "text", "text": "screenshot", "cache_control": EPHEMERAL}, image]
    assert "cache_control" not in out[3]


def test_empty_text_is_never_marked():
    out = _with_claude_cache_breakpoints(
        [{"role": "system", "content": SESSION_CONTEXT_MARKER + "tail"},
         {"role": "user", "content": [{"type": "text", "text": ""}]}], "sonnet-5-5")

    assert [p["text"] for p in out[0]["content"]] == [SESSION_CONTEXT_MARKER + "tail"]
    assert "cache_control" not in out[1]["content"][0]


HISTORY = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello"},
    {"role": "user", "content": "more"},
]


def _passthrough(mock_openai, result):
    create = mock_openai.AsyncOpenAI.return_value.chat.completions.create = AsyncMock(return_value=result)
    return OpenAIProvider(api_key="k", flavor=OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH), create


async def test_complete_marks_only_the_system_prompt():
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="ok", tool_calls=None), finish_reason="stop")]
    with patch("anton.core.llm.openai.openai") as mock_openai:
        provider, create = _passthrough(mock_openai, response)
        await provider.complete(model="sonnet-5-5", system=SYSTEM, messages=HISTORY)

    sent = create.call_args.kwargs["messages"]
    assert all(p["cache_control"] == EPHEMERAL for p in sent[0]["content"])
    assert [m["content"] for m in sent[1:]] == ["hi", "hello", "more"]


async def test_stream_marks_the_messages():
    async def chunks():
        yield SimpleNamespace(usage=None, choices=[SimpleNamespace(
            delta=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop")])

    with patch("anton.core.llm.openai.openai") as mock_openai:
        provider, create = _passthrough(mock_openai, chunks())
        [e async for e in provider.stream(model="sonnet-5-5", system=SYSTEM, messages=HISTORY)]

    sent = create.call_args.kwargs["messages"]
    assert sent[1]["content"] == [{"type": "text", "text": "hi", "cache_control": EPHEMERAL}]
    assert sent[-1]["content"] == [{"type": "text", "text": "more", "cache_control": EPHEMERAL}]
