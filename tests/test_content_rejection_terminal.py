"""ENG-2689 — a permanent content rejection ends the turn on the first attempt.

A `ContentValidationError` means the provider refused the request over content
that is ALREADY in conversation history. The request is rebuilt from that same
stored history on every attempt, so it is byte-identical each time and fails
identically each time — the type's own docstring says so, and the live incident
proved it: four attempts inside 33.7s carrying four images with four identical
Langfuse media IDs, differing only by an appended "SYSTEM: An error interrupted
execution" note that cannot make an image smaller.

It was retried anyway, because the type was not on any of the session's
non-retry paths. Three costs, all paid by the user who reported this: three
wasted full-context requests, a recovery note instructing the model to "diagnose
and fix" something it cannot reach, and a ~34s delay before the failure reached
the host — which is what triggers the history repair that actually unsticks the
conversation.

The raise sits AFTER `_seal_dangling_tool_uses` rather than with the early-raise
types at the top of the handler, because the refusal can land mid-tool-round and
an orphan `tool_use` left in history would 400 the NEXT turn too — converting a
repairable conversation into a differently-broken one.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from anton.chat import ChatSession
from anton.core.llm.provider import (
    ContentTooLargeError,
    ContentValidationError,
    StreamTextDelta,
)
from anton.core.session import ChatSessionConfig
from tests.conftest import make_mock_llm

_PROVIDER_COPY = (
    "An image in this conversation is too large for the model to accept. "
    "The provider said: The image you provided requires 32400 patches after "
    "processing, exceeding the limit of 30000. Please resize the image and "
    "try again."
)


def _session() -> ChatSession:
    # A session_id, deliberately: a turn with none and zero LLM calls is
    # dropped before the analytics sink as script traffic (ENG-1692).
    s = ChatSession(ChatSessionConfig(llm_client=make_mock_llm(), session_id="conv-2689"))
    s._llm.planning_model = "latest:sonnet"
    return s


def _counting_raiser(exc_factory, calls: list):
    """A `_stream_and_handle_tools` stand-in that records each attempt."""

    async def _gen(user_msg):
        calls.append(user_msg)
        raise exc_factory()
        yield  # pragma: no cover  (makes this an async generator)

    return _gen


def _dangling_tool_use_then_raise(exc_factory, calls: list, session: ChatSession):
    """Commits an assistant `tool_use` to history and THEN fails — the shape
    that leaves an orphan the next request would 400 on."""

    async def _gen(user_msg):
        calls.append(user_msg)
        session._append_history({
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "toolu_2689",
                "name": "read_file",
                "input": {"path": "/tmp/x"},
            }],
        })
        raise exc_factory()
        yield  # pragma: no cover

    return _gen


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: ContentValidationError("unsupported image block"),
        lambda: ContentTooLargeError(_PROVIDER_COPY),
    ],
    ids=["shape", "size"],
)
async def test_a_content_rejection_is_attempted_once(exc_factory):
    """The headline. Both families, because the size type is only correct by
    virtue of subclassing the shape one — a future refactor that broke the
    inheritance would keep the shape case green and silently restore the
    four-retry bug for the case this ticket is about."""
    s = _session()
    calls: list = []
    s._stream_and_handle_tools = _counting_raiser(exc_factory, calls)

    with pytest.raises(ContentValidationError):
        _ = [e async for e in s.turn_stream("what does this screenshot say?")]

    assert len(calls) == 1, f"expected a single attempt, got {len(calls)}"


async def test_no_recovery_note_is_injected_for_a_content_rejection():
    """The retry path appends "SYSTEM: An error interrupted execution … If you
    can diagnose and fix the issue, continue working". For a provider refusing
    an oversized image that instruction is unactionable — the model cannot
    resize what the user attached — and it permanently pollutes the history
    that every later turn replays."""
    s = _session()
    s._stream_and_handle_tools = _counting_raiser(
        lambda: ContentTooLargeError(_PROVIDER_COPY), []
    )

    with pytest.raises(ContentTooLargeError):
        _ = [e async for e in s.turn_stream("what does this screenshot say?")]

    text = " ".join(
        str(m.get("content", "")) for m in s._history if isinstance(m, dict)
    )
    assert "An error interrupted execution" not in text
    assert "Adjust your approach" not in text


async def test_a_dangling_tool_use_is_sealed_before_the_turn_fails():
    """Why this raise sits after the seal and not with the early-raise types.
    An orphan `tool_use` in stored history 400s the NEXT request too, so
    failing fast without sealing would trade a repairable conversation for a
    differently-broken one — and the repair downstream strips images, not
    orphan tool calls, so nothing else would fix it."""
    s = _session()
    calls: list = []
    s._stream_and_handle_tools = _dangling_tool_use_then_raise(
        lambda: ContentTooLargeError(_PROVIDER_COPY), calls, s
    )

    with pytest.raises(ContentTooLargeError):
        _ = [e async for e in s.turn_stream("what does this screenshot say?")]

    assert len(calls) == 1
    sealed = [
        b
        for m in s._history
        if isinstance(m, dict) and isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
        and b.get("tool_use_id") == "toolu_2689"
    ]
    assert sealed, "the orphan tool_use was left unsealed"


async def test_the_content_terminal_is_recorded_in_the_books():
    """`ended_by` and `error_type` cannot express WHY the retry flow stopped,
    so a turn killed by a content refusal would otherwise be indistinguishable
    from one that exhausted its retries against a flaky provider."""
    s = _session()
    s._stream_and_handle_tools = _counting_raiser(
        lambda: ContentTooLargeError(_PROVIDER_COPY), []
    )

    with patch("anton.analytics.send_event") as send:
        with pytest.raises(ContentTooLargeError):
            _ = [e async for e in s.turn_stream("what does this screenshot say?")]

    assert send.called, "no turn_completed event emitted"
    assert send.call_args.kwargs["retry_terminal_reason"] == "content_rejected"


async def test_the_refusal_reaches_the_caller_instead_of_becoming_prose():
    """The host maps this exception to a card and repairs the conversation.
    Both only happen if it propagates — wrapped into assistant text it becomes
    an untyped paragraph, and the conversation stays poisoned."""
    s = _session()
    s._stream_and_handle_tools = _counting_raiser(
        lambda: ContentTooLargeError(_PROVIDER_COPY), []
    )

    events: list = []
    with pytest.raises(ContentTooLargeError) as err:
        async for e in s.turn_stream("what does this screenshot say?"):
            events.append(e)

    assert "Please resize the image and try again." in str(err.value)
    text = "".join(e.text for e in events if isinstance(e, StreamTextDelta))
    assert "An unexpected error occurred" not in text


# ── the message's promise must be true for EVERY host (review of #484) ──────
# anton tells the user the image "will be removed automatically so the
# conversation can continue". cowork-server makes that true in its own store.
# The standalone CLI has no store — it keeps ONE long-lived session — so
# without an in-session repair the promise was false there and the next turn
# re-sent the same image forever.


def _image_history_turn(exc_factory, session, calls: list):
    async def _gen(user_msg):
        calls.append(user_msg)
        raise exc_factory()
        yield  # pragma: no cover

    return _gen


def _image_blocks(history) -> list:
    return [
        b
        for m in history
        if isinstance(m, dict) and isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "image"
    ]


async def test_the_offending_image_is_removed_from_history():
    s = _session()
    s._append_history({
        "role": "user",
        "content": [
            {"type": "text", "text": "what does this say?"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": "AAAA"}},
        ],
    })
    assert _image_blocks(s._history), "fixture did not seed an image"

    calls: list = []
    s._stream_and_handle_tools = _image_history_turn(
        lambda: ContentTooLargeError(_PROVIDER_COPY), s, calls
    )
    with pytest.raises(ContentTooLargeError):
        _ = [e async for e in s.turn_stream("what does this say?")]

    assert not _image_blocks(s._history), (
        "the image survived — the next turn would re-send it and fail identically"
    )
    text = " ".join(str(m.get("content", "")) for m in s._history if isinstance(m, dict))
    assert "removed" in text.lower(), "no placeholder explaining the removal"


async def test_an_image_in_a_tool_result_is_removed_too():
    """A screenshot returned by a tool is as poisonous as an attached one, and
    it hides one level deeper in the history."""
    s = _session()
    s._append_history({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": "toolu_x",
            "content": [{"type": "image", "source": {"type": "base64",
                                                     "media_type": "image/png",
                                                     "data": "AAAA"}}],
        }],
    })
    s._stream_and_handle_tools = _image_history_turn(
        lambda: ContentTooLargeError(_PROVIDER_COPY), s, []
    )
    with pytest.raises(ContentTooLargeError):
        _ = [e async for e in s.turn_stream("what does this say?")]

    nested = [
        b
        for m in s._history
        if isinstance(m, dict) and isinstance(m.get("content"), list)
        for blk in m["content"]
        if isinstance(blk, dict) and isinstance(blk.get("content"), list)
        for b in blk["content"]
        if isinstance(b, dict) and b.get("type") == "image"
    ]
    assert not nested, "an image nested in a tool_result survived the repair"


# ── the repair must cover every image shape, and outlive the process ────────
# Two gaps found reviewing the repair above (#484).


async def test_an_openai_shaped_image_url_block_is_removed_too():
    """`turn_stream` accepts `image_url` as public input and the provider
    translates it onward, so a repair that only knows the Anthropic `image`
    shape leaves the OpenAI one in history to be re-sent next turn — the exact
    failure the repair exists to stop. This file's own two other image sites
    already treat ("image", "image_url") as the pair."""
    s = _session()
    s._append_history({
        "role": "user",
        "content": [
            {"type": "text", "text": "read this"},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    })
    s._stream_and_handle_tools = _counting_raiser(
        lambda: ContentTooLargeError(_PROVIDER_COPY), []
    )
    with pytest.raises(ContentTooLargeError):
        _ = [e async for e in s.turn_stream("read this")]

    left = [
        b
        for m in s._history
        if isinstance(m, dict) and isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") in ("image", "image_url")
    ]
    assert not left, "an OpenAI-shaped image_url block survived the repair"


async def test_the_repair_is_persisted_so_resume_does_not_resend_it():
    """The turn-end `_persist_history()` sits below this raise and never runs,
    and `close()` does not save — so without an explicit save the repair lives
    only in memory and `/resume` reloads the original image and repeats the
    refusal forever."""
    saved: list = []

    class _Store:
        def save(self, session_id, history):
            saved.append((session_id, [dict(m) for m in history]))

        def load(self, session_id):
            return saved[-1][1] if saved else []

    s = _session()
    s._history_store = _Store()
    s._append_history({
        "role": "user",
        "content": [{"type": "image",
                     "source": {"type": "base64", "media_type": "image/png",
                                "data": "AAAA"}}],
    })
    s._stream_and_handle_tools = _counting_raiser(
        lambda: ContentTooLargeError(_PROVIDER_COPY), []
    )
    with pytest.raises(ContentTooLargeError):
        _ = [e async for e in s.turn_stream("what is this?")]

    assert saved, "the repaired history was never saved"
    resumed = saved[-1][1]
    survived = [
        b
        for m in resumed
        if isinstance(m, dict) and isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") in ("image", "image_url")
    ]
    assert not survived, "a resumed session would resend the rejected image"


async def test_a_failing_history_store_does_not_change_the_raised_error():
    """The save runs on the error path. An escape there would convert a
    handled, card-mappable failure into an unexpected one."""
    class _Broken:
        def save(self, session_id, history):
            raise OSError("disk full")

    s = _session()
    s._history_store = _Broken()
    s._append_history({
        "role": "user",
        "content": [{"type": "image",
                     "source": {"type": "base64", "media_type": "image/png",
                                "data": "AAAA"}}],
    })
    s._stream_and_handle_tools = _counting_raiser(
        lambda: ContentTooLargeError(_PROVIDER_COPY), []
    )
    with pytest.raises(ContentTooLargeError):
        _ = [e async for e in s.turn_stream("what is this?")]
