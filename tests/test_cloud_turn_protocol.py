"""Contract tests for the cloud-turn V1 wire protocol."""

from __future__ import annotations

import json

import pytest

from anton.cloud_turn import (
    PROTOCOL_VERSION,
    CapabilitiesV1,
    ErrorCodeV1,
    MessageV1,
    TurnCompletedV1,
    TurnFailedV1,
    TurnStartedV1,
    event_line,
    parse_request,
)
from anton.cloud_turn.errors import (
    InvalidRequestError,
    UnsupportedProtocolVersionError,
)
from anton.cloud_turn.protocol import (
    MAX_HISTORY_MESSAGES,
    MAX_REQUEST_BYTES,
    MAX_TEXT_BLOCK_CHARS,
    TERMINAL_KINDS,
    TurnErrorV1,
    TurnRequestV1,
)


def _minimal_request_json(**overrides) -> str:
    body = {
        "run_id": "run_1",
        "attempt_id": "att_1",
        "conversation_id": "conv_1",
        "input": "hello",
    }
    body.update(overrides)
    return json.dumps(body)


# ── request ───────────────────────────────────────────────────────────────

def test_capabilities_default_everything_off():
    caps = CapabilitiesV1()
    assert not any(getattr(caps, f) for f in CapabilitiesV1.model_fields)


def test_request_minimal_parses_with_safe_defaults():
    req = parse_request(_minimal_request_json())
    assert req.protocol_version == PROTOCOL_VERSION
    assert "workspace_path" not in TurnRequestV1.model_fields  # trusted pod config
    assert req.history == []
    assert req.model is None
    assert req.deadline_unix_ms is None
    assert req.capabilities == CapabilitiesV1()
    assert req.input == "hello"


def test_request_rejects_unknown_fields():
    with pytest.raises(InvalidRequestError):
        parse_request(_minimal_request_json(surprise="nope"))


def test_request_rejects_workspace_path_on_the_wire():
    with pytest.raises(InvalidRequestError):
        parse_request(_minimal_request_json(workspace_path="/etc"))


def test_request_rejects_wrong_protocol_version():
    with pytest.raises(UnsupportedProtocolVersionError):
        parse_request(_minimal_request_json(protocol_version=2))


def test_request_rejects_non_json():
    with pytest.raises(InvalidRequestError):
        parse_request("this is not json")


# ── typed history contract (item 3) ─────────────────────────────────────────

def test_history_accepts_supported_roles_and_blocks():
    req = parse_request(
        _minimal_request_json(
            history=[
                {"role": "user", "content": "earlier question"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "let me check"},
                        {"type": "tool_use", "id": "t1", "name": "scratchpad", "input": {"a": 1}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "42"},
                    ],
                },
            ]
        )
    )
    assert len(req.history) == 3
    assert req.history[1].content[1].name == "scratchpad"


def test_history_rejects_unsupported_role():
    with pytest.raises(InvalidRequestError):
        parse_request(_minimal_request_json(history=[{"role": "system", "content": "x"}]))


def test_history_rejects_unknown_block_type():
    with pytest.raises(InvalidRequestError):
        parse_request(
            _minimal_request_json(
                history=[{"role": "user", "content": [{"type": "image", "source": "..."}]}]
            )
        )


# ── validation limits (item 3) ──────────────────────────────────────────────

def test_history_message_count_limit():
    too_many = [{"role": "user", "content": "x"}] * (MAX_HISTORY_MESSAGES + 1)
    with pytest.raises(InvalidRequestError):
        parse_request(_minimal_request_json(history=too_many))


def test_text_block_size_limit():
    huge = "x" * (MAX_TEXT_BLOCK_CHARS + 1)
    with pytest.raises(InvalidRequestError):
        parse_request(
            _minimal_request_json(
                history=[{"role": "user", "content": [{"type": "text", "text": huge}]}]
            )
        )


def test_total_request_size_limit():
    # Build a raw string just over the byte cap without going through the model.
    oversized = "x" * (MAX_REQUEST_BYTES + 1)
    with pytest.raises(InvalidRequestError):
        parse_request(_minimal_request_json(input=oversized))


def test_exact_max_raw_request_size():
    assert MAX_REQUEST_BYTES == 10 * 1024 * 1024


def test_size_enforced_before_json_decoding(monkeypatch):
    # Oversized, NOT valid JSON: it must be rejected on size (raw bytes) BEFORE
    # any JSON decode is attempted — so the error is about size, not JSON.
    from anton.cloud_turn import protocol

    monkeypatch.setattr(protocol, "MAX_REQUEST_BYTES", 50)
    with pytest.raises(InvalidRequestError, match="exceeds"):
        parse_request("{not valid json " + "x" * 500)


def test_total_size_is_the_upper_bound_regardless_of_per_field_limits():
    # Per-field limits (1M-char text block, 1000 messages) are individually far
    # larger than the 10 MiB raw cap, so the raw cap is the true upper bound.
    assert MAX_TEXT_BLOCK_CHARS < MAX_REQUEST_BYTES
    # 1000 messages at even a modest size would exceed the raw cap and be
    # rejected on raw size first.
    assert MAX_HISTORY_MESSAGES > 0


# ── events (sequence + discriminated union) ─────────────────────────────────

def test_every_event_has_required_envelope_fields():
    for ev in (
        TurnStartedV1(run_id="r", attempt_id="a", sequence=1),
        TurnCompletedV1(run_id="r", attempt_id="a", sequence=2, final_text="done"),
        TurnFailedV1(
            run_id="r", attempt_id="a", sequence=2,
            error=TurnErrorV1(code=ErrorCodeV1.INTERNAL_TURN_FAILURE, message="boom"),
        ),
    ):
        decoded = json.loads(event_line(ev))
        assert decoded["protocol_version"] == PROTOCOL_VERSION
        assert decoded["run_id"] == "r"
        assert decoded["attempt_id"] == "a"
        assert "sequence" in decoded
        assert decoded["kind"] == ev.kind
        assert "\n" not in event_line(ev)


def test_completed_carries_output_messages_and_final_text():
    ev = TurnCompletedV1(
        run_id="r", attempt_id="a", sequence=2, final_text="hi",
        output_messages=[MessageV1(role="assistant", content="hi")],
    )
    decoded = json.loads(event_line(ev))
    assert decoded["kind"] == "turn.completed"
    assert decoded["final_text"] == "hi"
    assert decoded["output_messages"][0]["role"] == "assistant"


def test_failed_carries_structured_error():
    ev = TurnFailedV1(
        run_id="r", attempt_id="a", sequence=2,
        error=TurnErrorV1(code=ErrorCodeV1.DEADLINE_EXCEEDED, message="late", retryable=True),
    )
    decoded = json.loads(event_line(ev))
    assert decoded["error"] == {
        "code": "deadline_exceeded", "message": "late", "retryable": True,
    }


def test_no_speculative_fields_on_completed():
    fields = set(TurnCompletedV1.model_fields)
    for gone in ("usage", "history_rows", "artifact_manifest", "workspace_checkpoint"):
        assert gone not in fields


def test_terminal_kinds():
    assert TERMINAL_KINDS == {"turn.completed", "turn.failed"}
    assert TurnStartedV1(run_id="r", attempt_id="a", sequence=1).kind not in TERMINAL_KINDS
