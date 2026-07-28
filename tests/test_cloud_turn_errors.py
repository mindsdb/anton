"""Central exception → structured TurnErrorV1 mapping (item 4)."""

from __future__ import annotations

import pytest

from anton.cloud_turn.errors import (
    DeadlineExceededError,
    InvalidRequestError,
    UnsupportedCapabilityError,
    UnsupportedModelError,
    UnsupportedProtocolVersionError,
    classify_error,
)
from anton.core.llm.provider import (
    ContextOverflowError,
    ModelUnavailableError,
    ProviderOverloadedError,
    TokenLimitExceeded,
    TransientProviderError,
)


@pytest.mark.parametrize(
    "exc, code, retryable",
    [
        (InvalidRequestError("bad"), "invalid_request", False),
        (UnsupportedProtocolVersionError("v2"), "unsupported_protocol_version", False),
        (UnsupportedCapabilityError("connectors"), "unsupported_capability", False),
        (UnsupportedModelError("nope"), "unsupported_model", False),
        (DeadlineExceededError("late"), "deadline_exceeded", True),
        (ModelUnavailableError("gate", code="model_disabled", model="m"), "unsupported_model", False),
        (TransientProviderError("overloaded"), "model_provider_failure", True),
        (ProviderOverloadedError("exhausted"), "model_provider_failure", True),
        (TokenLimitExceeded("quota"), "model_provider_failure", False),
        (ContextOverflowError("too long"), "model_provider_failure", False),
        (ConnectionError("Invalid API key — check your ANTHROPIC_API_KEY"), "model_auth_failure", False),
        (ConnectionError("connection reset by peer"), "model_provider_failure", True),
        (RuntimeError("unexpected"), "internal_turn_failure", False),
    ],
)
def test_each_mapping(exc, code, retryable):
    err = classify_error(exc)
    assert err.code.value == code
    assert err.retryable is retryable
    assert err.message  # non-empty, short


def test_message_is_scrubbed_and_bounded():
    from anton.cloud_turn.errors import MAX_ERROR_MESSAGE_CHARS

    # A key-shaped token is redacted by scrub_credentials; the message is
    # truncated so nothing long/verbose reaches the wire.
    leaked_key = "sk-ant-api03-" + "A" * 80
    err = classify_error(RuntimeError(f"boom with key {leaked_key} " + "x" * 1000))
    assert leaked_key not in err.message
    assert len(err.message) <= MAX_ERROR_MESSAGE_CHARS
