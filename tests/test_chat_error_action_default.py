"""`_default_turn_error_action` — which action the interactive CLI's
turn-failure prompt defaults to (review feedback on ENG-1310).

ModelUnavailableError and EndpointConfigurationError are deterministic for
the identical request, so the CLI already steered them to "setup". A
provider-auth 401 is equally deterministic (ENG-1310 made it propagate
instead of flattening into chat text), but nothing steered its default away
from "retry" — a gap three past PRs (#236, #247, #288) flagged for this
error-default pattern. This pins the typed distinction.
"""

from __future__ import annotations

from anton.chat import _default_turn_error_action
from anton.core.llm.provider import (
    EndpointConfigurationError,
    ModelUnavailableError,
    ProviderAuthError,
    TokenLimitExceeded,
)

_AUTH_ERROR_MESSAGE = "Invalid API key — check your OpenAI API key configuration."


def test_provider_auth_error_defaults_to_setup():
    assert _default_turn_error_action(ProviderAuthError(_AUTH_ERROR_MESSAGE)) == "setup"


def test_invalid_key_text_without_the_canonical_type_defaults_to_retry():
    assert _default_turn_error_action(ConnectionError(_AUTH_ERROR_MESSAGE)) == "retry"


def test_generic_connection_error_still_defaults_to_retry():
    assert _default_turn_error_action(ConnectionError("temporarily unavailable")) == "retry"


def test_model_unavailable_error_defaults_to_setup():
    assert _default_turn_error_action(ModelUnavailableError("blocked", code="model_access_denied", model="sonnet")) == "setup"


def test_endpoint_configuration_error_defaults_to_setup():
    assert _default_turn_error_action(EndpointConfigurationError("bad endpoint")) == "setup"


def test_non_connection_error_defaults_to_setup():
    assert _default_turn_error_action(TokenLimitExceeded("out of tokens")) == "setup"
