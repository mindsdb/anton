"""The shared provider status-error mapper (ENG-598).

One mapper (`openai._raise_for_status_error`) now serves all four call paths
(chat/stream × completions/responses) — previously four copy-pasted blocks
that had already drifted in wording. These tests pin the mapping policy:

- 401 → ConnectionError with the exact invalid-key copy (cowork-server's
  provider_auth detection string-matches it).
- 429 + quota detail → TokenLimitExceeded (and it outranks any 403 logic).
- 403 + structured gateway code → ModelUnavailableError carrying code+model,
  with actionable copy per code.
- Any other 403 (BYOK region blocks, Anthropic permission errors, Cloudflare
  HTML) → the generic message, NEVER the plan copy.
"""

import pytest

from anton.core.llm.openai import _raise_for_status_error
from anton.core.llm.provider import ModelUnavailableError, TokenLimitExceeded


class _FakeStatusError(Exception):
    """Duck-typed stand-in for openai.APIStatusError (status_code + body)."""

    def __init__(self, status_code, body=None):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.body = body


def _gateway_403(code, model="sonnet"):
    return _FakeStatusError(403, body={"error": {
        "message": f"The model '{model}' is rejected.",
        "type": "invalid_request_error",
        "param": "model",
        "code": code,
    }})


# ── 401 ───────────────────────────────────────────────────────────────

def test_401_maps_to_invalid_key_connection_error():
    with pytest.raises(ConnectionError) as err:
        _raise_for_status_error(_FakeStatusError(401), "sonnet")
    # cowork-server's provider_auth detection keys on this exact phrase.
    assert "Invalid API key" in str(err.value)
    assert not isinstance(err.value, ModelUnavailableError)


# ── 429 (quota) ───────────────────────────────────────────────────────

def test_429_with_detail_maps_to_token_limit():
    exc = _FakeStatusError(429, body={"detail": "Monthly limit exceeded for tokens: 5/5"})
    with pytest.raises(TokenLimitExceeded) as err:
        _raise_for_status_error(exc, "sonnet")
    assert "Monthly limit exceeded" in str(err.value)
    assert "console.mindshub.ai" in str(err.value)


def test_bare_429_stays_generic():
    with pytest.raises(ConnectionError) as err:
        _raise_for_status_error(_FakeStatusError(429), "sonnet")
    assert "temporarily unavailable" in str(err.value)


# ── 403 with structured gateway codes ────────────────────────────────

def test_model_access_denied_maps_to_plan_copy():
    with pytest.raises(ModelUnavailableError) as err:
        _raise_for_status_error(_gateway_403("model_access_denied"), "sonnet")
    e = err.value
    assert e.code == "model_access_denied"
    assert e.model == "sonnet"
    assert "isn't included in your current MindsHub plan" in str(e)
    assert "upgrade" in str(e).lower()


def test_model_disabled_maps_to_hedged_copy():
    # Hedged until ENG-596's config lands everywhere: model_disabled can be
    # either a tier lock or an admin kill switch, so don't promise an upgrade
    # fixes it — but don't claim a transient outage either.
    with pytest.raises(ModelUnavailableError) as err:
        _raise_for_status_error(_gateway_403("model_disabled", model="opus"), "opus")
    e = err.value
    assert e.code == "model_disabled"
    assert e.model == "opus"
    assert "Switch models in Settings" in str(e)
    assert "temporarily unavailable. Try again" not in str(e)


def test_model_unavailable_is_a_connection_error():
    # Legacy call sites only know ConnectionError — the typed error must keep
    # flowing through them unchanged.
    with pytest.raises(ConnectionError):
        _raise_for_status_error(_gateway_403("model_disabled"), "sonnet")


# ── 403 WITHOUT the gateway codes: never the plan copy ────────────────

def test_byok_openai_403_falls_through_to_generic():
    # e.g. OpenAI region block — different error code.
    exc = _FakeStatusError(403, body={"error": {
        "code": "unsupported_country_region_territory",
        "message": "Country not supported.",
    }})
    with pytest.raises(ConnectionError) as err:
        _raise_for_status_error(exc, "gpt-4o")
    assert not isinstance(err.value, ModelUnavailableError)
    assert "temporarily unavailable" in str(err.value)


def test_html_403_falls_through_to_generic():
    # Cloudflare/WAF blocks have no JSON body at all.
    with pytest.raises(ConnectionError) as err:
        _raise_for_status_error(_FakeStatusError(403, body="<html>blocked</html>"), "sonnet")
    assert not isinstance(err.value, ModelUnavailableError)


def test_403_with_no_code_falls_through_to_generic():
    exc = _FakeStatusError(403, body={"error": {"message": "forbidden"}})
    with pytest.raises(ConnectionError) as err:
        _raise_for_status_error(exc, "sonnet")
    assert not isinstance(err.value, ModelUnavailableError)


# ── precedence: quota beats model-403 logic ───────────────────────────

def test_429_with_detail_wins_even_if_error_code_present():
    # A quota failure must stay token_limit, never be misread as model-403.
    exc = _FakeStatusError(429, body={
        "detail": "Monthly limit exceeded for tokens: 5/5",
        "error": {"code": "model_disabled"},
    })
    with pytest.raises(TokenLimitExceeded):
        _raise_for_status_error(exc, "sonnet")


# ── other statuses stay generic ───────────────────────────────────────

def test_500_stays_generic():
    with pytest.raises(ConnectionError) as err:
        _raise_for_status_error(_FakeStatusError(500), "sonnet")
    assert "Server returned 500" in str(err.value)
