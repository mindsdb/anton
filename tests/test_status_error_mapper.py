"""The shared provider status-error mapper (ENG-598, fixed in ENG-747).

One mapper (`openai._raise_for_status_error`) serves all four call paths
(chat/stream × completions/responses). These tests pin the mapping policy:

- 401 → ConnectionError with the exact invalid-key copy (cowork-server's
  provider_auth detection string-matches it).
- 429 + quota detail → TokenLimitExceeded (and it outranks any 403 logic).
- 403 + structured gateway code → ModelUnavailableError carrying code+model,
  with actionable copy per code.
- Any other 403 (BYOK region blocks, Anthropic permission errors, Cloudflare
  HTML) → the generic message, NEVER the plan copy.

ENG-747: every exception here is constructed by the REAL pinned OpenAI SDK
from a raw HTTP response (`httpx.MockTransport`), never hand-built. The
original suite duck-typed `exc.body` as the wire envelope
(``{"error": {...}}``) — but the SDK UNWRAPS that envelope
(``openai/_client.py``: ``data = body.get("error", body)``), so the tests
passed against a shape production never produces and the model-403 card
shipped dead. If the SDK's behavior ever changes, these tests notice;
hand-built fixtures cannot.
"""

import httpx
import openai
import pytest

from anton.core.llm.openai import _raise_for_status_error
from anton.core.llm.provider import ModelUnavailableError, TokenLimitExceeded


def _sdk_error(status_code, json_body=None, text_body=None):
    """Real `openai.APIStatusError`, built by the pinned SDK from a raw HTTP
    response — exactly what production call sites catch and hand to the
    mapper. This is the load-bearing difference from the original suite."""

    def handler(request: httpx.Request) -> httpx.Response:
        if text_body is not None:
            return httpx.Response(status_code, text=text_body)
        return httpx.Response(status_code, json=json_body if json_body is not None else {})

    client = openai.OpenAI(
        base_url="http://gateway.test/v1",
        api_key="test-key",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        client.chat.completions.create(
            model="latest:sonnet", max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
    except openai.APIStatusError as exc:
        return exc
    raise AssertionError(f"SDK did not raise for HTTP {status_code}")


def _gateway_403(code, model="sonnet"):
    """The gateway's OpenAI-style 403 envelope, byte-shaped like the live
    body captured from prod on 2026-07-13 (ENG-747)."""
    return _sdk_error(403, json_body={"error": {
        "message": f"The model '{model}' is rejected.",
        "type": "invalid_request_error",
        "param": "model",
        "code": code,
    }})


# ── the SDK-unwrap regression itself ──────────────────────────────────

def test_sdk_unwraps_error_envelope():
    """Documents the SDK behavior that broke ENG-598: `exc.body` is the
    INNER error object, not the wire envelope. If this ever fails, the SDK
    changed its parsing and the mapper's shape assumptions need re-auditing."""
    exc = _gateway_403("model_access_denied")
    assert isinstance(exc.body, dict)
    assert "error" not in exc.body
    assert exc.body["code"] == "model_access_denied"


# ── 401 ───────────────────────────────────────────────────────────────

def test_401_maps_to_invalid_key_connection_error():
    exc = _sdk_error(401, json_body={"error": {"message": "bad key"}})
    with pytest.raises(ConnectionError) as err:
        _raise_for_status_error(exc, "sonnet")
    # cowork-server's provider_auth detection keys on this exact phrase.
    assert "Invalid API key" in str(err.value)
    assert not isinstance(err.value, ModelUnavailableError)


def test_401_html_body_maps_to_invalid_key():
    # nginx auth walls return HTML — the 401 branch must not need a body.
    exc = _sdk_error(401, text_body="<html>401 Authorization Required</html>")
    with pytest.raises(ConnectionError) as err:
        _raise_for_status_error(exc, "sonnet")
    assert "Invalid API key" in str(err.value)


# ── 429 (quota) ───────────────────────────────────────────────────────

def test_429_fastapi_detail_maps_to_token_limit():
    # The gateway's CURRENT dialect: FastAPI HTTPException → {"detail": ...},
    # no envelope, so the SDK's unwrap is a no-op.
    exc = _sdk_error(429, json_body={"detail": "Monthly limit exceeded for tokens: 5/5"})
    with pytest.raises(TokenLimitExceeded) as err:
        _raise_for_status_error(exc, "sonnet")
    assert "Monthly limit exceeded" in str(err.value)
    assert "console.mindshub.ai" in str(err.value)


def test_429_enveloped_detail_also_maps_to_token_limit():
    # The time bomb (ENG-747): if the gateway ever standardizes its 429 onto
    # the OpenAI envelope, the SDK will unwrap it — the card must survive
    # both dialects.
    exc = _sdk_error(429, json_body={"error": {"detail": "Monthly limit exceeded for tokens: 5/5"}})
    with pytest.raises(TokenLimitExceeded) as err:
        _raise_for_status_error(exc, "sonnet")
    assert "Monthly limit exceeded" in str(err.value)


def test_bare_429_stays_generic():
    exc = _sdk_error(429, json_body={})
    with pytest.raises(ConnectionError) as err:
        _raise_for_status_error(exc, "sonnet")
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


def test_unwrapped_403_shape_also_maps():
    # Defensive: a proxy or SDK version that does NOT unwrap would hand the
    # mapper a top-level-code body directly; both shapes must classify.
    exc = _sdk_error(403, json_body={"code": "model_access_denied", "message": "no"})
    with pytest.raises(ModelUnavailableError):
        _raise_for_status_error(exc, "sonnet")


# ── 403 WITHOUT the gateway codes: never the plan copy ────────────────

def test_byok_openai_403_falls_through_to_generic():
    # e.g. OpenAI region block — different error code.
    exc = _sdk_error(403, json_body={"error": {
        "code": "unsupported_country_region_territory",
        "message": "Country not supported.",
    }})
    with pytest.raises(ConnectionError) as err:
        _raise_for_status_error(exc, "gpt-4o")
    assert not isinstance(err.value, ModelUnavailableError)
    assert "temporarily unavailable" in str(err.value)


def test_html_403_falls_through_to_generic():
    # Cloudflare/WAF blocks have no JSON body at all.
    exc = _sdk_error(403, text_body="<html>blocked</html>")
    with pytest.raises(ConnectionError) as err:
        _raise_for_status_error(exc, "sonnet")
    assert not isinstance(err.value, ModelUnavailableError)


def test_403_with_no_code_falls_through_to_generic():
    exc = _sdk_error(403, json_body={"error": {"message": "forbidden"}})
    with pytest.raises(ConnectionError) as err:
        _raise_for_status_error(exc, "sonnet")
    assert not isinstance(err.value, ModelUnavailableError)


# ── precedence: quota beats model-403 logic ───────────────────────────

def test_429_with_detail_wins_even_if_error_code_present():
    # A quota failure must stay token_limit, never be misread as model-403.
    # (The SDK unwraps this body to its "error" member, which carries both
    # fields — precedence must hold on the unwrapped shape too.)
    exc = _sdk_error(429, json_body={
        "error": {"detail": "Monthly limit exceeded for tokens: 5/5",
                  "code": "model_disabled"},
    })
    with pytest.raises(TokenLimitExceeded):
        _raise_for_status_error(exc, "sonnet")


# ── other statuses stay generic ───────────────────────────────────────

def test_500_stays_generic():
    exc = _sdk_error(500, json_body={"error": {"message": "boom"}})
    with pytest.raises(ConnectionError) as err:
        _raise_for_status_error(exc, "sonnet")
    assert "Server returned 500" in str(err.value)
