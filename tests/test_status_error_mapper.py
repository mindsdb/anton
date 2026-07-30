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
from anton.core.llm.provider import (
    EndpointConfigurationError,
    ModelUnavailableError,
    TokenLimitExceeded,
    TransientProviderError,
    classify_transient,
)


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
    # If the gateway ever moves its 429 into the OpenAI envelope while keeping
    # the `detail` field, the SDK unwraps it to top level — still classified.
    # Classified before the transient path (429+detail → quota), so ENG-673's
    # bare-429-is-transient change leaves this untouched.
    exc = _sdk_error(429, json_body={"error": {"detail": "Monthly limit exceeded for tokens: 5/5"}})
    with pytest.raises(TokenLimitExceeded) as err:
        _raise_for_status_error(exc, "sonnet")
    assert "Monthly limit exceeded" in str(err.value)


def test_bare_429_is_transient_fail_fast():
    # ENG-673: a plain 429 (no quota detail) is a retryable rate-limit, but the
    # SDK already retried it at request time → fail fast (no session backoff).
    # Replaces the old test_bare_429_stays_generic — this path is now typed.
    exc = _sdk_error(429, json_body={})
    with pytest.raises(TransientProviderError) as err:
        _raise_for_status_error(exc, "sonnet")
    assert err.value.session_backoff is False
    assert "rate-limiting" in str(err.value).lower()


def _openai_quota_429():
    """OpenAI's own quota dialect, byte-shaped like the live body captured
    from a zero-quota project key on 2026-07-28."""
    return _sdk_error(429, json_body={"error": {
        "message": (
            "You exceeded your current quota, please check your plan and "
            "billing details. For more information on this error, read the "
            "docs: https://platform.openai.com/docs/guides/error-codes/api-errors."
        ),
        "type": "insufficient_quota",
        "param": None,
        "code": "insufficient_quota",
    }})


def test_429_insufficient_quota_maps_to_token_limit():
    # BYOK OpenAI quota exhaustion is permanent for the identical request —
    # it must fail fast as a billing error, never enter the retry loop and
    # surface a misleading "provider overloaded" after the backoff budget.
    with pytest.raises(TokenLimitExceeded) as err:
        _raise_for_status_error(_openai_quota_429(), "gpt-4o")
    assert "platform.openai.com" in str(err.value)
    # BYOK error: the remedy is the user's OpenAI billing, not a MindsHub plan.
    assert "console.mindshub.ai" not in str(err.value)


def test_429_insufficient_quota_never_transient():
    # Defense for direct classify_transient callers (mid-stream paths): the
    # quota 429 has no `detail`, so without the code-exact guard it would
    # classify as a retryable plain rate-limit.
    exc = _openai_quota_429()
    assert classify_transient(429, exc.body, provider="openai", model="gpt-4o") is None


def test_429_list_detail_stays_generic():
    # FastAPI validation errors put a LIST in detail — its Python repr must
    # never reach user-facing copy with an upgrade CTA attached.
    exc = _sdk_error(429, json_body={"detail": [{"loc": ["body", "x"], "msg": "field required"}]})
    with pytest.raises(ConnectionError) as err:
        _raise_for_status_error(exc, "sonnet")
    assert "field required" not in str(err.value)
    assert not isinstance(err.value, TokenLimitExceeded)


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


def _wire_shaped_error(status_code, body):
    """Real `openai.APIStatusError` constructed DIRECTLY with an explicit
    body — bypassing the SDK's parse-and-unwrap on purpose. This is the only
    way to hand the mapper an envelope-shaped ``exc.body``: the pinned SDK
    always peels ``error`` (see test_sdk_unwraps_error_envelope), but
    anton's pyproject allows ``openai>=1.0`` and proxies exist that re-wrap,
    so the mapper's envelope fallback must stay pinned by a test that the
    MockTransport route physically cannot produce."""
    request = httpx.Request("POST", "http://gateway.test/v1/chat/completions")
    response = httpx.Response(status_code, json=body if isinstance(body, dict) else None, request=request)
    return openai.APIStatusError("wire-shaped", response=response, body=body)


def test_envelope_shaped_403_maps_via_fallback():
    # A client that does NOT unwrap delivers the wire envelope verbatim —
    # the mapper's `envelope.get("code")` fallback is what classifies it.
    exc = _wire_shaped_error(403, {"error": {"code": "model_access_denied", "message": "no"}})
    with pytest.raises(ModelUnavailableError):
        _raise_for_status_error(exc, "sonnet")


def test_envelope_shaped_429_detail_maps_via_fallback():
    exc = _wire_shaped_error(429, {"error": {"detail": "Monthly limit exceeded for tokens: 5/5"}})
    with pytest.raises(TokenLimitExceeded):
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


# ── 5xx / infra failures are transient (ENG-673) ──────────────────────

def test_500_is_transient_fail_fast():
    # ENG-673: a request-time 5xx is transient (retryable), but SDK-retried
    # already → fail fast with the honest typed message, no session backoff.
    exc = _sdk_error(500, json_body={})
    with pytest.raises(TransientProviderError) as err:
        _raise_for_status_error(exc, "sonnet")
    assert err.value.session_backoff is False
    assert "returned 500" in str(err.value)


# ── 404 model-not-found (ENG-1145) ────────────────────────────────────


def test_404_object_body_maps_to_model_unavailable():
    # A 404 is "model isn't served here", NOT a transient outage — it must be a
    # ModelUnavailableError (permanent, no retry copy), surfacing the provider's
    # message so the user switches models rather than waiting.
    exc = _sdk_error(404, json_body={"error": {
        "message": "models/foo is not found for API version v1beta.",
    }})
    with pytest.raises(ModelUnavailableError) as err:
        _raise_for_status_error(exc, "foo")
    assert err.value.code == "model_not_found"
    assert "is not found" in str(err.value)
    assert "temporarily unavailable" not in str(err.value)
    assert not isinstance(err.value, TransientProviderError)


def test_404_gemini_array_body_unwrapped_and_surfaced():
    # Gemini's OpenAI-compat CHAT errors arrive as a single-element ARRAY, which
    # the SDK stores verbatim (exc.body is a list, not a dict). The mapper must
    # unwrap it, or the model-not-found reason is lost to the generic message —
    # the exact ENG-1145 symptom ("Server returned 404 ... temporarily
    # unavailable" instead of Google's real copy).
    exc = _sdk_error(404, json_body=[{"error": {
        "message": "This model models/gemini-2.5-flash is no longer available to new users.",
        "code": 404,
        "status": "NOT_FOUND",
    }}])
    assert isinstance(exc.body, list)  # pin the SDK behavior this fix relies on
    with pytest.raises(ModelUnavailableError) as err:
        _raise_for_status_error(exc, "gemini-2.5-flash")
    assert "no longer available to new users" in str(err.value)
    assert "Switch models in Settings" in str(err.value)


def test_404_openai_model_not_found_code_maps_to_model_unavailable():
    # OpenAI's unknown-model 404 carries code=model_not_found (SDK-unwrapped to
    # the top level) — model-specific, so "switch models" is the right remedy.
    exc = _sdk_error(404, json_body={"error": {
        "message": "The model `gpt-x` does not exist or you do not have access to it.",
        "type": "invalid_request_error",
        "code": "model_not_found",
    }})
    with pytest.raises(ModelUnavailableError) as err:
        _raise_for_status_error(exc, "gpt-x")
    assert err.value.code == "model_not_found"


def test_404_bad_endpoint_is_config_error_not_model_unavailable():
    # A misrouted/misconfigured endpoint (bad base URL, missing /v1, proxy path)
    # also 404s, with a body that says nothing about the model — that must NOT
    # become "switch models". FastAPI-style {"detail": "Not Found"}.
    exc = _sdk_error(404, json_body={"detail": "Not Found"})
    with pytest.raises(EndpointConfigurationError) as err:
        _raise_for_status_error(exc, "sonnet")
    # A distinct type (still a ConnectionError) so the CLI defaults it to `setup`,
    # not `retry` — retry re-sends the same misrouted request (ENG-1145 review).
    assert isinstance(err.value, ConnectionError)
    assert not isinstance(err.value, ModelUnavailableError)
    assert "endpoint" in str(err.value).lower()
    assert "Switch models" not in str(err.value)


def test_404_html_body_is_config_error():
    # An nginx/proxy 404 returns HTML (SDK stores it as a string body, not dict).
    exc = _sdk_error(404, text_body="<html>404 Not Found</html>")
    with pytest.raises(EndpointConfigurationError) as err:
        _raise_for_status_error(exc, "sonnet")
    assert not isinstance(err.value, ModelUnavailableError)
    assert "endpoint" in str(err.value).lower()


def test_404_model_message_without_terminator_is_normalized():
    # A model-oriented 404 whose provider message has NO trailing punctuation
    # must not run into the appended sentence ("...available Switch models").
    # The mapper normalizes the terminator rather than trusting the provider's
    # punctuation (ENG-1145 review).
    exc = _sdk_error(404, json_body={"error": {
        "message": "models/foo is no longer available",  # no period
        "status": "NOT_FOUND",
    }})
    with pytest.raises(ModelUnavailableError) as err:
        _raise_for_status_error(exc, "foo")
    assert "available. Switch models" in str(err.value)
    assert "available Switch" not in str(err.value)
