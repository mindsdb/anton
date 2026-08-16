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

import anthropic
import httpx
import openai
import pytest

from anton.core.llm.anthropic import _raise_for_status_error as _raise_anthropic
from anton.core.llm.openai import _raise_for_status_error
from anton.core.llm.provider import (
    EndpointConfigurationError,
    ModelUnavailableError,
    TokenLimitExceeded,
    TransientProviderError,
    classify_transient,
    wallet_denial_code,
)
from anton.core.session import _is_provider_auth_error


def _sdk_error(status_code, json_body=None, text_body=None, headers=None):
    """Real `openai.APIStatusError`, built by the pinned SDK from a raw HTTP
    response — exactly what production call sites catch and hand to the
    mapper. This is the load-bearing difference from the original suite."""

    def handler(request: httpx.Request) -> httpx.Response:
        if text_body is not None:
            return httpx.Response(status_code, text=text_body, headers=headers)
        return httpx.Response(
            status_code, json=json_body if json_body is not None else {}, headers=headers
        )

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
    # session.py's own re-raise checks (ENG-1310) key on this predicate, not
    # the raw text — pin against the REAL mapper output so an edit to this
    # copy that drops "invalid api key" fails here too, not just silently in
    # production (review feedback on ENG-1310).
    assert _is_provider_auth_error(err.value)


def test_401_html_body_maps_to_invalid_key():
    # nginx auth walls return HTML — the 401 branch must not need a body.
    exc = _sdk_error(401, text_body="<html>401 Authorization Required</html>")
    with pytest.raises(ConnectionError) as err:
        _raise_for_status_error(exc, "sonnet")
    assert "Invalid API key" in str(err.value)
    assert _is_provider_auth_error(err.value)


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


def test_bare_429_is_transient_and_backs_off_in_session():
    # ENG-673 typed this path; ENG-1537 made it wait. A plain 429 is a velocity
    # limit — the one request-time status where the session must spend its
    # budget, because the SDK's retries fire seconds apart and the ceiling is
    # per-minute. Was session_backoff=False, which sent it down the count-based
    # path and re-issued the request twice with no delay.
    exc = _sdk_error(429, json_body={"code": "rate_limited"})
    with pytest.raises(TransientProviderError) as err:
        _raise_for_status_error(exc, "sonnet")
    assert err.value.session_backoff is True
    assert err.value.code == "rate_limited"
    assert "rate-limiting" in str(err.value).lower()


def test_bare_429_reads_retry_after_off_the_response():
    # ENG-1537: nothing in anton read this header before. The gateway sends it
    # on every velocity 429 as integer seconds; without it the session falls
    # back to a guessed curve instead of the interval the server named.
    exc = _sdk_error(429, json_body={}, headers={
        "Retry-After": "42", "X-MindsHub-Reason": "rate_limited",
    })
    with pytest.raises(TransientProviderError) as err:
        _raise_for_status_error(exc, "sonnet")
    assert err.value.retry_after == 42.0


def test_retry_after_http_date_form_is_ignored_not_misparsed():
    # The date form is legal but nothing in use emits it, and reading it as a
    # number would produce an absurd delay. Absent hint → jittered curve.
    exc = _sdk_error(429, json_body={}, headers={
        "Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT",
        "X-MindsHub-Reason": "rate_limited",
    })
    with pytest.raises(TransientProviderError) as err:
        _raise_for_status_error(exc, "sonnet")
    assert err.value.retry_after is None


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


# ── the M3 wallet taxonomy (ENG-1169) ─────────────────────────────────

def _gateway_402(code="wallet_empty", headers=None):
    """The M3 gateway's out-of-credits 402, byte-shaped like
    `minds/inference/errors.py:wallet_empty` (OpenAI lanes)."""
    return _sdk_error(402, json_body={"error": {
        "message": "Your wallet has no balance to cover the model 'sonnet'.",
        "type": "invalid_request_error",
        "param": None,
        "code": code,
    }}, headers=headers)


def test_gateway_402_wallet_empty_maps_to_token_limit():
    # The live shape: body code AND the X-MindsHub-Reason header. Before
    # ENG-1169 this fell to the generic "temporarily unavailable" copy, got
    # auto-retried, and the out-of-credits card never rendered.
    exc = _gateway_402(headers={
        "X-MindsHub-Reason": "wallet_empty",
        "X-MindsHub-Recovery-Url": "/billing",
    })
    with pytest.raises(TokenLimitExceeded) as err:
        _raise_for_status_error(exc, "sonnet")
    msg = str(err.value)
    assert "402" in msg and "credit" in msg.lower()
    assert "temporarily unavailable" not in msg
    assert "billing" in msg


def test_gateway_402_body_code_alone_maps_to_token_limit():
    # No headers (a proxy stripped them) — the body code is enough.
    exc = _gateway_402()
    with pytest.raises(TokenLimitExceeded):
        _raise_for_status_error(exc, "sonnet")


def test_gateway_402_header_alone_maps_to_token_limit():
    # Code-less body (the anthropic-dialect lane strips it) but the
    # X-MindsHub-Reason header survives — header is the fallback discriminator.
    exc = _sdk_error(
        402,
        json_body={"error": {"message": "Your wallet has no balance.",
                             "type": "invalid_request_error"}},
        headers={"X-MindsHub-Reason": "wallet_empty"},
    )
    with pytest.raises(TokenLimitExceeded):
        _raise_for_status_error(exc, "sonnet")


def test_byok_402_stays_generic():
    # A non-gateway 402 (e.g. OpenRouter's insufficient-credits dialect)
    # carries no wallet code — it must NOT get the MindsHub credits card/CTA;
    # the remedy is the user's own provider billing. Mirrors cowork-server's
    # test_byok_402_stays_generic.
    exc = _sdk_error(402, json_body={"error": {
        "message": "Insufficient credits. Add more at openrouter.ai.",
        "code": 402,
    }})
    with pytest.raises(ConnectionError) as err:
        _raise_for_status_error(exc, "sonnet")
    assert not isinstance(err.value, TokenLimitExceeded)
    assert "402" in str(err.value)


def test_gateway_429_allowance_exhausted_maps_to_token_limit():
    # The M3 allowance 429 carries a structured code but NO FastAPI `detail`,
    # so the legacy 429→TokenLimitExceeded branch misses it; before ENG-1169
    # it was misclassified as a retryable rate limit and surfaced as
    # "provider overloaded" after burning the backoff budget.
    exc = _sdk_error(429, json_body={"error": {
        "message": "Your included token allowance for 'sonnet' is exhausted.",
        "type": "rate_limit_error",
        "param": None,
        "code": "included_allowance_exhausted",
    }}, headers={"X-MindsHub-Reason": "included_allowance_exhausted"})
    with pytest.raises(TokenLimitExceeded) as err:
        _raise_for_status_error(exc, "sonnet")
    assert "allowance" in str(err.value)


def test_gateway_velocity_429_stays_transient():
    # The gate's velocity 429 (`rate_limited`, ENG-878 TPM/RPM) means "slow
    # down and retry" — it must stay transient, never the credits card.
    exc = _sdk_error(429, json_body={"error": {
        "message": "Rate limit exceeded for model 'sonnet'. Please slow down and retry.",
        "type": "rate_limit_error",
        "param": None,
        "code": "rate_limited",
    }}, headers={"X-MindsHub-Reason": "rate_limited", "Retry-After": "15"})
    with pytest.raises(TransientProviderError) as err:
        _raise_for_status_error(exc, "sonnet")
    assert err.value.code == "rate_limited"


def test_wallet_denial_code_reads_both_dialects():
    # The mid-stream guard reads the bare-APIError body directly: SDK-unwrapped
    # (top-level code) AND wire-envelope (nested) shapes must both resolve.
    assert wallet_denial_code({"code": "wallet_empty"}) == "wallet_empty"
    assert (
        wallet_denial_code({"error": {"code": "included_allowance_exhausted"}})
        == "included_allowance_exhausted"
    )
    assert wallet_denial_code({"error": {"code": "rate_limited"}}) is None
    assert wallet_denial_code({"code": 402}) is None
    assert wallet_denial_code(None) is None
    assert wallet_denial_code("<html>402</html>") is None


# ── the anthropic twin (ENG-1169) ─────────────────────────────────────

def test_anthropic_401_maps_to_invalid_key_connection_error():
    # No real-SDK 401 coverage existed for the anthropic mapper before this
    # (review feedback on ENG-1310) — only openai's 401 was pinned against
    # actual SDK output; anthropic's own "Invalid API key — …" copy was
    # untested except via a hand-built ConnectionError.
    exc = _anthropic_sdk_error(401, json_body={"type": "error", "error": {
        "type": "authentication_error",
        "message": "invalid x-api-key",
    }})
    with pytest.raises(ConnectionError) as err:
        _raise_anthropic(exc, model="claude-sonnet")
    assert "Invalid API key" in str(err.value)
    assert _is_provider_auth_error(err.value)

def _anthropic_sdk_error(status_code, json_body=None, headers=None):
    """Real `anthropic.APIStatusError` from the pinned SDK — the anthropic
    twin of `_sdk_error`. The anthropic SDK does NOT unwrap the error
    envelope (unlike openai's), so `exc.body` is the wire shape."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code, json=json_body if json_body is not None else {}, headers=headers
        )

    client = anthropic.Anthropic(
        base_url="http://gateway.test",
        api_key="test-key",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        client.messages.create(
            model="claude-sonnet", max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
    except anthropic.APIStatusError as exc:
        return exc
    raise AssertionError(f"anthropic SDK did not raise for HTTP {status_code}")


def test_anthropic_402_wallet_code_maps_to_token_limit():
    # Wire-envelope code (the anthropic SDK stores the envelope unmodified).
    exc = _anthropic_sdk_error(402, json_body={"type": "error", "error": {
        "type": "invalid_request_error",
        "message": "Your wallet has no balance to cover the model 'claude'.",
        "code": "wallet_empty",
    }})
    with pytest.raises(TokenLimitExceeded):
        _raise_anthropic(exc, model="claude-sonnet")


def test_anthropic_402_header_alone_maps_to_token_limit():
    # Today's live gateway anthropic lane strips the body code — if the
    # header survives (proxy/fixed gateway), it must still map.
    exc = _anthropic_sdk_error(402, json_body={"type": "error", "error": {
        "type": "invalid_request_error",
        "message": "Your wallet has no balance to cover the model 'claude'.",
    }}, headers={"X-MindsHub-Reason": "wallet_empty"})
    with pytest.raises(TokenLimitExceeded):
        _raise_anthropic(exc, model="claude-sonnet")


def test_anthropic_byok_402_stays_generic():
    # No wallet code, no reason header → generic, never the credits card.
    exc = _anthropic_sdk_error(402, json_body={"type": "error", "error": {
        "type": "invalid_request_error",
        "message": "Your credit balance is too low to access the Anthropic API.",
    }})
    with pytest.raises(ConnectionError) as err:
        _raise_anthropic(exc, model="claude-sonnet")
    assert not isinstance(err.value, TokenLimitExceeded)


def test_anthropic_404_not_found_error_maps_to_model_unavailable():
    # ENG-1139: a bare Anthropic 404 for an unknown model must become a
    # ModelUnavailableError with the provider's own reason folded in once,
    # not the generic "temporarily unavailable, try again" ConnectionError
    # (which then got double-wrapped by session.py's fallback prose).
    exc = _anthropic_sdk_error(404, json_body={"type": "error", "error": {
        "type": "not_found_error",
        "message": "model: nonexistent-model-xyz",
    }})
    with pytest.raises(ModelUnavailableError) as err:
        _raise_anthropic(exc, model="nonexistent-model-xyz")
    assert err.value.code == "model_not_found"
    assert str(err.value).count("nonexistent-model-xyz") == 2  # template + provider detail, not 3+
    assert "temporarily unavailable" not in str(err.value)
    assert "Switch models in Settings" in str(err.value)


def test_429_wallet_code_never_transient():
    # Defense for direct classify_transient callers (the mid-stream paths):
    # the M3 allowance 429 has no `detail`, so without the code-exact guard
    # it would classify as a retryable plain rate-limit — same precedent as
    # the insufficient_quota guard above (ENG-1169 self-review).
    body = {"message": "allowance exhausted", "type": "rate_limit_error",
            "code": "included_allowance_exhausted"}
    assert classify_transient(429, body, provider="gw", model="sonnet") is None


def test_unhashable_code_never_crashes_the_classifier():
    # A hostile/buggy OpenAI-compatible endpoint sending a NON-STRING `code`
    # (e.g. a list) must fall through to the generic mapping — frozenset
    # membership hashes the value, so without the isinstance guard this
    # raised TypeError from inside the error classifier (ENG-1169 review).
    assert wallet_denial_code({"code": ["wallet_empty"]}) is None
    assert wallet_denial_code({"error": {"code": {"c": "wallet_empty"}}}) is None
    exc = _sdk_error(402, json_body={"error": {"message": "denied", "code": ["wallet_empty"]}})
    with pytest.raises(ConnectionError) as err:
        _raise_for_status_error(exc, "sonnet")
    assert not isinstance(err.value, TokenLimitExceeded)
    assert classify_transient(429, {"code": ["wallet_empty"]}) is not None  # plain-429 path intact


def test_anthropic_mapper_also_reads_retry_after_and_the_velocity_signal():
    # ENG-1537 review finding 6: the anthropic twin's `retry_after=` passthrough
    # had no coverage — deleting it left all 87 mapper tests green, while the
    # same deletion in openai.py correctly reddened. The two mappers are meant
    # to be twins, so the untested one is where they drift.
    from anton.core.llm.anthropic import _raise_for_status_error as anthropic_mapper

    exc = _anthropic_sdk_error(429, json_body={"error": {"code": "rate_limited"}},
                               headers={"Retry-After": "42"})
    with pytest.raises(TransientProviderError) as err:
        anthropic_mapper(exc, provider="Anthropic", model="sonnet")
    assert err.value.retry_after == 42.0
    assert err.value.session_backoff is True
    assert err.value.code == "rate_limited"


def test_anthropic_mapper_does_not_wait_on_an_unconfirmed_429():
    # The twin must apply the same positive-signal rule, or the Gemini-dialect
    # quota hole exists on one door and not the other.
    from anton.core.llm.anthropic import _raise_for_status_error as anthropic_mapper

    exc = _anthropic_sdk_error(429, json_body={"error": {"code": 429}},
                               headers={"Retry-After": "42"})
    with pytest.raises(TransientProviderError) as err:
        anthropic_mapper(exc, provider="Anthropic", model="sonnet")
    assert err.value.session_backoff is False
    assert err.value.retry_after is None
