"""Minds REST API client helpers.

All Minds HTTP calls are kept here to make a future SDK migration easy.
Full migration is blocked on the SDK supporting custom request headers —
related to Cloudflare. Once the SDK exposes that,
this module can be replaced with a thin Client wrapper.
@TODO: check_minds_token_limits should be added to the SDK too
"""

from __future__ import annotations

import json as _json
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING

from anton.core.llm.openai import build_chat_completion_kwargs

if TYPE_CHECKING:
    from anton.config.settings import AntonSettings

# Tier-default models on MindsHub Cloud — bare catalogue ids, the same pair
# cowork-server's apply_model_defaults seeds. Used when /v1/models is not
# deployed on the target host (not every MindsHub host serves listing routes).
MINDS_DEFAULT_PLANNING_MODEL = "sonnet"
MINDS_DEFAULT_CODING_MODEL = "haiku"

# The free-bucket model present in every tier (auth's model_policy.json sets
# free_bucket only for it) — the last resort when the catalogue says the key
# cannot use the tier defaults, and cowork-server's probe model (ENG-576).
MINDS_FREE_TIER_MODEL = "mindshub_air"

# Dead mdb.ai smart-router aliases — never usable, whatever a server claims
# to serve (ENG-1140).
LEGACY_SMART_ROUTER_ALIASES = frozenset({"_reason_", "_code_"})


def minds_v1_base(base_url: str) -> str:
    """Host-aware OpenAI-compatible base URL.

    api.mindshub.ai serves the OpenAI-compatible API at /v1, the legacy
    mdb.ai host at /api/v1. Same rule as AntonSettings.model_post_init
    (ENG-436) and cowork-server's minds_chat_base_url; a shell twin lives in
    anton_services snapshots/cowork/setup.sh (mh_fetch_model_catalog) — keep
    them aligned.
    """
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base
    if "mdb.ai" in base:
        return f"{base}/api/v1"
    return f"{base}/v1"


def minds_request(
    url: str,
    api_key: str,
    *,
    method: str = "GET",
    payload: bytes | None = None,
    verify: bool = True,
    timeout: int = 30,
) -> bytes:
    """HTTP transport for all Minds API calls.

    Sets browser-like headers to pass through Cloudflare bot detection.
    This is why we use raw urllib instead of the minds-sdk (which uses
    plain requests with no such headers).
    """
    req = urllib.request.Request(url, data=payload, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header(
        "User-Agent",
        "Mozilla/5.0 (compatible; Anton/1.0;"
        " +https://github.com/mindsdb/anton)",
    )
    req.add_header("Accept-Language", "en-US,en;q=0.9")
    req.add_header("Accept-Encoding", "identity")
    req.add_header("Connection", "keep-alive")

    ctx = None
    if not verify:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        return resp.read()


def normalize_minds_url(url: str) -> str:
    """Add https:// if no scheme present, strip trailing slash."""
    url = url.strip()
    if url and not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    return url.rstrip("/")


def describe_minds_connection_error(err: Exception) -> tuple[str, str]:
    import socket
    import ssl

    if isinstance(err, urllib.error.HTTPError):
        reason = err.reason or "HTTP error"
        if err.code in (401, 403):
            return (
                f"Connection failed (HTTP {err.code}: {reason}). The server rejected the request.",
                "Common reasons: invalid or expired credentials, insufficient access, or the wrong server/endpoint.",
            )
        if 400 <= err.code < 500:
            return (
                f"Connection failed (HTTP {err.code}: {reason}). The server rejected the request.",
                "Common reasons: wrong URL, malformed request, or access restrictions on that endpoint.",
            )
        if err.code >= 500:
            return (
                f"Connection failed (HTTP {err.code}: {reason}). The server returned an error.",
                "Common reasons: server-side failure or a temporary outage.",
            )
        return (
            f"Connection failed (HTTP {err.code}: {reason}).",
            "Common reasons: a server response Anton could not use or a transient connectivity problem.",
        )

    if isinstance(err, urllib.error.URLError):
        reason = getattr(err, "reason", None)
        if isinstance(reason, ssl.SSLCertVerificationError):
            return (
                "Connection failed during TLS certificate verification.",
                "Common reasons: a self-signed, expired, or otherwise untrusted certificate.",
            )
        if (
            isinstance(reason, (TimeoutError, socket.timeout))
            or "timed out" in str(reason).lower()
        ):
            return (
                "Connection failed because the request timed out.",
                "Common reasons: the server is slow or unavailable, the URL is wrong, or there is a network path issue.",
            )
        return (
            f"Connection failed ({err}).",
            "Common reasons: network connectivity problems, DNS issues, or a server Anton could not reach.",
        )

    if "timed out" in str(err).lower():
        return (
            "Connection failed because the request timed out.",
            "Common reasons: the server is slow or unavailable, the URL is wrong, or there is a network path issue.",
        )

    return (
        f"Connection failed ({err}).",
        "Common reasons: network connectivity problems, authentication issues, or a server-side failure.",
    )


def list_minds(base_url: str, api_key: str, verify: bool = True) -> list[dict]:
    url = f"{base_url}/v1/minds"  # new format (legacy /api/v1 still works but /v1 is preferred)
    raw = minds_request(url, api_key, verify=verify)
    data = _json.loads(raw.decode())
    if isinstance(data, list):
        return data
    return data.get("minds", data if isinstance(data, list) else [])


def get_mind(
    base_url: str, api_key: str, mind_name: str, verify: bool = True
) -> dict | None:
    url = f"{base_url}/v1/minds/{mind_name}"
    try:
        raw = minds_request(url, api_key, verify=verify, timeout=15)
        return _json.loads(raw.decode())
    except Exception:
        return None


def refresh_knowledge(settings: AntonSettings, cortex) -> None:
    """Fetch the configured mind's parameters and update the memory topic file."""
    if not settings.minds_api_key or not settings.minds_mind_name or cortex is None:
        return

    mind = get_mind(
        normalize_minds_url(settings.minds_url),
        settings.minds_api_key,
        settings.minds_mind_name,
        verify=settings.minds_ssl_verify,
    )
    if not mind:
        return

    params = mind.get("parameters", {}) or {}
    parts = []
    if params.get("system_prompt"):
        parts.append(params["system_prompt"])
    if params.get("prompt_template"):
        parts.append(params["prompt_template"])

    if not parts:
        return

    knowledge = "\n\n".join(parts)
    topic_content = f"# Minds — {settings.minds_mind_name}\n\n{knowledge}\n"
    cortex.project_hc.encode_lesson(topic_content, topic='minds-datasource')


def list_datasources(
    base_url: str, api_key: str, verify: bool = True
) -> list[dict]:
    url = f"{base_url}/v1/datasources"
    raw = minds_request(url, api_key, verify=verify)
    data = _json.loads(raw.decode())
    if isinstance(data, list):
        return data
    return data.get("datasources", data if isinstance(data, list) else [])


def list_models(base_url: str, api_key: str, verify: bool = True) -> list[str]:
    """List the chat-model ids this key can actually use.

    Filters the catalogue on the fields that say whether a model is usable,
    not just listed: ``embedding`` models can't chat, and ``enabled`` is
    auth's wallet/allowance-aware access decision — a free-tier or
    wallet-empty key sees paid models with ``enabled: false`` (clients must
    not recompute access themselves; ENG-576). Entries without the flag are
    kept, so older hosts that don't send it still work. The dead smart-router
    aliases are dropped defensively.

    Short timeout: the catalogue is advisory (callers fall back to defaults),
    so it must not double the worst-case spinner hang of the probe itself.
    """
    url = f"{minds_v1_base(base_url)}/models"
    raw = minds_request(url, api_key, verify=verify, timeout=10)
    data = _json.loads(raw.decode())
    entries = data.get("data") if isinstance(data, dict) else data
    ids: list[str] = []
    for entry in entries or []:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        if entry.get("embedding") or entry.get("enabled") is False:
            continue
        model_id = str(entry["id"])
        if model_id in LEGACY_SMART_ROUTER_ALIASES:
            continue
        ids.append(model_id)
    return ids


def resolve_minds_models(
    base_url: str, api_key: str, verify: bool = True
) -> tuple[str, str]:
    """Pick the (planning, coding) model pair from the live catalogue.

    Falls back to the tier defaults when /v1/models is unreachable — listing
    routes are not deployed on every MindsHub host and can 404 even for valid
    keys — so setup never blocks on the catalogue being available.
    """
    try:
        ids = list_models(base_url, api_key, verify=verify)
    except Exception:
        ids = []
    if not ids:
        return (MINDS_DEFAULT_PLANNING_MODEL, MINDS_DEFAULT_CODING_MODEL)

    def pick(preferred: tuple[str, ...], fallback: str) -> str:
        for name in preferred:
            if name in ids:
                return name
        return fallback

    # The free-bucket model sits last in each chain: when the catalogue says
    # the key can't use the paid defaults (free tier / empty wallet), land on
    # the model it is entitled to instead of one that will 403 (ENG-576).
    planning = pick(
        (MINDS_DEFAULT_PLANNING_MODEL, "opus", "fable", MINDS_FREE_TIER_MODEL),
        ids[0],
    )
    coding = pick((MINDS_DEFAULT_CODING_MODEL, MINDS_FREE_TIER_MODEL), planning)
    return (planning, coding)


@dataclass
class LLMTestResult:
    """Outcome of an LLM connectivity probe, with the provider's own error.

    ``http_status`` is set when the server answered with an HTTP error — which
    proves the transport (TLS included) worked, so callers can skip
    no-SSL-verification retries that cannot change the outcome.
    """

    ok: bool
    rate_limited: bool = False
    error: str | None = None
    http_status: int | None = None


def _http_error_detail(e: urllib.error.HTTPError) -> str:
    """Extract the provider's error message from an HTTP error body.

    MindsHub returns OpenAI-shaped error envelopes ({"error": {"code":
    "model_not_found", "message": ...}}); surface that instead of a bare
    status so setup can tell the user what actually failed.
    """
    try:
        body = _json.loads(e.read().decode())
        err = body.get("error", body) if isinstance(body, dict) else None
        if isinstance(err, dict):
            message = err.get("message")
            code = err.get("code")
            if message:
                return f"{code}: {message}" if code else str(message)
        elif isinstance(err, str) and err:
            return err
    except Exception:
        pass
    return f"HTTP {e.code}: {e.reason or 'error'}"


def test_llm(
    base_url: str,
    api_key: str,
    verify: bool = True,
    model: str = MINDS_DEFAULT_CODING_MODEL,
) -> LLMTestResult:
    """Probe the server's chat-completions endpoint with a tiny request.

    Probes with the model setup is about to configure (resolved from the live
    catalogue by the caller), so a passing probe validates the actual config.

    max_tokens must be >= 16: the OpenAI-backed catalogue entries (gpt-*,
    mindshub_air) reject smaller values with integer_below_min_value, so a
    1-token probe was a deterministic false negative on those models. 20
    matches cowork-server's validate_minds.
    """
    payload = _json.dumps(build_chat_completion_kwargs(
        model=model,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=20,
    )).encode()

    url = f"{minds_v1_base(base_url)}/chat/completions"
    try:
        minds_request(url, api_key, method="POST", payload=payload, verify=verify)
        return LLMTestResult(ok=True)
    except urllib.error.HTTPError as e:
        if e.code == 429:
            return LLMTestResult(
                ok=False,
                rate_limited=True,
                error=_http_error_detail(e),
                http_status=429,
            )
        return LLMTestResult(ok=False, error=_http_error_detail(e), http_status=e.code)
    except Exception as e:
        headline, _advice = describe_minds_connection_error(e)
        return LLMTestResult(ok=False, error=headline)
