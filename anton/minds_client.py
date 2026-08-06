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
    """List model ids from the server's OpenAI-compatible /v1/models catalogue."""
    url = f"{base_url}/v1/models"
    raw = minds_request(url, api_key, verify=verify)
    data = _json.loads(raw.decode())
    entries = data.get("data") if isinstance(data, dict) else data
    ids: list[str] = []
    for entry in entries or []:
        if isinstance(entry, dict) and entry.get("id"):
            ids.append(str(entry["id"]))
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

    planning = pick((MINDS_DEFAULT_PLANNING_MODEL, "opus", "fable"), ids[0])
    coding = pick((MINDS_DEFAULT_CODING_MODEL,), planning)
    return (planning, coding)


@dataclass
class LLMTestResult:
    """Outcome of an LLM connectivity probe, with the provider's own error."""

    ok: bool
    rate_limited: bool = False
    error: str | None = None


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
    """Probe the server's chat-completions endpoint with a 1-token request.

    Probes with the model setup is about to configure (resolved from the live
    catalogue by the caller), so a passing probe validates the actual config.
    Uses the new /v1/ format (legacy /api/v1/ is still supported by the server).
    """
    payload = _json.dumps(build_chat_completion_kwargs(
        model=model,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=1,
    )).encode()

    url = f"{base_url}/v1/chat/completions"
    try:
        minds_request(url, api_key, method="POST", payload=payload, verify=verify)
        return LLMTestResult(ok=True)
    except urllib.error.HTTPError as e:
        if e.code == 429:
            return LLMTestResult(ok=False, rate_limited=True)
        return LLMTestResult(ok=False, error=_http_error_detail(e))
    except Exception as e:
        headline, _advice = describe_minds_connection_error(e)
        return LLMTestResult(ok=False, error=headline)
