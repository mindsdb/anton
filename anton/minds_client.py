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
from typing import TYPE_CHECKING

from anton.core.llm.openai import build_chat_completion_kwargs

if TYPE_CHECKING:
    from anton.config.settings import AntonSettings


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
    url = f"{base_url}/api/v1/minds/"  # trailing slash required
    raw = minds_request(url, api_key, verify=verify)
    data = _json.loads(raw.decode())
    if isinstance(data, list):
        return data
    return data.get("minds", data if isinstance(data, list) else [])


def get_mind(
    base_url: str, api_key: str, mind_name: str, verify: bool = True
) -> dict | None:
    url = f"{base_url}/api/v1/minds/{mind_name}"
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
    topic_path = cortex.project_hc._topics_dir / "minds-datasource.md"
    cortex.project_hc._topics_dir.mkdir(parents=True, exist_ok=True)
    cortex.project_hc._encode_with_lock(topic_path, topic_content, mode="write")


def list_datasources(
    base_url: str, api_key: str, verify: bool = True
) -> list[dict]:
    url = f"{base_url}/api/v1/datasources"
    raw = minds_request(url, api_key, verify=verify)
    data = _json.loads(raw.decode())
    if isinstance(data, list):
        return data
    return data.get("datasources", data if isinstance(data, list) else [])


def test_llm(base_url: str, api_key: str, verify: bool = True) -> bool:
    """Test if the Minds server supports LLM endpoints (_code_/_reason_ models)."""
    url = f"{base_url}/api/v1/chat/completions"
    payload = _json.dumps(build_chat_completion_kwargs(
        model="_code_",
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=1,
    )).encode()

    try:
        minds_request(url, api_key, method="POST", payload=payload, verify=verify)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 429:
            return "rate_limited"
        return False
    except Exception:
        return False
