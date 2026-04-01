from __future__ import annotations

import ssl
import urllib.request


def minds_request(
    url: str,
    api_key: str,
    *,
    method: str = "GET",
    payload: bytes | None = None,
    verify: bool = True,
    timeout: int = 30,
) -> bytes:
    """Shared HTTP helper for all Minds API calls.

    Sets headers that pass through Cloudflare bot detection.
    """
    req = urllib.request.Request(url, data=payload, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    # Browser-like headers to avoid Cloudflare bot detection
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
