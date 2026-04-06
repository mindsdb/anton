from __future__ import annotations

import enum
import json as _json
from dataclasses import dataclass

from anton.minds_client import minds_request


class TokenLimitStatus(enum.Enum):
    OK = "ok"
    WARNING = "warning"
    EXCEEDED = "exceeded"


@dataclass
class TokenLimitInfo:
    """Result of a token limit check, including the actual usage numbers."""

    status: TokenLimitStatus
    used: int = 0
    limit: int = 0
    period: str = ""
    lifetime_used: int = 0
    lifetime_limit: int = -1
    billing_cycle_used: int = 0
    billing_cycle_limit: int = -1


def check_minds_token_limits(
    base_url: str,
    api_key: str,
    *,
    verify: bool = True,
) -> TokenLimitInfo:
    """Fetch token limits from the Minds API and return usage status with actual numbers.

    Returns TokenLimitInfo with status EXCEEDED if any configured period (lifetime or
    monthly) has reached or exceeded 100% of its limit.

    Returns TokenLimitInfo with status WARNING if any configured period has reached or
    exceeded 80% but is still below 100%.

    Returns TokenLimitInfo with status OK if usage is below the warning threshold,
    limits are unlimited (-1 or 0), or if the endpoint is unreachable.

    Failures are treated as OK so that transient network issues never
    block the user from sending a message.

    lifetime_used and lifetime_limit are always populated when the API responds
    successfully — they can be used to display current usage regardless of threshold.
    """
    url = f"{base_url}/api/v1/limits/"
    try:
        raw = minds_request(url, api_key, verify=verify, timeout=5)
        data = _json.loads(raw.decode())
    except Exception:
        return TokenLimitInfo(status=TokenLimitStatus.OK)

    tokens = data.get("tokens", {})
    limits = tokens.get("limit", {})
    usage = tokens.get("usage", {})

    raw_lifetime_used = usage.get("lifetime", 0)
    if not isinstance(raw_lifetime_used, int):
        raw_lifetime_used = 0
    raw_lifetime_limit = limits.get("lifetime", -1)
    if not isinstance(raw_lifetime_limit, int):
        raw_lifetime_limit = -1

    raw_billing_cycle_used = usage.get("billing_cycle", 0)
    if not isinstance(raw_billing_cycle_used, int):
        raw_billing_cycle_used = 0
    raw_billing_cycle_limit = limits.get("monthly", -1)
    if not isinstance(raw_billing_cycle_limit, int):
        raw_billing_cycle_limit = -1

    def _make(status: TokenLimitStatus, used: int = 0, limit: int = 0, period: str = "") -> TokenLimitInfo:
        return TokenLimitInfo(
            status=status,
            used=used,
            limit=limit,
            period=period,
            lifetime_used=raw_lifetime_used,
            lifetime_limit=raw_lifetime_limit,
            billing_cycle_used=raw_billing_cycle_used,
            billing_cycle_limit=raw_billing_cycle_limit,
        )

    result = _make(TokenLimitStatus.OK)
    for period in ("lifetime", "monthly"):
        lim = limits.get(period, -1)
        if not isinstance(lim, int):
            lim = -1
        usage_key = "billing_cycle" if period == "monthly" else period
        used = usage.get(usage_key, 0)
        if not isinstance(used, int):
            used = 0

        if lim == -1 or lim <= 0:
            continue

        ratio = used / lim
        if ratio >= 1.0:
            return _make(TokenLimitStatus.EXCEEDED, used=used, limit=lim, period=period)
        if ratio >= 0.8:
            result = _make(TokenLimitStatus.WARNING, used=used, limit=lim, period=period)

    return result
