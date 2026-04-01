from __future__ import annotations

import json
from unittest.mock import patch

from anton.checks import TokenLimitStatus, check_minds_token_limits


def _make_raw(lifetime_lim=-1, lifetime_used=0, monthly_lim=-1, monthly_used=0) -> bytes:
    data = {
        "tokens": {
            "limit": {"lifetime": lifetime_lim, "monthly": monthly_lim},
            "usage": {"lifetime": lifetime_used, "billing_cycle": monthly_used},
        }
    }
    return json.dumps(data).encode()


def _patch_fetch(raw: bytes):
    return patch("anton.checks.minds_request", return_value=raw)


class TestCheckMindsTokenLimits:
    def test_ok_when_below_threshold(self):
        with _patch_fetch(_make_raw(lifetime_lim=1000, lifetime_used=100)):
            result = check_minds_token_limits("http://x", "key")
        assert result.status is TokenLimitStatus.OK
        assert result.lifetime_used == 100
        assert result.lifetime_limit == 1000
        assert result.billing_cycle_used == 0
        assert result.billing_cycle_limit == -1  # no monthly limit set

    def test_warning_at_80_percent(self):
        with _patch_fetch(_make_raw(lifetime_lim=1000, lifetime_used=850)):
            result = check_minds_token_limits("http://x", "key")
        assert result.status is TokenLimitStatus.WARNING
        assert result.used == 850
        assert result.limit == 1000
        assert result.period == "lifetime"
        assert result.lifetime_used == 850
        assert result.lifetime_limit == 1000

    def test_exceeded_at_100_percent(self):
        with _patch_fetch(_make_raw(lifetime_lim=1000, lifetime_used=1000)):
            result = check_minds_token_limits("http://x", "key")
        assert result.status is TokenLimitStatus.EXCEEDED
        assert result.used == 1000
        assert result.limit == 1000
        assert result.period == "lifetime"
        assert result.lifetime_used == 1000
        assert result.lifetime_limit == 1000

    def test_exceeded_over_100_percent(self):
        with _patch_fetch(_make_raw(lifetime_lim=1000, lifetime_used=1100)):
            result = check_minds_token_limits("http://x", "key")
        assert result.status is TokenLimitStatus.EXCEEDED
        assert result.used == 1100
        assert result.limit == 1000

    def test_monthly_exceeded_overrides_lifetime_ok(self):
        with _patch_fetch(
            _make_raw(
                lifetime_lim=10000, lifetime_used=100,
                monthly_lim=500, monthly_used=500,
            )
        ):
            result = check_minds_token_limits("http://x", "key")
        assert result.status is TokenLimitStatus.EXCEEDED
        assert result.used == 500
        assert result.limit == 500
        assert result.period == "monthly"
        # Raw lifetime counters still carried through
        assert result.lifetime_used == 100
        assert result.lifetime_limit == 10000

    def test_monthly_warning_lifetime_ok(self):
        with _patch_fetch(
            _make_raw(
                lifetime_lim=10000, lifetime_used=100,
                monthly_lim=500, monthly_used=450,
            )
        ):
            result = check_minds_token_limits("http://x", "key")
        assert result.status is TokenLimitStatus.WARNING
        assert result.used == 450
        assert result.limit == 500
        assert result.period == "monthly"
        assert result.lifetime_used == 100
        assert result.lifetime_limit == 10000

    def test_unlimited_lifetime_returns_ok(self):
        with _patch_fetch(_make_raw(lifetime_lim=-1, lifetime_used=9999)):
            result = check_minds_token_limits("http://x", "key")
        assert result.status is TokenLimitStatus.OK
        assert result.used == 0
        assert result.limit == 0
        assert result.lifetime_used == 9999
        assert result.lifetime_limit == -1  # unlimited

    def test_zero_limit_treated_as_unlimited(self):
        with _patch_fetch(_make_raw(lifetime_lim=0, lifetime_used=9999)):
            result = check_minds_token_limits("http://x", "key")
        assert result.status is TokenLimitStatus.OK

    def test_network_error_returns_ok(self):
        with patch(
            "anton.checks.minds_request", side_effect=OSError("timeout")
        ):
            result = check_minds_token_limits("http://x", "key")
        assert result.status is TokenLimitStatus.OK
        assert result.used == 0
        assert result.limit == 0
        assert result.lifetime_used == 0
        assert result.lifetime_limit == -1

    def test_malformed_json_returns_ok(self):
        with patch("anton.checks.minds_request", return_value=b"not json"):
            result = check_minds_token_limits("http://x", "key")
        assert result.status is TokenLimitStatus.OK

    def test_empty_tokens_returns_ok(self):
        with patch("anton.checks.minds_request", return_value=b"{}"):
            result = check_minds_token_limits("http://x", "key")
        assert result.status is TokenLimitStatus.OK
