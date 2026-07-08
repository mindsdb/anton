"""`_safe_error_message` framing policy for the local dispatch loop.

A spent token allowance is a quota condition, not a crash, so it must
surface anton's already-friendly message verbatim — without the
`[agent error]` prefix that reads like something broke. Every other
failure keeps the prefix (and the API-key redaction it already applied).
"""

from __future__ import annotations

from anton.core.dispatch.local_runtime import LocalScratchpadOrchestrator
from anton.core.llm.provider import TokenLimitExceeded


_TOKEN_LIMIT_MESSAGE = (
    "Server returned 429 — Monthly limit exceeded for tokens: 5000000/5000000 "
    "Visit https://console.mindshub.ai to upgrade or to top up your tokens."
)


def test_token_limit_message_has_no_agent_error_prefix():
    rendered = LocalScratchpadOrchestrator._safe_error_message(
        TokenLimitExceeded(_TOKEN_LIMIT_MESSAGE)
    )
    assert rendered == _TOKEN_LIMIT_MESSAGE
    assert "[agent error]" not in rendered


def test_generic_error_keeps_agent_error_prefix():
    rendered = LocalScratchpadOrchestrator._safe_error_message(ValueError("boom"))
    assert rendered.startswith("[agent error]")
    assert "boom" in rendered
