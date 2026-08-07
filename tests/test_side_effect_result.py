"""Contract checks for the side-effecting-tool result envelope (ENG-696)."""

import json

from anton.core.tools.registry import ToolOutcome
from anton.core.tools.side_effect import (
    SideEffectResult,
    hash_content,
    now_iso,
)

REQUIRED_FIELDS = {
    "success",
    "resource_id",
    "external_url",
    "idempotency_key",
    "committed_at",
    "content_hash",
}


def test_success_outcome_carries_every_required_field():
    res = SideEffectResult(
        success=True,
        message="Published successfully!",
        resource_id="rep-1",
        external_url="https://4nton.ai/view/x",
        idempotency_key="rep-1",
        committed_at=now_iso(),
        content_hash="md5:abc",
    )
    outcome = res.to_outcome()
    assert isinstance(outcome, ToolOutcome)
    # The verdict mirrors success so the ENG-1276 error streak keys on it.
    assert outcome.ok is True
    payload = json.loads(outcome.content)
    assert REQUIRED_FIELDS <= payload.keys()
    assert payload["success"] is True
    assert payload["external_url"] == "https://4nton.ai/view/x"
    assert payload["committed_at"] is not None


def test_failed_marks_no_commit():
    outcome = SideEffectResult.failed("PUBLISH FAILED: nope", reason="ValueError")
    assert outcome.ok is False
    assert outcome.reason == "ValueError"
    payload = json.loads(outcome.content)
    assert payload["success"] is False
    # Nothing committed → committed_at must be null, not a spurious timestamp.
    assert payload["committed_at"] is None
    assert "PUBLISH FAILED: nope" in payload["message"]


def test_hash_content_is_deterministic_and_prefixed():
    a = hash_content("hello")
    b = hash_content(b"hello")
    assert a == b
    assert a.startswith("sha256:")
    assert hash_content("hello") != hash_content("world")


if __name__ == "__main__":
    test_success_outcome_carries_every_required_field()
    test_failed_marks_no_commit()
    test_hash_content_is_deterministic_and_prefixed()
    print("ok")
