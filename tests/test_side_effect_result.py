"""Contract checks for the side-effecting-tool result envelope (ENG-696)."""

import asyncio
import json

from anton.core.tools.registry import ToolOutcome
from anton.core.tools.side_effect import SideEffectResult, now_iso
from anton.core.tools.tool_handlers import handle_create_artifact

REQUIRED_FIELDS = {
    "success",
    "resource_id",
    "external_url",
    "idempotency_key",
    "committed_at",
    "content_hash",
    "details",
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
        details={"port": 8080},
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
    # Tool-specific fields ride the machine-readable `details` channel.
    assert payload["details"] == {"port": 8080}


def test_failed_marks_no_commit():
    outcome = SideEffectResult.failed("PUBLISH FAILED: nope", reason="ValueError")
    assert outcome.ok is False
    assert outcome.reason == "ValueError"
    payload = json.loads(outcome.content)
    assert payload["success"] is False
    # Nothing committed → committed_at must be null, not a spurious timestamp.
    assert payload["committed_at"] is None
    assert "PUBLISH FAILED: nope" in payload["message"]


def test_environmental_failure_is_explicit_ok_false():
    # No workspace bound → store unavailable. This is now an explicit failure
    # verdict (ok=False), so it counts toward the ENG-1276 error streak and the
    # circuit breaker eventually tells the model to stop hammering a store that
    # will never appear. Pre-envelope this returned a plain string with none of
    # the legacy marker phrases → classified ok=None → never counted. This is a
    # deliberate behavior change, consistent with #308 (ENG-350's rejection was
    # likewise migrated to ok=False).
    outcome = asyncio.run(
        handle_create_artifact(object(), {"name": "x", "description": "y", "type": "html-app"})
    )
    assert isinstance(outcome, ToolOutcome)
    assert outcome.ok is False


def test_validation_failure_is_explicit_ok_false(tmp_path):
    class _WS:
        def __init__(self, d):
            self.artifacts_dir = d

    class _Sess:
        def __init__(self, d):
            self._workspace = _WS(d)

    # Missing `name` is a validation rejection — also an explicit failure now.
    outcome = asyncio.run(
        handle_create_artifact(_Sess(tmp_path), {"description": "y", "type": "html-app"})
    )
    assert outcome.ok is False


if __name__ == "__main__":
    test_success_outcome_carries_every_required_field()
    test_failed_marks_no_commit()
    print("ok")
