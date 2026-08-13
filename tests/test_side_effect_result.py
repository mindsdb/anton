"""Contract checks for the side-effecting-tool result envelope (ENG-696)."""

import asyncio
import json

from anton.core.memory.acc import _normalise_error_signature
from anton.core.tools.registry import ToolOutcome
from anton.core.tools.side_effect import SideEffectResult, now_iso
from anton.core.tools.tool_handlers import (
    handle_create_artifact,
    handle_launch_backend,
    handle_update_artifact_metadata,
)

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


class _WS:
    def __init__(self, d):
        self.artifacts_dir = d


class _Sess:
    def __init__(self, d):
        self._workspace = _WS(d)
        self._data_vault = None


def test_validation_failure_is_explicit_ok_false(tmp_path):
    # Missing `name` is a validation rejection — also an explicit failure now.
    outcome = asyncio.run(
        handle_create_artifact(_Sess(tmp_path), {"description": "y", "type": "html-app"})
    )
    assert outcome.ok is False


def test_create_artifact_success_serializes(tmp_path):
    # Success path must serialize cleanly — `details.path` is a Path, and
    # to_outcome() JSON-encodes the payload; an unserialized Path would crash
    # the tool and dispatch would report a spurious failure.
    outcome = asyncio.run(
        handle_create_artifact(
            _Sess(tmp_path), {"name": "My Art", "description": "y", "type": "html-app"}
        )
    )
    assert outcome.ok is True
    payload = json.loads(outcome.content)
    assert isinstance(payload["details"]["path"], str)
    assert payload["resource_id"] == payload["details"]["slug"]


def test_distinct_envelope_failures_stay_distinct():
    # The ACC dedupes failures by a normalised signature, and the normaliser
    # collapses every quoted run — raw envelope JSON is almost entirely quoted
    # runs, so three unrelated failures would hash to ONE signature and fire a
    # false "the same error repeats, retrying won't help" lesson.
    messages = [
        "Error: `name` is required.",
        "Error: no artifact found for slug `foo`.",
        "PUBLISH FAILED: settings module unavailable",
    ]
    sigs = {
        _normalise_error_signature(SideEffectResult.failed(m).content[:300])
        for m in messages
    }
    assert len(sigs) == 3


def test_error_signature_still_collapses_variable_parts():
    # Unwrapping the envelope must not cost the normaliser its actual job:
    # two failures differing only in a quoted token still share a signature.
    a = SideEffectResult.failed("Refusing to save record for engine='gmail-1'").content
    b = SideEffectResult.failed("Refusing to save record for engine='gmail-2'").content
    assert _normalise_error_signature(a) == _normalise_error_signature(b)
    # Plain prose is unaffected by the unwrap.
    assert _normalise_error_signature("Error: plain") == "Error: plain"


def test_envelope_unwrap_survives_leading_whitespace():
    # Leading whitespace must not defeat the unwrap (the `{`-prefix guard),
    # else two distinct messages collapse to one signature again.
    a = "  " + SideEffectResult.failed("Error: `name` is required.").content
    b = "  " + SideEffectResult.failed("PUBLISH FAILED: unsupported .docx").content
    assert _normalise_error_signature(a) != _normalise_error_signature(b)


def test_envelope_unwrap_ignores_empty_message():
    import json as _json

    from anton.core.memory.acc import _unwrap_envelope_message

    # An empty message carries no signal to sign — do NOT return "" (which would
    # sign every empty-message envelope alike); fall through to the raw text.
    envelope = _json.dumps({"success": False, "message": "", "reason": "a"})
    assert _unwrap_envelope_message(envelope) == envelope


# --- update_artifact -------------------------------------------------------


def test_update_artifact_not_found_is_failure(tmp_path):
    outcome = asyncio.run(
        handle_update_artifact_metadata(_Sess(tmp_path), {"slug": "nope", "primary": "a.html"})
    )
    assert outcome.ok is False
    assert outcome.reason == "artifact_not_found"
    assert json.loads(outcome.content)["committed_at"] is None


def test_update_artifact_invalid_port_is_failure(tmp_path):
    outcome = asyncio.run(
        handle_update_artifact_metadata(_Sess(tmp_path), {"slug": "x", "port": "not-a-number"})
    )
    assert outcome.ok is False
    assert outcome.reason == "invalid_port"


def test_update_artifact_success_carries_identity(tmp_path):
    sess = _Sess(tmp_path)
    created = json.loads(
        asyncio.run(
            handle_create_artifact(
                sess, {"name": "My Art", "description": "y", "type": "html-app"}
            )
        ).content
    )
    slug = created["resource_id"]

    outcome = asyncio.run(
        handle_update_artifact_metadata(sess, {"slug": slug, "port": 8080})
    )
    assert outcome.ok is True
    payload = json.loads(outcome.content)
    assert payload["resource_id"] == slug
    assert payload["idempotency_key"] == slug
    assert payload["committed_at"]
    assert payload["details"]["port"] == 8080


# --- launch_backend --------------------------------------------------------


def test_launch_backend_not_found_is_failure(tmp_path):
    outcome = asyncio.run(handle_launch_backend(_Sess(tmp_path), {"slug": "nope"}))
    assert outcome.ok is False
    assert outcome.reason == "artifact_not_found"
    assert json.loads(outcome.content)["committed_at"] is None


def test_launch_backend_invalid_health_timeout_is_failure(tmp_path):
    # The artifact must exist — the not-found guard runs before this one.
    sess = _Sess(tmp_path)
    created = json.loads(
        asyncio.run(
            handle_create_artifact(
                sess,
                {"name": "App3", "description": "y", "type": "fullstack-stateless-app"},
            )
        ).content
    )
    outcome = asyncio.run(
        handle_launch_backend(
            sess, {"slug": created["resource_id"], "health_timeout": "soon"}
        )
    )
    assert outcome.ok is False
    assert outcome.reason == "invalid_health_timeout"


def test_launch_backend_success_carries_url_and_details(tmp_path, monkeypatch):
    sess = _Sess(tmp_path)
    created = json.loads(
        asyncio.run(
            handle_create_artifact(
                sess,
                {"name": "App", "description": "y", "type": "fullstack-stateless-app"},
            )
        ).content
    )
    slug = created["resource_id"]
    sess._scratchpads = None

    async def _fake_launch(**kwargs):
        return {
            "slug": slug,
            "port": 8123,
            "pid": 4242,
            "url": "http://127.0.0.1:8123",
            "log_path": "/tmp/backend.log",
            "proc": object(),
        }

    monkeypatch.setattr(
        "anton.core.artifacts.backend_launcher.launch_artifact_backend", _fake_launch
    )
    outcome = asyncio.run(handle_launch_backend(sess, {"slug": slug}))
    assert outcome.ok is True
    payload = json.loads(outcome.content)
    assert payload["external_url"] == "http://127.0.0.1:8123"
    assert payload["resource_id"] == slug
    assert payload["details"]["port"] == 8123
    assert payload["details"]["pid"] == 4242
    assert payload["details"]["log_path"] == "/tmp/backend.log"


def test_launch_backend_launcher_error_is_failure(tmp_path, monkeypatch):
    sess = _Sess(tmp_path)
    created = json.loads(
        asyncio.run(
            handle_create_artifact(
                sess,
                {"name": "App2", "description": "y", "type": "fullstack-stateless-app"},
            )
        ).content
    )
    slug = created["resource_id"]
    sess._scratchpads = None

    async def _fake_launch(**kwargs):
        return "Error: backend exited early (rc=1) before binding to :8123."

    monkeypatch.setattr(
        "anton.core.artifacts.backend_launcher.launch_artifact_backend", _fake_launch
    )
    outcome = asyncio.run(handle_launch_backend(sess, {"slug": slug}))
    assert outcome.ok is False
    assert outcome.reason == "launch_failed"
    # The launcher rolls back on failure, so nothing committed.
    assert json.loads(outcome.content)["committed_at"] is None
