"""`python -m anton.cloud_turn capabilities` — stable machine-readable manifest."""

from __future__ import annotations

import json
import subprocess
import sys

from anton.cloud_turn import __main__ as entry
from anton.cloud_turn.protocol import PROTOCOL_VERSION
from anton.cloud_turn.session import CLOUD_TOOL_ALLOWLIST


def _manifest_via_subprocess() -> tuple[int, dict]:
    proc = subprocess.run(
        [sys.executable, "-m", "anton.cloud_turn", "capabilities"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
    )
    return proc.returncode, json.loads(proc.stdout.decode("utf-8"))


def test_capabilities_command_is_valid_json_and_exit_zero():
    code, manifest = _manifest_via_subprocess()
    assert code == 0
    assert isinstance(manifest, dict)
    assert manifest["ok"] is True


def test_capabilities_has_stable_required_fields():
    manifest = entry._capabilities()
    required = {
        "ok", "entrypoint", "cloud_turn_version", "anton_version",
        "protocol_versions", "tools", "content_block_types", "message_roles",
        "model_override", "memory", "connectors", "data_vault",
        "scratchpad_execution", "web_tools", "provider_sdks", "capabilities",
    }
    assert required <= set(manifest)
    assert manifest["entrypoint"] == "anton.cloud_turn"


def test_capabilities_reports_exact_cloud_tools():
    manifest = entry._capabilities()
    assert manifest["tools"] == sorted(CLOUD_TOOL_ALLOWLIST)


def test_capabilities_protocol_versions():
    manifest = entry._capabilities()
    assert manifest["protocol_versions"] == [PROTOCOL_VERSION]


def test_capabilities_reports_milestone_truthfully():
    m = entry._capabilities()
    # Only what THIS milestone actually enables.
    assert m["content_block_types"] == ["text", "tool_result", "tool_use"]  # sorted
    assert m["message_roles"] == ["assistant", "user"]
    assert m["scratchpad_execution"] is True
    assert m["memory"] == {"personal": False, "workspace": False}
    assert m["connectors"] is False
    assert m["data_vault"] is False
    assert m["web_tools"] is False
    assert m["model_override"]["policy"] == "trusted_allowlist"
    assert m["model_override"]["default"] == "deny"


def test_capabilities_reports_anthropic_sdk_present():
    # anthropic is a hard dependency — always installed.
    assert entry._capabilities()["provider_sdks"]["anthropic"] is True


def test_capabilities_reports_unavailable_sdk_accurately(monkeypatch):
    # Simulate the openai SDK being absent: __import__("openai") then fails.
    monkeypatch.setitem(sys.modules, "openai", None)
    assert entry._capabilities()["provider_sdks"]["openai"] is False
