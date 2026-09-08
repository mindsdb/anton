"""Scenario H — Credential scrubbing end-to-end."""

from __future__ import annotations

import pytest

from tests.e2e.harness import assert_exit_ok, base_env, run_anton
from tests.e2e.stub_server import StubServer


_SECRET_KEY = "DS_POSTGRES_PROD__PASSWORD"
_SECRET_VALUE = "super-secret-pw-123"


def _all_message_text(requests: list[dict]) -> str:
    parts: list[str] = []
    for req in requests:
        for msg in req.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        parts.append(str(block.get("content", "")))
                        parts.append(str(block.get("text", "")))
    return "\n".join(parts)


@pytest.mark.stub_only
def test_ds_secret_scrubbed_before_reaching_llm(tmp_path):
    """The secret arrives through the vault, the only supported channel: a pad's
    DS_* is derived from the vault, not inherited from the parent's environ."""
    from anton.core.datasources.data_vault import LocalDataVault

    home = tmp_path / "home"
    (home / ".anton").mkdir(parents=True)
    LocalDataVault(vault_dir=home / ".anton" / "data_vault").save(
        "postgres", "prod", {"password": _SECRET_VALUE}, secure_keys=["password"]
    )

    code = f"import os\nprint(os.environ['{_SECRET_KEY}'])\n"
    with StubServer() as stub:
        stub.queue_tool_call("scratchpad", {"action": "exec", "name": "cred_test", "code": code})
        stub.queue_text("The credential was redacted. SCRUBBED")
        stub.queue_verification_ok()
        env = base_env(stub)
        env["HOME"] = str(home)
        result = run_anton(["--folder", str(tmp_path)], ["print the db password", "exit"],
                           env=env, timeout=20)

    assert not result.timed_out
    assert_exit_ok(result)
    all_text = _all_message_text(stub.requests)
    assert _SECRET_VALUE not in all_text, \
        f"Secret leaked to LLM. Snippet: {all_text[max(0,all_text.find(_SECRET_VALUE)-100):all_text.find(_SECRET_VALUE)+100]!r}"
    assert f"[{_SECRET_KEY}]" in all_text, \
        f"Placeholder [{_SECRET_KEY}] not found. Text: {all_text[:500]!r}"
