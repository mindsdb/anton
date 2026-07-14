from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from anton.core.session import _scrub_user_input
from anton.utils.datasources import (
    _DS_KNOWN_VARS,
    _DS_SECRET_VARS,
    scrub_credentials,
)


@pytest.fixture(autouse=True)
def clean_ds_state():
    """Clear _DS_SECRET_VARS, _DS_KNOWN_VARS, and all DS_* env vars around each test."""
    def _clean():
        _DS_SECRET_VARS.clear()
        _DS_KNOWN_VARS.clear()
        for k in list(os.environ):
            if k.startswith("DS_"):
                del os.environ[k]

    _clean()
    yield
    _clean()


class TestScrubCredentials:
    """Focused regression tests for _scrub_credentials short-secret handling."""

    def test_registered_6char_secret_scrubbed(self, monkeypatch):
        """A 6-character registered secret is scrubbed regardless of length."""
        _DS_SECRET_VARS.add("DS_PASSWORD")
        monkeypatch.setenv("DS_PASSWORD", "abc123")
        result = scrub_credentials("auth failed: abc123")
        assert "abc123" not in result
        assert "[DS_PASSWORD]" in result

    def test_registered_8char_secret_scrubbed(self, monkeypatch):
        """An 8-character registered secret is scrubbed (was at the old threshold)."""
        _DS_SECRET_VARS.add("DS_API_KEY")
        monkeypatch.setenv("DS_API_KEY", "tok12345")
        result = scrub_credentials("token=tok12345 rejected")
        assert "tok12345" not in result
        assert "[DS_API_KEY]" in result

    def test_registered_1char_secret_scrubbed(self, monkeypatch):
        """A 1-character registered secret is scrubbed."""
        _DS_SECRET_VARS.add("DS_SECRET")
        monkeypatch.setenv("DS_SECRET", "x")
        result = scrub_credentials("value=x here")
        assert "=x " not in result
        assert "[DS_SECRET]" in result

    def test_non_secret_var_not_scrubbed(self, monkeypatch):
        """A known but non-secret DS_* var (e.g. DS_HOST) stays readable."""
        _DS_KNOWN_VARS.add("DS_HOST")
        monkeypatch.setenv("DS_HOST", "mydbhostname")
        result = scrub_credentials("host=mydbhostname")
        assert "mydbhostname" in result

    def test_unknown_short_ds_var_not_scrubbed(self, monkeypatch):
        """Unknown DS_* vars with short values are NOT scrubbed (heuristic threshold)."""
        monkeypatch.setenv("DS_ENABLE_FEATURE", "on")
        result = scrub_credentials("flag=on active")
        assert "on" in result


class TestScrubProviderKeys:
    """Provider API keys must never reach model context (ENG-463)."""

    MINDS_KEY = "mdb_dI2OzIgO.5t7QUxqGPdgrdg2wNwvFFDTUHPyYUZRH"

    def test_provider_key_value_scrubbed_with_label(self, monkeypatch):
        """A live provider key present in env is redacted with its var label."""
        monkeypatch.setenv("ANTON_MINDS_API_KEY", self.MINDS_KEY)
        result = scrub_credentials(f'api_key = "{self.MINDS_KEY}"')
        assert self.MINDS_KEY not in result
        assert "[ANTON_MINDS_API_KEY]" in result

    def test_openai_key_value_scrubbed(self, monkeypatch):
        key = "sk-proj-abcDEF1234567890abcDEF1234567890"
        monkeypatch.setenv("OPENAI_API_KEY", key)
        result = scrub_credentials(f"OPENAI_API_KEY={key}")
        assert key not in result
        assert "[OPENAI_API_KEY]" in result

    def test_mdb_key_scrubbed_by_pattern_without_env(self):
        """A key the model already emitted (not in any env var) is caught by shape."""
        result = scrub_credentials("here it is: mdb_AAAAAAAAAA.BBBBBBBBBBBBCCCC")
        assert "mdb_AAAAAAAAAA" not in result
        assert "[REDACTED_API_KEY]" in result

    def test_sk_and_gemini_keys_scrubbed_by_pattern(self):
        text = "k1=sk-ant-api03-abcdefghij1234567890XYZ k2=AIzaSyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q"
        result = scrub_credentials(text)
        assert "sk-ant-api03" not in result
        assert "AIzaSy" not in result

    def test_short_sk_and_base_url_left_readable(self, monkeypatch):
        """Short `sk-` strings and non-secret base URLs are not over-redacted."""
        monkeypatch.setenv("ANTON_OPENAI_BASE_URL", "https://api.openai.com/v1")
        result = scrub_credentials("sk-abc connecting to https://api.openai.com/v1")
        assert "sk-abc" in result
        assert "https://api.openai.com/v1" in result


class TestScrubUserInput:
    """User messages are scrubbed before entering session history (ENG-583)."""

    def test_string_input_key_redacted(self):
        result = _scrub_user_input(
            "use this key: sk-ant-api03-abcdefghij1234567890XYZ"
        )
        assert "sk-ant-api03" not in result
        assert "[REDACTED_API_KEY]" in result

    def test_plain_string_unchanged(self):
        text = "please connect me to my staging database"
        assert _scrub_user_input(text) == text

    def test_text_blocks_scrubbed_other_blocks_untouched(self):
        blocks = [
            {"type": "text", "text": "key is mdb_AAAAAAAAAA.BBBBBBBBBBBBCCCC"},
            {"type": "image", "source": {"type": "base64", "data": "aGk="}},
        ]
        result = _scrub_user_input(blocks)
        assert "mdb_AAAAAAAAAA" not in result[0]["text"]
        assert "[REDACTED_API_KEY]" in result[0]["text"]
        assert result[1] is blocks[1]

    def test_known_secret_env_value_redacted_with_label(self, monkeypatch):
        """A pasted value matching a stored provider secret gets its var label."""
        key = "sk-proj-abcDEF1234567890abcDEF1234567890"
        monkeypatch.setenv("OPENAI_API_KEY", key)
        result = _scrub_user_input(f"my key is {key}")
        assert key not in result
        assert "[OPENAI_API_KEY]" in result


class TestCustomEngineRegistration:
    """ENG-688: connections of engines not in the registry (custom engines,
    connector-spec saves) must register their fields so non-secret values
    (base_url, host, ...) stay readable instead of leaking as markers."""

    def _vault(self, tmp_path):
        from anton.core.datasources.data_vault import LocalDataVault

        return LocalDataVault(tmp_path / "vault")

    def test_custom_engine_base_url_readable_secret_scrubbed(self, tmp_path):
        from anton.utils.datasources import restore_namespaced_env

        vault = self._vault(tmp_path)
        vault.save(
            "acme_crm", "prod",
            {"base_url": "https://api.acme-crm.example", "token": "tok_1234567890abcdef"},
            secure_keys=["token"],
        )
        restore_namespaced_env(vault)

        result = scrub_credentials(
            "GET https://api.acme-crm.example failed with token tok_1234567890abcdef"
        )
        assert "https://api.acme-crm.example" in result
        assert "tok_1234567890abcdef" not in result
        assert "[DS_ACME_CRM_PROD__TOKEN]" in result

    def test_custom_engine_without_secure_keys_uses_name_heuristic(self, tmp_path):
        from anton.utils.datasources import restore_namespaced_env

        vault = self._vault(tmp_path)
        vault.save(
            "acme_crm", "legacy",
            {"base_url": "https://legacy.acme-crm.example", "api_key": "ak_1234567890abcdef"},
        )
        restore_namespaced_env(vault)

        result = scrub_credentials(
            "base https://legacy.acme-crm.example key ak_1234567890abcdef"
        )
        assert "https://legacy.acme-crm.example" in result
        assert "ak_1234567890abcdef" not in result
        assert "[DS_ACME_CRM_LEGACY__API_KEY]" in result

    def test_custom_engine_legacy_passphrase_is_scrubbed(self, tmp_path):
        from anton.utils.datasources import restore_namespaced_env

        vault = self._vault(tmp_path)
        passphrase = "correct horse battery staple"
        vault.save(
            "acme_crm",
            "legacy",
            {
                "base_url": "https://legacy.acme-crm.example",
                "passphrase": passphrase,
            },
        )
        restore_namespaced_env(vault)

        result = scrub_credentials(
            f"base https://legacy.acme-crm.example passphrase {passphrase}"
        )
        assert "https://legacy.acme-crm.example" in result
        assert passphrase not in result
        assert "[DS_ACME_CRM_LEGACY__PASSPHRASE]" in result
