from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

import pytest

from anton.core.session import _scrub_user_input
from anton.utils.datasources import (
    _DS_KNOWN_VARS,
    _DS_SECRET_VARS,
    _reset_registered_ds_vars,
    scrub_credentials,
)


@pytest.fixture(autouse=True)
def clean_ds_state():
    """Clear _DS_SECRET_VARS, _DS_KNOWN_VARS, and all DS_* env vars around each test."""
    def _clean():
        _reset_registered_ds_vars()
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


class TestConcurrentTurnIsolation:
    """The ticket's own acceptance criterion: two turns running concurrently
    on divergent vaults must never cross-contaminate scrub state."""

    def _vault(self, tmp_path, subdir: str):
        from anton.core.datasources.data_vault import LocalDataVault

        return LocalDataVault(tmp_path / subdir)

    async def test_two_concurrent_turns_scrub_only_their_own_secret(self, tmp_path):
        import asyncio

        from anton.utils.datasources import restore_namespaced_env

        vault_a = self._vault(tmp_path, "vault_a")
        vault_a.save("acme_crm", "prod", {"token": "secret-for-turn-a"}, secure_keys=["token"])

        vault_b = self._vault(tmp_path, "vault_b")
        vault_b.save("acme_crm", "prod", {"token": "secret-for-turn-b"}, secure_keys=["token"])

        async def turn(vault, own_secret: str, other_secret: str) -> str:
            restore_namespaced_env(vault)
            await asyncio.sleep(0)  # yield, so the other turn's setup interleaves here
            return scrub_credentials(f"own={own_secret} other={other_secret}")

        result_a, result_b = await asyncio.gather(
            turn(vault_a, "secret-for-turn-a", "secret-for-turn-b"),
            turn(vault_b, "secret-for-turn-b", "secret-for-turn-a"),
        )

        assert "secret-for-turn-a" not in result_a
        assert "[DS_ACME_CRM_PROD__TOKEN]" in result_a
        assert "secret-for-turn-b" not in result_b
        assert "[DS_ACME_CRM_PROD__TOKEN]" in result_b
class TestOAuthEngineRegistryCollision:
    """A cloud gmail OAuth connection shares its engine name with the
    registry's legacy IMAP gmail connector (datasources.md) — its OAuth
    fields (access_token, account_email, ...) must classify against the
    vault's own secure_keys, not the IMAP entry's (email, app_password)."""

    def test_gmail_oauth_fields_classify_against_the_vault_not_the_imap_registry_entry(self, monkeypatch):
        from anton.core.datasources.data_vault import TurnKeyDataVault
        from anton.utils.datasources import restore_namespaced_env

        # Long enough (> 8 chars) to actually exercise the coarse "unknown
        # DS_* var" bucket if misclassified — a short value dodges it either
        # way and would pass this test for the wrong reason.
        email = "someone@example.com"

        def fake(url, api_key, *, method="GET", payload=None, verify=True, timeout=30):
            return f'{{"access_token": "ya29.live-token", "account_email": "{email}", "scope": "gmail.readonly"}}'.encode()

        monkeypatch.setattr("anton.minds_client.minds_request", fake)
        vault = TurnKeyDataVault({"turn_key": "tk_abc", "connections": [{"engine": "gmail", "name": "primary"}]})
        restore_namespaced_env(vault)

        result = scrub_credentials(f"token ya29.live-token for {email} scope gmail.readonly")
        assert "ya29.live-token" not in result
        assert "[DS_GMAIL_PRIMARY__ACCESS_TOKEN]" in result
        # Non-secret metadata must stay visible — before the fix, both fell
        # into the coarse "unknown DS_* var, len > 8" bucket instead, since
        # the IMAP registry entry's fields (email, app_password) never
        # registered these as known.
        assert email in result
        assert "gmail.readonly" in result


class TestTurnMapIsAuthoritative:
    """Once a turn has a value map, os.environ is not consulted for a missing
    key: another turn may hold a different value under the same name."""

    def test_a_missing_key_is_not_read_from_environ(self, monkeypatch):
        from anton.utils.datasources import set_ds_env_values

        key = "DS_ACME_CRM_PROD__TOKEN"
        _DS_SECRET_VARS.add(key)
        _DS_KNOWN_VARS.add(key)

        # Another turn's value for the same var name, still in the process env.
        monkeypatch.setenv(key, "another-turns-token")
        # This turn has a map, and the key is not in it (its lookup failed).
        set_ds_env_values({})

        result = scrub_credentials("log line mentioning another-turns-token")

        # Redacting here would both miss this turn's secret and confirm the
        # other turn's value by substituting it.
        assert "another-turns-token" in result

    def test_the_coarse_net_still_reads_environ_after_a_map_opens(self, monkeypatch):
        """Only the labeled lookup is gated. A DS_* var the vault never
        registered still gets caught by the unknown-DS_* net, which is what
        keeps an operator-exported credential out of model context.
        """
        from anton.utils.datasources import set_ds_env_values

        # Registered as secret but NOT as known, i.e. the coarse-net case.
        _DS_SECRET_VARS.add("DS_SHELL_SET__PASSWORD")
        monkeypatch.setenv("DS_SHELL_SET__PASSWORD", "shell-value")

        set_ds_env_values({})
        result = scrub_credentials("pw shell-value")

        assert "shell-value" not in result

    def test_a_caller_that_never_set_a_map_still_reads_environ(self, monkeypatch):
        """The CLI and os.environ-based tests keep working unchanged."""
        _DS_SECRET_VARS.add("DS_MANUAL__PASSWORD")
        monkeypatch.setenv("DS_MANUAL__PASSWORD", "shell-set-secret")

        result = scrub_credentials("pw shell-set-secret")

        assert "shell-set-secret" not in result
        assert "[DS_MANUAL__PASSWORD]" in result


class TestRegistrationFromInsideAToolCall:
    """Every tool call runs in its own task (`asyncio.create_task` in
    session.py), and that copies the context. A connect made mid-turn must
    still be scrubbed from the rest of the turn's output, so the per-turn
    state has to be mutated in place rather than reassigned.
    """

    async def test_a_mid_turn_connect_is_scrubbed_by_its_own_turn(self, tmp_path):
        from anton.core.datasources.data_vault import LocalDataVault
        from anton.utils.datasources import restore_namespaced_env

        vault = LocalDataVault(vault_dir=tmp_path / "vault")

        # The turn opens its scope, as the host does before any tool runs.
        restore_namespaced_env(vault)

        # Mid-turn the user hands the agent a password; the tool handler that
        # saves it runs in its own task.
        async def connect_tool() -> None:
            vault.save(
                "postgres", "prod", {"password": "given-mid-turn"}, secure_keys=["password"]
            )
            restore_namespaced_env(vault)

        await asyncio.create_task(connect_tool())

        # Back in the turn's own context, which is what scrubs cell output.
        assert "DS_POSTGRES_PROD__PASSWORD" in _DS_SECRET_VARS
        result = scrub_credentials("traceback: password=given-mid-turn")
        assert "given-mid-turn" not in result
        assert "[DS_POSTGRES_PROD__PASSWORD]" in result

    async def test_a_concurrent_turn_still_cannot_see_it(self, tmp_path):
        """Mutating in place must not reintroduce cross-turn bleed: a sibling
        turn opens its own scope and sees only its own vault."""
        from anton.core.datasources.data_vault import LocalDataVault
        from anton.utils.datasources import restore_namespaced_env, set_ds_env_values

        vault_a = LocalDataVault(vault_dir=tmp_path / "a")
        vault_a.save("postgres", "prod", {"password": "turn-a-pw"}, secure_keys=["password"])
        vault_b = LocalDataVault(vault_dir=tmp_path / "b")

        async def turn(vault) -> str:
            restore_namespaced_env(vault)
            await asyncio.sleep(0)
            return scrub_credentials("saw turn-a-pw")

        # Each turn runs as its own task, the way the streaming producers do.
        out_a, out_b = await asyncio.gather(
            asyncio.create_task(turn(vault_a)),
            asyncio.create_task(turn(vault_b)),
        )

        assert "turn-a-pw" not in out_a
        # Turn B has no such connection, so it has no value to redact with and
        # must not have acquired turn A's.
        assert "turn-a-pw" in out_b
