"""Tests for scratchpad env-var isolation (Phase 1 security).

Verifies that LocalScratchpadRuntime correctly filters os.environ when
allowed_env_keys is set, ensuring unrelated DS_* credentials are not
visible inside the scratchpad subprocess.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from anton.core.backends.local import LocalScratchpadRuntime, local_scratchpad_runtime_factory
from anton.core.backends.manager import ScratchpadManager


_SCRATCHPAD_DEFAULTS = dict(
    coding_provider="anthropic",
    coding_model="",
    coding_api_key="",
    coding_base_url="",
)


def make_scratchpad(name: str, **kwargs) -> LocalScratchpadRuntime:
    return LocalScratchpadRuntime(name=name, **{**_SCRATCHPAD_DEFAULTS, **kwargs})


# ---------------------------------------------------------------------------
# Unit tests — no subprocess needed, just inspect the env dict built in start()
# ---------------------------------------------------------------------------

class TestEnvFiltering:
    """Verify the env dict passed to the subprocess is correctly filtered."""

    def _build_env(self, pad: LocalScratchpadRuntime, environ: dict) -> dict:
        """Simulate the env-building logic from start() against a given environ."""
        if pad._allowed_env_keys is None:
            return environ.copy()
        _SYSTEM_KEYS = {
            "PATH", "HOME", "USER", "LOGNAME", "SHELL",
            "TMPDIR", "TEMP", "TMP",
            "LANG", "LC_ALL", "LC_CTYPE",
            "PYTHONPATH", "PYTHONHASHSEED",
            "SystemRoot", "COMSPEC",
        }
        return {
            k: v for k, v in environ.items()
            if k in _SYSTEM_KEYS or k in pad._allowed_env_keys
        }

    def test_none_allowed_env_keys_copies_everything(self):
        """allowed_env_keys=None → full env copy (legacy / CLI behaviour)."""
        pad = make_scratchpad("test-none")
        assert pad._allowed_env_keys is None

        fake_env = {
            "PATH": "/usr/bin",
            "DS_POSTGRES_PROD__PASSWORD": "secret",
            "DS_SLACK_MAIN__BOT_TOKEN": "xoxb-123",
            "SOME_OTHER_VAR": "value",
        }
        result = self._build_env(pad, fake_env)
        assert result == fake_env  # everything passes through

    def test_empty_allowed_env_keys_strips_all_ds_vars(self):
        """allowed_env_keys={} → DS_* vars stripped, system keys kept."""
        pad = make_scratchpad("test-empty", allowed_env_keys=set())

        fake_env = {
            "PATH": "/usr/bin",
            "HOME": "/Users/test",
            "DS_POSTGRES_PROD__PASSWORD": "secret",
            "DS_SLACK_MAIN__BOT_TOKEN": "xoxb-123",
        }
        result = self._build_env(pad, fake_env)
        assert "DS_POSTGRES_PROD__PASSWORD" not in result
        assert "DS_SLACK_MAIN__BOT_TOKEN" not in result
        assert result["PATH"] == "/usr/bin"
        assert result["HOME"] == "/Users/test"

    def test_specific_allowed_keys_only_those_ds_vars_pass(self):
        """Only the explicitly allowed DS_* var should be visible."""
        allowed = {"DS_POSTGRES_PROD__HOST", "DS_POSTGRES_PROD__PASSWORD"}
        pad = make_scratchpad("test-specific", allowed_env_keys=allowed)

        fake_env = {
            "PATH": "/usr/bin",
            "DS_POSTGRES_PROD__HOST": "db.example.com",
            "DS_POSTGRES_PROD__PASSWORD": "s3cr3t",
            "DS_SLACK_MAIN__BOT_TOKEN": "xoxb-should-not-appear",
            "DS_WHATSAPP_DEFAULT__ACCESS_TOKEN": "wh-should-not-appear",
        }
        result = self._build_env(pad, fake_env)

        # Allowed DS vars — present
        assert result["DS_POSTGRES_PROD__HOST"] == "db.example.com"
        assert result["DS_POSTGRES_PROD__PASSWORD"] == "s3cr3t"
        # Unrelated DS vars — blocked
        assert "DS_SLACK_MAIN__BOT_TOKEN" not in result
        assert "DS_WHATSAPP_DEFAULT__ACCESS_TOKEN" not in result
        # System key — always present
        assert result["PATH"] == "/usr/bin"

    def test_system_keys_always_pass_regardless_of_allowed_set(self):
        """PATH, HOME etc. must always be present even with a restrictive allowlist."""
        pad = make_scratchpad("test-sys", allowed_env_keys=set())  # empty — block all DS_*

        fake_env = {
            "PATH": "/usr/bin:/usr/local/bin",
            "HOME": "/Users/test",
            "USER": "test",
            "LANG": "en_US.UTF-8",
            "DS_SOME_SECRET__KEY": "should-be-blocked",
        }
        result = self._build_env(pad, fake_env)
        assert result["PATH"] == "/usr/bin:/usr/local/bin"
        assert result["HOME"] == "/Users/test"
        assert result["USER"] == "test"
        assert result["LANG"] == "en_US.UTF-8"
        assert "DS_SOME_SECRET__KEY" not in result


# ---------------------------------------------------------------------------
# Integration tests — actual subprocess execution
# ---------------------------------------------------------------------------

class TestEnvIsolationSubprocess:
    """Run real scratchpad cells and verify env var visibility."""

    async def test_blocked_ds_var_not_visible_in_subprocess(self):
        """A DS_* var NOT in allowed_env_keys must not appear in the subprocess."""
        sentinel = "SUPER_SECRET_TOKEN_XYZ_12345"
        env_key = "DS_BLOCKED_SERVICE__TOKEN"

        # Plant the secret in the parent process env
        os.environ[env_key] = sentinel
        try:
            pad = make_scratchpad(
                "test-blocked",
                allowed_env_keys=set(),  # block all DS_* vars
            )
            await pad.start()
            try:
                cell = await pad.execute_streaming(
                    f"import os; print(os.environ.get({env_key!r}, 'NOT_FOUND'))",
                    description="check blocked var",
                    estimated_seconds=5,
                ).__anext__()
                # Drain the generator to get the final Cell
                async for item in pad.execute_streaming(
                    f"import os; print(os.environ.get({env_key!r}, 'NOT_FOUND'))",
                    description="check blocked var",
                    estimated_seconds=5,
                ):
                    last = item
                assert "NOT_FOUND" in last.stdout
                assert sentinel not in last.stdout
            finally:
                await pad.close()
        finally:
            del os.environ[env_key]

    async def test_allowed_ds_var_visible_in_subprocess(self):
        """A DS_* var IN allowed_env_keys must be visible in the subprocess."""
        sentinel = "ALLOWED_TOKEN_ABC_67890"
        env_key = "DS_ALLOWED_SERVICE__TOKEN"

        os.environ[env_key] = sentinel
        try:
            pad = make_scratchpad(
                "test-allowed",
                allowed_env_keys={env_key},
            )
            await pad.start()
            try:
                async for item in pad.execute_streaming(
                    f"import os; print(os.environ.get({env_key!r}, 'NOT_FOUND'))",
                    description="check allowed var",
                    estimated_seconds=5,
                ):
                    last = item
                assert last.stdout.strip() == sentinel
                assert last.error is None
            finally:
                await pad.close()
        finally:
            del os.environ[env_key]

    async def test_legacy_none_mode_passes_all_ds_vars(self):
        """allowed_env_keys=None (default) → DS_* vars still visible (CLI compat)."""
        sentinel = "CLI_COMPAT_SECRET_99999"
        env_key = "DS_LEGACY_SERVICE__TOKEN"

        os.environ[env_key] = sentinel
        try:
            pad = make_scratchpad(
                "test-legacy",
                allowed_env_keys=None,  # explicit None = legacy behaviour
            )
            await pad.start()
            try:
                async for item in pad.execute_streaming(
                    f"import os; print(os.environ.get({env_key!r}, 'NOT_FOUND'))",
                    description="check legacy var",
                    estimated_seconds=5,
                ):
                    last = item
                assert last.stdout.strip() == sentinel
                assert last.error is None
            finally:
                await pad.close()
        finally:
            del os.environ[env_key]


# ---------------------------------------------------------------------------
# ScratchpadManager threading test
# ---------------------------------------------------------------------------

class TestManagerThreadsAllowedEnvKeys:
    """Verify ScratchpadManager correctly threads allowed_env_keys to new runtimes."""

    def test_manager_stores_allowed_env_keys(self):
        mgr = ScratchpadManager(
            runtime_factory=local_scratchpad_runtime_factory,
            coding_provider="anthropic",
            coding_model="",
            coding_api_key="",
            coding_base_url="",
            allowed_env_keys={"DS_POSTGRES_PROD__HOST"},
        )
        assert mgr._allowed_env_keys == {"DS_POSTGRES_PROD__HOST"}

    def test_manager_none_allowed_env_keys_is_default(self):
        mgr = ScratchpadManager(
            runtime_factory=local_scratchpad_runtime_factory,
            coding_provider="anthropic",
            coding_model="",
            coding_api_key="",
            coding_base_url="",
        )
        assert mgr._allowed_env_keys is None

    async def test_manager_passes_allowed_env_keys_to_runtime(self):
        """Runtime created by manager should have the same allowed_env_keys."""
        allowed = {"DS_POSTGRES_PROD__HOST", "DS_POSTGRES_PROD__PASSWORD"}
        mgr = ScratchpadManager(
            runtime_factory=local_scratchpad_runtime_factory,
            coding_provider="anthropic",
            coding_model="",
            coding_api_key="",
            coding_base_url="",
            allowed_env_keys=allowed,
        )
        pad = await mgr.get_or_create("test-mgr")
        try:
            assert pad._allowed_env_keys == allowed
        finally:
            await mgr.close_all()
