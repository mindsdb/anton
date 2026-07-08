from __future__ import annotations

import os
from pathlib import Path

import pytest

from anton.config.settings import AntonSettings


_ANTON_MODEL_KEYS = [
    "ANTON_PLANNING_PROVIDER",
    "ANTON_PLANNING_MODEL",
    "ANTON_CODING_PROVIDER",
    "ANTON_CODING_MODEL",
]


class TestAntonSettingsDefaults:
    def test_default_planning_provider(self, monkeypatch):
        for k in _ANTON_MODEL_KEYS:
            monkeypatch.delenv(k, raising=False)
        s = AntonSettings(anthropic_api_key="test", _env_file=None)
        assert s.planning_provider == "anthropic"

    def test_default_planning_model(self, monkeypatch):
        for k in _ANTON_MODEL_KEYS:
            monkeypatch.delenv(k, raising=False)
        s = AntonSettings(anthropic_api_key="test", _env_file=None)
        assert s.planning_model == "claude-sonnet-4-6"

    def test_default_coding_provider(self, monkeypatch):
        for k in _ANTON_MODEL_KEYS:
            monkeypatch.delenv(k, raising=False)
        s = AntonSettings(anthropic_api_key="test", _env_file=None)
        assert s.coding_provider == "anthropic"

    def test_default_coding_model(self, monkeypatch):
        for k in _ANTON_MODEL_KEYS:
            monkeypatch.delenv(k, raising=False)
        s = AntonSettings(anthropic_api_key="test", _env_file=None)
        assert s.coding_model == "claude-haiku-4-5-20251001"

    def test_default_memory_dir(self):
        s = AntonSettings(anthropic_api_key="test")
        assert s.memory_dir == ".anton"

    def test_default_context_dir(self):
        s = AntonSettings(anthropic_api_key="test")
        assert s.context_dir == ".anton/context"

    def test_default_api_key_is_none(self):
        s = AntonSettings(_env_file=None)
        assert s.anthropic_api_key is None


class TestAntonSettingsEnvOverride:
    def test_env_overrides_planning_model(self, monkeypatch):
        monkeypatch.setenv("ANTON_PLANNING_MODEL", "custom-model")
        s = AntonSettings(_env_file=None)
        assert s.planning_model == "custom-model"

    def test_env_overrides_api_key(self, monkeypatch):
        monkeypatch.setenv("ANTON_ANTHROPIC_API_KEY", "sk-test-key")
        s = AntonSettings(_env_file=None)
        assert s.anthropic_api_key == "sk-test-key"

class TestWorkspaceResolution:
    def test_resolve_workspace_defaults_to_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        s = AntonSettings(anthropic_api_key="test", _env_file=None)
        s.resolve_workspace()

        assert s.workspace_path == tmp_path
        assert Path(s.memory_dir) == tmp_path / ".anton"
        assert Path(s.context_dir) == tmp_path / ".anton" / "context"

    def test_resolve_workspace_with_explicit_folder(self, tmp_path):
        s = AntonSettings(anthropic_api_key="test", _env_file=None)
        s.resolve_workspace(str(tmp_path))

        assert s.workspace_path == tmp_path
        assert Path(s.memory_dir) == tmp_path / ".anton"
        assert Path(s.context_dir) == tmp_path / ".anton" / "context"

    def test_resolve_workspace_does_not_create_anton_dir(self, tmp_path):
        """resolve_workspace only resolves paths — directory creation is deferred to initialize()."""
        s = AntonSettings(anthropic_api_key="test", _env_file=None)
        s.resolve_workspace(str(tmp_path))

        assert not (tmp_path / ".anton").exists()

    def test_resolve_workspace_preserves_absolute_paths(self, tmp_path):
        s = AntonSettings(
            anthropic_api_key="test",
            memory_dir="/absolute/path",
            _env_file=None,
        )
        s.resolve_workspace(str(tmp_path))

        # Absolute path should not be changed
        assert s.memory_dir == "/absolute/path"

    def test_workspace_path_before_resolve(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        s = AntonSettings(anthropic_api_key="test", _env_file=None)
        # Before resolve, workspace_path falls back to cwd
        assert s.workspace_path == tmp_path


class TestMindsOpenAIBaseUrlDerivation:
    """model_post_init derives a host-aware openai_base_url from minds_url.

    Regression for the mdb.ai-era hardcoded /api/v1 (ENG-436): api.mindshub.ai
    must derive /v1, mdb.ai keeps /api/v1, and an already-suffixed URL is
    preserved. Derivation only fires for an openai-compatible provider when
    no openai key/url is already set.
    """

    def _derive(self, minds_url, monkeypatch):
        for k in _ANTON_MODEL_KEYS + [
            "ANTON_OPENAI_API_KEY",
            "ANTON_OPENAI_BASE_URL",
            "ANTON_MINDS_API_KEY",
            "ANTON_MINDS_URL",
        ]:
            monkeypatch.delenv(k, raising=False)
        return AntonSettings(
            minds_api_key="mdb_test",
            minds_url=minds_url,
            planning_provider="openai-compatible",
            coding_provider="openai-compatible",
            _env_file=None,
        )

    def test_mindshub_derives_v1(self, monkeypatch):
        s = self._derive("https://api.mindshub.ai", monkeypatch)
        assert s.openai_api_key == "mdb_test"
        assert s.openai_base_url == "https://api.mindshub.ai/v1"

    def test_legacy_mdb_ai_derives_api_v1(self, monkeypatch):
        s = self._derive("https://mdb.ai", monkeypatch)
        assert s.openai_base_url == "https://mdb.ai/api/v1"

    def test_already_suffixed_url_preserved(self, monkeypatch):
        s = self._derive("https://api.mindshub.ai/v1", monkeypatch)
        assert s.openai_base_url == "https://api.mindshub.ai/v1"

    def test_no_derivation_when_openai_key_present(self, monkeypatch):
        for k in _ANTON_MODEL_KEYS + ["ANTON_OPENAI_BASE_URL"]:
            monkeypatch.delenv(k, raising=False)
        s = AntonSettings(
            minds_api_key="mdb_test",
            minds_url="https://api.mindshub.ai",
            openai_api_key="sk-real-user-key",
            planning_provider="openai-compatible",
            _env_file=None,
        )
        # Derivation is skipped because openai_api_key is already set.
        assert s.openai_api_key == "sk-real-user-key"
        assert s.openai_base_url is None


class TestMindsCloudProviderNormalization:
    """A shared/desktop config sets provider = 'minds-cloud', but the CLI's
    LLMClient registry only has 'openai-compatible' (MindsHub speaks the
    OpenAI-compatible API). Without normalization the CLI crashed with
    'Unknown planning provider: minds-cloud' (ENG-655)."""

    def test_minds_cloud_planning_maps_to_openai_compatible(self, monkeypatch):
        for k in _ANTON_MODEL_KEYS:
            monkeypatch.delenv(k, raising=False)
        s = AntonSettings(planning_provider="minds-cloud", _env_file=None)
        assert s.planning_provider == "openai-compatible"

    def test_minds_cloud_coding_maps_to_openai_compatible(self, monkeypatch):
        for k in _ANTON_MODEL_KEYS:
            monkeypatch.delenv(k, raising=False)
        s = AntonSettings(coding_provider="minds-cloud", _env_file=None)
        assert s.coding_provider == "openai-compatible"

    def test_underscore_spelling_also_maps(self, monkeypatch):
        for k in _ANTON_MODEL_KEYS:
            monkeypatch.delenv(k, raising=False)
        s = AntonSettings(planning_provider="minds_cloud", _env_file=None)
        assert s.planning_provider == "openai-compatible"

    def test_case_and_whitespace_tolerant(self, monkeypatch):
        for k in _ANTON_MODEL_KEYS:
            monkeypatch.delenv(k, raising=False)
        for variant in ("MINDS-CLOUD", " Minds_Cloud ", "minds_cloud"):
            s = AntonSettings(planning_provider=variant, _env_file=None)
            assert s.planning_provider == "openai-compatible", variant

    def test_other_providers_pass_through(self, monkeypatch):
        for k in _ANTON_MODEL_KEYS:
            monkeypatch.delenv(k, raising=False)
        for p in ("anthropic", "openai", "openai-compatible"):
            s = AntonSettings(planning_provider=p, _env_file=None)
            assert s.planning_provider == p

    def test_from_settings_does_not_crash_on_minds_cloud(self, monkeypatch):
        # The exact regression: building the LLM client from a minds-cloud
        # config must not raise "Unknown planning provider".
        for k in _ANTON_MODEL_KEYS:
            monkeypatch.delenv(k, raising=False)
        from anton.core.llm.client import LLMClient
        s = AntonSettings(
            planning_provider="minds-cloud",
            coding_provider="minds-cloud",
            minds_api_key="mdb_dummy",
            minds_url="https://api.mindshub.ai/v1",
            _env_file=None,
        )
        # Should build without raising; creds derived from the minds_* fields.
        LLMClient.from_settings(s)
        assert s.openai_base_url == "https://api.mindshub.ai/v1"

    def test_minds_cloud_router_maps_to_openai_compatible(self, monkeypatch):
        # ENG-660: the router role is validated by from_settings identically to
        # planning/coding, so it must be normalized too — else a shared
        # minds-cloud config re-crashes with "Unknown router provider".
        for k in _ANTON_MODEL_KEYS:
            monkeypatch.delenv(k, raising=False)
        s = AntonSettings(router_provider="minds_cloud", _env_file=None)
        assert s.router_provider == "openai-compatible"

    def test_from_settings_does_not_crash_on_minds_cloud_router(self, monkeypatch):
        for k in _ANTON_MODEL_KEYS:
            monkeypatch.delenv(k, raising=False)
        from anton.core.llm.client import LLMClient
        s = AntonSettings(
            planning_provider="minds-cloud",
            coding_provider="minds-cloud",
            router_provider="minds-cloud",
            minds_api_key="mdb_dummy",
            minds_url="https://api.mindshub.ai/v1",
            _env_file=None,
        )
        # Must build without "Unknown router provider: minds-cloud".
        LLMClient.from_settings(s)
