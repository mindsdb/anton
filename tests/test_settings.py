from __future__ import annotations

import os

from anton.config.settings import AntonSettings


class TestAntonSettingsDefaults:
    def test_default_planning_provider(self):
        s = AntonSettings(anthropic_api_key="test")
        assert s.planning_provider == "anthropic"

    def test_default_planning_model(self):
        s = AntonSettings(anthropic_api_key="test")
        assert s.planning_model == "claude-sonnet-4-6"

    def test_default_coding_provider(self):
        s = AntonSettings(anthropic_api_key="test")
        assert s.coding_provider == "anthropic"

    def test_default_coding_model(self):
        s = AntonSettings(anthropic_api_key="test")
        assert s.coding_model == "claude-opus-4-6"

    def test_default_skills_dir(self):
        s = AntonSettings(anthropic_api_key="test")
        assert s.skills_dir == "skills"

    def test_default_user_skills_dir(self):
        s = AntonSettings(anthropic_api_key="test")
        assert s.user_skills_dir == "~/.anton/skills"

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

    def test_env_overrides_skills_dir(self, monkeypatch):
        monkeypatch.setenv("ANTON_SKILLS_DIR", "/custom/skills")
        s = AntonSettings(_env_file=None)
        assert s.skills_dir == "/custom/skills"
