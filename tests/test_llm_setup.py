from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from anton.config.settings import AntonSettings
from anton.llm.ollama import OllamaModelInfo
from anton.llm.setup import configure_llm_settings
from anton.workspace import Workspace


@pytest.fixture(autouse=True)
def clean_llm_env(monkeypatch):
    for key in (
        "ANTON_ANTHROPIC_API_KEY",
        "ANTON_OPENAI_API_KEY",
        "ANTON_OPENAI_BASE_URL",
        "ANTON_OLLAMA_BASE_URL",
        "ANTON_PLANNING_PROVIDER",
        "ANTON_CODING_PROVIDER",
        "ANTON_PLANNING_MODEL",
        "ANTON_CODING_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)


class TestConfigureLlmSettings:
    @patch("anton.llm.setup.list_ollama_models")
    @patch("anton.llm.setup.Prompt.ask")
    def test_ollama_discovery_success(self, mock_ask, mock_list, tmp_path):
        mock_ask.side_effect = ["3", "http://localhost:11434", "1", "2"]
        mock_list.return_value = [
            OllamaModelInfo(name="qwen3.5:4b", parameter_size="4.7B", quantization="Q4_K_M"),
            OllamaModelInfo(name="qwen3:0.6b", parameter_size="751.63M", quantization="Q4_K_M"),
        ]

        console = MagicMock()
        workspace = Workspace(tmp_path)
        settings = AntonSettings(_env_file=None)

        applied = configure_llm_settings(console, settings, workspace, show_current_config=False)

        assert applied is True
        assert settings.planning_provider == "ollama"
        assert settings.coding_provider == "ollama"
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.planning_model == "qwen3.5:4b"
        assert settings.coding_model == "qwen3:0.6b"
        assert workspace.get_secret("ANTON_OLLAMA_BASE_URL") == "http://localhost:11434"

    @patch("anton.llm.setup.list_ollama_models")
    @patch("anton.llm.setup.Prompt.ask")
    def test_ollama_manual_fallback_when_discovery_fails(self, mock_ask, mock_list, tmp_path):
        mock_ask.side_effect = ["3", "localhost:11434/v1", "qwen3.5:4b", "qwen3.5:4b"]
        mock_list.side_effect = ConnectionError("boom")

        console = MagicMock()
        workspace = Workspace(tmp_path)
        settings = AntonSettings(_env_file=None)

        applied = configure_llm_settings(console, settings, workspace, show_current_config=False)

        assert applied is True
        assert settings.planning_provider == "ollama"
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.planning_model == "qwen3.5:4b"
        assert settings.coding_model == "qwen3.5:4b"

    @patch("anton.llm.setup.Prompt.ask")
    def test_anthropic_setup_persists_api_key(self, mock_ask, tmp_path):
        mock_ask.side_effect = ["1", "sk-ant-test", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"]

        console = MagicMock()
        workspace = Workspace(tmp_path)
        settings = AntonSettings(_env_file=None)

        applied = configure_llm_settings(console, settings, workspace, show_current_config=False)

        assert applied is True
        assert settings.planning_provider == "anthropic"
        assert settings.anthropic_api_key == "sk-ant-test"
        assert workspace.get_secret("ANTON_ANTHROPIC_API_KEY") == "sk-ant-test"
