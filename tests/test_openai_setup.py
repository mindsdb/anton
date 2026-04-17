from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import openai

import anton.minds_client as minds_client
from anton.cli import _setup_openai, _validate_openai_probe_response
from anton.config.settings import AntonSettings


def test_minds_test_llm_uses_modern_openai_token_parameter(monkeypatch):
    captured: dict = {}

    def fake_minds_request(url, api_key, method="GET", payload=None, verify=True, **kwargs):
        captured["url"] = url
        captured["api_key"] = api_key
        captured["method"] = method
        captured["payload"] = payload
        captured["verify"] = verify
        return b"{}"

    monkeypatch.setattr("anton.minds_client.minds_request", fake_minds_request)

    assert minds_client.test_llm("https://example.com", "test-key") is True

    payload = json.loads(captured["payload"].decode())
    assert payload["model"] == "_code_"
    assert payload["max_completion_tokens"] == 1
    assert "max_tokens" not in payload


def test_setup_openai_uses_modern_openai_token_parameter(monkeypatch):
    settings = AntonSettings(_env_file=None)
    workspace = MagicMock()
    prompts = iter(["test-key", "gpt-5.4"])
    mock_create = MagicMock()
    mock_create.return_value = MagicMock(choices=[MagicMock(
        finish_reason="stop",
        message=MagicMock(content="pong"),
    )])
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    monkeypatch.setattr("anton.cli._setup_prompt", lambda *args, **kwargs: next(prompts))
    monkeypatch.setattr("anton.cli._validate_with_spinner", lambda console, model, fn: fn())
    monkeypatch.setattr(openai, "OpenAI", lambda api_key: mock_client)

    _setup_openai(settings, workspace)

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-5.4"
    assert call_kwargs["messages"] == [{"role": "user", "content": "Reply with exactly: pong"}]
    assert call_kwargs["max_completion_tokens"] == 16
    assert "max_tokens" not in call_kwargs
    assert settings.openai_api_key == "test-key"
    assert settings.planning_model == "gpt-5.4"
    assert settings.coding_model == "gpt-5.4"


def test_validate_openai_probe_response_accepts_exact_pong():
    response = MagicMock()
    response.choices = [MagicMock(
        finish_reason="stop",
        message=MagicMock(content="pong"),
    )]

    _validate_openai_probe_response(response)


def test_validate_openai_probe_response_accepts_truncated_nonempty_output():
    response = MagicMock()
    response.choices = [MagicMock(
        finish_reason="length",
        message=MagicMock(content="po"),
    )]

    _validate_openai_probe_response(response)


class TestSetupCustomOpenAIAzure:
    def _make_probe_response(self, content="pong"):
        return MagicMock(choices=[MagicMock(
            finish_reason="stop",
            message=MagicMock(content=content),
        )])

    def test_azure_url_with_api_version_uses_azure_client(self, monkeypatch):
        """Azure URL + api-version → AzureOpenAI, endpoint stripped to scheme+host."""
        from anton.cli import _setup_custom_openai

        settings = AntonSettings(_env_file=None)
        workspace = MagicMock()

        # base_url (with path+query), api_key, model, api_version
        prompts = iter([
            "https://myresource.cognitiveservices.azure.com/openai/responses?api-version=2024-06-01",
            "azure-api-key",
            "gpt-4.1-mini",
            "2024-12-01-preview",
        ])
        monkeypatch.setattr("anton.cli._setup_prompt", lambda *a, **kw: next(prompts))
        monkeypatch.setattr("anton.cli._validate_with_spinner", lambda _c, _l, fn: fn())

        captured: dict = {}
        mock_azure_client = MagicMock()
        mock_azure_client.chat.completions.create.return_value = self._make_probe_response()

        def fake_azure_openai(**kwargs):
            captured.update(kwargs)
            return mock_azure_client

        with patch("anton.cli.AzureOpenAI", fake_azure_openai):
            _setup_custom_openai(settings, workspace)

        assert captured["api_version"] == "2024-12-01-preview"
        assert captured["api_key"] == "azure-api-key"
        # Path and query must have been stripped
        assert captured["azure_endpoint"] == "https://myresource.cognitiveservices.azure.com"

    def test_azure_flow_saves_api_version_to_settings(self, monkeypatch):
        """api_version must be persisted on settings and written to workspace."""
        from anton.cli import _setup_custom_openai

        settings = AntonSettings(_env_file=None)
        workspace = MagicMock()

        prompts = iter([
            "https://myresource.cognitiveservices.azure.com",
            "azure-key",
            "gpt-4.1-mini",
            "2024-12-01-preview",
        ])
        monkeypatch.setattr("anton.cli._setup_prompt", lambda *a, **kw: next(prompts))
        monkeypatch.setattr("anton.cli._validate_with_spinner", lambda _c, _l, fn: fn())

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_probe_response()

        with patch("anton.cli.AzureOpenAI", return_value=mock_client):
            _setup_custom_openai(settings, workspace)

        assert settings.openai_api_version == "2024-12-01-preview"
        assert settings.planning_model == "gpt-4.1-mini"
        workspace.set_secret.assert_any_call("ANTON_OPENAI_API_VERSION", "2024-12-01-preview")

    def test_no_api_version_uses_standard_client(self, monkeypatch):
        """Blank api-version → regular openai.OpenAI, no AzureOpenAI."""
        from anton.cli import _setup_custom_openai

        settings = AntonSettings(_env_file=None)
        workspace = MagicMock()

        # base_url, api_key, model, api_version (blank)
        prompts = iter(["http://localhost:11434/v1", "not-needed", "llama3", ""])
        monkeypatch.setattr("anton.cli._setup_prompt", lambda *a, **kw: next(prompts))
        monkeypatch.setattr("anton.cli._validate_with_spinner", lambda _c, _l, fn: fn())

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_probe_response()

        azure_called = []
        with patch("anton.cli.AzureOpenAI", side_effect=lambda **kw: azure_called.append(kw)), \
             patch("anton.cli.openai") as mock_openai_mod:
            mock_openai_mod.OpenAI.return_value = mock_client
            _setup_custom_openai(settings, workspace)

        assert not azure_called
        assert settings.openai_api_version is None

    def test_non_azure_endpoint_with_api_version_uses_standard_client(self, monkeypatch):
        """Non-Azure URL + api-version → openai.OpenAI with default_query, not AzureOpenAI."""
        from anton.cli import _setup_custom_openai

        settings = AntonSettings(_env_file=None)
        workspace = MagicMock()

        # base_url, api_key, model, api_version
        prompts = iter(["https://api.example.com/v1", "key", "my-model", "2025-01"])
        monkeypatch.setattr("anton.cli._setup_prompt", lambda *a, **kw: next(prompts))
        monkeypatch.setattr("anton.cli._validate_with_spinner", lambda _c, _l, fn: fn())

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_probe_response()

        azure_called = []
        with patch("anton.cli.AzureOpenAI", side_effect=lambda **kw: azure_called.append(kw)), \
             patch("anton.cli.openai") as mock_openai_mod:
            mock_openai_mod.OpenAI.return_value = mock_client
            _setup_custom_openai(settings, workspace)

        assert not azure_called
        assert settings.openai_api_version == "2025-01"
