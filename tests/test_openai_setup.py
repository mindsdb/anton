from __future__ import annotations

import json
from unittest.mock import MagicMock

import openai

from anton.minds_client import test_llm as _minds_test_llm
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

    assert _minds_test_llm("https://example.com", "test-key") is True

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
