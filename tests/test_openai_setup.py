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

    result = minds_client.test_llm("https://example.com", "test-key")
    assert result.ok is True

    payload = json.loads(captured["payload"].decode())
    assert payload["model"] == minds_client.MINDS_DEFAULT_CODING_MODEL
    # >= 16: the OpenAI-backed catalogue models (gpt-*, mindshub_air) reject
    # smaller values with integer_below_min_value — a 1-token probe was a
    # deterministic false negative on them.
    assert payload["max_completion_tokens"] == 20
    assert "max_tokens" not in payload


def _http_error(code: int, body: dict | None = None, reason: str = "err"):
    import io
    import urllib.error

    raw = json.dumps(body).encode() if body is not None else b""
    return urllib.error.HTTPError(
        "https://example.com/v1/chat/completions", code, reason, {}, io.BytesIO(raw)
    )


def test_minds_test_llm_probes_with_given_model(monkeypatch):
    captured: dict = {}

    def fake_minds_request(url, api_key, method="GET", payload=None, verify=True, **kwargs):
        captured["payload"] = payload
        return b"{}"

    monkeypatch.setattr("anton.minds_client.minds_request", fake_minds_request)

    assert minds_client.test_llm("https://example.com", "k", model="sonnet").ok
    assert json.loads(captured["payload"].decode())["model"] == "sonnet"


def test_minds_test_llm_propagates_provider_error_message(monkeypatch):
    """Regression for ENG-1140: a 404 model_not_found must surface the
    provider's message, not collapse into a bare False that gets rendered as
    'Check your API key and URL'."""

    err = _http_error(404, {"error": {"code": "model_not_found",
                                      "message": "The model '_code_' does not exist"}})

    def fake_minds_request(*args, **kwargs):
        raise err

    monkeypatch.setattr("anton.minds_client.minds_request", fake_minds_request)

    result = minds_client.test_llm("https://example.com", "test-key")
    assert result.ok is False
    assert result.rate_limited is False
    assert "model_not_found" in result.error
    assert "_code_" in result.error


def test_minds_test_llm_flags_rate_limit_and_keeps_detail(monkeypatch):
    """A 429 is usually the TPM/RPM limiter, not an empty wallet — the
    provider's own message must survive so the caller doesn't hardcode
    'buy tokens' advice at a user who has tokens."""

    def fake_minds_request(*args, **kwargs):
        raise _http_error(
            429, {"error": {"code": "rate_limit", "message": "Rate limit exceeded for model 'sonnet'"}}
        )

    monkeypatch.setattr("anton.minds_client.minds_request", fake_minds_request)

    result = minds_client.test_llm("https://example.com", "test-key")
    assert result.ok is False
    assert result.rate_limited is True
    assert "Rate limit exceeded" in result.error


def test_minds_test_llm_falls_back_to_status_when_body_unparseable(monkeypatch):
    def fake_minds_request(*args, **kwargs):
        import io
        import urllib.error

        raise urllib.error.HTTPError(
            "https://example.com", 502, "Bad Gateway", {}, io.BytesIO(b"<html>")
        )

    monkeypatch.setattr("anton.minds_client.minds_request", fake_minds_request)

    result = minds_client.test_llm("https://example.com", "test-key")
    assert result.ok is False
    assert "502" in result.error


class TestResolveMindsModels:
    def _catalog(self, ids):
        """Entries deliberately omit enabled/embedding — older hosts don't
        send the flags, and absent flags must mean 'usable'."""
        return json.dumps({"data": [{"id": i} for i in ids]}).encode()

    def _rich_catalog(self, entries):
        return json.dumps({"data": entries}).encode()

    def test_prefers_tier_defaults_from_catalog(self, monkeypatch):
        monkeypatch.setattr(
            "anton.minds_client.minds_request",
            lambda *a, **kw: self._catalog(
                ["mindshub_air", "haiku", "sonnet", "opus", "fable"]
            ),
        )
        r = minds_client.resolve_minds_models("https://x", "k")
        assert (r.planning, r.coding) == ("sonnet", "haiku")
        assert r.probe == r.coding  # catalogue-backed probe validates the config

    def test_falls_back_within_catalog_when_defaults_missing(self, monkeypatch):
        monkeypatch.setattr(
            "anton.minds_client.minds_request",
            lambda *a, **kw: self._catalog(["fable", "mindshub_air"]),
        )
        r = minds_client.resolve_minds_models("https://x", "k")
        assert (r.planning, r.coding) == ("fable", "mindshub_air")

    def test_coding_follows_planning_when_no_cheap_model_exists(self, monkeypatch):
        monkeypatch.setattr(
            "anton.minds_client.minds_request",
            lambda *a, **kw: self._catalog(["fable", "grok"]),
        )
        r = minds_client.resolve_minds_models("https://x", "k")
        assert (r.planning, r.coding) == ("fable", "fable")

    def test_falls_back_to_defaults_when_models_route_missing(self, monkeypatch):
        """/v1/models is not deployed on every MindsHub host — a 404 there
        must not block setup (same caution as cowork-server's validate_minds)."""

        def fake_minds_request(*args, **kwargs):
            raise _http_error(404)

        monkeypatch.setattr("anton.minds_client.minds_request", fake_minds_request)

        r = minds_client.resolve_minds_models("https://x", "k")
        assert (r.planning, r.coding) == (
            minds_client.MINDS_DEFAULT_PLANNING_MODEL,
            minds_client.MINDS_DEFAULT_CODING_MODEL,
        )
        # No catalogue to trust → probe with the universal free-bucket model,
        # not a paid default that 403s on free-tier keys (ENG-576).
        assert r.probe == minds_client.MINDS_FREE_TIER_MODEL

    def test_never_returns_dead_smart_router_aliases(self, monkeypatch):
        """The legacy mdb.ai aliases must never be picked, whatever the server
        says — even a catalogue consisting only of them falls back to defaults."""
        monkeypatch.setattr(
            "anton.minds_client.minds_request",
            lambda *a, **kw: self._catalog(["_reason_", "_code_"]),
        )
        r = minds_client.resolve_minds_models("https://x", "k")
        assert (r.planning, r.coding) == (
            minds_client.MINDS_DEFAULT_PLANNING_MODEL,
            minds_client.MINDS_DEFAULT_CODING_MODEL,
        )

    def test_free_tier_key_lands_on_the_free_bucket_model(self, monkeypatch):
        """enabled=false is auth's wallet/allowance-aware access decision: a
        free-tier key sees paid models disabled and must land on mindshub_air,
        not on a model that will 403 (the ENG-576 shape)."""
        monkeypatch.setattr(
            "anton.minds_client.minds_request",
            lambda *a, **kw: self._rich_catalog([
                {"id": "sonnet", "enabled": False, "embedding": False},
                {"id": "haiku", "enabled": False, "embedding": False},
                {"id": "opus", "enabled": False, "embedding": False},
                {"id": "mindshub_air", "enabled": True, "embedding": False},
            ]),
        )
        r = minds_client.resolve_minds_models("https://x", "k")
        assert (r.planning, r.coding, r.probe) == (
            "mindshub_air", "mindshub_air", "mindshub_air",
        )

    def test_embedding_models_are_never_picked(self, monkeypatch):
        """The ids[0] fallback must not land on an embeddings model configured
        as planning/coding."""
        monkeypatch.setattr(
            "anton.minds_client.minds_request",
            lambda *a, **kw: self._rich_catalog([
                {"id": "embed-small", "enabled": True, "embedding": True},
                {"id": "grok", "enabled": True, "embedding": False},
            ]),
        )
        r = minds_client.resolve_minds_models("https://x", "k")
        assert (r.planning, r.coding) == ("grok", "grok")


class TestMindsV1Base:
    """Host-aware base rule — same as AntonSettings.model_post_init (ENG-436)
    and cowork-server's minds_chat_base_url."""

    def test_mindshub_host_uses_v1(self):
        assert minds_client.minds_v1_base("https://api.mindshub.ai") == "https://api.mindshub.ai/v1"

    def test_legacy_mdb_host_uses_api_v1(self):
        assert minds_client.minds_v1_base("https://mdb.ai") == "https://mdb.ai/api/v1"

    def test_explicit_v1_suffix_kept_as_is(self):
        assert minds_client.minds_v1_base("https://host.example/v1") == "https://host.example/v1"


def test_setup_minds_skips_no_ssl_retry_on_http_error(monkeypatch):
    """An HTTP-level failure (server answered → TLS worked) must not trigger
    the verify=False retry — it can only repeat the same error."""
    import anton.cli as cli

    settings = AntonSettings(_env_file=None)
    workspace = MagicMock()
    probe_calls = []

    monkeypatch.setattr("anton.cli._setup_prompt", lambda *a, **kw: "mdb_test-key")
    monkeypatch.setattr("anton.cli.Confirm.ask", lambda *a, **kw: False)  # no retry
    monkeypatch.setattr(
        "anton.cli.resolve_minds_models",
        lambda *a, **kw: minds_client.MindsModels("sonnet", "haiku", "haiku"),
    )

    def fake_test_llm(*a, **kw):
        probe_calls.append(kw)
        return minds_client.LLMTestResult(
            ok=False, error="model_not_found: nope", http_status=404
        )

    monkeypatch.setattr("anton.cli.test_llm", fake_test_llm)

    import pytest

    with pytest.raises(cli._SetupRetry):
        cli._setup_minds(settings, workspace)

    assert len(probe_calls) == 1


def test_setup_minds_writes_catalog_resolved_models(monkeypatch):
    """End-to-end through _setup_minds: on a passing probe the resolved pair —
    not the dead _reason_/_code_ aliases — is persisted (ENG-1140)."""
    import anton.cli as cli

    settings = AntonSettings(_env_file=None)
    workspace = MagicMock()

    monkeypatch.setattr("anton.cli._setup_prompt", lambda *a, **kw: "mdb_test-key")
    monkeypatch.setattr("anton.cli.Confirm.ask", lambda *a, **kw: True)
    monkeypatch.setattr(
        "anton.cli.resolve_minds_models",
        lambda *a, **kw: minds_client.MindsModels("sonnet", "haiku", "haiku"),
    )
    monkeypatch.setattr(
        "anton.cli.test_llm",
        lambda *a, **kw: minds_client.LLMTestResult(ok=True),
    )

    cli._setup_minds(settings, workspace)

    assert settings.planning_model == "sonnet"
    assert settings.coding_model == "haiku"
    workspace.set_secret.assert_any_call("ANTON_PLANNING_MODEL", "sonnet")
    workspace.set_secret.assert_any_call("ANTON_CODING_MODEL", "haiku")


def test_setup_minds_honors_env_url_override(monkeypatch):
    """ANTON_MINDS_URL is the only path to a non-default host (staging,
    self-hosted) — setup must use it, not clobber it back to prod."""
    import anton.cli as cli

    settings = AntonSettings(_env_file=None)
    workspace = MagicMock()

    monkeypatch.setenv("ANTON_MINDS_URL", "https://api.staging.mindshub.ai/")
    monkeypatch.setattr("anton.cli._setup_prompt", lambda *a, **kw: "mdb_test-key")
    monkeypatch.setattr("anton.cli.Confirm.ask", lambda *a, **kw: True)
    monkeypatch.setattr(
        "anton.cli.resolve_minds_models",
        lambda *a, **kw: minds_client.MindsModels("sonnet", "haiku", "haiku"),
    )
    monkeypatch.setattr(
        "anton.cli.test_llm", lambda *a, **kw: minds_client.LLMTestResult(ok=True)
    )

    cli._setup_minds(settings, workspace)

    assert settings.minds_url == "https://api.staging.mindshub.ai"
    workspace.set_secret.assert_any_call(
        "ANTON_MINDS_URL", "https://api.staging.mindshub.ai"
    )


def test_setup_minds_persists_the_base_url_the_probe_validated(monkeypatch):
    """Invariant: the persisted ANTON_OPENAI_BASE_URL must be the base the
    probe hit. With a /v1-suffixed ANTON_MINDS_URL the probe absorbs the
    suffix (minds_v1_base) — persisting f"{url}/v1" would double it and save
    a runtime config the probe never validated ('Connected', then 404s)."""
    import anton.cli as cli

    settings = AntonSettings(_env_file=None)
    workspace = MagicMock()
    probed: dict = {}

    monkeypatch.setenv("ANTON_MINDS_URL", "https://api.staging.mindshub.ai/v1")
    monkeypatch.setattr("anton.cli._setup_prompt", lambda *a, **kw: "mdb_test-key")
    monkeypatch.setattr("anton.cli.Confirm.ask", lambda *a, **kw: True)
    monkeypatch.setattr(
        "anton.cli.resolve_minds_models",
        lambda *a, **kw: minds_client.MindsModels("sonnet", "haiku", "haiku"),
    )

    def fake_test_llm(base_url, api_key, verify=True, model=None):
        probed["base"] = minds_client.minds_v1_base(base_url)
        return minds_client.LLMTestResult(ok=True)

    monkeypatch.setattr("anton.cli.test_llm", fake_test_llm)

    cli._setup_minds(settings, workspace)

    assert probed["base"] == "https://api.staging.mindshub.ai/v1"
    assert settings.openai_base_url == probed["base"]
    workspace.set_secret.assert_any_call("ANTON_OPENAI_BASE_URL", probed["base"])


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


# ─────────────────────────────────────────────────────────────────────────────
# Search provider helpers — pure-function corners worth pinning
# ─────────────────────────────────────────────────────────────────────────────


class TestCurrentSearchLabel:
    """Locks in the masked format used by the ``Currently:`` line in
    ``_setup_search_provider`` — a regression here would silently leak a
    different number of key characters into the chat output.
    """

    def test_none_when_unconfigured(self):
        from anton.cli import _current_search_label
        from types import SimpleNamespace

        s = SimpleNamespace(external_search_provider=None, exa_api_key=None, brave_api_key=None)
        assert _current_search_label(s) == "none"

    def test_exa_with_full_key_masks_to_last_four(self):
        from anton.cli import _current_search_label
        from types import SimpleNamespace

        s = SimpleNamespace(
            external_search_provider="exa",
            exa_api_key="abcd-1234-wxyz",
            brave_api_key=None,
        )
        assert _current_search_label(s) == "Exa.ai (key: ****wxyz)"

    def test_brave_with_full_key_masks_to_last_four(self):
        from anton.cli import _current_search_label
        from types import SimpleNamespace

        s = SimpleNamespace(
            external_search_provider="brave",
            brave_api_key="brv-key-9876",
            exa_api_key=None,
        )
        assert _current_search_label(s) == "Brave Search (key: ****9876)"

    def test_short_key_omits_the_mask_to_avoid_revealing_length(self):
        from anton.cli import _current_search_label
        from types import SimpleNamespace

        s = SimpleNamespace(external_search_provider="exa", exa_api_key="ab", brave_api_key=None)
        assert _current_search_label(s) == "Exa.ai"

    def test_unknown_provider_falls_back_to_raw_value(self):
        from anton.cli import _current_search_label
        from types import SimpleNamespace

        s = SimpleNamespace(
            external_search_provider="serper", exa_api_key=None, brave_api_key=None
        )
        assert _current_search_label(s) == "serper"
