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
        # No catalogue to trust → probe the model we would actually CONFIGURE,
        # and expose the free bucket as the escalation target. Probing the free
        # model directly (the old behaviour) passed for free-tier keys and then
        # persisted an unvalidated paid pair that 403s on first real use —
        # "Connected" masking a broken install (#317 re-review). Escalation is
        # what keeps free-tier keys unblocked; see resolve_and_probe.
        assert r.probe == r.coding
        assert r.free_fallback == minds_client.MINDS_FREE_TIER_MODEL

    def test_catalogue_resolved_pair_has_no_escalation_target(self, monkeypatch):
        """A catalogue-derived pair is already access-aware — /v1/models only
        lists what the key may use — so there is nothing to escalate to."""
        monkeypatch.setattr(
            "anton.minds_client.list_models", lambda *a, **k: ["sonnet", "haiku"]
        )
        r = minds_client.resolve_minds_models("https://x", "k")
        assert (r.planning, r.coding, r.probe) == ("sonnet", "haiku", "haiku")
        assert r.free_fallback is None

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


class TestMindsPassthroughFlavorDetection:
    """The gateway must be detected as passthrough for every base URL
    ``model_post_init`` can derive (#317 review).

    Getting this wrong is silent: the provider falls back to
    ``FLAVOR_OPENAI_COMPATIBLE_GENERIC``, whose ``native_web_tools()`` returns
    an empty set, so the session routes web_search to a handler ToolDef needing
    an Exa/Brave key that MindsHub users don't have. Web search just stops
    working, with no error anywhere.
    """

    def _flavor(self, minds_url: str) -> str:
        from anton.core.llm.client import _resolve_openai_compatible_flavor

        settings = AntonSettings(
            planning_provider="openai-compatible",
            coding_provider="openai-compatible",
            minds_api_key="k",
            minds_url=minds_url,
            openai_base_url=None,
            openai_api_key=None,
        )
        return _resolve_openai_compatible_flavor(settings)

    def _passthrough(self) -> str:
        from anton.core.llm.openai import OpenAIProvider

        return OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH

    def test_mindshub_host_is_passthrough(self):
        # The canonical value setup writes. This is the case that was broken:
        # minds_url has no suffix, model_post_init derives .../v1, and the
        # detection only knew about the bare host and /api/v1.
        assert self._flavor("https://api.mindshub.ai") == self._passthrough()

    def test_legacy_mdb_host_is_passthrough(self):
        assert self._flavor("https://mdb.ai") == self._passthrough()

    def test_explicit_v1_suffix_is_passthrough(self):
        assert self._flavor("https://api.mindshub.ai/v1") == self._passthrough()

    def test_third_party_endpoint_stays_generic(self):
        # The guard against over-matching: a generic endpoint must NOT get the
        # passthrough, or anton sends web-tool types it can't understand.
        from anton.core.llm.openai import OpenAIProvider

        settings = AntonSettings(
            planning_provider="openai-compatible",
            coding_provider="openai-compatible",
            minds_api_key="k",
            minds_url="https://api.mindshub.ai",
            openai_base_url="https://some-proxy.example/v1",
            openai_api_key="k",
        )
        from anton.core.llm.client import _resolve_openai_compatible_flavor

        assert (
            _resolve_openai_compatible_flavor(settings)
            == OpenAIProvider.FLAVOR_OPENAI_COMPATIBLE_GENERIC
        )

    def test_passthrough_actually_enables_native_web_tools(self):
        # The property that matters downstream — the flavor is only a means.
        from anton.core.llm.openai import OpenAIProvider

        provider = OpenAIProvider(
            api_key="k",
            base_url="https://api.mindshub.ai/v1",
            flavor=self._passthrough(),
        )
        assert provider.native_web_tools() == {"web_search", "web_fetch"}


class TestSetupErrorTextIsMarkupSafe:
    """Provider error text is interpolated into Rich markup; an unescaped
    ``[...]`` that Rich reads as a style tag raises MarkupError and crashes
    setup — the opposite of the error-surfacing this PR adds (#317 review).
    """

    def _run_setup_with_error(self, monkeypatch, detail: str):
        import anton.cli as cli

        monkeypatch.setattr(
            cli, "resolve_and_probe",
            lambda *a, **k: (
                minds_client.MindsModels(
                    planning="sonnet", coding="haiku", probe="haiku"
                ),
                minds_client.LLMTestResult(ok=False, error=detail, http_status=404),
            ),
        )
        monkeypatch.setattr(cli, "_setup_prompt", lambda *a, **k: "key")
        monkeypatch.setattr(cli.Confirm, "ask", staticmethod(lambda *a, **k: False))
        settings = AntonSettings(minds_url="https://api.mindshub.ai")
        ws = MagicMock()
        # Declining the retry exits via _SetupRetry — that's the normal control
        # flow and not what this test is about. The assertion is narrow on
        # purpose: MarkupError must never escape, whatever the provider said.
        from rich.errors import MarkupError

        try:
            cli._setup_minds(settings, ws)
        except MarkupError:
            raise
        except Exception:
            pass

    # Fixture choice matters: Rich TOLERATES an unmatched OPENING tag
    # ("[sonnet-4-6]" prints fine), so those strings prove nothing. Only a
    # CLOSING tag ("[/...]") raises — verified against rich directly. A route
    # or JSON-pointer echoed in a gateway error is exactly that shape.
    def test_route_in_error_text_does_not_crash_setup(self, monkeypatch):
        self._run_setup_with_error(
            monkeypatch, "model_not_found: unknown route [/v1/chat/completions]"
        )

    def test_json_pointer_in_error_text_does_not_crash_setup(self, monkeypatch):
        self._run_setup_with_error(
            monkeypatch, "validation error at [/model] — field required"
        )


class TestResolveAndProbeEscalation:
    """The invariant: never persist a pair no probe covered (#317 re-review).

    On the no-catalogue branch the paid default is probed first. A model-access
    denial escalates ONCE to the free bucket and returns the free pair — so a
    free-tier key is neither blocked (ENG-576) nor handed an unvalidated paid
    pair that 403s on the first real request.
    """

    def _probes(self, monkeypatch, results: dict):
        """Stub the catalogue away (no-catalogue branch) and script the probes."""
        monkeypatch.setattr("anton.minds_client.list_models", lambda *a, **k: [])
        seen: list[str] = []

        def fake_test_llm(base_url, api_key, verify=True, model=None):
            seen.append(model)
            return results[model]

        monkeypatch.setattr("anton.minds_client.test_llm", fake_test_llm)
        return seen

    def test_paid_probe_passing_persists_the_paid_pair(self, monkeypatch):
        seen = self._probes(
            monkeypatch, {"haiku": minds_client.LLMTestResult(ok=True)}
        )
        models, result = minds_client.resolve_and_probe("https://x", "k")
        assert result.ok
        assert (models.planning, models.coding) == ("sonnet", "haiku")
        assert seen == ["haiku"], "no needless second probe"

    def test_denied_paid_probe_escalates_and_persists_the_free_pair(self, monkeypatch):
        seen = self._probes(monkeypatch, {
            "haiku": minds_client.LLMTestResult(
                ok=False, error="model_access_denied", http_status=403
            ),
            "mindshub_air": minds_client.LLMTestResult(ok=True),
        })
        models, result = minds_client.resolve_and_probe("https://x", "k")
        assert result.ok
        # The pair PERSISTED is the one that was validated — the whole point.
        assert models.planning == models.coding == minds_client.MINDS_FREE_TIER_MODEL
        assert models.probe == minds_client.MINDS_FREE_TIER_MODEL
        assert seen == ["haiku", "mindshub_air"]

    def test_model_not_found_also_escalates(self, monkeypatch):
        self._probes(monkeypatch, {
            "haiku": minds_client.LLMTestResult(
                ok=False, error="model_not_found", http_status=404
            ),
            "mindshub_air": minds_client.LLMTestResult(ok=True),
        })
        models, result = minds_client.resolve_and_probe("https://x", "k")
        assert result.ok and models.coding == minds_client.MINDS_FREE_TIER_MODEL

    def test_bad_key_does_not_escalate(self, monkeypatch):
        # A 401 is not an entitlement signal — escalating would burn a second
        # call and could report the wrong cause.
        seen = self._probes(monkeypatch, {
            "haiku": minds_client.LLMTestResult(
                ok=False, error="Invalid API key", http_status=401
            ),
        })
        models, result = minds_client.resolve_and_probe("https://x", "k")
        assert not result.ok and result.http_status == 401
        assert seen == ["haiku"]

    def test_rate_limit_does_not_escalate(self, monkeypatch):
        seen = self._probes(monkeypatch, {
            "haiku": minds_client.LLMTestResult(
                ok=False, rate_limited=True, error="Rate limit", http_status=429
            ),
        })
        _, result = minds_client.resolve_and_probe("https://x", "k")
        assert result.rate_limited
        assert seen == ["haiku"]

    def test_transport_failure_does_not_escalate(self, monkeypatch):
        # http_status None = never reached the server; says nothing about models.
        seen = self._probes(monkeypatch, {
            "haiku": minds_client.LLMTestResult(ok=False, error="timed out"),
        })
        _, result = minds_client.resolve_and_probe("https://x", "k")
        assert not result.ok and result.http_status is None
        assert seen == ["haiku"]

    def test_both_denied_reports_the_original_denial(self, monkeypatch):
        # The first error names the model the user asked to be set up with.
        self._probes(monkeypatch, {
            "haiku": minds_client.LLMTestResult(
                ok=False, error="paid model denied", http_status=403
            ),
            "mindshub_air": minds_client.LLMTestResult(
                ok=False, error="free model denied too", http_status=403
            ),
        })
        models, result = minds_client.resolve_and_probe("https://x", "k")
        assert result.error == "paid model denied"
        assert models.coding == "haiku"

    def test_catalogue_pair_never_escalates(self, monkeypatch):
        monkeypatch.setattr(
            "anton.minds_client.list_models", lambda *a, **k: ["sonnet", "haiku"]
        )
        seen: list[str] = []

        def fake_test_llm(base_url, api_key, verify=True, model=None):
            seen.append(model)
            return minds_client.LLMTestResult(
                ok=False, error="denied", http_status=403
            )

        monkeypatch.setattr("anton.minds_client.test_llm", fake_test_llm)
        _, result = minds_client.resolve_and_probe("https://x", "k")
        assert not result.ok
        assert seen == ["haiku"], "catalogue pair is already access-aware"


def test_setup_minds_skips_no_ssl_retry_on_http_error(monkeypatch):
    """An HTTP-level failure (server answered → TLS worked) must not trigger
    the verify=False retry — it can only repeat the same error."""
    import anton.cli as cli

    settings = AntonSettings(_env_file=None)
    workspace = MagicMock()
    probe_calls = []

    monkeypatch.setattr("anton.cli._setup_prompt", lambda *a, **kw: "mdb_test-key")
    monkeypatch.setattr("anton.cli.Confirm.ask", lambda *a, **kw: False)  # no retry
    def fake_resolve_and_probe(*a, **kw):
        probe_calls.append(kw)
        return (
            minds_client.MindsModels("sonnet", "haiku", "haiku"),
            minds_client.LLMTestResult(
                ok=False, error="model_not_found: nope", http_status=404
            ),
        )

    monkeypatch.setattr("anton.cli.resolve_and_probe", fake_resolve_and_probe)

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
        "anton.cli.resolve_and_probe",
        lambda *a, **kw: (
            minds_client.MindsModels("sonnet", "haiku", "haiku"),
            minds_client.LLMTestResult(ok=True),
        ),
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
        "anton.cli.resolve_and_probe",
        lambda *a, **kw: (
            minds_client.MindsModels("sonnet", "haiku", "haiku"),
            minds_client.LLMTestResult(ok=True),
        ),
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
    def fake_resolve_and_probe(base_url, api_key, verify=True):
        probed["base"] = minds_client.minds_v1_base(base_url)
        return (
            minds_client.MindsModels("sonnet", "haiku", "haiku"),
            minds_client.LLMTestResult(ok=True),
        )

    monkeypatch.setattr("anton.cli.resolve_and_probe", fake_resolve_and_probe)

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
