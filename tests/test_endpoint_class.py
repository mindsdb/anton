"""`endpoint_class` — where a turn's tokens actually went (ENG-1689).

The defect these cover is not arithmetic, it is *which field gets consulted*.
`llm_provider` reported the emitting host's vocabulary rather than the
destination, so our own gateway arrived under two labels and a gateway-share
query ran 18% low. The tests therefore pin three things: the host partition,
the provider gate in front of it (a vendor-fixed provider must never be
classified from a base URL it does not read), and the emit-site seam.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from anton.core.llm.endpoints import (
    ENDPOINT_LOCAL,
    ENDPOINT_MINDSHUB,
    ENDPOINT_THIRD_PARTY,
    ENDPOINT_UNKNOWN,
    VALID_ENDPOINT_CLASSES,
    classify_base_url,
    classify_endpoint,
)


def _settings(provider: str, base_url: str = "") -> SimpleNamespace:
    """Only the two attributes `classify_endpoint` reads."""
    return SimpleNamespace(planning_provider=provider, openai_base_url=base_url)


# ─── the host partition ──────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "base_url, expected",
    [
        # Loopback, in the shapes people actually paste.
        ("http://localhost:11434/v1", ENDPOINT_LOCAL),
        ("http://127.0.0.1:1234/v1", ENDPOINT_LOCAL),
        ("http://[::1]:8080/v1", ENDPOINT_LOCAL),
        ("http://0.0.0.0:11434/v1", ENDPOINT_LOCAL),
        # No scheme — `urlsplit` parses this as a path and finds no host
        # unless `//` is prepended, which is why `_host_of` does that.
        ("127.0.0.1:11434", ENDPOINT_LOCAL),
        # Private / LAN. An Ollama box on the network is still local, and
        # calling it `third-party` would understate exactly what we measure.
        ("http://192.168.1.40:11434/v1", ENDPOINT_LOCAL),
        ("http://10.0.0.7:8000/v1", ENDPOINT_LOCAL),
        ("http://172.16.5.5:8000/v1", ENDPOINT_LOCAL),
        ("http://169.254.10.1:8000/v1", ENDPOINT_LOCAL),
        ("http://mac-studio.local:11434/v1", ENDPOINT_LOCAL),
        # Our gateway, in all three shapes `model_post_init` can build.
        ("https://api.mindshub.ai/v1", ENDPOINT_MINDSHUB),
        ("https://mdb.ai/api/v1", ENDPOINT_MINDSHUB),
        ("https://mindshub.ai", ENDPOINT_MINDSHUB),
        ("https://api.mindshub.ai/v1/", ENDPOINT_MINDSHUB),
        ("HTTPS://API.MINDSHUB.AI/v1", ENDPOINT_MINDSHUB),
        # Somebody else's.
        ("https://openrouter.ai/api/v1", ENDPOINT_THIRD_PARTY),
        ("https://api.openai.com/v1", ENDPOINT_THIRD_PARTY),
        ("https://acme-corp.databricks.com/serving-endpoints", ENDPOINT_THIRD_PARTY),
        # Nothing configured still reaches a third party: the OpenAI SDK
        # defaults to api.openai.com, so `third-party` is the true answer
        # here rather than a fallback.
        ("", ENDPOINT_THIRD_PARTY),
        ("   ", ENDPOINT_THIRD_PARTY),
        ("not a url at all", ENDPOINT_THIRD_PARTY),
    ],
)
def test_host_partition(base_url, expected):
    assert classify_base_url(base_url) == expected


def test_api_mindshub_v1_is_recognised_the_shape_the_old_helper_missed():
    """`cli.py`'s `_looks_like_mdb_ai` matches `{minds_url}` and
    `{minds_url}/api/v1` but not `{minds_url}/v1` — which is precisely what
    `model_post_init` builds for `https://api.mindshub.ai`, our production
    gateway. Building this property on that helper would have reproduced the
    bug in a new field (ENG-1695).
    """
    assert classify_base_url("https://api.mindshub.ai/v1") == ENDPOINT_MINDSHUB


@pytest.mark.parametrize(
    "hostile",
    [
        "https://mindshub.ai.attacker.example/v1",
        "https://notmindshub.ai/v1",
        "https://mdb.ai.evil.test/v1",
        "https://fake-mdb.ai.co/v1",
    ],
)
def test_domain_match_is_not_a_substring(hostile):
    """`LLMClient`'s vision gate uses `"mindshub.ai" in host`, which these
    satisfy. This property matches on exact host or dot-delimited subdomain,
    so a lookalike cannot inflate gateway share.
    """
    assert classify_base_url(hostile) == ENDPOINT_THIRD_PARTY


# ─── the provider gate in front of the host partition ────────────────────────

def test_minds_cloud_is_the_gateway_whatever_the_base_url_says():
    """The desktop DB overlay writes `minds-cloud` via `setattr`, bypassing
    the validator. It means the gateway by definition, so the URL is not
    consulted.
    """
    assert classify_endpoint(_settings("minds-cloud", "")) == ENDPOINT_MINDSHUB
    assert (
        classify_endpoint(_settings("minds-cloud", "http://localhost:1234/v1"))
        == ENDPOINT_MINDSHUB
    )


def test_vendor_fixed_provider_is_never_classified_from_a_stale_base_url():
    """The trap this gate exists for. `model_post_init` derives
    `openai_base_url` from `minds_url` — so the field can be set and pointed
    at us while an `anthropic` turn goes to api.anthropic.com and never
    touches our gateway. Reading the URL here would invent gateway traffic.
    """
    stale = _settings("anthropic", "https://api.mindshub.ai/v1")
    assert classify_endpoint(stale) == ENDPOINT_THIRD_PARTY
    assert classify_endpoint(_settings("gemini", "https://api.mindshub.ai/v1")) == (
        ENDPOINT_THIRD_PARTY
    )


@pytest.mark.parametrize("provider", ["openai", "openai-compatible"])
def test_url_reading_providers_are_classified_from_the_url(provider):
    """Both of these are constructed with `base_url=settings.openai_base_url`
    in `LLMClient.from_settings`, so both can legitimately point at us.
    Omitting plain `openai` would misfile a user who does.
    """
    assert classify_endpoint(_settings(provider, "https://api.mindshub.ai/v1")) == (
        ENDPOINT_MINDSHUB
    )
    assert classify_endpoint(_settings(provider, "http://localhost:11434/v1")) == (
        ENDPOINT_LOCAL
    )
    assert classify_endpoint(_settings(provider, "https://openrouter.ai/api/v1")) == (
        ENDPOINT_THIRD_PARTY
    )


def test_unrecognised_provider_is_absent_not_guessed():
    """A provider nobody has taught this module about must not be assigned a
    value: absent is queryable as absent, a wrong label is not.
    """
    assert classify_endpoint(_settings("some-future-provider", "")) == ENDPOINT_UNKNOWN
    assert classify_endpoint(_settings("", "")) == ENDPOINT_UNKNOWN


def test_case_and_whitespace_in_the_provider_do_not_defeat_the_gate():
    assert classify_endpoint(_settings("  Minds-Cloud  ", "")) == ENDPOINT_MINDSHUB


def test_never_raises_on_a_hostile_settings_object():
    """Called from the analytics path, which must never disturb the turn."""

    class _Exploding:
        @property
        def planning_provider(self):  # pragma: no cover - raises by design
            raise RuntimeError("settings blew up")

    assert classify_endpoint(_Exploding()) == ENDPOINT_UNKNOWN


# ─── the privacy constraint ──────────────────────────────────────────────────

@pytest.mark.parametrize(
    "base_url",
    [
        "https://llm.internal.acme-corp.example/v1",
        "http://gpu-box-7.eng.acme.internal:8000/v1",
        "https://api.mindshub.ai/v1",
        "http://192.168.1.40:11434/v1",
    ],
)
def test_no_hostname_ever_leaves_the_classifier(base_url):
    """Internal corporate hostnames must not land in analytics — the reason
    this is a class rather than the raw URL.
    """
    result = classify_base_url(base_url)
    # The label is one of a closed set of three, so nothing host-derived can
    # ride along. Checked structurally rather than by forbidden substrings:
    # the label "mindshub" legitimately contains a domain fragment, so a
    # substring blocklist would either fail on it or have to special-case it.
    assert result in VALID_ENDPOINT_CLASSES
    assert not any(ch.isdigit() for ch in result), "no address octet survives"
    for structural in (".", ":", "/", "acme", "gpu-box", "internal", "example"):
        assert structural not in result


# ─── the real settings object, not a stub ────────────────────────────────────

@pytest.mark.parametrize(
    "minds_url, expected",
    [
        ("https://api.mindshub.ai", ENDPOINT_MINDSHUB),  # derives /v1
        ("https://mdb.ai", ENDPOINT_MINDSHUB),           # derives /api/v1
        ("https://api.mindshub.ai/v1", ENDPOINT_MINDSHUB),  # kept as-is
    ],
)
def test_all_three_derived_url_shapes_classify_as_the_gateway(minds_url, expected):
    """Exercised through the real `AntonSettings.model_post_init` derivation
    rather than hand-written URLs, so a change to that derivation surfaces
    here instead of silently reclassifying gateway traffic.
    """
    from anton.config.settings import AntonSettings

    settings = AntonSettings(
        minds_url=minds_url,
        minds_api_key="mdb_test_key",
        openai_api_key=None,
        planning_provider="openai-compatible",
        coding_provider="openai-compatible",
    )
    assert settings.openai_base_url, "derivation did not run — test is vacuous"
    assert classify_endpoint(settings) == expected


# ─── the emit-site seam ──────────────────────────────────────────────────────

def test_emit_site_actually_sends_both_new_properties():
    """A call-site guard, not a helper test. Deleting the emit-site argument
    for a comparable property once passed the entire 1,448-test suite, because
    no test drove that path — so the seam is pinned structurally.

    AST rather than a substring search: a mention in a comment or a stale
    import must not satisfy it. The keywords have to be on the `send_event`
    call inside `_emit_turn_cost`.
    """
    src = Path(
        __import__("anton.core.session", fromlist=["session"]).__file__
    ).read_text()
    emit = next(
        (
            n
            for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.FunctionDef) and n.name == "_emit_turn_cost"
        ),
        None,
    )
    assert emit is not None, "_emit_turn_cost not found — rename?"

    send_calls = [
        n
        for n in ast.walk(emit)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "send_event"
    ]
    assert send_calls, "no send_event call inside _emit_turn_cost"
    keywords = {kw.arg for call in send_calls for kw in call.keywords}
    assert "endpoint_class" in keywords, "endpoint_class is not emitted"
    assert "error_type" in keywords, "error_type is not emitted"


def test_the_seam_guard_can_fail():
    """Positive control for the guard above: the same AST predicate run
    against a source that omits the keyword must report absence. Without this
    a bug in the predicate would make the guard silently vacuous.
    """
    src = (
        "def _emit_turn_cost(self):\n"
        "    send_event(settings, 'turn_completed', ended_by='x')\n"
    )
    emit = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "_emit_turn_cost"
    )
    keywords = {
        kw.arg
        for n in ast.walk(emit)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "send_event"
        for kw in n.keywords
    }
    assert "endpoint_class" not in keywords
    assert "ended_by" in keywords
