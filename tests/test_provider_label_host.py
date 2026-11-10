"""The setup summary must name the endpoint that will actually serve requests.

A MindsHub key and URL are set for anyone signed in, so deciding the label from
`minds_url` told a user pointed at a local model that they were on MindsHub —
the claim they installed a local model to be able to trust.
"""
from anton.cli import is_minds_host


def test_local_endpoint_is_not_minds_even_when_signed_in():
    assert (
        is_minds_host("http://192.168.1.100:1234/v1", "https://api.mindshub.ai")
        is False
    )


def test_public_minds_hosts():
    assert is_minds_host("https://api.mindshub.ai/v1") is True
    assert is_minds_host("https://llm.mdb.ai/api/v1") is True


def test_self_hosted_gateway_matches_by_minds_url():
    assert is_minds_host("http://gateway.internal:8080/v1", "http://gateway.internal:8080") is True


def test_same_host_different_port_is_not_the_gateway():
    assert is_minds_host("http://gateway.internal:1234/v1", "http://gateway.internal:8080") is False


def test_lookalike_host_is_not_minds():
    assert is_minds_host("https://notmindshub.ai/v1") is False


def test_unset_base_is_not_minds():
    assert is_minds_host("", "https://api.mindshub.ai") is False


# ── The label itself ────────────────────────────────────────────────────
#
# The helper tests above pin the host comparison. These pin the thing the bug
# was actually about: WHICH setting the label is read from. Reverting the call
# site to the old `minds_url` substring check turns these red.

from types import SimpleNamespace

from anton.cli import openai_compatible_label


def _cfg(base: str, minds_url: str = ""):
    return SimpleNamespace(openai_base_url=base, minds_url=minds_url)


def test_label_names_the_local_endpoint_not_mindshub():
    label = openai_compatible_label(
        _cfg("http://192.168.1.100:1234/v1", "https://api.mindshub.ai")
    )
    assert label != "MindsHub"
    assert "192.168.1.100:1234" in label


def test_label_says_mindshub_for_the_gateway():
    assert (
        openai_compatible_label(_cfg("https://api.mindshub.ai/v1", "https://api.mindshub.ai"))
        == "MindsHub"
    )


def test_label_never_claims_mindshub_without_an_endpoint_to_point_to():
    """No base URL names no endpoint, so the label must not assert one.

    A real MindsHub config does not reach here with an empty base: settings
    derive it from ``minds_url``. So this shape means the endpoint is unknown,
    and the old check answered "MindsHub" for it purely because a MindsHub URL
    was configured — the over-claim this label exists to stop.
    """
    assert openai_compatible_label(_cfg("", "https://api.mindshub.ai")) == "OpenAI-compatible"


def test_label_recognises_gemini():
    base = "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert openai_compatible_label(_cfg(base, "https://api.mindshub.ai")) == "Google Gemini"


async def test_setup_summary_uses_the_same_label(tmp_path, monkeypatch):
    """/setup must not disagree with the startup summary about the endpoint."""
    from rich.console import Console

    from anton.commands import setup

    monkeypatch.setenv("HOME", str(tmp_path))

    async def _back(*a, **kw):
        return "q"

    monkeypatch.setattr(setup, "prompt_or_cancel", _back)
    settings = SimpleNamespace(
        planning_provider="openai-compatible",
        planning_model="gemini-2.5-pro",
        coding_model="gemini-2.5-pro",
        openai_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        minds_url="https://api.mindshub.ai",
    )
    console = Console(record=True, width=200)
    session = object()
    result = await setup.handle_setup_models(console, settings, None, {}, None, None, session)
    assert result is session
    assert "Provider: Google Gemini" in console.export_text()
