"""Tests for the reasoning-effort capability catalog."""

import pytest
from fastapi import HTTPException

from minds.inference.effort import EffortCapability, get_effort_capability, validate_effort


class TestGetEffortCapability:
    """Longest-prefix lookup against the in-code catalog."""

    def test_exact_entry_beats_family_fallback(self):
        # claude-opus-4-6 has its own entry (no xhigh) even though the
        # claude-opus- family fallback also matches.
        cap = get_effort_capability("claude-opus-4-6")
        assert cap.levels == ("low", "medium", "high", "max")

    def test_dated_snapshot_matches_exact_prefix(self):
        cap = get_effort_capability("claude-opus-4-8-20270101")
        assert cap.levels == ("low", "medium", "high", "xhigh", "max")

    def test_unknown_model_version_gets_family_fallback(self):
        # "New model day": a version newer than the catalog still resolves to
        # the conservative family baseline.
        cap = get_effort_capability("claude-opus-4-9")
        assert cap.levels == ("low", "medium", "high")
        assert cap.default == "high"

    def test_explicit_unsupported_entry_shadows_family_fallback(self):
        cap = get_effort_capability("claude-sonnet-4-5")
        assert cap is not None
        assert not cap.supported

    def test_haiku_unsupported(self):
        cap = get_effort_capability("claude-haiku-4-5-20251001")
        assert cap is not None
        assert not cap.supported

    def test_unknown_model_returns_none(self):
        assert get_effort_capability("totally-unknown-model") is None

    def test_openai_levels(self):
        assert get_effort_capability("gpt-5.5-2026-04-23").levels == ("none", "low", "medium", "high", "xhigh")
        assert get_effort_capability("gpt-5.5-2026-04-23").default == "medium"

    def test_openai_family_fallback_for_future_version(self):
        cap = get_effort_capability("gpt-5.9")
        assert cap.levels == ("low", "medium", "high")

    def test_fireworks_deepseek_full_ladder(self):
        cap = get_effort_capability("accounts/fireworks/models/deepseek-v4-pro")
        assert cap.levels == ("none", "low", "medium", "high", "xhigh", "max")

    def test_fireworks_kimi_has_no_entry(self):
        assert get_effort_capability("accounts/fireworks/models/kimi-k2p6") is None


class TestEffortOverrides:
    """Statsig override layer merged over the in-code catalog."""

    def test_override_beats_catalog_on_same_prefix(self):
        overrides = {"claude-opus-4-8": EffortCapability(("low", "high"), "low")}
        cap = get_effort_capability("claude-opus-4-8", overrides)
        assert cap.levels == ("low", "high")
        assert cap.default == "low"

    def test_longer_override_beats_shorter_catalog_entry(self):
        overrides = {"claude-opus-4-9": EffortCapability(("low", "medium", "high", "xhigh", "max"), "high")}
        cap = get_effort_capability("claude-opus-4-9", overrides)
        assert "xhigh" in cap.levels

    def test_shorter_override_does_not_beat_longer_catalog_entry(self):
        # An override on the family prefix must not shadow a more-specific
        # in-code entry.
        overrides = {"claude-opus-": EffortCapability(("low",), "low")}
        cap = get_effort_capability("claude-opus-4-8", overrides)
        assert cap.levels == ("low", "medium", "high", "xhigh", "max")

    def test_override_enables_unknown_model(self):
        overrides = {"acme-llm-1": EffortCapability(("low", "turbo"), "turbo")}
        cap = get_effort_capability("acme-llm-1-preview", overrides)
        assert cap.levels == ("low", "turbo")

    def test_empty_levels_override_is_kill_switch(self):
        overrides = {"gpt-5.5": EffortCapability((), None)}
        cap = get_effort_capability("gpt-5.5", overrides)
        assert not cap.supported


class TestValidateEffort:
    def test_valid_effort_passes(self):
        validate_effort("xhigh", "claude-opus-4-8", "opus")

    def test_invalid_level_raises_400_listing_allowed(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_effort("xhigh", "claude-sonnet-4-6", "sonnet")
        assert exc_info.value.status_code == 400
        assert "low, medium, high, max" in exc_info.value.detail
        assert "latest:sonnet" in exc_info.value.detail

    def test_unsupported_model_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_effort("low", "claude-haiku-4-5", "haiku")
        assert exc_info.value.status_code == 400
        assert "does not support reasoning effort" in exc_info.value.detail

    def test_unknown_model_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_effort("low", "mystery-model", "mystery")
        assert exc_info.value.status_code == 400

    def test_opaque_new_level_accepted_via_override(self):
        # A level name the code has never seen ships via Statsig alone.
        overrides = {"gpt-5.6": EffortCapability(("none", "low", "medium", "high", "xhigh", "max"), "medium")}
        validate_effort("max", "gpt-5.6", "gpt", overrides)
