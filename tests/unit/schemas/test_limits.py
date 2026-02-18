"""
Unit tests for limits schemas.

Tests the Pydantic models used for resource usage limits including:
- Default values (UNLIMITED = -1)
- Custom values and validation
- Parsing from Statsig dynamic config payloads
- Serialization / deserialization
- Independence of default_factory instances
- UsageConfig with lifetime and billing_cycle fields
"""

import pytest
from pydantic import ValidationError

from minds.schemas.limits import (
    UNLIMITED,
    LimitsConfig,
    MindLimitsConfig,
    ResourceUsageConfig,
    UsageConfig,
)


class TestUnlimitedConstant:
    """Tests for the UNLIMITED sentinel value."""

    def test_unlimited_value(self):
        assert UNLIMITED == -1, f"UNLIMITED should be -1, got {UNLIMITED}"


class TestLimitsConfig:
    """Tests for LimitsConfig model."""

    def test_defaults_are_unlimited(self):
        config = LimitsConfig()
        assert config.lifetime == UNLIMITED, f"lifetime should default to {UNLIMITED}, got {config.lifetime}"
        assert config.monthly == UNLIMITED, f"monthly should default to {UNLIMITED}, got {config.monthly}"

    def test_custom_values(self):
        config = LimitsConfig(lifetime=100, monthly=50)
        assert config.lifetime == 100, f"lifetime should be 100, got {config.lifetime}"
        assert config.monthly == 50, f"monthly should be 50, got {config.monthly}"

    def test_partial_override(self):
        config = LimitsConfig(monthly=250)
        assert config.lifetime == UNLIMITED, f"lifetime should remain {UNLIMITED}, got {config.lifetime}"
        assert config.monthly == 250, f"monthly should be 250, got {config.monthly}"

    def test_zero_is_valid(self):
        config = LimitsConfig(lifetime=0, monthly=0)
        assert config.lifetime == 0, f"lifetime should be 0, got {config.lifetime}"
        assert config.monthly == 0, f"monthly should be 0, got {config.monthly}"

    def test_string_integers_are_coerced(self):
        """Statsig returns string values; Pydantic should coerce them."""
        config = LimitsConfig(lifetime="3", monthly="250")
        assert config.lifetime == 3, f"lifetime should coerce '3' to 3, got {config.lifetime}"
        assert config.monthly == 250, f"monthly should coerce '250' to 250, got {config.monthly}"

    def test_invalid_type_raises(self):
        with pytest.raises(ValidationError, match="lifetime"):
            LimitsConfig(lifetime="not_a_number")

    def test_serialization_roundtrip(self):
        config = LimitsConfig(lifetime=10, monthly=20)
        dumped = config.model_dump()
        restored = LimitsConfig(**dumped)
        assert restored == config, f"Roundtrip failed: {restored} != {config}"

    def test_json_roundtrip(self):
        config = LimitsConfig(lifetime=5, monthly=100)
        json_str = config.model_dump_json()
        restored = LimitsConfig.model_validate_json(json_str)
        assert restored == config, f"JSON roundtrip failed: {restored} != {config}"


class TestUsageConfig:
    """Tests for UsageConfig model."""

    def test_defaults_are_zero(self):
        config = UsageConfig()
        assert config.lifetime == 0, f"lifetime should default to 0, got {config.lifetime}"
        assert config.billing_cycle == 0, f"billing_cycle should default to 0, got {config.billing_cycle}"

    def test_custom_values(self):
        config = UsageConfig(lifetime=100, billing_cycle=25)
        assert config.lifetime == 100
        assert config.billing_cycle == 25

    def test_serialization_roundtrip(self):
        config = UsageConfig(lifetime=42, billing_cycle=7)
        dumped = config.model_dump()
        restored = UsageConfig(**dumped)
        assert restored == config

    def test_independent_instances(self):
        a = UsageConfig()
        b = UsageConfig()
        a.lifetime = 99
        assert b.lifetime == 0


class TestResourceUsageConfig:
    """Tests for ResourceUsageConfig model."""

    def test_defaults(self):
        config = ResourceUsageConfig()
        assert config.limit.lifetime == UNLIMITED, f"limit.lifetime should be {UNLIMITED}, got {config.limit.lifetime}"
        assert config.limit.monthly == UNLIMITED, f"limit.monthly should be {UNLIMITED}, got {config.limit.monthly}"
        assert config.usage.lifetime == 0, f"usage.lifetime should default to 0, got {config.usage.lifetime}"
        assert config.usage.billing_cycle == 0, (
            f"usage.billing_cycle should default to 0, got {config.usage.billing_cycle}"
        )

    def test_custom_limit_and_usage(self):
        config = ResourceUsageConfig(
            limit=LimitsConfig(lifetime=10, monthly=5),
            usage=UsageConfig(lifetime=3, billing_cycle=1),
        )
        assert config.limit.lifetime == 10, f"limit.lifetime should be 10, got {config.limit.lifetime}"
        assert config.limit.monthly == 5, f"limit.monthly should be 5, got {config.limit.monthly}"
        assert config.usage.lifetime == 3, f"usage.lifetime should be 3, got {config.usage.lifetime}"
        assert config.usage.billing_cycle == 1, f"usage.billing_cycle should be 1, got {config.usage.billing_cycle}"

    def test_nested_dict_parsing(self):
        """Simulate the shape that comes from Statsig dynamic config."""
        raw = {"limit": {"lifetime": "1", "monthly": "1"}}
        config = ResourceUsageConfig(**raw)
        assert config.limit.lifetime == 1, f"limit.lifetime should coerce '1' to 1, got {config.limit.lifetime}"
        assert config.limit.monthly == 1, f"limit.monthly should coerce '1' to 1, got {config.limit.monthly}"
        assert config.usage.lifetime == 0, "usage.lifetime should default to 0 when absent"
        assert config.usage.billing_cycle == 0, "usage.billing_cycle should default to 0 when absent"

    def test_default_factory_returns_independent_instances(self):
        """Each default should be a fresh object, not shared."""
        a = ResourceUsageConfig()
        b = ResourceUsageConfig()
        a.usage.lifetime = 99
        assert b.usage.lifetime == 0, f"b.usage.lifetime should be 0 (independent instance), got {b.usage.lifetime}"

    def test_serialization_roundtrip(self):
        config = ResourceUsageConfig(
            limit=LimitsConfig(lifetime=3, monthly=3),
            usage=UsageConfig(lifetime=2, billing_cycle=1),
        )
        dumped = config.model_dump()
        restored = ResourceUsageConfig(**dumped)
        assert restored == config, f"Roundtrip failed: {restored} != {config}"


def _resource_fields() -> list[str]:
    """Return all field names on MindLimitsConfig that are ResourceUsageConfig.

    Tests use this to auto-discover resource fields, so adding a new one
    (e.g. 'embeddings') is automatically covered without updating tests.
    """
    return [
        name
        for name, field_info in MindLimitsConfig.model_fields.items()
        if field_info.annotation is ResourceUsageConfig
    ]


# Known resource types — if a new one is added to the model but not here, the
# test_no_unexpected_resource_fields / test_expected_resource_fields tests will fail.
EXPECTED_RESOURCE_FIELDS = {"tokens", "minds", "datasources", "questions"}


class TestMindLimitsConfig:
    """Tests for MindLimitsConfig model."""

    def test_expected_resource_fields_present(self):
        """Every expected resource field must exist on the model."""
        actual = set(_resource_fields())
        missing = EXPECTED_RESOURCE_FIELDS - actual
        assert not missing, f"Expected resource fields missing from MindLimitsConfig: {missing}"

    def test_no_unexpected_resource_fields(self):
        """Catch new ResourceUsageConfig fields that need test coverage."""
        actual = set(_resource_fields())
        unexpected = actual - EXPECTED_RESOURCE_FIELDS
        assert not unexpected, (
            f"New ResourceUsageConfig fields detected on MindLimitsConfig: {unexpected}. "
            f"Add them to EXPECTED_RESOURCE_FIELDS and update STATSIG_PAYLOAD in "
            f"test_mind_limits.py so they are covered by tests."
        )

    def test_defaults_are_all_unlimited(self):
        config = MindLimitsConfig()
        for resource in _resource_fields():
            section = getattr(config, resource)
            assert section.limit.lifetime == UNLIMITED, f"{resource}.limit.lifetime should be unlimited"
            assert section.limit.monthly == UNLIMITED, f"{resource}.limit.monthly should be unlimited"
            assert section.usage.lifetime == 0, f"{resource}.usage.lifetime should default to 0"
            assert section.usage.billing_cycle == 0, f"{resource}.usage.billing_cycle should default to 0"

    def test_from_statsig_payload(self):
        """Parse a real Statsig dynamic config value dict."""
        statsig_value = {
            "tokens": {"limit": {"lifetime": "-1", "monthly": "1000000"}},
            "minds": {"limit": {"lifetime": "1", "monthly": "1"}},
            "datasources": {"limit": {"lifetime": "3", "monthly": "3"}},
            "questions": {"limit": {"lifetime": "-1", "monthly": "250"}},
        }
        config = MindLimitsConfig(**statsig_value)

        assert config.tokens.limit.lifetime == -1, (
            f"tokens.limit.lifetime should be -1, got {config.tokens.limit.lifetime}"
        )
        assert config.tokens.limit.monthly == 1_000_000, (
            f"tokens.limit.monthly should be 1000000, got {config.tokens.limit.monthly}"
        )
        assert config.minds.limit.lifetime == 1, f"minds.limit.lifetime should be 1, got {config.minds.limit.lifetime}"
        assert config.minds.limit.monthly == 1, f"minds.limit.monthly should be 1, got {config.minds.limit.monthly}"
        assert config.datasources.limit.lifetime == 3, (
            f"datasources.limit.lifetime should be 3, got {config.datasources.limit.lifetime}"
        )
        assert config.datasources.limit.monthly == 3, (
            f"datasources.limit.monthly should be 3, got {config.datasources.limit.monthly}"
        )
        assert config.questions.limit.lifetime == -1, (
            f"questions.limit.lifetime should be -1, got {config.questions.limit.lifetime}"
        )
        assert config.questions.limit.monthly == 250, (
            f"questions.limit.monthly should be 250, got {config.questions.limit.monthly}"
        )

        # Usage defaults to zero when not in payload
        for resource in _resource_fields():
            section = getattr(config, resource)
            assert section.usage.lifetime == 0, (
                f"{resource}.usage.lifetime should default to 0 when absent from payload"
            )
            assert section.usage.billing_cycle == 0, (
                f"{resource}.usage.billing_cycle should default to 0 when absent from payload"
            )

    def test_partial_statsig_payload(self):
        """Only some resource types present — others stay at defaults."""
        config = MindLimitsConfig(
            tokens={"limit": {"lifetime": "-1", "monthly": "500000"}},
        )
        assert config.tokens.limit.monthly == 500_000, (
            f"tokens.limit.monthly should be 500000, got {config.tokens.limit.monthly}"
        )
        # Others remain unlimited
        for resource in _resource_fields():
            if resource == "tokens":
                continue
            section = getattr(config, resource)
            assert section.limit.monthly == UNLIMITED, f"{resource}.limit.monthly should remain unlimited"

    def test_default_factory_instances_are_independent(self):
        a = MindLimitsConfig()
        b = MindLimitsConfig()
        a.tokens.usage.lifetime = 42
        assert b.tokens.usage.lifetime == 0, (
            f"b.tokens.usage.lifetime should be 0 (independent instance), got {b.tokens.usage.lifetime}"
        )

    def test_serialization_roundtrip(self):
        config = MindLimitsConfig(
            tokens=ResourceUsageConfig(
                limit=LimitsConfig(lifetime=-1, monthly=1_000_000),
                usage=UsageConfig(lifetime=500, billing_cycle=100),
            ),
            minds=ResourceUsageConfig(
                limit=LimitsConfig(lifetime=1, monthly=1),
                usage=UsageConfig(lifetime=1, billing_cycle=1),
            ),
        )
        dumped = config.model_dump()
        restored = MindLimitsConfig(**dumped)
        assert restored == config, f"Roundtrip failed: {restored} != {config}"

    def test_json_serialization(self):
        config = MindLimitsConfig()
        json_str = config.model_dump_json()
        restored = MindLimitsConfig.model_validate_json(json_str)
        assert restored == config, f"JSON roundtrip failed: {restored} != {config}"

    def test_invalid_nested_type_raises(self):
        with pytest.raises(ValidationError, match="tokens"):
            MindLimitsConfig(tokens="not_a_dict")
