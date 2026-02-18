"""
Unit tests for AppSettings, focusing on DeploymentMode.

Tests:
- DeploymentMode enum values
- Default deployment mode is self-hosted
- Parsing deployment mode from environment variables
"""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from minds.common.settings.app_settings import AppSettings, DeploymentMode


class TestDeploymentMode:
    """Tests for the DeploymentMode enum."""

    def test_enum_values(self):
        assert DeploymentMode.SELF_HOSTED.value == "self-hosted", "SELF_HOSTED value should be 'self-hosted'"
        assert DeploymentMode.CLOUD.value == "cloud", "CLOUD value should be 'cloud'"

    def test_enum_members_count(self):
        assert len(DeploymentMode) == 2, f"Expected 2 DeploymentMode members, got {len(DeploymentMode)}"

    def test_is_string_enum(self):
        """DeploymentMode inherits from str so it compares naturally with strings."""
        assert DeploymentMode.SELF_HOSTED == "self-hosted", "SELF_HOSTED should compare equal to 'self-hosted'"
        assert DeploymentMode.CLOUD == "cloud", "CLOUD should compare equal to 'cloud'"


class TestAppSettingsDeploymentMode:
    """Tests for deployment_mode in AppSettings."""

    def test_default_is_self_hosted(self):
        settings = AppSettings()
        assert settings.deployment_mode == DeploymentMode.SELF_HOSTED, (
            f"Default deployment_mode should be SELF_HOSTED, got {settings.deployment_mode}"
        )

    @patch.dict(os.environ, {"DEPLOYMENT_MODE": "cloud"}, clear=False)
    def test_cloud_from_env(self):
        settings = AppSettings()
        assert settings.deployment_mode == DeploymentMode.CLOUD, (
            f"deployment_mode should be CLOUD when DEPLOYMENT_MODE=cloud, got {settings.deployment_mode}"
        )

    @patch.dict(os.environ, {"DEPLOYMENT_MODE": "self-hosted"}, clear=False)
    def test_self_hosted_from_env(self):
        settings = AppSettings()
        assert settings.deployment_mode == DeploymentMode.SELF_HOSTED, (
            f"deployment_mode should be SELF_HOSTED when DEPLOYMENT_MODE=self-hosted, got {settings.deployment_mode}"
        )

    @patch.dict(os.environ, {"DEPLOYMENT_MODE": "invalid"}, clear=False)
    def test_invalid_deployment_mode_raises(self):
        with pytest.raises(ValidationError, match="deployment_mode"):
            AppSettings()
