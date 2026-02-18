"""
Unit tests for Statsig user builder.

Tests building a StatsigUser from request context.
"""

from unittest.mock import patch
from uuid import UUID

from minds.common.constants import (
    CONTEXT_FIELD_ENV,
    CONTEXT_FIELD_ORGANIZATION_ID,
    CONTEXT_FIELD_USER_ID,
    CONTEXT_FIELD_USER_ROLES,
)
from minds.common.settings.app_settings import AppSettings
from minds.common.statsig.users import build_statsig_user
from minds.requests.context import Context


def _make_context(**overrides) -> Context:
    defaults = {
        "user_id": UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
        "organization_id": UUID("11111111-2222-3333-4444-555555555555"),
        "user_email": "alice@example.com",
        "user_roles": ["pro", "admin"],
    }
    defaults.update(overrides)
    return Context(**defaults)


class TestBuildStatsigUser:
    """Tests for build_statsig_user."""

    def test_user_id_is_set(self):
        ctx = _make_context()
        user = build_statsig_user(context=ctx, settings=AppSettings())
        assert user.user_id == str(ctx.user_id), f"user_id should be '{ctx.user_id}', got '{user.user_id}'"

    def test_email_is_set(self):
        ctx = _make_context()
        user = build_statsig_user(context=ctx, settings=AppSettings())
        assert user.email == ctx.user_email, f"email should be '{ctx.user_email}', got '{user.email}'"

    def test_custom_ids_contain_organization(self):
        ctx = _make_context()
        user = build_statsig_user(context=ctx, settings=AppSettings())
        assert user.custom_ids[CONTEXT_FIELD_ORGANIZATION_ID] == str(ctx.organization_id), (
            f"custom_ids['{CONTEXT_FIELD_ORGANIZATION_ID}'] should be '{ctx.organization_id}', "
            f"got '{user.custom_ids.get(CONTEXT_FIELD_ORGANIZATION_ID)}'"
        )

    def test_custom_fields(self):
        ctx = _make_context()
        settings = AppSettings(env="staging")
        user = build_statsig_user(context=ctx, settings=settings)

        assert user.custom[CONTEXT_FIELD_USER_ID] == str(ctx.user_id), (
            f"custom['{CONTEXT_FIELD_USER_ID}'] should be '{ctx.user_id}', "
            f"got '{user.custom.get(CONTEXT_FIELD_USER_ID)}'"
        )
        assert user.custom[CONTEXT_FIELD_ORGANIZATION_ID] == str(ctx.organization_id), (
            f"custom['{CONTEXT_FIELD_ORGANIZATION_ID}'] should be '{ctx.organization_id}', "
            f"got '{user.custom.get(CONTEXT_FIELD_ORGANIZATION_ID)}'"
        )
        assert user.custom[CONTEXT_FIELD_ENV] == "staging", (
            f"custom['{CONTEXT_FIELD_ENV}'] should be 'staging', got '{user.custom.get(CONTEXT_FIELD_ENV)}'"
        )
        assert user.custom[CONTEXT_FIELD_USER_ROLES] == ["pro", "admin"], (
            f"custom['{CONTEXT_FIELD_USER_ROLES}'] should be ['pro', 'admin'], "
            f"got {user.custom.get(CONTEXT_FIELD_USER_ROLES)}"
        )

    def test_uses_default_settings_when_none(self):
        """When settings=None is passed, it falls back to get_app_settings()."""
        ctx = _make_context()
        with patch("minds.common.statsig.users.get_app_settings", return_value=AppSettings(env="test")):
            user = build_statsig_user(context=ctx, settings=None)
        assert user.custom[CONTEXT_FIELD_ENV] == "test", (
            f"custom['{CONTEXT_FIELD_ENV}'] should fall back to 'test', got '{user.custom.get(CONTEXT_FIELD_ENV)}'"
        )
