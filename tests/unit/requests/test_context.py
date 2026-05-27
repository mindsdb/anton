from datetime import datetime, timezone
from unittest.mock import Mock
from uuid import UUID

import pytest
from fastapi import HTTPException, Request

from minds.common.constants import (
    HEADER_BILLING_PERIOD_END,
    HEADER_BILLING_PERIOD_START,
    HEADER_ORGANIZATION_ID,
    HEADER_USER_EMAIL,
    HEADER_USER_ID,
    HEADER_USER_ROLES,
)
from minds.requests.context import (
    Context,
    LangfuseContext,
    LangfuseContextMetadata,
    create_langfuse_context,
    extract_context_from_request,
)


class TestContext:
    """Test cases for Context model."""

    def test_context_initialization_with_defaults(self):
        """Test Context initialization with default values."""
        context = Context()

        assert context.request_id == UUID("00000000-0000-0000-0000-000000000000")
        assert context.user_id == UUID("00000000-0000-0000-0000-000000000000")
        assert context.organization_id == UUID("00000000-0000-0000-0000-000000000000")
        assert context.user_email == ""
        assert context.user_roles == []
        assert context.billing_cycle_start is None
        assert context.billing_cycle_end is None

    def test_context_initialization_with_values(self):
        """Test Context initialization with provided values."""
        context = Context(
            request_id=UUID("00000000-0000-0000-0000-000000000099"),
            user_id=UUID("00000000-0000-0000-0000-000000000001"),
            organization_id=UUID("00000000-0000-0000-0000-000000000002"),
            user_email="user@example.com",
            user_roles=["admin", "editor"],
        )

        assert context.request_id == UUID("00000000-0000-0000-0000-000000000099")
        assert context.user_id == UUID("00000000-0000-0000-0000-000000000001")
        assert context.organization_id == UUID("00000000-0000-0000-0000-000000000002")
        assert context.user_email == "user@example.com"
        assert context.user_roles == ["admin", "editor"]

    def test_context_is_basemodel(self):
        """Test that Context inherits from BaseModel."""
        context = Context()

        # Should have BaseModel methods
        assert hasattr(context, "model_dump")
        assert hasattr(context, "model_validate")


class TestExtractContextFromRequest:
    """Test cases for extract_context_from_request function."""

    def test_extract_context_with_all_headers(self):
        """Test extracting context when all headers are present."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            HEADER_USER_ID: "00000000-0000-0000-0000-000000000001",
            HEADER_ORGANIZATION_ID: "00000000-0000-0000-0000-000000000002",
            HEADER_USER_EMAIL: "user@example.com",
            HEADER_USER_ROLES: "admin,editor",
        }

        context = extract_context_from_request(mock_request)

        assert context.user_id == UUID("00000000-0000-0000-0000-000000000001")
        assert context.organization_id == UUID("00000000-0000-0000-0000-000000000002")
        assert context.user_email == "user@example.com"
        assert context.user_roles == ["admin", "editor"]
        # request_id should be a valid UUID (auto-generated)
        assert isinstance(context.request_id, UUID)

    def test_extract_context_with_single_role(self):
        """Test extracting context when a single role is provided."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            HEADER_USER_ID: "00000000-0000-0000-0000-000000000001",
            HEADER_ORGANIZATION_ID: "00000000-0000-0000-0000-000000000002",
            HEADER_USER_EMAIL: "user@example.com",
            HEADER_USER_ROLES: "viewer",
        }

        context = extract_context_from_request(mock_request)

        assert context.user_roles == ["viewer"]

    def test_extract_context_with_empty_roles(self):
        """Test extracting context when roles header is empty."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            HEADER_USER_ID: "00000000-0000-0000-0000-000000000001",
            HEADER_ORGANIZATION_ID: "00000000-0000-0000-0000-000000000002",
            HEADER_USER_EMAIL: "user@example.com",
            HEADER_USER_ROLES: "",
        }

        context = extract_context_from_request(mock_request)

        assert context.user_roles == []

    def test_extract_context_with_missing_user_id_raises_http_exception(self):
        """Test that a missing user ID header raises HTTPException."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            HEADER_USER_ID: None,
            HEADER_ORGANIZATION_ID: "00000000-0000-0000-0000-000000000002",
        }

        with pytest.raises(HTTPException) as exc_info:
            extract_context_from_request(mock_request)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Unauthorized"

    def test_extract_context_with_missing_organization_id_raises_http_exception(self):
        """Test that a missing organization ID header raises HTTPException."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            HEADER_USER_ID: "00000000-0000-0000-0000-000000000001",
            HEADER_ORGANIZATION_ID: None,
        }

        with pytest.raises(HTTPException) as exc_info:
            extract_context_from_request(mock_request)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Unauthorized"

    def test_extract_context_with_both_missing_raises_http_exception(self):
        """Test that missing both headers raises HTTPException."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {HEADER_USER_ID: None, HEADER_ORGANIZATION_ID: None}

        with pytest.raises(HTTPException) as exc_info:
            extract_context_from_request(mock_request)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Unauthorized"

    def test_extract_context_with_invalid_user_id(self):
        """Test extracting context when user ID is not a valid UUID."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            HEADER_USER_ID: "not-a-uuid",
            HEADER_ORGANIZATION_ID: "00000000-0000-0000-0000-000000000002",
            HEADER_USER_EMAIL: "user@example.com",
            HEADER_USER_ROLES: "",
        }

        with pytest.raises(ValueError):
            extract_context_from_request(mock_request)

    def test_extract_context_with_invalid_organization_id(self):
        """Test extracting context when organization ID is not a valid UUID."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            HEADER_USER_ID: "00000000-0000-0000-0000-000000000001",
            HEADER_ORGANIZATION_ID: "not-a-uuid",
            HEADER_USER_EMAIL: "user@example.com",
            HEADER_USER_ROLES: "",
        }

        with pytest.raises(ValueError):
            extract_context_from_request(mock_request)

    def test_extract_context_generates_unique_request_ids(self):
        """Test that each call generates a unique request_id."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            HEADER_USER_ID: "00000000-0000-0000-0000-000000000001",
            HEADER_ORGANIZATION_ID: "00000000-0000-0000-0000-000000000002",
            HEADER_USER_EMAIL: "user@example.com",
            HEADER_USER_ROLES: "",
        }

        context1 = extract_context_from_request(mock_request)
        context2 = extract_context_from_request(mock_request)

        assert context1.request_id != context2.request_id

    def test_extract_context_with_billing_period_start(self):
        """Test extracting billing_period_start from header."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            HEADER_USER_ID: "00000000-0000-0000-0000-000000000001",
            HEADER_ORGANIZATION_ID: "00000000-0000-0000-0000-000000000002",
            HEADER_USER_EMAIL: "user@example.com",
            HEADER_USER_ROLES: "",
            HEADER_BILLING_PERIOD_START: "2026-02-01T00:00:00+00:00",
        }

        context = extract_context_from_request(mock_request)

        assert context.billing_cycle_start == datetime(2026, 2, 1, tzinfo=timezone.utc)

    def test_extract_context_with_billing_period_start_naive(self):
        """Test extracting billing_period_start without timezone info (as sent by the gateway)."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            HEADER_USER_ID: "00000000-0000-0000-0000-000000000001",
            HEADER_ORGANIZATION_ID: "00000000-0000-0000-0000-000000000002",
            HEADER_USER_EMAIL: "user@example.com",
            HEADER_USER_ROLES: "",
            HEADER_BILLING_PERIOD_START: "2026-04-01T17:25:59",
        }

        context = extract_context_from_request(mock_request)

        assert context.billing_cycle_start == datetime(2026, 4, 1, 17, 25, 59)

    def test_extract_context_with_billing_period_start_z_suffix(self):
        """Test extracting billing_period_start with trailing Z (as sent by the gateway)."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            HEADER_USER_ID: "00000000-0000-0000-0000-000000000001",
            HEADER_ORGANIZATION_ID: "00000000-0000-0000-0000-000000000002",
            HEADER_USER_EMAIL: "user@example.com",
            HEADER_USER_ROLES: "",
            HEADER_BILLING_PERIOD_START: "2026-02-17T00:00:00Z",
        }

        context = extract_context_from_request(mock_request)

        assert context.billing_cycle_start == datetime(2026, 2, 17, tzinfo=timezone.utc)

    def test_extract_context_with_billing_period_start_whitespace(self):
        """Test that leading/trailing whitespace in billing period header is handled."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            HEADER_USER_ID: "00000000-0000-0000-0000-000000000001",
            HEADER_ORGANIZATION_ID: "00000000-0000-0000-0000-000000000002",
            HEADER_USER_EMAIL: "user@example.com",
            HEADER_USER_ROLES: "",
            HEADER_BILLING_PERIOD_START: " 2026-04-01T17:25:59 ",
        }

        context = extract_context_from_request(mock_request)

        assert context.billing_cycle_start == datetime(2026, 4, 1, 17, 25, 59)

    def test_extract_context_without_billing_period_start(self):
        """Test that missing billing period header defaults to None."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            HEADER_USER_ID: "00000000-0000-0000-0000-000000000001",
            HEADER_ORGANIZATION_ID: "00000000-0000-0000-0000-000000000002",
            HEADER_USER_EMAIL: "user@example.com",
            HEADER_USER_ROLES: "",
        }

        context = extract_context_from_request(mock_request)

        assert context.billing_cycle_start is None

    def test_extract_context_with_invalid_billing_period_start(self):
        """Test that an invalid billing period header is handled gracefully."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            HEADER_USER_ID: "00000000-0000-0000-0000-000000000001",
            HEADER_ORGANIZATION_ID: "00000000-0000-0000-0000-000000000002",
            HEADER_USER_EMAIL: "user@example.com",
            HEADER_USER_ROLES: "",
            HEADER_BILLING_PERIOD_START: "not-a-date",
        }

        context = extract_context_from_request(mock_request)

        assert context.billing_cycle_start is None

    def test_extract_context_with_billing_period_end(self):
        """Test extracting billing_period_end from header."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            HEADER_USER_ID: "00000000-0000-0000-0000-000000000001",
            HEADER_ORGANIZATION_ID: "00000000-0000-0000-0000-000000000002",
            HEADER_USER_EMAIL: "user@example.com",
            HEADER_USER_ROLES: "",
            HEADER_BILLING_PERIOD_END: "2026-03-01T00:00:00+00:00",
        }

        context = extract_context_from_request(mock_request)

        assert context.billing_cycle_end == datetime(2026, 3, 1, tzinfo=timezone.utc)

    def test_extract_context_with_billing_period_end_naive(self):
        """Test extracting billing_period_end without timezone info (as sent by the gateway)."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            HEADER_USER_ID: "00000000-0000-0000-0000-000000000001",
            HEADER_ORGANIZATION_ID: "00000000-0000-0000-0000-000000000002",
            HEADER_USER_EMAIL: "user@example.com",
            HEADER_USER_ROLES: "",
            HEADER_BILLING_PERIOD_END: "2026-05-01T17:25:59",
        }

        context = extract_context_from_request(mock_request)

        assert context.billing_cycle_end == datetime(2026, 5, 1, 17, 25, 59)

    def test_extract_context_with_billing_period_end_z_suffix(self):
        """Test extracting billing_period_end with trailing Z."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            HEADER_USER_ID: "00000000-0000-0000-0000-000000000001",
            HEADER_ORGANIZATION_ID: "00000000-0000-0000-0000-000000000002",
            HEADER_USER_EMAIL: "user@example.com",
            HEADER_USER_ROLES: "",
            HEADER_BILLING_PERIOD_END: "2026-03-17T00:00:00Z",
        }

        context = extract_context_from_request(mock_request)

        assert context.billing_cycle_end == datetime(2026, 3, 17, tzinfo=timezone.utc)

    def test_extract_context_with_billing_period_end_whitespace(self):
        """Test that leading/trailing whitespace in billing period end header is handled."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            HEADER_USER_ID: "00000000-0000-0000-0000-000000000001",
            HEADER_ORGANIZATION_ID: "00000000-0000-0000-0000-000000000002",
            HEADER_USER_EMAIL: "user@example.com",
            HEADER_USER_ROLES: "",
            HEADER_BILLING_PERIOD_END: " 2026-05-01T17:25:59 ",
        }

        context = extract_context_from_request(mock_request)

        assert context.billing_cycle_end == datetime(2026, 5, 1, 17, 25, 59)

    def test_extract_context_without_billing_period_end(self):
        """Test that missing billing period end header defaults to None."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            HEADER_USER_ID: "00000000-0000-0000-0000-000000000001",
            HEADER_ORGANIZATION_ID: "00000000-0000-0000-0000-000000000002",
            HEADER_USER_EMAIL: "user@example.com",
            HEADER_USER_ROLES: "",
        }

        context = extract_context_from_request(mock_request)

        assert context.billing_cycle_end is None

    def test_extract_context_with_invalid_billing_period_end(self):
        """Test that an invalid billing period end header is handled gracefully."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            HEADER_USER_ID: "00000000-0000-0000-0000-000000000001",
            HEADER_ORGANIZATION_ID: "00000000-0000-0000-0000-000000000002",
            HEADER_USER_EMAIL: "user@example.com",
            HEADER_USER_ROLES: "",
            HEADER_BILLING_PERIOD_END: "not-a-date",
        }

        context = extract_context_from_request(mock_request)

        assert context.billing_cycle_end is None

    def test_extract_context_with_both_billing_period_headers(self):
        """Test extracting both billing period start and end from headers."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            HEADER_USER_ID: "00000000-0000-0000-0000-000000000001",
            HEADER_ORGANIZATION_ID: "00000000-0000-0000-0000-000000000002",
            HEADER_USER_EMAIL: "user@example.com",
            HEADER_USER_ROLES: "",
            HEADER_BILLING_PERIOD_START: "2026-02-01T00:00:00+00:00",
            HEADER_BILLING_PERIOD_END: "2026-03-01T00:00:00+00:00",
        }

        context = extract_context_from_request(mock_request)

        assert context.billing_cycle_start == datetime(2026, 2, 1, tzinfo=timezone.utc)
        assert context.billing_cycle_end == datetime(2026, 3, 1, tzinfo=timezone.utc)

    def test_extract_context_with_both_billing_period_headers_naive(self):
        """Test extracting both billing periods as naive datetimes matching real gateway format."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            HEADER_USER_ID: "00000000-0000-0000-0000-000000000001",
            HEADER_ORGANIZATION_ID: "00000000-0000-0000-0000-000000000002",
            HEADER_USER_EMAIL: "user@example.com",
            HEADER_USER_ROLES: "",
            HEADER_BILLING_PERIOD_START: "2026-04-01T17:25:59",
            HEADER_BILLING_PERIOD_END: "2026-05-01T17:25:59",
        }

        context = extract_context_from_request(mock_request)

        assert context.billing_cycle_start == datetime(2026, 4, 1, 17, 25, 59)
        assert context.billing_cycle_end == datetime(2026, 5, 1, 17, 25, 59)


class TestLangfuseContextMetadata:
    """Test cases for LangfuseContextMetadata model."""

    def test_langfuse_context_metadata_defaults(self):
        """Test LangfuseContextMetadata initialization with defaults."""
        metadata = LangfuseContextMetadata()

        assert metadata.request_id == UUID("00000000-0000-0000-0000-000000000000")
        assert metadata.user_id == UUID("00000000-0000-0000-0000-000000000000")
        assert metadata.organization_id == UUID("00000000-0000-0000-0000-000000000000")
        assert metadata.user_email == ""
        assert metadata.user_roles == []

    def test_langfuse_context_metadata_with_values(self):
        """Test LangfuseContextMetadata initialization with values."""
        metadata = LangfuseContextMetadata(
            request_id=UUID("00000000-0000-0000-0000-000000000099"),
            user_id=UUID("00000000-0000-0000-0000-000000000001"),
            organization_id=UUID("00000000-0000-0000-0000-000000000002"),
            user_email="user@example.com",
            user_roles=["admin"],
        )

        assert metadata.request_id == UUID("00000000-0000-0000-0000-000000000099")
        assert metadata.user_id == UUID("00000000-0000-0000-0000-000000000001")
        assert metadata.organization_id == UUID("00000000-0000-0000-0000-000000000002")
        assert metadata.user_email == "user@example.com"
        assert metadata.user_roles == ["admin"]

    def test_langfuse_context_metadata_inherits_from_context(self):
        """Test that LangfuseContextMetadata inherits from Context."""
        assert issubclass(LangfuseContextMetadata, Context)


class TestLangfuseContext:
    """Test cases for LangfuseContext model."""

    def test_langfuse_context_defaults(self):
        """Test LangfuseContext initialization with defaults."""
        langfuse_context = LangfuseContext()

        assert langfuse_context.user_id == UUID("00000000-0000-0000-0000-000000000000")
        assert langfuse_context.metadata.user_id == UUID("00000000-0000-0000-0000-000000000000")
        assert langfuse_context.metadata.organization_id == UUID("00000000-0000-0000-0000-000000000000")
        assert langfuse_context.tags == []

    def test_langfuse_context_with_values(self):
        """Test LangfuseContext initialization with values."""
        metadata = LangfuseContextMetadata(
            user_id=UUID("00000000-0000-0000-0000-000000000001"),
            organization_id=UUID("00000000-0000-0000-0000-000000000002"),
            user_email="user@example.com",
            user_roles=["admin"],
        )

        langfuse_context = LangfuseContext(
            user_id=UUID("00000000-0000-0000-0000-000000000001"),
            metadata=metadata,
            tags=["tag1", "tag2"],
        )

        assert langfuse_context.user_id == UUID("00000000-0000-0000-0000-000000000001")
        assert langfuse_context.metadata == metadata
        assert langfuse_context.tags == ["tag1", "tag2"]


class TestCreateLangfuseContext:
    """Test cases for create_langfuse_context function."""

    def test_create_langfuse_context_from_context(self):
        """Test creating LangfuseContext from Context."""
        context = Context(
            user_id=UUID("00000000-0000-0000-0000-000000000001"),
            organization_id=UUID("00000000-0000-0000-0000-000000000002"),
            user_email="user@example.com",
            user_roles=["admin", "editor"],
        )

        langfuse_context = create_langfuse_context(context)

        assert langfuse_context.user_id == UUID("00000000-0000-0000-0000-000000000001")
        assert langfuse_context.metadata.user_id == UUID("00000000-0000-0000-0000-000000000001")
        assert langfuse_context.metadata.organization_id == UUID("00000000-0000-0000-0000-000000000002")
        assert langfuse_context.metadata.user_email == "user@example.com"
        assert langfuse_context.metadata.user_roles == ["admin", "editor"]
        assert langfuse_context.tags == [
            f"user_id:{UUID('00000000-0000-0000-0000-000000000001')}",
            f"organization_id:{UUID('00000000-0000-0000-0000-000000000002')}",
            "local",
            f"request_id:{context.request_id}",
            "user@example.com",
        ]

    def test_create_langfuse_context_with_empty_context(self):
        """Test creating LangfuseContext from empty Context."""
        context = Context()

        langfuse_context = create_langfuse_context(context)

        assert langfuse_context.user_id == UUID("00000000-0000-0000-0000-000000000000")
        assert langfuse_context.metadata.user_id == UUID("00000000-0000-0000-0000-000000000000")
        assert langfuse_context.metadata.organization_id == UUID("00000000-0000-0000-0000-000000000000")
        assert langfuse_context.metadata.user_email == ""
        assert langfuse_context.metadata.user_roles == []
        assert langfuse_context.tags == [
            f"user_id:{UUID('00000000-0000-0000-0000-000000000000')}",
            f"organization_id:{UUID('00000000-0000-0000-0000-000000000000')}",
            "local",
            f"request_id:{context.request_id}",
            "",
        ]

    def test_create_langfuse_context_tags_format(self):
        """Test that tags are created in the correct format."""
        context = Context(
            user_id=UUID("00000000-0000-0000-0000-000000000001"),
            organization_id=UUID("00000000-0000-0000-0000-000000000002"),
            user_email="test@test.com",
            user_roles=["viewer"],
        )

        langfuse_context = create_langfuse_context(context)

        expected_tags = [
            f"user_id:{UUID('00000000-0000-0000-0000-000000000001')}",
            f"organization_id:{UUID('00000000-0000-0000-0000-000000000002')}",
            "local",
            f"request_id:{context.request_id}",
            "test@test.com",
        ]
        assert langfuse_context.tags == expected_tags
        assert len(langfuse_context.tags) == 5

    def test_create_langfuse_context_metadata_consistency(self):
        """Test that metadata fields match the context fields."""
        context = Context(
            user_id=UUID("00000000-0000-0000-0000-000000000001"),
            organization_id=UUID("00000000-0000-0000-0000-000000000002"),
            user_email="user@example.com",
            user_roles=["admin"],
        )

        langfuse_context = create_langfuse_context(context)

        # All corresponding fields should match
        assert langfuse_context.user_id == context.user_id
        assert langfuse_context.metadata.user_id == context.user_id
        assert langfuse_context.metadata.organization_id == context.organization_id
        assert langfuse_context.metadata.request_id == context.request_id
        assert langfuse_context.metadata.user_email == context.user_email
        assert langfuse_context.metadata.user_roles == context.user_roles
