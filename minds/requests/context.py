import uuid
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

from minds.common.constants import (
    CONTEXT_FIELD_ORGANIZATION_ID,
    CONTEXT_FIELD_REQUEST_ID,
    CONTEXT_FIELD_USER_ID,
    HEADER_BILLING_PERIOD_START,
    HEADER_ORGANIZATION_ID,
    HEADER_USER_EMAIL,
    HEADER_USER_ID,
    HEADER_USER_ROLES,
)
from minds.common.logger import setup_logging
from minds.common.settings.app_settings import get_app_settings

# Set up logging
logger = setup_logging()
settings = get_app_settings()


class Context(BaseModel):
    """
    Context for the application.
    """

    request_id: UUID = Field(default=UUID("00000000-0000-0000-0000-000000000000"), description="The request ID")
    user_id: UUID = Field(default=UUID("00000000-0000-0000-0000-000000000000"), description="The user ID")
    organization_id: UUID = Field(
        default=UUID("00000000-0000-0000-0000-000000000000"), description="The organization ID"
    )
    user_email: str = Field(default="", description="The user email")
    user_roles: list[str] = Field(default=[], description="The user roles")
    billing_period_start: datetime | None = Field(
        default=None, description="Start of the current billing period for monthly usage calculations"
    )


def extract_context_from_request(request: Request) -> Context:
    """
    Extract the context from the request headers.
    """

    if request.headers.get(HEADER_USER_ID) is None or request.headers.get(HEADER_ORGANIZATION_ID) is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    x_user_id = str(request.headers.get(HEADER_USER_ID))
    x_organization_id = str(request.headers.get(HEADER_ORGANIZATION_ID))
    x_user_email = str(request.headers.get(HEADER_USER_EMAIL))
    x_user_roles = str(request.headers.get(HEADER_USER_ROLES))

    user_id = UUID(x_user_id)
    organization_id = UUID(x_organization_id)
    user_email = x_user_email
    user_roles = x_user_roles.split(",") if x_user_roles else []

    billing_period_start: datetime | None = None
    x_billing_period_start = request.headers.get(HEADER_BILLING_PERIOD_START)
    if x_billing_period_start:
        try:
            # Python 3.10's fromisoformat() does not accept the "Z" suffix;
            # normalize to "+00:00" so all supported versions parse correctly.
            normalized = x_billing_period_start.replace("Z", "+00:00")
            billing_period_start = datetime.fromisoformat(normalized)
        except (ValueError, TypeError):
            logger.warning(f"Invalid {HEADER_BILLING_PERIOD_START} header value: {x_billing_period_start}")

    request_id = uuid.uuid4()

    logger.debug(
        f"Extracted context from request: request_id={request_id}, user_id={user_id}, "
        f"organization_id={organization_id}, user_email={user_email}, user_roles={user_roles}, "
        f"billing_period_start={billing_period_start}"
    )

    return Context(
        user_id=user_id,
        organization_id=organization_id,
        request_id=request_id,
        user_email=user_email,
        user_roles=user_roles,
        billing_period_start=billing_period_start,
    )


class LangfuseContextMetadata(Context):
    """
    Metadata for the Langfuse context.
    Attributes:
        request_id: UUID
        user_id: UUID
        organization_id: UUID
    """

    pass


class LangfuseContext(BaseModel):
    """
    Context for Langfuse, including user and metadata.
    Attributes:
        user_id: str
        metadata: LangfuseContextMetadata
        tags: list[str]
    """

    user_id: UUID = UUID("00000000-0000-0000-0000-000000000000")
    metadata: LangfuseContextMetadata = LangfuseContextMetadata()
    tags: list[Any] = []
    trace_id: str | None = None


def create_langfuse_context(context: Context) -> LangfuseContext:
    """
    Create a Langfuse context from request context.
    """
    tags = [
        f"{CONTEXT_FIELD_USER_ID}:{context.user_id}",
        f"{CONTEXT_FIELD_ORGANIZATION_ID}:{context.organization_id}",
        settings.env,
        f"{CONTEXT_FIELD_REQUEST_ID}:{context.request_id}",
        context.user_email,
    ]

    return LangfuseContext(
        user_id=context.user_id,
        metadata=LangfuseContextMetadata(
            user_id=context.user_id,
            organization_id=context.organization_id,
            request_id=context.request_id,
            user_email=context.user_email,
            user_roles=context.user_roles,
        ),
        tags=tags,
    )
