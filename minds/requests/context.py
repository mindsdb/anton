import uuid
from typing import Any
from uuid import UUID

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

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
    tenant_id: UUID = Field(default=UUID("00000000-0000-0000-0000-000000000000"), description="The tenant ID")
    user_email: str = Field(default="", description="The user email")


def extract_context_from_request(request: Request) -> Context:
    """
    Extract the context from the request headers.
    """

    # TODO: Temporary solution while lucas.koontz finishes working on Auth API

    if settings.auth.disable:
        logger.debug(f"Extracting context from request with auth disabled: {settings.auth.disable}")
        return Context(
            user_id=UUID("00000000-0000-0000-0000-000000000000"),
            tenant_id=UUID("00000000-0000-0000-0000-000000000000"),
            user_email="",
        )

    if request.headers.get("x-user-id") is None or request.headers.get("x-company-id") is None:
        raise HTTPException(status_code=400, detail="Missing required authentication")

    x_user_id = str(request.headers.get("x-user-id"))
    x_tenant_id = str(request.headers.get("x-company-id"))
    x_user_email = str(request.headers.get("x-user-email"))

    user_id = UUID(int=int(x_user_id))
    tenant_id = UUID(int=int(x_tenant_id))
    user_email = x_user_email

    request_id = uuid.uuid4()

    logger.debug(f"Extracted context from request: user_id={user_id}, tenant_id={tenant_id}, user_email={user_email}")

    return Context(user_id=user_id, tenant_id=tenant_id, request_id=request_id, user_email=user_email)


class LangfuseContextMetadata(Context):
    """
    Metadata for the Langfuse context.
    Attributes:
        request_id: UUID
        user_id: UUID
        tenant_id: UUID
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
        f"user_id:{context.user_id}",
        f"tenant_id:{context.tenant_id}",
        settings.env,
        f"request_id:{context.request_id}",
        context.user_email,
    ]

    return LangfuseContext(
        user_id=context.user_id,
        metadata=LangfuseContextMetadata(
            user_id=context.user_id,
            tenant_id=context.tenant_id,
            request_id=context.request_id,
            user_email=context.user_email,
        ),
        tags=tags,
    )
