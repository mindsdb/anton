from typing import Any
from uuid import UUID

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

from minds.common.logger import setup_logging
from minds.common.vars import DISABLE_AUTH

# Set up logging
logger = setup_logging()


class Context(BaseModel):
    """
    Context for the application.
    """

    user_id: UUID = Field(default=UUID("00000000-0000-0000-0000-000000000000"), description="The user ID")
    tenant_id: UUID = Field(default=UUID("00000000-0000-0000-0000-000000000000"), description="The tenant ID")


def extract_context_from_request(request: Request) -> Context:
    """
    Extract the context from the request headers.
    """
    # TODO: Temporary solution while lucas.koontz finishes working on Auth API

    if DISABLE_AUTH:
        logger.debug(f"Extracting context from request with DISABLE_AUTH: {DISABLE_AUTH}")
        return Context(
            user_id=UUID("00000000-0000-0000-0000-000000000000"),
            tenant_id=UUID("00000000-0000-0000-0000-000000000000"),
        )

    if request.headers.get("x-user-id") is None or request.headers.get("x-company-id") is None:
        raise HTTPException(status_code=400, detail="Missing required authentication")

    x_user_id = str(request.headers.get("x-user-id"))
    x_tenant_id = str(request.headers.get("x-company-id"))

    user_id = UUID(int=int(x_user_id))
    tenant_id = UUID(int=int(x_tenant_id))

    return Context(user_id=user_id, tenant_id=tenant_id)


class LangfuseContextMetadata(BaseModel):
    """
    Metadata for the Langfuse context.
    Attributes:
        user_id: UUID
    """

    user_id: UUID = Field(default=UUID("00000000-0000-0000-0000-000000000000"), description="The user ID")
    tenant_id: UUID = Field(default=UUID("00000000-0000-0000-0000-000000000000"), description="The tenant ID")


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
    tags = [context.user_id, context.tenant_id]

    return LangfuseContext(
        user_id=context.user_id,
        metadata=LangfuseContextMetadata(
            user_id=context.user_id,
            tenant_id=context.tenant_id,
        ),
        tags=tags,
    )
