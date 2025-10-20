from typing import Any
from uuid import UUID
from fastapi import Request
from pydantic import BaseModel, Field


class Context(BaseModel):
    """
    Context for the application.
    """

    user_id: UUID = Field(default="", description="The user ID")
    tenant_id: UUID = Field(default="", description="The tenant ID")


def extract_context_from_request(request: Request) -> Context:
    """
    Extract the context from the request headers.
    """
    # TODO: Temporary solution while lucas.koontz finishes working on Auth API
    
    x_user_id = str(request.headers.get("x-user-id", ""))
    x_tenant_id = str(request.headers.get("x-company-id", ""))
    
    user_id = UUID(int=x_user_id)
    tenant_id = UUID(int=x_tenant_id)

    return Context(user_id=user_id, tenant_id=tenant_id)

class LangfuseContextMetadata(BaseModel):
    """
    Metadata for the Langfuse context.
    Attributes:
        user_id: int
        user_email: str
    """

    user_id: str = Field(default="", description="The user ID")
    user_email: str = Field(default="", description="The user email")


class LangfuseContext(BaseModel):
    """
    Context for Langfuse, including user and metadata.
    Attributes:
        user_id: str
        metadata: LangfuseContextMetadata
        tags: list[str]
    """

    user_id: str = ""
    metadata: LangfuseContextMetadata = LangfuseContextMetadata()
    tags: list[Any] = []
    trace_id: str | None = None


def create_langfuse_context(context: Context) -> LangfuseContext:
    """
    Create a Langfuse context from request context.
    """
    tags = [context.user_email]

    return LangfuseContext(
        user_id=context.user_id,
        metadata=LangfuseContextMetadata(
            user_id=context.user_id,
            user_email=context.user_email,
        ),
        tags=tags,
    )
