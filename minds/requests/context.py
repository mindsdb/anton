from typing import Any, Optional

from pydantic import BaseModel, Field
from fastapi import Request


class Context(BaseModel):
    """
    Context for the application.
    """

    user_id: int = 0
    user_email: str = ""
    company_id: int = 0


def extract_context_from_request(request: Request) -> Context:
    """
    Extract the context from the request headers.
    """

    user_id = int(request.headers.get("x-user-id", 0))
    user_email = request.headers.get("x-user-email", "")
    company_id = int(request.headers.get("x-company-id", 0))

    return Context(user_id=user_id, user_email=user_email, company_id=company_id)


class LangfuseContextMetadata(BaseModel):
    """
    Metadata for the Langfuse context.
    Attributes:
        user_id: int
        user_email: str
        company_id: int
    """

    user_id: int = Field(default=0, description="The user ID")
    user_email: str = Field(default="", description="The user email")
    company_id: int = Field(default=0, description="The company ID")


class LangfuseContext(BaseModel):
    """
    Context for Langfuse, including user and metadata.
    Attributes:
        user_id: str
        metadata: LangfuseContextMetadata
        tags: list[str]
    """

    user_id: int = 0
    metadata: LangfuseContextMetadata = LangfuseContextMetadata()
    tags: list[Any] = []
    trace_id: Optional[str] = None


def create_langfuse_context(context: Context) -> LangfuseContext:
    """
    Create a Langfuse context from request context.
    """
    tags = [context.user_email, context.company_id]

    return LangfuseContext(
        user_id=context.user_id,
        metadata=LangfuseContextMetadata(
            user_id=context.user_id,
            user_email=context.user_email,
            company_id=context.company_id,
        ),
        tags=tags,
    )
