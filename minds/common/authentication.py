from uuid import UUID

from fastapi import HTTPException, Request, status
from pydantic import BaseModel

from minds.requests.context import Context


class AuthHeaders(BaseModel):
    """Model for authorization headers."""

    authorization: str | None = None


class MindsDBHeaders(BaseModel):
    organization_id: UUID
    user_id: UUID

    def to_dict(self) -> dict:
        return {
            "company-id": str(
                self.organization_id
            ),  # Going to rename to organization_id in MindsDB in the future for consistency
            "user-id": str(self.user_id),
            "enforce-user-id": "false",
        }


def get_authorization_bearer_token(
    request_or_headers: Request | dict | AuthHeaders,
) -> str:
    """
    Extract the Bearer token from authorization header.

    Args:
        request_or_headers: FastAPI Request object, dict with headers, or AuthHeaders model

    Returns:
        str: The API key without 'Bearer ' prefix

    Raises:
        HTTPException: If no authorization header is found or invalid format
    """
    auth_header = None

    if isinstance(request_or_headers, Request):
        # FastAPI Request object
        auth_header = request_or_headers.headers.get("authorization")
    elif isinstance(request_or_headers, dict):
        # Dictionary with headers
        auth_header = request_or_headers.get("authorization") or request_or_headers.get("Authorization")
    elif isinstance(request_or_headers, AuthHeaders):
        # Pydantic model
        auth_header = request_or_headers.authorization
    else:
        raise ValueError(f"Unsupported type: {type(request_or_headers)}. Expected Request, dict, or AuthHeaders")

    if not auth_header:
        return None

    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must start with 'Bearer '",
        )

    # Remove 'Bearer ' prefix and return the API key
    api_key = auth_header.replace("Bearer ", "")

    if not api_key:
        return None

    return api_key


def get_api_key_from_request(request: Request) -> str:
    """
    Convenience function to get API key from FastAPI Request object.

    Args:
        request: FastAPI Request object

    Returns:
        str: The API key
    """
    return get_authorization_bearer_token(request)


def get_api_key_from_headers(headers: dict) -> str:
    """
    Convenience function to get API key from headers dictionary.

    Args:
        headers: Dictionary containing headers

    Returns:
        str: The API key
    """
    return get_authorization_bearer_token(headers)


def get_headers_for_mindsdb_client(context: Context) -> dict:
    """
    Get the headers for MindsDB client from the context.

    Args:
        context: The request context.

    Returns:
        dict: The headers for MindsDB client.
    """
    return MindsDBHeaders(organization_id=context.organization_id, user_id=context.user_id).to_dict()
