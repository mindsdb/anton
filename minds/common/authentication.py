import hashlib
from uuid import UUID

from fastapi import HTTPException, Request, status
from pydantic import BaseModel

from minds.requests.context import Context


class AuthHeaders(BaseModel):
    """Model for authorization headers."""

    authorization: str | None = None


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


def get_company_id(user_id: UUID, tenant_id: UUID) -> str:
    """
    Get the company ID from the context.
    This will be a combination of tenant ID and user ID, hashed to produce a consistent
    integer value. It is also modulo'd to fit within 32-bit signed integer range.
    Ideally, MindsDB should support multi-tenant natively in the future.

    Args:
        user_id: The user ID
        tenant_id: The tenant ID

    Returns:
        str: The company ID
    """
    digest = hashlib.sha256((str(tenant_id) + str(user_id)).encode()).digest()
    company_id = int.from_bytes(digest[:4], byteorder="big", signed=False) % (2**31 - 1)
    return str(company_id)


def get_headers_for_mindsdb_client(context: Context) -> dict:
    """
    Get the headers for MindsDB client from the context.

    Args:
        context: The request context.

    Returns:
        dict: The headers for MindsDB client.
    """
    return {
        "company-id": get_company_id(context.user_id, context.tenant_id),
    }
