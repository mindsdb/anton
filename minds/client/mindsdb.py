from fastapi import Request
from mindsdb_sdk import connect
from mindsdb_sdk.server import Server

from minds.common.vars import MINDSDB_URL, MINDSDB_API_KEY
from minds.common import get_authorization_bearer_token


def create_mindsdb_client_from_env() -> Server:
    """
    Create a MindsDB client from the environment.
    """
    return create_mindsdb_client(api_key=MINDSDB_API_KEY)


def create_mindsdb_client_from_request(request: Request) -> Server:
    """
    Create a MindsDB client from a request.

    Args:
      request: The request to use for the client.
    """
    api_key = get_authorization_bearer_token(request)

    return create_mindsdb_client(api_key)


def create_mindsdb_client(api_key: str) -> Server:
    """
    Create a MindsDB client.

    Args:
      api_key: The API key to use for the client.

    Returns:
      A MindsDB client.
    """

    if api_key is None:
        raise ValueError("API key is required")

    if not isinstance(api_key, str):
        raise ValueError("API key must be a string")

    if not api_key or not api_key.strip():
        raise ValueError("API key is required")

    return connect(
        url=MINDSDB_URL,
        api_key=api_key,
    )
