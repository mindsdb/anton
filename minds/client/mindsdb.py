from fastapi import Request
from mindsdb_sdk import connect
from mindsdb_sdk.server import Server

from minds.common import get_authorization_bearer_token, get_headers_for_mindsdb_client
from minds.common.vars import MINDSDB_API_KEY, MINDSDB_LOGIN, MINDSDB_PASSWORD, MINDSDB_URL
from minds.requests.context import Context


def create_mindsdb_client_from_env() -> Server:
    """
    Create a MindsDB client from the environment.
    """
    return create_mindsdb_client(api_key=MINDSDB_API_KEY)


def create_mindsdb_client_from_request(request: Request, context: Context) -> Server:
    """
    Create a MindsDB client from a request.

    Args:
      request: The request to use for the client.
    """
    api_key = get_authorization_bearer_token(request)
    headers = get_headers_for_mindsdb_client(context)

    return create_mindsdb_client(api_key, headers=headers)


def create_mindsdb_client(api_key: str | None, headers: dict | None) -> Server:
    """
    Create a MindsDB client.

    Args:
        api_key: The API key to use for the client. Can be None if MindsDB doesn't require auth.
        headers: Additional headers to pass to the MindsDB client.

    Returns:
        A MindsDB client.
    """

    # If no API key is provided, try connecting without authentication
    if api_key is None or not api_key or not api_key.strip():
        # For MindsDB without authentication, don't pass login/password at all
        if not MINDSDB_PASSWORD:
            return connect(url=MINDSDB_URL, headers=headers)
        else:
            return connect(
                url=MINDSDB_URL,
                login=MINDSDB_LOGIN,
                password=MINDSDB_PASSWORD,
                headers=headers,
            )

    if not isinstance(api_key, str):
        raise ValueError("API key must be a string")

    return connect(
        url=MINDSDB_URL,
        api_key=api_key,
        headers=headers,
    )


# This is used in the data catalog loader flow
# TODO: Remove thise once the settings are configured better
def create_mindsdb_client_with_credentials(
    url: str,
    api_key: str | None = None,
    login: str | None = None,
    password: str | None = None,
) -> Server:
    """
    Create a MindsDB client with explicit credentials.

    Args:
      url: The MindsDB server URL.
      api_key: The API key to use for the client. Can be None if MindsDB doesn't require auth.
      login: The login username for authentication.
      password: The password for authentication.

    Returns:
      A MindsDB client.
    """

    # If no API key is provided, try connecting without authentication
    if api_key is None or not api_key or not api_key.strip():
        # For MindsDB without authentication, don't pass login/password at all
        if not password:
            return connect(url=url)
        else:
            return connect(
                url=url,
                login=login,
                password=password,
            )

    if not isinstance(api_key, str):
        raise ValueError("API key must be a string")

    return connect(
        url=url,
        api_key=api_key,
    )
