from fastapi import Request
from mindsdb_sdk import connect
from mindsdb_sdk.server import Server

from minds.common import get_authorization_bearer_token, get_headers_for_mindsdb_client
from minds.common.authentication import MindsDBHeaders
from minds.common.logger import setup_logging
from minds.common.settings.app_settings import get_app_settings
from minds.requests.context import Context

# Set up logging
logger = setup_logging()
settings = get_app_settings()


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
        if not settings.mindsdb.password:
            logger.debug(f"Creating MindsDB client without authentication with URL: {settings.mindsdb.url}")
            return connect(url=settings.mindsdb.url, headers=headers)
        else:
            logger.debug(f"Creating MindsDB client with authentication with URL: {settings.mindsdb.url}")
            return connect(
                url=settings.mindsdb.url,
                login=settings.mindsdb.login,
                password=settings.mindsdb.password,
                headers=headers,
            )

    if not isinstance(api_key, str):
        raise ValueError("API key must be a string")

    return connect(
        url=settings.mindsdb.url,
        api_key=api_key,
        headers=headers,
    )


def create_mindsdb_client_with_credentials(
    url: str,
    api_key: str | None = None,
    login: str | None = None,
    password: str | None = None,
    organization_id: str | None = None,
    user_id: str | None = None,
) -> Server:
    """
    Create a MindsDB client with explicit credentials.

    Args:
      url: The MindsDB server URL.
      api_key: The API key to use for the client. Can be None if MindsDB doesn't require auth.
      login: The login username for authentication.
      password: The password for authentication.
      organization_id: The organization ID.
      user_id: The user ID.
    Returns:
      A MindsDB client.
    """

    headers = MindsDBHeaders(organization_id=organization_id, user_id=user_id).to_dict()

    # If no API key is provided, try connecting without authentication
    if api_key is None or not api_key or not api_key.strip():
        # For MindsDB without authentication, don't pass login/password at all
        if not password:
            return connect(url=url, headers=headers)
        else:
            return connect(
                url=url,
                login=login,
                password=password,
                headers=headers,
            )

    if not isinstance(api_key, str):
        raise ValueError("API key must be a string")

    return connect(
        url=url,
        api_key=api_key,
        headers=headers,
    )
