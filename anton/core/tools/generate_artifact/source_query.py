"""Strict OAuth client for documented scenario/column-row query sources.

The caller supplies the connection, query path, scenario and statement. No
table names, field names, business measures or customer data live here.
Uses only the standard library, including inside a scratchpad environment.
"""
import json
import os
import time
import math
from collections.abc import Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urljoin
from urllib.request import Request, HTTPRedirectHandler, build_opener


class SourceQueryError(ValueError):
    pass


class SourceHTTPError(SourceQueryError):
    def __init__(self, status):
        self.status = status
        super().__init__(f'Source request failed (HTTP {status}).')


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def _origin(url):
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError()
        return parsed.scheme, parsed.hostname.casefold(), parsed.port or (443 if parsed.scheme == 'https' else 80)
    except ValueError:
        raise SourceQueryError('Source connection needs HTTP(S) URLs without embedded credentials.') from None


class ScenarioRowsClient:
    def __init__(self, *, base_url, token_url, client_id, client_secret, timeout=30):
        for url in (base_url, token_url):
            _origin(url)
        if urlsplit(base_url).query or urlsplit(base_url).fragment:
            raise SourceQueryError('Source base URL must not contain a query or fragment.')
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise SourceQueryError('Source timeout must be a positive finite number.')
        self._base = base_url.rstrip('/')
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._timeout = timeout
        self._token = None
        self._expires_at = 0
        self._opener = build_opener(_NoRedirect())

    def __repr__(self):
        return 'ScenarioRowsClient()'

    @classmethod
    def from_environment(cls, prefix=None):
        """Use an injected DS_ connection; require an explicit prefix if several match."""
        required = ('BASE_URL', 'TOKEN_URL', 'CLIENT_ID', 'CLIENT_SECRET')
        prefixes = {key.removesuffix('__BASE_URL') for key in os.environ
                    if key.startswith('DS_') and key.endswith('__BASE_URL')}
        complete = [name for name in sorted(prefixes)
                    if all(os.environ.get(name + '__' + field) for field in required)]
        if prefix is not None:
            complete = [name for name in complete if name == prefix]
        if len(complete) != 1:
            raise SourceQueryError('Choose the exact injected DS_ environment prefix of one complete OAuth connection.')
        values = {field.lower(): os.environ[complete[0] + '__' + field] for field in required}
        return cls(**values)

    def _json(self, url, data, headers):
        try:
            with self._opener.open(Request(url, data=data, headers=headers, method='POST'), timeout=self._timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as error:
            # Do not echo response bodies, URLs, headers or credential values.
            raise SourceHTTPError(error.code) from None
        except (URLError, OSError):
            raise SourceQueryError('Source request could not complete.') from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise SourceQueryError('Source response exceeds the bounded query response size; narrow or partition the query.')
        try:
            def invalid_number(value):
                raise ValueError('Nonfinite JSON number')
            result = json.loads(raw, parse_constant=invalid_number)
        except (ValueError, UnicodeError):
            raise SourceQueryError('Source response is not valid JSON.') from None
        if not isinstance(result, dict):
            raise SourceQueryError('Source response must be a JSON object.')
        return result

    def _authenticate(self):
        data = urlencode({'grant_type':'client_credentials', 'client_id':self._client_id,
                          'client_secret':self._client_secret}).encode()
        result = self._json(self._token_url, data, {'Content-Type':'application/x-www-form-urlencoded'})
        token = result.get('access_token')
        if not isinstance(token, str) or not token or '\r' in token or '\n' in token:
            raise SourceQueryError('Source authentication returned no usable access token.')
        self._token = token
        ttl = result.get('expires_in')
        if ttl is None:
            self._expires_at = math.inf
        else:
            try:
                seconds = float(ttl)
            except (TypeError, ValueError):
                raise SourceQueryError('Source token expiry is invalid.') from None
            if isinstance(ttl, bool) or not math.isfinite(seconds) or seconds <= 0:
                raise SourceQueryError('Source token expiry is invalid.')
            self._expires_at = time.monotonic() + seconds - min(30, seconds * .1)

    def query_rows(self, path, *, scenario, query):
        """Return complete list[dict] rows; accept a relative path or same-origin URL."""
        if not isinstance(path, str) or not path.strip():
            raise SourceQueryError('Provide the query URL/path from the source procedure, not a connection name.')
        endpoint = urljoin(self._base + '/', path.strip())
        if _origin(endpoint) != _origin(self._base):
            raise SourceQueryError('Query URL/path must stay on the configured source origin.')
        if not isinstance(scenario, Mapping) or not scenario or not isinstance(query, str) or not query.strip():
            raise SourceQueryError('Provide one scenario object and a nonempty query string.')
        if self._token is None or time.monotonic() >= self._expires_at:
            self._authenticate()
        data = json.dumps({'Scenarios':[dict(scenario)], 'QueryString':query}, allow_nan=False).encode()
        def request():
            return self._json(endpoint, data,
                              {'Content-Type':'application/json', 'Authorization':'Bearer ' + self._token})
        try:
            result = request()
        except SourceHTTPError as error:
            if error.status != 401:
                raise
            self._authenticate()
            result = request()  # one bounded token-renewal retry, never an alternate source
        if 'Truncated' in result and not isinstance(result['Truncated'], bool):
            raise SourceQueryError('Source truncation flag is invalid.')
        if result.get('Truncated'):
            raise SourceQueryError('Source query was truncated; partition it before calculating totals.')
        columns, rows = result.get('Columns'), result.get('Rows')
        if (not isinstance(columns, list) or not isinstance(rows, list)
                or not all(isinstance(c, dict) and isinstance(c.get('ColumnName'), str) and c['ColumnName'] for c in columns)):
            raise SourceQueryError('Source response needs Columns with ColumnName and a Rows list; an error or another response shape is not an empty dataset.')
        names = [column['ColumnName'] for column in columns]
        if len(names) != len(set(names)):
            raise SourceQueryError('Source column names are duplicated.')
        if not all(isinstance(row, list) and len(row) == len(names) for row in rows):
            raise SourceQueryError('Source row widths do not match the declared columns.')
        return [dict(zip(names, row)) for row in rows]

    def close(self):
        self._token = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
