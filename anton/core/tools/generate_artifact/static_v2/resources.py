"""Bounded reference data and pinned display libraries, embedded before author code."""
import hashlib
import json
import re
from typing import Literal
from urllib.parse import urljoin, urlsplit

import httpx2 as httpx
from pydantic import Field

from anton.core.artifacts.static.data import Model, Name
from anton.core.artifacts.static.render import _script_json, _inject_head
from anton.core.artifacts.static.views import validate_ids
from anton.core.tools.web_tools import _check_url_ssrf, _TransientFetchError

MAX_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 4 * 1024 * 1024


class Resource(Model):
    name: Name
    url: str = Field(max_length=2048)
    label: str = Field(min_length=1, max_length=200)
    kind: Literal['json', 'script'] = 'json'


TOOL = {'name': 'load_static_resource', 'description': 'Embed supporting resources offline. kind=json loads authentic public reference data, such as geometry, topology, or diagram structure, as window.ARTIFACT_RESOURCES[name]. kind=script loads a browser-ready library bundle from a pinned npm version on cdn.jsdelivr.net and runs it before author scripts. Downloads use the normal web URL policy, are bounded, and never require a view-time fetch. Use public HTTPS URLs without credentials or query parameters. Never invent real-world geometry or replace the designated business evidence.', 'input_schema': Resource.model_json_schema()}


async def load(resource: Resource) -> dict:
    url = resource.url
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=False) as client:
            for _ in range(6):
                parsed = urlsplit(url)
                if parsed.scheme != 'https' or not parsed.hostname or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
                    raise ValueError('A resource needs a public HTTPS URL without credentials, query, or fragment')
                if resource.kind == 'script' and (parsed.hostname != 'cdn.jsdelivr.net' or not parsed.path.startswith('/npm/') or not re.search(r'/[^/]+@\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?/', parsed.path)):
                    raise ValueError('A script needs a pinned npm package version on the preview-approved cdn.jsdelivr.net host')
                if error := _check_url_ssrf(url):
                    raise ValueError(error)
                async with client.stream('GET', url, headers={'User-Agent': 'AntonBot/1.0'}) as response:
                    if response.is_redirect:
                        location = response.headers.get('location')
                        if not location:
                            raise ValueError('Resource redirect has no destination')
                        url = urljoin(url, location)
                        continue
                    response.raise_for_status()
                    chunks, size = [], 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > MAX_BYTES:
                            raise ValueError('Resource exceeds 2 MiB; use a smaller reference asset')
                        chunks.append(chunk)
                    raw = b''.join(chunks)
                    if resource.kind == 'script':
                        if response.headers.get('content-type', '').split(';')[0].lower() not in ('application/javascript', 'text/javascript', 'application/ecmascript', 'text/ecmascript'):
                            raise ValueError('Script resource must be served as JavaScript')
                        data = raw.decode('utf-8-sig')
                        if re.search(r'</script', data, re.I):
                            raise ValueError('Bundle contains an HTML script terminator; choose a browser-safe distribution')
                    else:
                        data = json.loads(raw)
                        if not isinstance(data, (dict, list)):
                            raise ValueError('Resource must be a JSON object or array')
                        # Reject non-finite parser values rather than changing their meaning.
                        json.dumps({'data': {resource.name: data}}, allow_nan=False)
                    return {'data': data, 'source': {'label': resource.label, 'url': url, 'kind': resource.kind,
                        'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': size}}
            raise ValueError('Resource exceeds five redirects')
    except httpx.HTTPStatusError as error:
        raise ValueError(f'Reference resource returned HTTP {error.response.status_code}') from error
    except (httpx.HTTPError, UnicodeError, RecursionError, _TransientFetchError) as error:
        raise ValueError(f'Reference resource could not be loaded: {type(error).__name__}') from error


def embed(html, manifest, resources):
    if not resources:
        return html
    manifest['resources'] = {name: item['source'] for name, item in resources.items()}
    payload = {'data': {name: item['data'] for name, item in resources.items() if item['source'].get('kind', 'json') == 'json'}, 'sources': manifest['resources']}
    script = '<script id="_artifact-resources" type="application/json">' + _script_json(payload) + '</script>'
    script += '<script>window.ARTIFACT_RESOURCES=JSON.parse(document.getElementById("_artifact-resources").textContent).data;</script>'
    script += ''.join(f'<script data-artifact-resource="{name}">{item["data"]}</script>' for name, item in resources.items() if item['source'].get('kind') == 'script')
    html = _inject_head(html, script)
    validate_ids(html)
    return html
