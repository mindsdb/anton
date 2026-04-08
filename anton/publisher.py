"""Publish HTML reports to the anton-services web host."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import zipfile
from pathlib import Path

from anton.minds_client import minds_request
from anton.utils.datasources import scrub_credentials

# LLM API key env vars whose values must be stripped from published files.
_LLM_SECRET_VARS = (
    "ANTON_ANTHROPIC_API_KEY",
    "ANTON_OPENAI_API_KEY",
    "ANTON_MINDS_API_KEY",
)

# File extensions treated as text and subject to credential scrubbing.
_TEXT_EXTENSIONS = {".html", ".htm", ".js", ".css"}


DEFAULT_PUBLISH_URL = "https://4nton.ai"

# Patterns that capture relative paths from HTML attributes and CSS url()
_REF_PATTERNS = [
    re.compile(r'(?:src|href)\s*=\s*"([^":#?]+)"', re.IGNORECASE),
    re.compile(r"(?:src|href)\s*=\s*'([^':#?]+)'", re.IGNORECASE),
    re.compile(r'url\(\s*["\']?([^"\':#?)]+)["\']?\s*\)', re.IGNORECASE),
]


def _find_referenced_files(html_path: Path) -> list[Path]:
    """Scan an HTML file for relative references and return existing sibling paths."""
    try:
        html = html_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    parent = html_path.parent
    refs: set[Path] = set()

    for pattern in _REF_PATTERNS:
        for match in pattern.finditer(html):
            ref = match.group(1).strip()
            # Skip absolute URLs, data URIs, anchors, protocol-relative
            if not ref or ref.startswith(("/", "http:", "https:", "data:", "//")):
                continue
            candidate = (parent / ref).resolve()
            # Only include files that exist and are under the parent directory
            if candidate.is_file() and str(candidate).startswith(str(parent.resolve())):
                refs.add(candidate)

    return sorted(refs)


def _scrub_content(text: str) -> str:
    """Strip LLM API keys and DB credentials from text before it enters the archive."""
    for var in _LLM_SECRET_VARS:
        value = os.environ.get(var, "")
        if value:
            text = text.replace(value, "")
    return scrub_credentials(text)


def _write_scrubbed(zf: zipfile.ZipFile, src: Path, arc_name: str) -> None:
    """Add *src* to *zf* as *arc_name*, scrubbing credentials from text files."""
    if src.suffix.lower() in _TEXT_EXTENSIONS:
        raw = src.read_text(encoding="utf-8", errors="ignore")
        zf.writestr(arc_name, _scrub_content(raw))
    else:
        zf.write(src, arc_name)


def _zip_html(path: Path) -> bytes:
    """Create a ZIP archive from an HTML file (with referenced siblings) or a directory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if path.is_file():
            _write_scrubbed(zf, path, "index.html")
            # Bundle any referenced sibling files (JS, CSS, images, etc.)
            parent = path.resolve().parent
            for ref in _find_referenced_files(path):
                arc_name = str(ref.relative_to(parent))
                _write_scrubbed(zf, ref, arc_name)
        else:
            # Directory — include all files
            for f in sorted(path.rglob("*")):
                if f.is_file():
                    _write_scrubbed(zf, f, str(f.relative_to(path)))
    return buf.getvalue()


def publish(
    file_path: Path,
    *,
    api_key: str,
    publish_url: str = DEFAULT_PUBLISH_URL,
    ssl_verify: bool = True,
) -> dict:
    """Zip and upload an HTML file/directory. Returns the upload response dict.

    Response keys: user_prefix, md5, view_url, files
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Path not found: {file_path}")

    zipped = _zip_html(file_path)
    payload = json.dumps({"file_payload": base64.b64encode(zipped).decode()}).encode()

    url = f"{publish_url.rstrip('/')}/upload"
    raw = minds_request(url, api_key, method="POST", payload=payload, verify=ssl_verify)
    return json.loads(raw)


def list_published(
    *,
    api_key: str,
    publish_url: str = DEFAULT_PUBLISH_URL,
    ssl_verify: bool = True,
) -> list[dict]:
    """List all published reports for the authenticated user.

    Returns list of dicts with keys: md5, title, view_url, files, last_modified, uploaded_at
    """
    url = f"{publish_url.rstrip('/')}/list"
    raw = minds_request(url, api_key, method="GET", verify=ssl_verify)
    data = json.loads(raw)
    return data.get("reports", [])


def unpublish(
    md5: str,
    *,
    api_key: str,
    publish_url: str = DEFAULT_PUBLISH_URL,
    ssl_verify: bool = True,
) -> dict:
    """Delete a published report by its md5 hash. User is derived from the token.

    Returns dict with keys: deleted, md5, files_deleted
    """
    url = f"{publish_url.rstrip('/')}/delete/{md5}"
    raw = minds_request(url, api_key, method="DELETE", verify=ssl_verify)
    return json.loads(raw)
