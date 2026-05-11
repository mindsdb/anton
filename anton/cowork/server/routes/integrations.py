"""Curated integration catalogue for Anton CoWork."""

from __future__ import annotations

import base64
import hashlib
import html as html_lib
import json
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

from .cowork_state import load_state, update_state
from .settings import _get_env

logger = logging.getLogger(__name__)

router = APIRouter()

MANAGED_BEGIN = "# >>> Anton CoWork managed integrations >>>"
MANAGED_END = "# <<< Anton CoWork managed integrations <<<"
USER_DATASOURCES_PATH = Path.home() / ".anton" / "datasources.md"
GOOGLE_DRIVE_ENGINE = "google_drive"
GOOGLE_DRIVE_OAUTH_SCOPES = (
    "openid",
    "email",
    "profile",
    "https://www.googleapis.com/auth/drive",
)
GOOGLE_OAUTH_STATE_KEY = "google_drive_oauth"

GOOGLE_CALENDAR_ENGINE = "google_calendar"
GOOGLE_CALENDAR_OAUTH_SCOPES = (
    "openid",
    "email",
    "profile",
    "https://www.googleapis.com/auth/calendar",
)
GOOGLE_CALENDAR_OAUTH_STATE_KEY = "google_calendar_oauth"

GMAIL_ENGINE = "gmail"
GMAIL_OAUTH_SCOPES = (
    "openid",
    "email",
    "profile",
    "https://www.googleapis.com/auth/gmail.modify",
)
GMAIL_OAUTH_STATE_KEY = "gmail_oauth"

GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_ENDPOINT = "https://openidconnect.googleapis.com/v1/userinfo"

GOOGLE_DRIVE_BLOCK = dedent(
    """
    ## Google Drive

    ```yaml
    engine: google_drive
    display_name: Google Drive
    pip: google-api-python-client google-auth google-auth-httplib2 google-auth-oauthlib
    popular: true
    fields:
      - { name: service_account_keys, required: true, secret: false, description: "path to a Google service account JSON key file on disk" }
      - { name: shared_drive_id, required: false, secret: false, description: "optional shared drive ID to search within" }
      - { name: root_folder_id, required: false, secret: false, description: "optional folder ID to scope file access" }
    test_snippet: |
      import os
      from google.oauth2 import service_account
      from googleapiclient.discovery import build

      key_path = os.environ.get('DS_SERVICE_ACCOUNT_KEYS', '').strip()
      if not key_path:
          raise RuntimeError('Provide a service account JSON key path.')

      creds = service_account.Credentials.from_service_account_file(
          key_path,
          scopes=['https://www.googleapis.com/auth/drive.readonly'],
      )
      service = build('drive', 'v3', credentials=creds, cache_discovery=False)

      shared_drive_id = os.environ.get('DS_SHARED_DRIVE_ID', '').strip()
      root_folder_id = os.environ.get('DS_ROOT_FOLDER_ID', '').strip()
      query = 'trashed = false'
      if root_folder_id:
          query += f" and '{root_folder_id}' in parents"

      params = {
          'pageSize': 1,
          'fields': 'files(id, name, mimeType)',
          'q': query,
          'includeItemsFromAllDrives': True,
          'supportsAllDrives': True,
      }
      if shared_drive_id:
          params.update({'driveId': shared_drive_id, 'corpora': 'drive'})

      service.files().list(**params).execute()
      print('ok')
    ```
    """
).strip()

GOOGLE_CALENDAR_BLOCK = dedent(
    """
    ## Google Calendar

    ```yaml
    engine: google_calendar
    display_name: Google Calendar
    pip: google-api-python-client google-auth google-auth-httplib2 google-auth-oauthlib
    popular: true
    fields:
      - { name: access_token, required: false, secret: true, description: "OAuth access token (managed by Anton)" }
    test_snippet: |
      import os
      from google.oauth2.credentials import Credentials
      from googleapiclient.discovery import build

      creds = Credentials(token=os.environ.get('DS_ACCESS_TOKEN', ''))
      service = build('calendar', 'v3', credentials=creds, cache_discovery=False)
      result = service.calendarList().list(maxResults=1).execute()
      print('ok — calendars:', len(result.get('items', [])))
    ```
    """
).strip()


GMAIL_BLOCK = dedent(
    """
    ## Gmail

    ```yaml
    engine: gmail
    display_name: Gmail
    pip: google-api-python-client google-auth google-auth-httplib2 google-auth-oauthlib
    popular: true
    fields:
      - { name: access_token, required: false, secret: true, description: "OAuth access token (managed by Anton)" }
    test_snippet: |
      import os
      from google.oauth2.credentials import Credentials
      from googleapiclient.discovery import build

      creds = Credentials(token=os.environ.get('DS_ACCESS_TOKEN', ''))
      service = build('gmail', 'v1', credentials=creds, cache_discovery=False)
      result = service.users().getProfile(userId='me').execute()
      print('ok — email:', result.get('emailAddress', ''))
    ```
    """
).strip()


def _replace_managed_block(existing: str, managed_body: str) -> str:
    managed_section = f"{MANAGED_BEGIN}\n\n{managed_body}\n\n{MANAGED_END}\n"
    if MANAGED_BEGIN in existing and MANAGED_END in existing:
        before, _, remainder = existing.partition(MANAGED_BEGIN)
        _, _, after = remainder.partition(MANAGED_END)
        updated = before.rstrip()
        if updated:
            updated += "\n\n"
        updated += managed_section
        trailing = after.lstrip("\n")
        if trailing:
            updated += trailing
        return updated

    updated = existing.rstrip()
    if updated:
        updated += "\n\n"
    updated += managed_section
    return updated


def ensure_managed_integrations() -> Path:
    """Ensure app-managed datasource definitions exist without clobbering user edits."""
    USER_DATASOURCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = (
        USER_DATASOURCES_PATH.read_text(encoding="utf-8")
        if USER_DATASOURCES_PATH.exists()
        else ""
    )

    base_without_managed = existing
    if MANAGED_BEGIN in existing and MANAGED_END in existing:
        before, _, remainder = existing.partition(MANAGED_BEGIN)
        _, _, after = remainder.partition(MANAGED_END)
        base_without_managed = (before.rstrip() + "\n\n" + after.lstrip("\n")).strip()

    managed_blocks: list[str] = []
    if "engine: google_drive" not in base_without_managed:
        managed_blocks.append(GOOGLE_DRIVE_BLOCK)
    if "engine: google_calendar" not in base_without_managed:
        managed_blocks.append(GOOGLE_CALENDAR_BLOCK)
    if "engine: gmail" not in base_without_managed:
        managed_blocks.append(GMAIL_BLOCK)

    managed_body = "\n\n".join(block for block in managed_blocks if block).strip()
    updated = (
        _replace_managed_block(existing, managed_body)
        if managed_body
        else base_without_managed.strip() + ("\n" if base_without_managed.strip() else "")
    )

    if updated != existing:
        USER_DATASOURCES_PATH.write_text(updated, encoding="utf-8")

    return USER_DATASOURCES_PATH


def _field_payload(field) -> dict[str, Any]:
    return {
        "name": field.name,
        "required": field.required,
        "secret": field.secret,
        "description": field.description,
        "default": field.default,
    }


def _server_origin() -> str:
    configured = _get_env("ANTON_GOOGLE_OAUTH_REDIRECT_BASE", "").strip()
    if configured:
        return configured.rstrip("/")
    host = os.environ.get("ANTON_SERVER_HOST", "127.0.0.1").strip() or "127.0.0.1"
    if host in {"0.0.0.0", "::", "[::]"}:
        host = "127.0.0.1"
    port = int(os.environ.get("ANTON_SERVER_PORT", "8765"))
    return f"http://{host}:{port}"


def _google_redirect_uri() -> str:
    return f"{_server_origin()}/v1/integrations/google-drive/oauth/callback"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _google_oauth_meta() -> dict[str, Any]:
    state = load_state()
    utility_state = state.get("utility_state") if isinstance(state, dict) else {}
    meta = utility_state.get(GOOGLE_OAUTH_STATE_KEY) if isinstance(utility_state, dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    if not isinstance(meta.get("pending"), dict):
        meta["pending"] = {}
    return meta


def _write_google_oauth_meta(**updates: Any) -> dict[str, Any]:
    def mutate(state: dict[str, Any]) -> dict[str, Any]:
        utility_state = state.setdefault("utility_state", {})
        meta = utility_state.get(GOOGLE_OAUTH_STATE_KEY)
        if not isinstance(meta, dict):
            meta = {}
        meta.update(updates)
        if not isinstance(meta.get("pending"), dict):
            meta["pending"] = {}
        utility_state[GOOGLE_OAUTH_STATE_KEY] = meta
        return dict(meta)

    return update_state(mutate)


def _clear_google_oauth_pending(**updates: Any) -> dict[str, Any]:
    return _write_google_oauth_meta(pending={}, **updates)


GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")

def _google_oauth_config() -> dict[str, str | bool]:
    client_id = GOOGLE_CLIENT_ID.strip()
    client_secret = GOOGLE_CLIENT_SECRET.strip()
    ready = bool(client_id and client_secret
                 and client_id != "YOUR_CLIENT_ID_HERE"
                 and client_secret != "YOUR_CLIENT_SECRET_HERE")
    return {
        "ready": ready,
        "client_id": client_id,
        "client_secret": client_secret,
        "error": "" if ready else "Google OAuth credentials are not configured.",
    }


def _google_calendar_oauth_config() -> dict[str, str | bool]:
    return _google_oauth_config()


def _google_calendar_oauth_meta() -> dict[str, Any]:
    state = load_state()
    utility_state = state.get("utility_state") if isinstance(state, dict) else {}
    meta = utility_state.get(GOOGLE_CALENDAR_OAUTH_STATE_KEY) if isinstance(utility_state, dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    if not isinstance(meta.get("pending"), dict):
        meta["pending"] = {}
    return meta


def _write_google_calendar_oauth_meta(**updates: Any) -> dict[str, Any]:
    def mutate(state: dict[str, Any]) -> dict[str, Any]:
        utility_state = state.setdefault("utility_state", {})
        meta = utility_state.get(GOOGLE_CALENDAR_OAUTH_STATE_KEY)
        if not isinstance(meta, dict):
            meta = {}
        meta.update(updates)
        if not isinstance(meta.get("pending"), dict):
            meta["pending"] = {}
        utility_state[GOOGLE_CALENDAR_OAUTH_STATE_KEY] = meta
        return dict(meta)
    return update_state(mutate)


def _clear_google_calendar_oauth_pending(**updates: Any) -> dict[str, Any]:
    return _write_google_calendar_oauth_meta(pending={}, **updates)


def _google_calendar_redirect_uri() -> str:
    return f"{_server_origin()}/v1/integrations/google-calendar/oauth/callback"


def _pkce_verifier() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(48)).decode("ascii").rstrip("=")


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _json_request(url: str, *, method: str = "GET", data: dict[str, str] | None = None, headers: dict[str, str] | None = None) -> dict[str, Any]:
    request_headers = {"Accept": "application/json", **(headers or {})}
    body = None
    if data is not None:
        request_headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
        body = urlencode(data).encode("utf-8")
    request = Request(url, data=body, headers=request_headers, method=method)
    try:
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        detail = raw
        try:
            payload = json.loads(raw)
            detail = (
                payload.get("error_description")
                or payload.get("error", {}).get("message")
                or payload.get("error")
                or raw
            )
        except json.JSONDecodeError:
            pass
        raise HTTPException(status_code=502, detail=f"Google OAuth request failed: {detail}") from exc
    except URLError as exc:
        raise HTTPException(status_code=502, detail="Could not reach Google OAuth services") from exc


GOOGLE_REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke"
GOOGLE_OAUTH_ENGINES = {GOOGLE_DRIVE_ENGINE, GOOGLE_CALENDAR_ENGINE, GMAIL_ENGINE}


def revoke_google_token(engine: str, name: str) -> None:
    """Revoke Google OAuth tokens for a connection, removing it from the user's Google permissions page.

    Fire-and-forget — errors are logged but never raised so the caller's
    delete flow is never blocked by a failed revocation.
    """
    import logging
    log = logging.getLogger("integrations.revoke")

    if engine not in GOOGLE_OAUTH_ENGINES:
        return

    try:
        from anton.core.datasources.data_vault import LocalDataVault
        fields = LocalDataVault().load(engine, name) or {}
    except Exception:
        return

    if fields.get("auth_type") != "oauth":
        return

    token = fields.get("refresh_token", "").strip() or fields.get("access_token", "").strip()
    if not token:
        return

    try:
        from urllib.request import urlopen, Request
        req = Request(
            f"{GOOGLE_REVOKE_ENDPOINT}?token={token}",
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urlopen(req, timeout=10):
            pass
        log.info("Revoked Google token for %s/%s", engine, name)
    except Exception as exc:
        log.warning("Could not revoke Google token for %s/%s: %s", engine, name, exc)


def refresh_google_oauth_tokens() -> None:
    """Proactively refresh Google OAuth tokens that are close to expiry.

    Runs in a background thread every 30 minutes. Skips connections that
    are still valid; logs warnings on failure so the app doesn't crash.
    """
    import logging
    log = logging.getLogger("integrations.token-refresh")

    oauth_config = _google_oauth_config()
    if not oauth_config["ready"]:
        return

    try:
        from anton.core.datasources.data_vault import LocalDataVault
        vault = LocalDataVault()
    except Exception:
        return

    engines = {GOOGLE_DRIVE_ENGINE, GOOGLE_CALENDAR_ENGINE, GMAIL_ENGINE}
    now = datetime.now(timezone.utc)
    refresh_threshold = now + timedelta(minutes=10)

    for item in vault.list_connections():
        engine = item.get("engine")
        name = item.get("name")
        if engine not in engines or not name:
            continue

        try:
            fields = vault.load(engine, name) or {}
        except Exception:
            continue

        if fields.get("auth_type") != "oauth":
            continue

        refresh_token = fields.get("refresh_token", "").strip()
        if not refresh_token:
            continue

        expires_at_str = fields.get("expires_at", "").strip()
        if expires_at_str:
            try:
                expires_dt = datetime.fromisoformat(expires_at_str)
                if expires_dt > refresh_threshold:
                    continue  # still valid for >10 minutes, skip
            except ValueError:
                pass  # unparseable — refresh to be safe

        try:
            token_data = _json_request(
                GOOGLE_TOKEN_ENDPOINT,
                method="POST",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": str(oauth_config["client_id"]),
                    "client_secret": str(oauth_config["client_secret"]),
                },
            )
            new_access_token = str(token_data.get("access_token", "")).strip()
            if not new_access_token:
                continue

            expires_in = int(token_data.get("expires_in", 0) or 0)
            new_expires_at = (
                datetime.now(timezone.utc) + timedelta(seconds=expires_in)
            ).isoformat() if expires_in else ""

            updated = {**fields, "access_token": new_access_token, "expires_at": new_expires_at}
            new_refresh_token = str(token_data.get("refresh_token", "")).strip()
            if new_refresh_token:
                updated["refresh_token"] = new_refresh_token

            vault.save(engine, name, updated)
            log.info("Refreshed %s/%s token (expires %s)", engine, name, new_expires_at)
        except Exception as exc:
            log.warning("Could not refresh %s/%s token: %s", engine, name, exc)


def _google_drive_oauth_connections(vault) -> list[dict[str, Any]]:
    connections = []
    for item in vault.list_connections():
        if item.get("engine") != GOOGLE_DRIVE_ENGINE or not item.get("name"):
            continue
        fields = vault.load(GOOGLE_DRIVE_ENGINE, item["name"]) or {}
        if fields.get("auth_type") != "oauth":
            continue
        display_name = fields.get("account_name", "").strip() or fields.get("account_email", "").strip() or item["name"]
        subtitle = fields.get("account_email", "").strip() or item["name"]
        connections.append(
            {
                "engine": GOOGLE_DRIVE_ENGINE,
                "name": item["name"],
                "slug": f"{GOOGLE_DRIVE_ENGINE}-{item['name']}",
                "label": display_name,
                "subtitle": subtitle,
                "connectedVia": "browser_oauth",
                "createdAt": item.get("created_at", ""),
            }
        )
    return connections


def _google_calendar_oauth_connections(vault) -> list[dict[str, Any]]:
    connections = []
    for item in vault.list_connections():
        if item.get("engine") != GOOGLE_CALENDAR_ENGINE or not item.get("name"):
            continue
        fields = vault.load(GOOGLE_CALENDAR_ENGINE, item["name"]) or {}
        if fields.get("auth_type") != "oauth":
            continue
        display_name = fields.get("account_name", "").strip() or fields.get("account_email", "").strip() or item["name"]
        subtitle = fields.get("account_email", "").strip() or item["name"]
        connections.append(
            {
                "engine": GOOGLE_CALENDAR_ENGINE,
                "name": item["name"],
                "slug": f"{GOOGLE_CALENDAR_ENGINE}-{item['name']}",
                "label": display_name,
                "subtitle": subtitle,
                "connectedVia": "browser_oauth",
                "createdAt": item.get("created_at", ""),
            }
        )
    return connections


def _calendar_integration_item(vault) -> dict[str, Any]:
    oauth_config = _google_calendar_oauth_config()
    oauth_meta = _google_calendar_oauth_meta()
    calendar_connections = _google_calendar_oauth_connections(vault)
    return {
        "id": GOOGLE_CALENDAR_ENGINE,
        "title": "Google Calendar",
        "engine": GOOGLE_CALENDAR_ENGINE,
        "status": "connected" if calendar_connections else ("available" if oauth_config["ready"] else "needs_config"),
        "description": "Connect your Google Calendar account so Anton can read and manage your events.",
        "setupMode": "browser_oauth",
        "connections": calendar_connections,
        "connectionCount": len(calendar_connections),
        "engineAvailable": True,
        "oauth": {
            "ready": oauth_config["ready"],
            "configError": oauth_config["error"],
            "pending": bool(oauth_meta.get("pending")),
            "lastSuccessAt": oauth_meta.get("lastSuccessAt", ""),
            "lastError": oauth_meta.get("lastError", ""),
            "lastErrorAt": oauth_meta.get("lastErrorAt", ""),
            "launchLabel": "Connect Google Calendar",
            "redirectUri": _google_calendar_redirect_uri(),
        },
    }


def _integration_item(vault) -> dict[str, Any]:
    oauth_config = _google_oauth_config()
    oauth_meta = _google_oauth_meta()
    drive_connections = _google_drive_oauth_connections(vault)
    notes = [
        "Click Connect Google Drive to open Google sign-in in your browser.",
        "Anton stores the returned Google OAuth credentials in its local data vault under ~/.anton/data_vault/.",
        "Google Drive only shows as connected after the OAuth callback succeeds.",
    ]
    return {
        "id": GOOGLE_DRIVE_ENGINE,
        "title": "Google Drive",
        "engine": GOOGLE_DRIVE_ENGINE,
        "status": "connected" if drive_connections else ("available" if oauth_config["ready"] else "needs_config"),
        "description": "Connect your Google Drive account with Google sign-in so Anton can work with Drive files, Docs, and Sheets.",
        "setupMode": "browser_oauth",
        "connections": drive_connections,
        "connectionCount": len(drive_connections),
        "engineAvailable": True,
        "notes": notes,
        "oauth": {
            "ready": oauth_config["ready"],
            "configError": oauth_config["error"],
            "pending": bool(oauth_meta.get("pending")),
            "lastSuccessAt": oauth_meta.get("lastSuccessAt", ""),
            "lastError": oauth_meta.get("lastError", ""),
            "lastErrorAt": oauth_meta.get("lastErrorAt", ""),
            "launchLabel": "Connect Google Drive",
            "redirectUri": _google_redirect_uri(),
        },
    }


@router.get("")
async def list_integrations():
    ensure_managed_integrations()

    try:
        from anton.core.datasources.data_vault import LocalDataVault
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Anton integration catalogue is unavailable") from exc

    vault = LocalDataVault()
    return {"items": [_integration_item(vault), _calendar_integration_item(vault), _gmail_integration_item(vault)]}


@router.post("/google-drive/oauth/start")
async def start_google_drive_oauth():
    oauth_config = _google_oauth_config()
    if not oauth_config["ready"]:
        raise HTTPException(status_code=400, detail=str(oauth_config["error"]))

    verifier = _pkce_verifier()
    challenge = _pkce_challenge(verifier)
    started_at = _iso_now()
    state = secrets.token_urlsafe(24)
    redirect_uri = _google_redirect_uri()

    _write_google_oauth_meta(
        pending={
            "state": state,
            "verifier": verifier,
            "redirectUri": redirect_uri,
            "startedAt": started_at,
        },
        lastError="",
        lastErrorAt="",
    )

    auth_url = (
        f"{GOOGLE_AUTH_ENDPOINT}?"
        + urlencode(
            {
                "client_id": oauth_config["client_id"],
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "access_type": "offline",
                "include_granted_scopes": "true",
                "prompt": "consent",
                "scope": " ".join(GOOGLE_DRIVE_OAUTH_SCOPES),
                "state": state,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
        )
    )
    return {
        "status": "ok",
        "authUrl": auth_url,
        "redirectUri": redirect_uri,
        "startedAt": started_at,
    }


def _callback_page(title: str, message: str, *, success: bool) -> HTMLResponse:
    accent = "#0f766e" if success else "#b42318"
    safe_title = html_lib.escape(title)
    safe_message = html_lib.escape(message)
    safe_state = "Connected" if success else "Connection failed"
    page_html = dedent(
        f"""
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>{safe_title}</title>
          <style>
            body {{
              margin: 0;
              min-height: 100vh;
              display: grid;
              place-items: center;
              background: #f6f5f1;
              color: #161616;
              font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }}
            main {{
              width: min(92vw, 520px);
              background: #fff;
              border: 1px solid #e7e3da;
              border-radius: 18px;
              padding: 28px 28px 24px;
              box-shadow: 0 20px 60px rgba(20, 17, 12, 0.08);
            }}
            h1 {{
              margin: 0 0 10px;
              font-size: 24px;
              line-height: 1.15;
            }}
            p {{
              margin: 0;
              font-size: 15px;
              line-height: 1.55;
              color: #55514a;
            }}
            .pill {{
              display: inline-flex;
              align-items: center;
              gap: 8px;
              margin-bottom: 14px;
              padding: 6px 10px;
              border-radius: 999px;
              background: #f7f2ee;
              color: {accent};
              font-size: 12px;
              font-weight: 600;
              letter-spacing: 0.02em;
              text-transform: uppercase;
            }}
          </style>
        </head>
        <body>
          <main>
            <div class="pill">{safe_state}</div>
            <h1>{safe_title}</h1>
            <p>{safe_message}</p>
          </main>
        </body>
        </html>
        """
    ).strip()
    return HTMLResponse(content=page_html)


@router.get("/google-drive/oauth/callback")
async def google_drive_oauth_callback(
    code: str = Query(default=""),
    state: str = Query(default=""),
    error: str = Query(default=""),
):
    oauth_meta = _google_oauth_meta()
    pending = oauth_meta.get("pending") or {}
    if error:
        _clear_google_oauth_pending(lastError=f"Google sign-in returned: {error}", lastErrorAt=_iso_now())
        return _callback_page(
            "Google Drive connection was cancelled",
            "You can return to Anton CoWork and try the connection again whenever you are ready.",
            success=False,
        )

    if not pending:
        return _callback_page(
            "Google Drive sign-in expired",
            "Anton CoWork could not find a pending Google Drive sign-in request. Start the connection again from Customize.",
            success=False,
        )

    pending_state = str(pending.get("state", "")).strip()
    if not state or state != pending_state:
        _clear_google_oauth_pending(lastError="Google sign-in state did not match the pending request.", lastErrorAt=_iso_now())
        return _callback_page(
            "Google Drive connection could not be verified",
            "Anton CoWork rejected the callback because the Google sign-in state did not match.",
            success=False,
        )

    if not code:
        _clear_google_oauth_pending(lastError="Google sign-in did not return an authorization code.", lastErrorAt=_iso_now())
        return _callback_page(
            "Google Drive connection could not be completed",
            "Google did not return an authorization code to Anton CoWork.",
            success=False,
        )

    oauth_config = _google_oauth_config()
    if not oauth_config["ready"]:
        _clear_google_oauth_pending(lastError=str(oauth_config["error"]), lastErrorAt=_iso_now())
        return _callback_page(
            "Google Drive connection is not configured",
            str(oauth_config["error"]),
            success=False,
        )

    started_at = str(pending.get("startedAt", "")).strip()
    if started_at:
        try:
            started_dt = datetime.fromisoformat(started_at)
            if datetime.now(timezone.utc) - started_dt > timedelta(minutes=20):
                _clear_google_oauth_pending(lastError="Google sign-in timed out before it completed.", lastErrorAt=_iso_now())
                return _callback_page(
                    "Google Drive sign-in expired",
                    "That Google sign-in request took too long. Start the connection again from Customize.",
                    success=False,
                )
        except ValueError:
            pass

    try:
        token_data = _json_request(
            GOOGLE_TOKEN_ENDPOINT,
            method="POST",
            data={
                "code": code,
                "client_id": str(oauth_config["client_id"]),
                "client_secret": str(oauth_config["client_secret"]),
                "redirect_uri": str(pending.get("redirectUri") or _google_redirect_uri()),
                "grant_type": "authorization_code",
                "code_verifier": str(pending.get("verifier", "")),
            },
        )
        access_token = str(token_data.get("access_token", "")).strip()
        if not access_token:
            raise HTTPException(status_code=502, detail="Google OAuth token exchange did not return an access token.")

        userinfo = _json_request(
            GOOGLE_USERINFO_ENDPOINT,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        account_email = str(userinfo.get("email", "")).strip()
        account_name = str(userinfo.get("name", "")).strip()
        connection_name = account_email or "google_drive"
        expires_in = int(token_data.get("expires_in", 0) or 0)
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        ).isoformat() if expires_in else ""

        try:
            from anton.core.datasources.data_vault import LocalDataVault
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Anton data vault is unavailable") from exc

        LocalDataVault().save(
            GOOGLE_DRIVE_ENGINE,
            connection_name,
            {
                "auth_type": "oauth",
                "access_token": access_token,
                "refresh_token": str(token_data.get("refresh_token", "")).strip(),
                "token_type": str(token_data.get("token_type", "Bearer")).strip(),
                "scope": str(token_data.get("scope", "")).strip(),
                "expires_at": expires_at,
                "account_email": account_email,
                "account_name": account_name,
            },
        )
    except HTTPException as exc:
        _clear_google_oauth_pending(lastError=str(exc.detail), lastErrorAt=_iso_now())
        return _callback_page(
            "Google Drive connection failed",
            str(exc.detail),
            success=False,
        )
    except Exception as exc:
        _clear_google_oauth_pending(lastError=str(exc), lastErrorAt=_iso_now())
        return _callback_page(
            "Google Drive connection failed",
            "Anton CoWork could not finish the Google sign-in flow.",
            success=False,
        )

    _clear_google_oauth_pending(lastError="", lastErrorAt="", lastSuccessAt=_iso_now())
    return _callback_page(
        "Google Drive connected",
        f"{account_name or account_email or 'Your Google account'} is now connected. You can close this tab and return to Anton CoWork.",
        success=True,
    )


@router.post("/google-calendar/oauth/start")
async def start_google_calendar_oauth():
    oauth_config = _google_calendar_oauth_config()
    if not oauth_config["ready"]:
        raise HTTPException(status_code=400, detail=str(oauth_config["error"]))

    verifier = _pkce_verifier()
    challenge = _pkce_challenge(verifier)
    started_at = _iso_now()
    state = secrets.token_urlsafe(24)
    redirect_uri = _google_calendar_redirect_uri()

    _write_google_calendar_oauth_meta(
        pending={
            "state": state,
            "verifier": verifier,
            "redirectUri": redirect_uri,
            "startedAt": started_at,
        },
        lastError="",
        lastErrorAt="",
    )

    auth_url = (
        f"{GOOGLE_AUTH_ENDPOINT}?"
        + urlencode(
            {
                "client_id": oauth_config["client_id"],
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "access_type": "offline",
                "include_granted_scopes": "true",
                "prompt": "consent",
                "scope": " ".join(GOOGLE_CALENDAR_OAUTH_SCOPES),
                "state": state,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
        )
    )
    return {
        "status": "ok",
        "authUrl": auth_url,
        "redirectUri": redirect_uri,
        "startedAt": started_at,
    }


@router.get("/google-calendar/oauth/callback")
async def google_calendar_oauth_callback(
    code: str = Query(default=""),
    state: str = Query(default=""),
    error: str = Query(default=""),
):
    oauth_meta = _google_calendar_oauth_meta()
    pending = oauth_meta.get("pending") or {}
    if error:
        _clear_google_calendar_oauth_pending(lastError=f"Google sign-in returned: {error}", lastErrorAt=_iso_now())
        return _callback_page(
            "Google Calendar connection was cancelled",
            "You can return to Anton CoWork and try the connection again whenever you are ready.",
            success=False,
        )

    if not pending:
        return _callback_page(
            "Google Calendar sign-in expired",
            "Anton CoWork could not find a pending Google Calendar sign-in request. Start the connection again from Customize.",
            success=False,
        )

    pending_state = str(pending.get("state", "")).strip()
    if not state or state != pending_state:
        _clear_google_calendar_oauth_pending(lastError="Google sign-in state did not match.", lastErrorAt=_iso_now())
        return _callback_page(
            "Google Calendar connection could not be verified",
            "Anton CoWork rejected the callback because the Google sign-in state did not match.",
            success=False,
        )

    if not code:
        _clear_google_calendar_oauth_pending(lastError="Google sign-in did not return an authorization code.", lastErrorAt=_iso_now())
        return _callback_page(
            "Google Calendar connection could not be completed",
            "Google did not return an authorization code to Anton CoWork.",
            success=False,
        )

    oauth_config = _google_calendar_oauth_config()
    if not oauth_config["ready"]:
        _clear_google_calendar_oauth_pending(lastError=str(oauth_config["error"]), lastErrorAt=_iso_now())
        return _callback_page(
            "Google Calendar connection is not configured",
            str(oauth_config["error"]),
            success=False,
        )

    started_at = str(pending.get("startedAt", "")).strip()
    if started_at:
        try:
            started_dt = datetime.fromisoformat(started_at)
            if datetime.now(timezone.utc) - started_dt > timedelta(minutes=20):
                _clear_google_calendar_oauth_pending(lastError="Google sign-in timed out.", lastErrorAt=_iso_now())
                return _callback_page(
                    "Google Calendar sign-in expired",
                    "That Google sign-in request took too long. Start the connection again from Customize.",
                    success=False,
                )
        except ValueError:
            pass

    try:
        token_data = _json_request(
            GOOGLE_TOKEN_ENDPOINT,
            method="POST",
            data={
                "code": code,
                "client_id": str(oauth_config["client_id"]),
                "client_secret": str(oauth_config["client_secret"]),
                "redirect_uri": str(pending.get("redirectUri") or _google_calendar_redirect_uri()),
                "grant_type": "authorization_code",
                "code_verifier": str(pending.get("verifier", "")),
            },
        )
        access_token = str(token_data.get("access_token", "")).strip()
        if not access_token:
            raise HTTPException(status_code=502, detail="Google OAuth token exchange did not return an access token.")

        userinfo = _json_request(
            GOOGLE_USERINFO_ENDPOINT,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        account_email = str(userinfo.get("email", "")).strip()
        account_name = str(userinfo.get("name", "")).strip()
        connection_name = account_email or "google_calendar"
        expires_in = int(token_data.get("expires_in", 0) or 0)
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        ).isoformat() if expires_in else ""

        try:
            from anton.core.datasources.data_vault import LocalDataVault
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Anton data vault is unavailable") from exc

        LocalDataVault().save(
            GOOGLE_CALENDAR_ENGINE,
            connection_name,
            {
                "auth_type": "oauth",
                "access_token": access_token,
                "refresh_token": str(token_data.get("refresh_token", "")).strip(),
                "token_type": str(token_data.get("token_type", "Bearer")).strip(),
                "scope": str(token_data.get("scope", "")).strip(),
                "expires_at": expires_at,
                "account_email": account_email,
                "account_name": account_name,
            },
        )
    except HTTPException as exc:
        _clear_google_calendar_oauth_pending(lastError=str(exc.detail), lastErrorAt=_iso_now())
        return _callback_page(
            "Google Calendar connection failed",
            str(exc.detail),
            success=False,
        )
    except Exception as exc:
        logger.exception("Google Calendar OAuth callback failed")
        _clear_google_calendar_oauth_pending(lastError=str(exc), lastErrorAt=_iso_now())
        return _callback_page(
            "Google Calendar connection failed",
            "Anton CoWork could not finish the Google sign-in flow.",
            success=False,
        )

    _clear_google_calendar_oauth_pending(lastError="", lastErrorAt="", lastSuccessAt=_iso_now())
    return _callback_page(
        "Google Calendar connected",
        f"{account_name or account_email or 'Your Google account'} is now connected. You can close this tab and return to Anton CoWork.",
        success=True,
    )


# ── Gmail OAuth ──────────────────────────────────────────────────────────────

def _gmail_oauth_config() -> dict[str, str | bool]:
    return _google_oauth_config()


def _gmail_redirect_uri() -> str:
    return f"{_server_origin()}/v1/integrations/gmail/oauth/callback"


def _gmail_oauth_meta() -> dict[str, Any]:
    state = load_state()
    utility_state = state.get("utility_state") if isinstance(state, dict) else {}
    meta = utility_state.get(GMAIL_OAUTH_STATE_KEY) if isinstance(utility_state, dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    if not isinstance(meta.get("pending"), dict):
        meta["pending"] = {}
    return meta


def _write_gmail_oauth_meta(**updates: Any) -> dict[str, Any]:
    def mutate(state: dict[str, Any]) -> dict[str, Any]:
        utility_state = state.setdefault("utility_state", {})
        meta = utility_state.get(GMAIL_OAUTH_STATE_KEY)
        if not isinstance(meta, dict):
            meta = {}
        meta.update(updates)
        if not isinstance(meta.get("pending"), dict):
            meta["pending"] = {}
        utility_state[GMAIL_OAUTH_STATE_KEY] = meta
        return dict(meta)
    return update_state(mutate)


def _clear_gmail_oauth_pending(**updates: Any) -> dict[str, Any]:
    return _write_gmail_oauth_meta(pending={}, **updates)


def _gmail_oauth_connections(vault) -> list[dict[str, Any]]:
    connections = []
    for item in vault.list_connections():
        if item.get("engine") != GMAIL_ENGINE or not item.get("name"):
            continue
        fields = vault.load(GMAIL_ENGINE, item["name"]) or {}
        if fields.get("auth_type") != "oauth":
            continue
        display_name = fields.get("account_name", "").strip() or fields.get("account_email", "").strip() or item["name"]
        subtitle = fields.get("account_email", "").strip() or item["name"]
        connections.append(
            {
                "engine": GMAIL_ENGINE,
                "name": item["name"],
                "slug": f"{GMAIL_ENGINE}-{item['name']}",
                "label": display_name,
                "subtitle": subtitle,
                "connectedVia": "browser_oauth",
                "createdAt": item.get("created_at", ""),
            }
        )
    return connections


def _gmail_integration_item(vault) -> dict[str, Any]:
    oauth_config = _gmail_oauth_config()
    oauth_meta = _gmail_oauth_meta()
    gmail_connections = _gmail_oauth_connections(vault)
    return {
        "id": GMAIL_ENGINE,
        "title": "Gmail",
        "engine": GMAIL_ENGINE,
        "status": "connected" if gmail_connections else ("available" if oauth_config["ready"] else "needs_config"),
        "description": "Connect your Gmail account so Anton can read, search, and send email on your behalf.",
        "setupMode": "browser_oauth",
        "connections": gmail_connections,
        "connectionCount": len(gmail_connections),
        "engineAvailable": True,
        "oauth": {
            "ready": oauth_config["ready"],
            "configError": oauth_config["error"],
            "pending": bool(oauth_meta.get("pending")),
            "lastSuccessAt": oauth_meta.get("lastSuccessAt", ""),
            "lastError": oauth_meta.get("lastError", ""),
            "lastErrorAt": oauth_meta.get("lastErrorAt", ""),
            "launchLabel": "Connect Gmail",
            "redirectUri": _gmail_redirect_uri(),
        },
    }


@router.post("/gmail/oauth/start")
async def start_gmail_oauth():
    oauth_config = _gmail_oauth_config()
    if not oauth_config["ready"]:
        raise HTTPException(status_code=400, detail=str(oauth_config["error"]))

    verifier = _pkce_verifier()
    challenge = _pkce_challenge(verifier)
    started_at = _iso_now()
    state = secrets.token_urlsafe(24)
    redirect_uri = _gmail_redirect_uri()

    _write_gmail_oauth_meta(
        pending={
            "state": state,
            "verifier": verifier,
            "redirectUri": redirect_uri,
            "startedAt": started_at,
        },
        lastError="",
        lastErrorAt="",
    )

    auth_url = (
        f"{GOOGLE_AUTH_ENDPOINT}?"
        + urlencode(
            {
                "client_id": oauth_config["client_id"],
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "access_type": "offline",
                "include_granted_scopes": "true",
                "prompt": "consent",
                "scope": " ".join(GMAIL_OAUTH_SCOPES),
                "state": state,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
        )
    )
    return {
        "status": "ok",
        "authUrl": auth_url,
        "redirectUri": redirect_uri,
        "startedAt": started_at,
    }


@router.get("/gmail/oauth/callback")
async def gmail_oauth_callback(
    code: str = Query(default=""),
    state: str = Query(default=""),
    error: str = Query(default=""),
):
    oauth_meta = _gmail_oauth_meta()
    pending = oauth_meta.get("pending") or {}
    if error:
        _clear_gmail_oauth_pending(lastError=f"Google sign-in returned: {error}", lastErrorAt=_iso_now())
        return _callback_page(
            "Gmail connection was cancelled",
            "You can return to Anton CoWork and try the connection again whenever you are ready.",
            success=False,
        )

    if not pending:
        return _callback_page(
            "Gmail sign-in expired",
            "Anton CoWork could not find a pending Gmail sign-in request. Start the connection again from Customize.",
            success=False,
        )

    pending_state = str(pending.get("state", "")).strip()
    if not state or state != pending_state:
        _clear_gmail_oauth_pending(lastError="Google sign-in state did not match.", lastErrorAt=_iso_now())
        return _callback_page(
            "Gmail connection could not be verified",
            "Anton CoWork rejected the callback because the Google sign-in state did not match.",
            success=False,
        )

    if not code:
        _clear_gmail_oauth_pending(lastError="Google sign-in did not return an authorization code.", lastErrorAt=_iso_now())
        return _callback_page(
            "Gmail connection could not be completed",
            "Google did not return an authorization code to Anton CoWork.",
            success=False,
        )

    oauth_config = _gmail_oauth_config()
    if not oauth_config["ready"]:
        _clear_gmail_oauth_pending(lastError=str(oauth_config["error"]), lastErrorAt=_iso_now())
        return _callback_page(
            "Gmail connection is not configured",
            str(oauth_config["error"]),
            success=False,
        )

    started_at = str(pending.get("startedAt", "")).strip()
    if started_at:
        try:
            started_dt = datetime.fromisoformat(started_at)
            if datetime.now(timezone.utc) - started_dt > timedelta(minutes=20):
                _clear_gmail_oauth_pending(lastError="Google sign-in timed out.", lastErrorAt=_iso_now())
                return _callback_page(
                    "Gmail sign-in expired",
                    "That Google sign-in request took too long. Start the connection again from Customize.",
                    success=False,
                )
        except ValueError:
            pass

    try:
        token_data = _json_request(
            GOOGLE_TOKEN_ENDPOINT,
            method="POST",
            data={
                "code": code,
                "client_id": str(oauth_config["client_id"]),
                "client_secret": str(oauth_config["client_secret"]),
                "redirect_uri": str(pending.get("redirectUri") or _gmail_redirect_uri()),
                "grant_type": "authorization_code",
                "code_verifier": str(pending.get("verifier", "")),
            },
        )
        access_token = str(token_data.get("access_token", "")).strip()
        if not access_token:
            raise HTTPException(status_code=502, detail="Google OAuth token exchange did not return an access token.")

        userinfo = _json_request(
            GOOGLE_USERINFO_ENDPOINT,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        account_email = str(userinfo.get("email", "")).strip()
        account_name = str(userinfo.get("name", "")).strip()
        connection_name = account_email or "gmail"
        expires_in = int(token_data.get("expires_in", 0) or 0)
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        ).isoformat() if expires_in else ""

        try:
            from anton.core.datasources.data_vault import LocalDataVault
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Anton data vault is unavailable") from exc

        LocalDataVault().save(
            GMAIL_ENGINE,
            connection_name,
            {
                "auth_type": "oauth",
                "access_token": access_token,
                "refresh_token": str(token_data.get("refresh_token", "")).strip(),
                "token_type": str(token_data.get("token_type", "Bearer")).strip(),
                "scope": str(token_data.get("scope", "")).strip(),
                "expires_at": expires_at,
                "account_email": account_email,
                "account_name": account_name,
            },
        )
    except HTTPException as exc:
        _clear_gmail_oauth_pending(lastError=str(exc.detail), lastErrorAt=_iso_now())
        return _callback_page(
            "Gmail connection failed",
            str(exc.detail),
            success=False,
        )
    except Exception as exc:
        _clear_gmail_oauth_pending(lastError=str(exc), lastErrorAt=_iso_now())
        return _callback_page(
            "Gmail connection failed",
            str(exc),
            success=False,
        )

    _clear_gmail_oauth_pending(lastError="", lastErrorAt="", lastSuccessAt=_iso_now())
    return _callback_page(
        "Gmail connected",
        f"{account_name or account_email or 'Your Google account'} is now connected. You can close this tab and return to Anton CoWork.",
        success=True,
    )
