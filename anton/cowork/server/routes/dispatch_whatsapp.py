"""WhatsApp Business Cloud channel adapter + HTTP routes.

Components:
    WhatsAppBridge   — concrete :class:`ChatBridgeBase` that also satisfies
                       :class:`anton.core.dispatch.adapter.ChannelAdapter`.
                       Webhook-driven; no outbound socket.
    GET  /whatsapp/webhook — Meta verification handshake
                             (``hub.mode=subscribe&hub.verify_token=…&hub.challenge=…``).
    POST /whatsapp/webhook — inbound message delivery (X-Hub-Signature-256).
    GET/PUT /whatsapp/config — operator config panel.

Environment expected (via DataVault, ``DS_WHATSAPP_<ACCOUNT>__*``):
    phone_number_id   — Meta-issued ID of *our* sending phone number. Required.
    access_token      — system-user or temporary user access token. Required.
    verify_token      — operator-chosen string Meta echoes during webhook subscribe. Required.
    app_secret        — used to verify X-Hub-Signature-256 on inbound POSTs. Required.
    business_account_id — WABA id; optional, only used for display.

Optional env vars:
    ANTON_WHATSAPP_ACCOUNT — pin a specific vault account (default: first
                             ``whatsapp-*`` vault entry, then "default").

WhatsApp Cloud notes:
    - Free-form outbound text is allowed only within the 24-hour
      customer-care window after the user last messaged us. Outside that
      window only pre-approved template messages can be sent — we surface
      a clear error instead of silently failing.
    - Text limit is 4 096 chars; :func:`channels.text.split_for_limit`
      handles splitting automatically.
    - Reply anchoring (``context.message_id``) is a future-phase task —
      v1 sends without quoting.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.request import Request as URLRequest, urlopen

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

# anton.core.dispatch ships only on the dispatch-enabled branch of anton.
# When the installed anton lacks it, leave the router mountable but skip
# registration — `from __future__ import annotations` stringifies the type
# hints below so missing names don't break class definition.
try:
    from anton.core.dispatch.adapter import (
        ChannelAdapter,
        ChannelSetup,
        InboundEvent,
        InboundMessage,
        OutboundMessage,
        PlatformAddress,
    )
    from anton.core.dispatch.registry import register_channel_adapter
    _DISPATCH_AVAILABLE = True
except ImportError:
    _DISPATCH_AVAILABLE = False

from ..channels import (
    ChatBridgeBase,
    SignatureMismatch,
    WebhookHandshake,
    load_channel_secrets,
    verify_whatsapp,
)
from .settings import _read_dotenv, _write_dotenv, GLOBAL_ENV_PATH

logger = logging.getLogger(__name__)
router = APIRouter()

GRAPH_API_BASE = "https://graph.facebook.com/v19.0"
WHATSAPP_MAX_TEXT = 4096
# Meta's "customer-care window" — bots may send free-form text within
# this window after the user's last inbound message. Outside, only
# pre-approved templates are permitted.
CUSTOMER_CARE_WINDOW = timedelta(hours=24)


# ---------------------------------------------------------------------------
# WhatsAppBridge
# ---------------------------------------------------------------------------


class WhatsAppBridge(ChatBridgeBase):
    """WhatsApp Business Cloud adapter built on :class:`ChatBridgeBase`.

    Webhook-driven: Meta POSTs every inbound message and status update to
    the configured callback URL. We verify the payload's HMAC, parse out
    text messages, and forward to dispatch.

    Outbound replies are HTTP POSTs to ``{phone_number_id}/messages``
    on the Graph API, gated by the 24-hour customer-care window.
    """

    max_text_length = WHATSAPP_MAX_TEXT
    supports_threads = False

    def __init__(self, account: str, channel_secrets: dict[str, str]) -> None:
        super().__init__(
            channel_type="whatsapp",
            account=account,
            secrets=channel_secrets,
        )
        self._setup: ChannelSetup | None = None
        # Track the user's last inbound timestamp so outbound free-form can
        # check the 24-hour window. Keyed by user phone (E.164, no '+').
        self._last_inbound: dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # ChannelAdapter Protocol
    # ------------------------------------------------------------------

    async def setup(self, setup: ChannelSetup) -> None:
        self._setup = setup
        # Structured fields only — no access_token, app_secret, or verify_token
        # is ever passed to a logger anywhere in this module.
        logger.info(
            "WhatsAppBridge ready",
            extra={
                "channel_type": "whatsapp",
                "account": self.account,
                "phone_number_id_set": bool(self.secrets.get("phone_number_id")),
            },
        )

    async def shutdown(self) -> None:
        self._setup = None

    async def open_dm(self, user_handle: str) -> PlatformAddress:
        """WhatsApp: the user's E.164 phone number IS the chat address.

        Returns a :class:`PlatformAddress` pointing at the user's phone so
        callers can DM without a separate channel-open step. WhatsApp has
        no notion of "opening" a DM — every chat is implicitly a 1:1.
        """
        return PlatformAddress(
            channel_type="whatsapp",
            platform_id=user_handle.lstrip("+"),
            thread_id=None,
        )

    async def deliver(self, message: OutboundMessage) -> None:
        await self.chunked_send(address=message.address, text=message.text)

    async def show_action_card(self, address: PlatformAddress, card: Any) -> None:
        """Render an action card as plain text (interactive list buttons are a follow-on)."""
        bullets = "\n".join(f"  • {o.label}" for o in getattr(card, "options", []))
        text = f"*{card.title}*\n{card.question}\n{bullets}".strip()
        await self.chunked_send(address=address, text=text)

    # ------------------------------------------------------------------
    # ChatBridgeBase abstracts
    # ------------------------------------------------------------------

    def try_handshake(
        self,
        *,
        method: str,
        body: bytes,
        headers,
        query,
    ) -> WebhookHandshake:
        # Meta's webhook verification: GET with hub.mode=subscribe and a
        # verify_token chosen by us. Echo the challenge back as plain text
        # when the token matches; respond 403 when it doesn't, so a
        # misconfigured app doesn't silently re-subscribe.
        if method != "GET":
            return WebhookHandshake(handled=False)
        if query.get("hub.mode") != "subscribe":
            return WebhookHandshake(handled=False)
        expected = (self.secrets.get("verify_token") or "").strip()
        provided = query.get("hub.verify_token", "")
        if not expected or provided != expected:
            return WebhookHandshake(
                handled=True,
                response_body="forbidden",
                content_type="text/plain",
                status_code=403,
            )
        return WebhookHandshake(
            handled=True,
            response_body=query.get("hub.challenge", ""),
            content_type="text/plain",
        )

    def verify_signature(self, *, body: bytes, headers) -> None:
        app_secret = (self.secrets.get("app_secret") or "").strip()
        signature = headers.get("x-hub-signature-256", "")
        verify_whatsapp(body, app_secret=app_secret, signature_header=signature)

    async def parse_inbound(
        self,
        *,
        body: bytes,
        headers,
    ) -> list[InboundEvent]:
        """Translate a WhatsApp webhook envelope into InboundEvents."""
        try:
            payload = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return []

        events: list[InboundEvent] = []
        for entry in (payload.get("entry") or []):
            for change in (entry.get("changes") or []):
                value = change.get("value") or {}
                # Status updates (delivered/read/failed) and template events
                # come down the same webhook — ignore everything but messages.
                for raw in (value.get("messages") or []):
                    event = self._normalize_message(raw)
                    if event is not None:
                        events.append(event)
        return events

    def _normalize_message(self, msg: dict) -> InboundEvent | None:
        """Translate one inbound message dict into an :class:`InboundEvent`.

        Only text messages produce events for v1; media (image, audio,
        document) arrive but are skipped so the bot doesn't echo a confused
        reply. Future phases can map media → :class:`InboundMessage`
        attachments.
        """
        if msg.get("type") != "text":
            return None

        text = ((msg.get("text") or {}).get("body") or "").strip()
        if not text:
            return None

        sender = (msg.get("from") or "").strip()  # E.164 phone, no '+'
        message_id = str(msg.get("id", ""))
        if not sender or not message_id:
            return None

        ts_raw = msg.get("timestamp", "")
        try:
            timestamp = datetime.fromtimestamp(int(ts_raw), timezone.utc)
        except (TypeError, ValueError):
            timestamp = datetime.now(timezone.utc)

        # Refresh the customer-care window every time the user messages us.
        self._last_inbound[sender] = timestamp

        return InboundEvent(
            address=PlatformAddress(
                channel_type="whatsapp",
                platform_id=sender,
                thread_id=None,  # WhatsApp has no thread concept in 1:1 chats
            ),
            message=InboundMessage(
                id=message_id,
                content=text,
                timestamp=timestamp,
                kind="chat",
                sender_id=sender,
                # 1:1 chats only for v1 — every inbound is implicitly a mention.
                is_mention=True,
                is_group=False,
            ),
        )

    # ------------------------------------------------------------------
    # Outbound: send_text
    # ------------------------------------------------------------------

    async def send_text(self, *, address: PlatformAddress, text: str) -> str:
        """POST one text chunk via the Graph API; returns the message_id.

        Refuses to send if the customer-care window has lapsed, since Meta
        will reject the call anyway and the operator should see why.

        Maps 429/5xx Meta responses to :class:`ConnectionError` so the
        bridge's ``send_with_retry`` wrapper retries transient platform
        errors; 4xx responses (other than 429) raise :class:`RuntimeError`
        and fail fast so the operator sees the underlying issue.
        """
        phone_number_id = self.require_secret("phone_number_id")
        access_token = self.require_secret("access_token")

        recipient = address.platform_id
        if not recipient:
            raise RuntimeError("whatsapp outbound missing recipient phone")

        last = self._last_inbound.get(recipient)
        now = datetime.now(timezone.utc)
        if last is None or (now - last) > CUSTOMER_CARE_WINDOW:
            raise RuntimeError(
                "WhatsApp customer-care window expired for this contact; "
                "only pre-approved templates can be sent."
            )

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient,
            "type": "text",
            "text": {"body": text, "preview_url": False},
        }

        result = await asyncio.to_thread(
            self._call_graph_api,
            access_token,
            phone_number_id,
            payload,
        )
        if result.get("error"):
            err = result["error"]
            # Meta returns {"error": {"code": ..., "type": ..., "message": ...}}.
            # Code 4 / 17 / 80007 are documented rate-limit codes; treat them
            # as transient. Subcodes 2200 / 2207 / etc. (server) likewise.
            code = err.get("code") if isinstance(err, dict) else None
            if code in (4, 17, 32, 80007) or err.get("type") == "OAuthException" and code == 17:
                raise ConnectionError(f"whatsapp rate-limit: {err.get('message', err)}")
            raise RuntimeError(
                f"whatsapp send failed: {err.get('message', err) if isinstance(err, dict) else err}"
            )
        msgs = result.get("messages") or []
        if not msgs:
            raise RuntimeError(f"whatsapp send returned no message id: {result}")
        return str(msgs[0].get("id", ""))

    @staticmethod
    def _call_graph_api(token: str, phone_number_id: str, payload: dict) -> dict:
        """Synchronous Graph API call (run via :func:`asyncio.to_thread`).

        Returns the parsed JSON body on both 2xx and 4xx (Meta encodes errors
        as JSON so the caller can inspect the ``error.code``). Re-raises
        5xx/429 ``HTTPError`` as :class:`ConnectionError` so the bridge's
        send-with-retry wrapper retries transient platform errors.
        """
        from urllib.error import HTTPError as _HTTPError

        from channels.bridge import raise_if_retryable

        url = f"{GRAPH_API_BASE}/{phone_number_id}/messages"
        body = json.dumps(payload).encode("utf-8")
        req = URLRequest(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
                "User-Agent": "AntonCoWork/1.0",
            },
        )
        try:
            with urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except _HTTPError as exc:
            raise_if_retryable(exc)
            try:
                return json.loads(exc.read().decode("utf-8"))
            except Exception:
                raise exc from None


# ---------------------------------------------------------------------------
# Adapter factory — runs on import so the registry sees WhatsApp immediately
# ---------------------------------------------------------------------------


async def _whatsapp_adapter_factory() -> ChannelAdapter | None:
    """Build a WhatsAppBridge from the first usable WhatsApp connection.

    Resolution order:
      1. ``ANTON_WHATSAPP_ACCOUNT`` env var — explicit account name.
      2. Any ``whatsapp-*`` entry in the local DataVault.
      3. ``DS_WHATSAPP_DEFAULT__*`` env vars — pure env-var config.
    """
    try:
        from anton.core.datasources.data_vault import LocalDataVault
        vault = LocalDataVault()
        vault_conns = [
            c for c in vault.list_connections() if c.get("engine") == "whatsapp"
        ]
    except Exception:
        vault, vault_conns = None, []

    explicit = os.environ.get("ANTON_WHATSAPP_ACCOUNT", "").strip()
    if explicit:
        candidates = [explicit]
    elif vault_conns:
        candidates = [c["name"] for c in vault_conns]
    else:
        candidates = ["default"]

    for name in candidates:
        fields: dict[str, str] = {}
        if vault is not None:
            try:
                stored = vault.load("whatsapp", name) or {}
                fields.update({k: str(v) for k, v in stored.items() if v})
            except Exception:
                pass
        fields.update(load_channel_secrets("whatsapp", name))
        if fields.get("phone_number_id") and fields.get("access_token"):
            logger.info(
                "WhatsAppBridge factory selected account=%s",
                name,
            )
            return WhatsAppBridge(account=name, channel_secrets=fields)

    logger.debug("No usable WhatsApp credentials found; adapter not started.")
    return None


if _DISPATCH_AVAILABLE:
    register_channel_adapter("whatsapp", _whatsapp_adapter_factory)
else:
    logger.warning(
        "anton.core.dispatch not available; WhatsApp adapter not registered. "
        "Install anton from a dispatch-enabled build to enable WhatsApp dispatch."
    )


# ---------------------------------------------------------------------------
# Webhook routes
# ---------------------------------------------------------------------------


@router.get("/whatsapp/webhook")
async def whatsapp_webhook_verify(request: Request):
    """GET handshake — Meta calls this once when subscribing the webhook URL.

    Always returns text; never JSON. A 403 here is the only signal Meta gives
    to the operator that ``verify_token`` doesn't match.
    """
    if not _DISPATCH_AVAILABLE:
        raise HTTPException(503, detail="anton dispatch module not available")
    from anton.core.dispatch.registry import get_active_adapter

    adapter = get_active_adapter("whatsapp")
    if adapter is None or not isinstance(adapter, WhatsAppBridge):
        raise HTTPException(503, detail="whatsapp adapter not configured")

    handshake = adapter.try_handshake(
        method="GET",
        body=b"",
        headers={k.lower(): v for k, v in request.headers.items()},
        query=dict(request.query_params),
    )
    if not handshake.handled:
        raise HTTPException(400, detail="not a verification request")
    return Response(
        content=handshake.response_body,
        media_type=handshake.content_type,
        status_code=handshake.status_code,
    )


@router.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request):
    """POST endpoint — Meta delivers every inbound message + status update here.

    Order of operations:
        1. Read raw body — needed for HMAC verify.
        2. Verify X-Hub-Signature-256.
        3. Parse + dispatch via ChannelSetup.
        4. Always 200 — Meta retries aggressively on non-2xx and replays
           can amplify a transient parse bug into hours of duplicates.
    """
    if not _DISPATCH_AVAILABLE:
        raise HTTPException(503, detail="anton dispatch module not available")
    from anton.core.dispatch.registry import get_active_adapter

    adapter = get_active_adapter("whatsapp")
    if adapter is None or not isinstance(adapter, WhatsAppBridge):
        raise HTTPException(503, detail="whatsapp adapter not configured")

    body = await request.body()
    headers_lower = {k.lower(): v for k, v in request.headers.items()}

    try:
        adapter.verify_signature(body=body, headers=headers_lower)
    except SignatureMismatch as exc:
        logger.warning("whatsapp signature mismatch: %s", exc)
        raise HTTPException(401, detail="signature mismatch") from exc

    try:
        events = await adapter.parse_inbound(body=body, headers=headers_lower)
    except Exception:
        logger.exception("whatsapp webhook: parse_inbound failed")
        return Response(content="parse error", status_code=200)

    for event in events:
        try:
            if adapter._setup is not None:
                await adapter._setup.on_inbound(event)
        except Exception:
            logger.exception("whatsapp webhook: on_inbound failed")

    return {"ok": True, "events": len(events)}


# ---------------------------------------------------------------------------
# Config panel — read/write WhatsApp credentials from the UI
# ---------------------------------------------------------------------------

# WhatsApp Cloud has no in-app OAuth flow we'd drive — the operator sets up
# the app + system user in Meta Business Manager, then pastes the resulting
# credentials. We persist them under the DataVault env-var convention
# (DS_WHATSAPP_DEFAULT__*) so the existing factory at
# _whatsapp_adapter_factory finds them via load_channel_secrets().
WHATSAPP_PHONE_NUMBER_ID_KEY = "DS_WHATSAPP_DEFAULT__PHONE_NUMBER_ID"
WHATSAPP_ACCESS_TOKEN_KEY = "DS_WHATSAPP_DEFAULT__ACCESS_TOKEN"
WHATSAPP_VERIFY_TOKEN_KEY = "DS_WHATSAPP_DEFAULT__VERIFY_TOKEN"
WHATSAPP_APP_SECRET_KEY = "DS_WHATSAPP_DEFAULT__APP_SECRET"
WHATSAPP_BUSINESS_ACCOUNT_ID_KEY = "DS_WHATSAPP_DEFAULT__BUSINESS_ACCOUNT_ID"

WHATSAPP_ENV_KEYS = (
    WHATSAPP_PHONE_NUMBER_ID_KEY,
    WHATSAPP_ACCESS_TOKEN_KEY,
    WHATSAPP_VERIFY_TOKEN_KEY,
    WHATSAPP_APP_SECRET_KEY,
    WHATSAPP_BUSINESS_ACCOUNT_ID_KEY,
)


class WhatsAppConfigPatch(BaseModel):
    phone_number_id: str | None = None
    access_token: str | None = None
    verify_token: str | None = None
    app_secret: str | None = None
    business_account_id: str | None = None


def _whatsapp_env_value(key: str) -> str:
    """Resolve a WhatsApp env var from process env first, then ~/.anton/.env."""
    val = os.environ.get(key, "").strip()
    if val:
        return val
    return _read_dotenv(GLOBAL_ENV_PATH).get(key, "").strip()


@router.get("/whatsapp/config")
async def whatsapp_get_config():
    """Return whether each WhatsApp env var is set. Never returns the values themselves."""
    flags = {key: bool(_whatsapp_env_value(key)) for key in WHATSAPP_ENV_KEYS}
    return {
        "phone_number_id_set":      flags[WHATSAPP_PHONE_NUMBER_ID_KEY],
        "access_token_set":         flags[WHATSAPP_ACCESS_TOKEN_KEY],
        "verify_token_set":         flags[WHATSAPP_VERIFY_TOKEN_KEY],
        "app_secret_set":           flags[WHATSAPP_APP_SECRET_KEY],
        "business_account_id_set":  flags[WHATSAPP_BUSINESS_ACCOUNT_ID_KEY],
        # Minimum viable: phone_number_id + access_token to call Graph API,
        # verify_token + app_secret to accept and verify Meta's webhook.
        "install_ready": all(
            flags[k] for k in (
                WHATSAPP_PHONE_NUMBER_ID_KEY,
                WHATSAPP_ACCESS_TOKEN_KEY,
                WHATSAPP_VERIFY_TOKEN_KEY,
                WHATSAPP_APP_SECRET_KEY,
            )
        ),
    }


@router.put("/whatsapp/config")
async def whatsapp_put_config(patch: WhatsAppConfigPatch):
    """Persist WhatsApp credentials to ~/.anton/.env and mirror to os.environ.

    Empty strings clear the corresponding key. Restart the server (or trigger
    an adapter refresh elsewhere) for an already-instantiated bridge to pick
    up new secrets — this route does not hot-reload the live bridge.
    """
    fields: dict[str, str | None] = {
        WHATSAPP_PHONE_NUMBER_ID_KEY:     patch.phone_number_id,
        WHATSAPP_ACCESS_TOKEN_KEY:        patch.access_token,
        WHATSAPP_VERIFY_TOKEN_KEY:        patch.verify_token,
        WHATSAPP_APP_SECRET_KEY:          patch.app_secret,
        WHATSAPP_BUSINESS_ACCOUNT_ID_KEY: patch.business_account_id,
    }

    writes: dict[str, str] = {}
    deletes: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue  # field not in the patch — leave existing value alone
        trimmed = value.strip()
        if trimmed:
            writes[key] = trimmed
        else:
            deletes.append(key)

    _write_dotenv(GLOBAL_ENV_PATH, writes, delete_keys=tuple(deletes))

    os.environ.update(writes)
    for key in deletes:
        os.environ.pop(key, None)

    return await whatsapp_get_config()
