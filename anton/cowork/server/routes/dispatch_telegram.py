"""Telegram channel adapter + HTTP routes (long-polling ingress).

Components:
    TelegramBridge    — concrete :class:`ChatBridgeBase` that also satisfies
                        :class:`anton.core.dispatch.adapter.ChannelAdapter`,
                        so the dispatch registry can hand it to the router.
    Long-polling task — on ``setup()``, spawns an asyncio task that calls
                        ``getUpdates`` with ``timeout=30`` (long-poll) so no
                        public webhook URL is required. The same
                        ``setup.on_inbound`` callback the webhook route would
                        use is called for every inbound message.
    POST /webhook     — optional webhook ingress for hosted deployments where
                        a public HTTPS URL is available. Mutually exclusive
                        with long-polling; the bridge auto-detects based on
                        whether ``TELEGRAM_WEBHOOK_URL`` is set.

Environment expected (via DataVault, ``DS_TELEGRAM_<ACCOUNT>__*``) :
    bot_token    — the ``12345:ABC-...`` token from @BotFather. Required.

Optional env vars:
    ANTON_TELEGRAM_ACCOUNT   — pin a specific vault account (default: first
                               ``telegram-*`` vault entry, then "default").
    TELEGRAM_WEBHOOK_URL     — if set, the bridge skips long-polling and
                               registers this URL with Telegram instead.

Telegram API notes:
    - ``getUpdates?offset=N&timeout=30&allowed_updates=["message"]``
      long-polls for up to 30 s then returns.  Setting offset to
      ``last_update_id + 1`` marks previous updates consumed.
    - Text limit is 4 096 chars; :class:`~channels.text.split_for_limit`
      handles splitting automatically.
    - No signing secret — authentication is the bot token in the URL itself.
    - Bot echoes its own messages back via ``from.is_bot``; we skip those.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any
from urllib.request import Request as URLRequest, urlopen


def hmac_compare(a: str, b: str) -> bool:
    """Constant-time string equality for secret-token comparison."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

# anton.core.dispatch ships only on the dispatch-enabled branch of anton.
# When the installed anton lacks it, define the module-level router as a stub
# so main.py can still mount the prefix; the adapter just never registers.
# The class body below uses these symbols only via `from __future__ import
# annotations` (PEP 563), so missing names don't break class definition —
# they're only resolved if/when the bridge actually runs.
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
)
from .settings import _read_dotenv, _write_dotenv, GLOBAL_ENV_PATH

logger = logging.getLogger(__name__)
router = APIRouter()

TELEGRAM_API_BASE = "https://api.telegram.org/bot"
TELEGRAM_MAX_TEXT = 4096
# Long-poll timeout in seconds — Telegram holds the connection open for up to
# this long before returning an empty update list.  Keep < 60 s to play well
# with server-side connection limits.
LONG_POLL_TIMEOUT = 30
# How long to wait before retrying after a network / API error.
POLL_ERROR_BACKOFF = 5.0


# ---------------------------------------------------------------------------
# TelegramBridge
# ---------------------------------------------------------------------------


class TelegramBridge(ChatBridgeBase):
    """Telegram adapter built on :class:`ChatBridgeBase`.

    Uses long-polling (``getUpdates``) by default so no public URL is
    needed.  Drop-in webhook support is available for hosted deployments
    by setting ``TELEGRAM_WEBHOOK_URL`` before startup.
    """

    max_text_length = TELEGRAM_MAX_TEXT

    def __init__(self, account: str, channel_secrets: dict[str, str]) -> None:
        super().__init__(
            channel_type="telegram",
            account=account,
            secrets=channel_secrets,
        )
        self._setup: ChannelSetup | None = None
        self._poll_task: asyncio.Task | None = None
        # Tracks the highest update_id we've confirmed consumed so the next
        # getUpdates call advances the offset and Telegram doesn't re-deliver.
        self._offset: int = 0

    # ------------------------------------------------------------------
    # ChannelAdapter Protocol
    # ------------------------------------------------------------------

    async def setup(self, setup: ChannelSetup) -> None:
        self._setup = setup
        webhook_url = os.environ.get("TELEGRAM_WEBHOOK_URL", "").strip()
        if webhook_url:
            await self._register_webhook(webhook_url)
            logger.info(
                "TelegramBridge ready for account=%s (webhook mode, url=%s)",
                self.account,
                webhook_url,
            )
        else:
            # Delete any previously registered webhook so Telegram delivers
            # updates to our long-poll loop instead.
            await self._delete_webhook()
            self._poll_task = asyncio.create_task(
                self._run_long_poll(),
                name=f"telegram-poll-{self.account}",
            )
            logger.info(
                "TelegramBridge ready for account=%s (long-polling)",
                self.account,
            )

    async def shutdown(self) -> None:
        if self._poll_task is not None and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):
                pass
        self._poll_task = None
        self._setup = None

    async def open_dm(self, user_handle: str) -> PlatformAddress:
        """Telegram: the numeric chat_id IS the DM address.

        Returns a :class:`PlatformAddress` for the chat_id directly. Telegram
        has no separate "open DM" handshake — once the user has DMed the bot
        once, the bot can DM them back using their chat id.
        """
        return PlatformAddress(
            channel_type="telegram",
            platform_id=user_handle,
            thread_id=None,
        )

    async def deliver(self, message: OutboundMessage) -> None:
        await self.chunked_send(address=message.address, text=message.text)

    async def show_action_card(self, address: PlatformAddress, card: Any) -> None:
        """Render an action card as plain text (inline keyboard is a follow-on)."""
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
        # Telegram has no challenge-response handshake — just return not-handled
        # so the webhook route proceeds to parse_inbound.
        return WebhookHandshake(handled=False)

    def verify_signature(self, *, body: bytes, headers) -> None:
        """Verify the X-Telegram-Bot-Api-Secret-Token header (constant-time).

        Telegram doesn't HMAC-sign payloads. The lightweight authenticator is
        the ``secret_token`` you pass to ``setWebhook`` — Telegram echoes it
        back in the ``X-Telegram-Bot-Api-Secret-Token`` header on every
        delivery. When a ``secret_token`` is configured (vault field or env
        var ``DS_TELEGRAM_<ACCOUNT>__SECRET_TOKEN``) we require it; absent
        the secret, webhook ingress is disabled entirely so the route can't
        be spoofed.
        """
        expected = (self.secrets.get("secret_token") or "").strip()
        if not expected:
            raise SignatureMismatch(
                "telegram webhook secret_token not configured; "
                "webhook ingress refuses unauthenticated payloads"
            )
        provided = headers.get("x-telegram-bot-api-secret-token", "")
        if not hmac_compare(expected, provided):
            raise SignatureMismatch("telegram secret_token mismatch")

    async def parse_inbound(
        self,
        *,
        body: bytes,
        headers,
    ) -> list[InboundEvent]:
        """Parse a Telegram Update JSON payload into InboundEvents."""
        try:
            data = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return []
        return self._updates_to_events([data])

    # ------------------------------------------------------------------
    # Long-polling loop
    # ------------------------------------------------------------------

    async def _run_long_poll(self) -> None:
        """Poll getUpdates indefinitely, forwarding messages to the router.

        Uses the recommended long-poll pattern:
          - ``timeout=30`` — Telegram holds the TCP connection open for up to
            30 s waiting for updates before returning an empty list.
          - ``offset=self._offset`` — tells Telegram we've consumed all
            updates up to (offset - 1) so they won't be re-delivered.
          - ``allowed_updates=["message"]`` — ignore other update types
            (inline queries, callback buttons, etc.) for now.
        """
        bot_token = self.secrets.get("bot_token") or ""
        logger.info("Telegram long-poll loop starting (account=%s)", self.account)

        while True:
            try:
                updates = await asyncio.to_thread(
                    self._fetch_updates, bot_token, self._offset
                )
                events = self._updates_to_events(updates)
                for event in events:
                    if self._setup is not None:
                        try:
                            await self._setup.on_inbound(event)
                        except Exception:
                            logger.exception(
                                "router.on_inbound raised for telegram update"
                            )
                # Advance offset past the last consumed update.
                if updates:
                    self._offset = max(u["update_id"] for u in updates) + 1

            except asyncio.CancelledError:
                logger.info(
                    "Telegram long-poll loop cancelled (account=%s)", self.account
                )
                raise
            except Exception:
                logger.exception(
                    "Telegram long-poll error (account=%s), retrying in %.0fs",
                    self.account,
                    POLL_ERROR_BACKOFF,
                )
                await asyncio.sleep(POLL_ERROR_BACKOFF)

    def _fetch_updates(self, bot_token: str, offset: int) -> list[dict]:
        """Synchronous getUpdates call (run via ``asyncio.to_thread``)."""
        url = (
            f"{TELEGRAM_API_BASE}{bot_token}/getUpdates"
            f"?offset={offset}&timeout={LONG_POLL_TIMEOUT}"
            f'&allowed_updates=%5B%22message%22%5D'  # URL-encoded ["message"]
        )
        req = URLRequest(url, headers={"User-Agent": "AntonCoWork/1.0"})
        with urlopen(req, timeout=LONG_POLL_TIMEOUT + 10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not data.get("ok"):
            raise RuntimeError(f"getUpdates failed: {data.get('description', data)}")
        return data.get("result", [])

    # ------------------------------------------------------------------
    # Shared event normalisation (used by both long-poll and webhook paths)
    # ------------------------------------------------------------------

    def _updates_to_events(self, updates: list[dict]) -> list[InboundEvent]:
        """Translate a list of Telegram Update objects into InboundEvents."""
        events: list[InboundEvent] = []
        for update in updates:
            event = self._normalize_update(update)
            if event is not None:
                events.append(event)
        return events

    def _normalize_update(self, update: dict) -> InboundEvent | None:
        """Translate one Telegram Update into an InboundEvent, or None to skip.

        Skips:
          - Updates that carry no ``message`` key (edited_message, poll, etc.)
          - Messages from bots (``from.is_bot == True``) to avoid echo loops.
          - Messages with no text (photo, sticker, document, etc. — future work).
        """
        msg = update.get("message")
        if not isinstance(msg, dict):
            return None

        sender = msg.get("from") or {}
        if sender.get("is_bot"):
            return None

        text = (msg.get("text") or "").strip()
        if not text:
            return None

        chat = msg.get("chat") or {}
        chat_id = str(chat.get("id", ""))
        chat_type = chat.get("type", "private")        # private | group | supergroup | channel
        is_group = chat_type in ("group", "supergroup", "channel")

        message_id = str(msg.get("message_id", ""))
        sender_id = str(sender.get("id", "")) or None

        date = msg.get("date")
        try:
            timestamp = (
                datetime.fromtimestamp(float(date), timezone.utc)
                if date
                else datetime.now(timezone.utc)
            )
        except (TypeError, ValueError):
            timestamp = datetime.now(timezone.utc)

        # In groups the bot must be @mentioned to respond; in private chats
        # every message is implicitly a mention.
        bot_username = self.secrets.get("bot_username", "")
        is_mention = (
            not is_group
            or (bot_username and f"@{bot_username}" in text)
        )

        return InboundEvent(
            address=PlatformAddress(
                channel_type="telegram",
                platform_id=chat_id,
                thread_id=None,  # Telegram threads (topics) are future work
            ),
            message=InboundMessage(
                id=message_id,
                content=text,
                timestamp=timestamp,
                kind="chat",
                sender_id=sender_id,
                is_mention=is_mention,
                is_group=is_group,
            ),
        )

    # ------------------------------------------------------------------
    # Outbound: send_text
    # ------------------------------------------------------------------

    async def send_text(self, *, address: PlatformAddress, text: str) -> str:
        """Send one text chunk via ``sendMessage``; returns the message_id.

        Maps 429/5xx + ``ok: false, error_code in {420, 429, 500..599}`` to
        :class:`ConnectionError` so the bridge retries transient errors.
        """
        bot_token = self.require_secret("bot_token")

        payload = {
            "chat_id": address.platform_id,
            "text": text,
            "parse_mode": "Markdown",
        }

        result = await asyncio.to_thread(self._call_send_message, bot_token, payload)
        if not result.get("ok"):
            # Markdown parse errors are common (e.g. unbalanced backticks).
            # Retry once without parse_mode before giving up.
            if result.get("error_code") == 400:
                payload.pop("parse_mode", None)
                result = await asyncio.to_thread(
                    self._call_send_message, bot_token, payload
                )
        if not result.get("ok"):
            code = result.get("error_code")
            if code in (420, 429) or (isinstance(code, int) and 500 <= code < 600):
                raise ConnectionError(
                    f"telegram transient error code={code}: "
                    f"{result.get('description', '')}"
                )
            raise RuntimeError(
                f"telegram sendMessage failed: {result.get('description', result)}"
            )
        return str((result.get("result") or {}).get("message_id", ""))

    @staticmethod
    def _call_send_message(bot_token: str, payload: dict) -> dict:
        """Synchronous ``sendMessage`` call (run via ``asyncio.to_thread``).

        Returns Telegram's JSON body for both 2xx and 4xx — Telegram encodes
        errors as ``{"ok": false, "error_code": ..., "description": ...}``.
        Re-raises 429/5xx ``HTTPError`` as :class:`ConnectionError` so the
        bridge's send-with-retry wrapper kicks in on transient platform
        errors.
        """
        from urllib.error import HTTPError as _HTTPError

        from channels.bridge import raise_if_retryable

        url = f"{TELEGRAM_API_BASE}{bot_token}/sendMessage"
        body = json.dumps(payload).encode("utf-8")
        req = URLRequest(
            url,
            data=body,
            headers={
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

    # ------------------------------------------------------------------
    # Webhook registration helpers
    # ------------------------------------------------------------------

    async def _register_webhook(self, webhook_url: str) -> None:
        """Tell Telegram to POST updates to ``webhook_url``.

        Passes the configured ``secret_token`` so Telegram echoes it back in
        the ``X-Telegram-Bot-Api-Secret-Token`` header on every delivery,
        which ``verify_signature`` then compares constant-time.
        """
        bot_token = self.secrets.get("bot_token") or ""
        payload: dict[str, Any] = {
            "url": webhook_url,
            "allowed_updates": ["message"],
            "drop_pending_updates": False,
        }
        secret_token = (self.secrets.get("secret_token") or "").strip()
        if secret_token:
            payload["secret_token"] = secret_token
        result = await asyncio.to_thread(
            self._call_api, bot_token, "setWebhook", payload
        )
        if not result.get("ok"):
            logger.error(
                "Failed to register Telegram webhook at %s: %s",
                webhook_url,
                result.get("description"),
            )
        else:
            logger.info("Telegram webhook registered at %s", webhook_url)

    async def _delete_webhook(self) -> None:
        """Remove any previously registered webhook so long-polling works."""
        bot_token = self.secrets.get("bot_token") or ""
        try:
            await asyncio.to_thread(
                self._call_api, bot_token, "deleteWebhook", {"drop_pending_updates": False}
            )
        except Exception:
            logger.debug("deleteWebhook failed (non-fatal)", exc_info=True)

    @staticmethod
    def _call_api(bot_token: str, method: str, payload: dict) -> dict:
        url = f"{TELEGRAM_API_BASE}{bot_token}/{method}"
        body = json.dumps(payload).encode("utf-8")
        req = URLRequest(
            url,
            data=body,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "User-Agent": "AntonCoWork/1.0",
            },
        )
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Adapter factory — runs on import so the registry sees Telegram immediately
# ---------------------------------------------------------------------------


async def _telegram_adapter_factory() -> ChannelAdapter | None:
    """Build a TelegramBridge from the first usable Telegram connection.

    Resolution order:
      1. ``ANTON_TELEGRAM_ACCOUNT`` env var — explicit account name.
      2. Any ``telegram-*`` entry in the local DataVault.
      3. ``DS_TELEGRAM_DEFAULT__*`` env vars — pure env-var config.
    """
    try:
        from anton.core.datasources.data_vault import LocalDataVault
        vault = LocalDataVault()
        vault_conns = [
            c for c in vault.list_connections() if c.get("engine") == "telegram"
        ]
    except Exception:
        vault, vault_conns = None, []

    explicit = os.environ.get("ANTON_TELEGRAM_ACCOUNT", "").strip()
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
                stored = vault.load("telegram", name) or {}
                fields.update({k: str(v) for k, v in stored.items() if v})
            except Exception:
                pass
        fields.update(load_channel_secrets("telegram", name))
        if fields.get("bot_token"):
            logger.info(
                "TelegramBridge factory selected account=%s",
                name,
            )
            return TelegramBridge(account=name, channel_secrets=fields)

    logger.debug("No usable Telegram credentials found; adapter not started.")
    return None


if _DISPATCH_AVAILABLE:
    register_channel_adapter("telegram", _telegram_adapter_factory)
else:
    logger.warning(
        "anton.core.dispatch not available; Telegram adapter not registered. "
        "Install anton from a dispatch-enabled build to enable Telegram dispatch."
    )


@router.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    """Telegram webhook entry point (hosted / ngrok deployments).

    Telegram sends a POST with a JSON Update object.  We parse it and
    forward to the dispatch router via the bridge — same path as long-polling
    but driven by HTTP push instead of polling.

    Order of operations matches the other adapters — handshake (no-op for
    Telegram) → verify_signature (X-Telegram-Bot-Api-Secret-Token compare) →
    parse_inbound. Without a configured ``secret_token`` the verify step
    raises and the route returns 401 — refuses unauthenticated payloads
    instead of trusting the URL secrecy alone.
    """
    if not _DISPATCH_AVAILABLE:
        raise HTTPException(503, detail="anton dispatch module not available")
    from anton.core.dispatch.registry import get_active_adapter

    adapter = get_active_adapter("telegram")
    if adapter is None or not isinstance(adapter, TelegramBridge):
        raise HTTPException(503, detail="telegram adapter not configured")

    body = await request.body()
    headers_lower = {k.lower(): v for k, v in request.headers.items()}

    try:
        adapter.verify_signature(body=body, headers=headers_lower)
    except SignatureMismatch as exc:
        logger.warning("telegram secret_token mismatch: %s", exc)
        raise HTTPException(401, detail="signature mismatch") from exc

    try:
        events = await adapter.parse_inbound(
            body=body,
            headers=headers_lower,
        )
    except Exception:
        logger.exception("telegram webhook: parse_inbound failed")
        # Always return 200 so Telegram doesn't retry indefinitely.
        return Response(content="parse error", status_code=200)

    for event in events:
        try:
            if adapter._setup is not None:
                await adapter._setup.on_inbound(event)
        except Exception:
            logger.exception("telegram webhook: on_inbound failed")

    return Response(content="ok", status_code=200)


# ---------------------------------------------------------------------------
# Config panel — read/write Telegram credentials from the UI
# ---------------------------------------------------------------------------

# Telegram has no OAuth flow — the operator pastes the bot token from
# @BotFather and (optionally) the bot's @username. We persist them under the
# DataVault env-var convention (DS_TELEGRAM_DEFAULT__BOT_TOKEN /
# DS_TELEGRAM_DEFAULT__BOT_USERNAME) so the existing factory at
# _telegram_adapter_factory finds them via load_channel_secrets() without
# any extra wiring. TELEGRAM_WEBHOOK_URL stays as a plain env var because
# setup() reads it directly from os.environ.
TELEGRAM_BOT_TOKEN_KEY = "DS_TELEGRAM_DEFAULT__BOT_TOKEN"
TELEGRAM_BOT_USERNAME_KEY = "DS_TELEGRAM_DEFAULT__BOT_USERNAME"
TELEGRAM_WEBHOOK_URL_KEY = "TELEGRAM_WEBHOOK_URL"

TELEGRAM_ENV_KEYS = (
    TELEGRAM_BOT_TOKEN_KEY,
    TELEGRAM_BOT_USERNAME_KEY,
    TELEGRAM_WEBHOOK_URL_KEY,
)


class TelegramConfigPatch(BaseModel):
    bot_token: str | None = None
    bot_username: str | None = None
    webhook_url: str | None = None


def _telegram_env_value(key: str) -> str:
    """Resolve a Telegram env var from process env first, then ~/.anton/.env."""
    val = os.environ.get(key, "").strip()
    if val:
        return val
    return _read_dotenv(GLOBAL_ENV_PATH).get(key, "").strip()


@router.get("/telegram/config")
async def telegram_get_config():
    """Return whether each Telegram env var is set. Never returns the values themselves."""
    flags = {key: bool(_telegram_env_value(key)) for key in TELEGRAM_ENV_KEYS}
    return {
        "bot_token_set": flags[TELEGRAM_BOT_TOKEN_KEY],
        "bot_username_set": flags[TELEGRAM_BOT_USERNAME_KEY],
        "webhook_url_set": flags[TELEGRAM_WEBHOOK_URL_KEY],
        # The bot can poll as soon as the token is set; everything else is optional.
        "install_ready": flags[TELEGRAM_BOT_TOKEN_KEY],
        # Long-poll is the default; webhook mode kicks in only if the URL is set.
        "mode": "webhook" if flags[TELEGRAM_WEBHOOK_URL_KEY] else "long-poll",
    }


@router.put("/telegram/config")
async def telegram_put_config(patch: TelegramConfigPatch):
    """Persist Telegram credentials to ~/.anton/.env and mirror to os.environ.

    Empty strings clear the corresponding key. Restart the server (or trigger an
    adapter refresh elsewhere) for an already-instantiated bridge to pick up
    new secrets — this route does not hot-reload the live bridge.
    """
    fields: dict[str, str | None] = {
        TELEGRAM_BOT_TOKEN_KEY:    patch.bot_token,
        TELEGRAM_BOT_USERNAME_KEY: patch.bot_username,
        TELEGRAM_WEBHOOK_URL_KEY:  patch.webhook_url,
    }

    writes: dict[str, str] = {}
    deletes: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue   # field not in the patch — leave existing value alone
        trimmed = value.strip()
        # Bot username is normally entered with a leading "@"; strip it so the
        # mention-detection check (`f"@{bot_username}" in text`) in
        # _normalize_update doesn't double up.
        if key == TELEGRAM_BOT_USERNAME_KEY and trimmed.startswith("@"):
            trimmed = trimmed[1:]
        if trimmed:
            writes[key] = trimmed
        else:
            deletes.append(key)

    # Auto-fill bot_username from Telegram's getMe when the operator pasted a
    # token but didn't supply the username — matches the convenience of Slack's
    # OAuth callback handing you the bot identity. Bot username is needed for
    # @mention detection in group chats (see _normalize_update). Skipped when
    # the user explicitly cleared the username (deletes contains the key) or
    # already provided one. Failures are non-fatal: the credentials still save,
    # the user just has to fill the username manually.
    effective_token = writes.get(TELEGRAM_BOT_TOKEN_KEY) or _telegram_env_value(TELEGRAM_BOT_TOKEN_KEY)
    needs_username_lookup = (
        effective_token
        and TELEGRAM_BOT_USERNAME_KEY not in writes
        and TELEGRAM_BOT_USERNAME_KEY not in deletes
        and not _telegram_env_value(TELEGRAM_BOT_USERNAME_KEY)
    )
    if needs_username_lookup:
        try:
            me = await asyncio.to_thread(_telegram_get_me, effective_token)
            resolved = (me.get("username") or "").strip()
            if resolved:
                writes[TELEGRAM_BOT_USERNAME_KEY] = resolved
        except Exception:
            logger.debug("getMe lookup failed; bot_username left blank", exc_info=True)

    _write_dotenv(GLOBAL_ENV_PATH, writes, delete_keys=tuple(deletes))

    os.environ.update(writes)
    for key in deletes:
        os.environ.pop(key, None)

    return await telegram_get_config()


def _telegram_get_me(bot_token: str) -> dict:
    """Synchronous getMe call — returns the ``result`` dict on success.

    Raises if Telegram returns ``ok: false`` so the caller can decide whether
    to surface the failure or swallow it (auto-username lookup swallows it).
    """
    url = f"{TELEGRAM_API_BASE}{bot_token}/getMe"
    req = URLRequest(url, headers={"User-Agent": "AntonCoWork/1.0"})
    with urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not data.get("ok"):
        raise RuntimeError(f"getMe failed: {data.get('description', data)}")
    return data.get("result") or {}