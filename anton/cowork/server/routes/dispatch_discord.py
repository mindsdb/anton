"""Discord channel adapter + HTTP routes.

Components:
    DiscordBridge      — concrete :class:`ChatBridgeBase` that also satisfies
                         :class:`anton.core.dispatch.adapter.ChannelAdapter`.
    Gateway task       — on ``setup()`` opens an outbound WebSocket via
                         :mod:`discord` (a.k.a. discord.py). Passive listening
                         on all channels the bot can see; no public webhook
                         URL required. Mirrors Slack's Socket Mode shape.
    POST /interactions — Discord interaction endpoint (slash commands /
                         buttons). Verifies Ed25519 signature; the v1
                         body is a stub that just acks PING.
    GET  /discord/oauth/install — returns the install URL for adding the bot
                         to a guild. Discord does NOT mint a per-install
                         bot token; the bot token is set once per app, so
                         there's no token-exchange callback to write here.
    GET/PUT /discord/config — operator config panel.

Environment expected (via DataVault, ``DS_DISCORD_<ACCOUNT>__*``):
    bot_token    — the ``Bot xxxx`` token from the Developer Portal. Required.
    public_key   — application public key (hex). Required for /interactions.
    client_id    — used by the install URL builder.

Optional env vars:
    ANTON_DISCORD_ACCOUNT — pin a specific vault account (default: first
                            ``discord-*`` vault entry, then "default").

Discord API notes:
    - Text limit per message is 2 000 codepoints.
    - Threads are first-class channels — ``msg.channel.id`` already points to
      the thread when a user posts in one, so replies naturally land there
      without explicit ``thread_id`` plumbing.
    - Reply anchoring (``message_reference``) and the slash-command surface
      are tracked as separate phase-1 follow-ons.
"""
from __future__ import annotations

import asyncio
import html as html_lib
import json
import logging
import os
import secrets as secrets_mod
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request as URLRequest, urlopen

from fastapi import APIRouter, HTTPException, Query, Request, Response
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
)
from .cowork_state import load_state, update_state
from .settings import _read_dotenv, _write_dotenv, GLOBAL_ENV_PATH

logger = logging.getLogger(__name__)
router = APIRouter()

DISCORD_API_BASE = "https://discord.com/api/v10"
DISCORD_AUTHORIZE_URL = "https://discord.com/oauth2/authorize"
DISCORD_MAX_TEXT = 2000
# Bot scope plus message-content reading. ``applications.commands`` reserves
# the slash-command surface for a future phase without forcing the operator
# to re-authorize when we add it.
DISCORD_OAUTH_SCOPES = ("bot", "applications.commands")
# View Channels (0x400) + Send Messages (0x800) + Read Message History (0x10000).
DISCORD_BOT_PERMISSIONS = 0x400 | 0x800 | 0x10000
DISCORD_OAUTH_STATE_KEY = "discord_oauth"


# ---------------------------------------------------------------------------
# DiscordBridge
# ---------------------------------------------------------------------------


class DiscordBridge(ChatBridgeBase):
    """Discord adapter built on :class:`ChatBridgeBase`.

    Uses the Gateway for ingress (outbound WebSocket via :mod:`discord`) so
    no public URL is needed. Outbound replies hit the REST API directly via
    :func:`urllib.request.urlopen` for consistency with the other adapters
    and to avoid pulling in another HTTP client.
    """

    max_text_length = DISCORD_MAX_TEXT
    supports_threads = True

    def __init__(self, account: str, channel_secrets: dict[str, str]) -> None:
        super().__init__(
            channel_type="discord",
            account=account,
            secrets=channel_secrets,
        )
        self._setup: ChannelSetup | None = None
        self._gateway_task: Any = None
        self._client: Any = None
        # Cached user id of the bot — populated after the Gateway READY event.
        # Used so on_message can detect @-mentions without re-reading state.
        self._bot_user_id: int | None = None

    # ------------------------------------------------------------------
    # ChannelAdapter Protocol
    # ------------------------------------------------------------------

    async def setup(self, setup: ChannelSetup) -> None:
        self._setup = setup
        bot_token = (self.secrets.get("bot_token") or "").strip()
        if not bot_token:
            logger.warning(
                "DiscordBridge bot_token missing — Gateway not started for account=%s",
                self.account,
            )
            return
        self._gateway_task = asyncio.create_task(
            self._run_gateway(bot_token),
            name=f"discord-gateway-{self.account}",
        )
        logger.info(
            "DiscordBridge ready for account=%s (Gateway mode)",
            self.account,
        )

    async def shutdown(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                logger.debug("Discord client close failed", exc_info=True)
            self._client = None
        if self._gateway_task is not None and not self._gateway_task.done():
            self._gateway_task.cancel()
            try:
                await self._gateway_task
            except (asyncio.CancelledError, Exception):
                pass
        self._gateway_task = None
        self._setup = None

    async def deliver(self, message: OutboundMessage) -> None:
        await self.chunked_send(address=message.address, text=message.text)

    async def show_action_card(self, address: PlatformAddress, card: Any) -> None:
        """Render an action card as plain text (Discord components are a follow-on)."""
        bullets = "\n".join(f"  • {o.label}" for o in getattr(card, "options", []))
        text = f"**{card.title}**\n{card.question}\n{bullets}".strip()
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
        # Discord's only "handshake" is the interaction PING (type=1) on the
        # /interactions endpoint, and that's verified + handled by the route
        # itself so the signature check runs first. Gateway ingress never
        # touches this code path.
        return WebhookHandshake(handled=False)

    def verify_signature(self, *, body: bytes, headers) -> None:
        """Verify a Discord interaction payload via Ed25519.

        Discord uses Ed25519 (not HMAC) for the interactions webhook —
        ``X-Signature-Ed25519`` is hex(sign(application_private_key,
        timestamp || body)). The verifier needs PyNaCl; we import lazily
        and surface a clear error when it's missing so the operator knows
        to add ``PyNaCl`` to requirements rather than silently failing.
        """
        public_key = (self.secrets.get("public_key") or "").strip()
        signature = headers.get("x-signature-ed25519", "")
        timestamp = headers.get("x-signature-timestamp", "")
        if not public_key:
            raise SignatureMismatch("missing discord public_key")
        if not signature or not timestamp:
            raise SignatureMismatch("missing discord interaction headers")
        try:
            from nacl.signing import VerifyKey
            from nacl.exceptions import BadSignatureError
        except ImportError as exc:
            raise SignatureMismatch(
                "PyNaCl not installed; pip install pynacl to enable Discord interactions"
            ) from exc
        try:
            VerifyKey(bytes.fromhex(public_key)).verify(
                timestamp.encode("utf-8") + body,
                bytes.fromhex(signature),
            )
        except (BadSignatureError, ValueError) as exc:
            raise SignatureMismatch("ed25519 verify failed") from exc

    async def parse_inbound(
        self,
        *,
        body: bytes,
        headers,
    ) -> list[InboundEvent]:
        """Translate one Discord interaction payload into InboundEvents.

        v1 only forwards APPLICATION_COMMAND (type=2) interactions, mapped
        as @-mention-equivalent chat events. PING (type=1) is handled by
        the route handler before this method is called.
        """
        try:
            data = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return []
        if not isinstance(data, dict):
            return []
        if data.get("type") != 2:  # APPLICATION_COMMAND
            return []

        channel_id = str(data.get("channel_id", "") or "")
        if not channel_id:
            return []

        user_obj = (data.get("member") or {}).get("user") or data.get("user") or {}
        sender_id = str(user_obj.get("id", "")) or None
        interaction_id = str(data.get("id", "") or "")

        # Slash command options come as a list; concatenate string values into
        # a single utterance so the bot sees `/anton ask "..."` and dispatches
        # something coherent. Future phases can replace this with real
        # command parsing.
        opts = ((data.get("data") or {}).get("options") or [])
        utterance = " ".join(
            str(o.get("value", "")).strip()
            for o in opts
            if isinstance(o, dict) and o.get("value")
        ).strip()
        if not utterance:
            utterance = "/" + str((data.get("data") or {}).get("name", "")).strip()

        return [
            InboundEvent(
                address=PlatformAddress(
                    channel_type="discord",
                    platform_id=channel_id,
                    thread_id=None,
                ),
                message=InboundMessage(
                    id=interaction_id,
                    content=utterance,
                    timestamp=datetime.now(timezone.utc),
                    kind="chat",
                    sender_id=sender_id,
                    is_mention=True,
                    is_group=bool(data.get("guild_id")),
                ),
            )
        ]

    # ------------------------------------------------------------------
    # Gateway — listens for chat messages over an outbound WebSocket
    # ------------------------------------------------------------------

    async def _run_gateway(self, bot_token: str) -> None:
        """Open and serve a Discord Gateway connection via discord.py.

        Lazy import so the cowork server starts cleanly when discord.py
        isn't installed — Slack/Telegram bridges still work; only Discord
        ingress sits idle until the dependency is added.

        The library handles heartbeats, sequence numbers, RESUME on
        reconnect, and intent gating; we just attach an on_message handler
        and route through the bridge's ChannelSetup.
        """
        try:
            import discord  # type: ignore[import-not-found]
        except ImportError:
            logger.error(
                "discord.py not installed; Discord Gateway unavailable. "
                "Add `discord.py` to requirements.txt to enable the bridge."
            )
            return

        intents = discord.Intents.default()
        intents.message_content = True   # Required for non-mention message text
        intents.guild_messages = True
        intents.dm_messages = True

        client = discord.Client(intents=intents)
        self._client = client
        bridge = self

        @client.event
        async def on_ready() -> None:
            bridge._bot_user_id = client.user.id if client.user else None
            logger.info(
                "Discord Gateway ready as %s (id=%s)",
                client.user,
                bridge._bot_user_id,
            )

        @client.event
        async def on_message(msg: "discord.Message") -> None:
            if msg.author.bot:
                return
            event = bridge._message_to_event(msg, discord)
            if event is None or bridge._setup is None:
                return
            try:
                await bridge._setup.on_inbound(event)
            except Exception:
                logger.exception(
                    "router.on_inbound raised for discord message %s",
                    event.message.id,
                )

        try:
            await client.start(bot_token)
        except asyncio.CancelledError:
            try:
                await client.close()
            except Exception:
                pass
            raise
        except Exception:
            logger.exception(
                "Discord Gateway error for account=%s", self.account,
            )

    def _message_to_event(self, msg: Any, discord_mod: Any) -> InboundEvent | None:
        """Translate a ``discord.Message`` into an :class:`InboundEvent`.

        Returns None for messages we should skip (empty text, our own echoes).
        Threads count as standalone channels — ``msg.channel.id`` already
        points to the thread when applicable, so replies anchor there.
        """
        content = (msg.content or "").strip()
        if not content:
            return None

        # DMs in discord.py: msg.guild is None and isinstance(msg.channel, DMChannel)
        is_dm = msg.guild is None
        is_mention = (
            is_dm
            or self._bot_user_id is not None
            and any(getattr(m, "id", None) == self._bot_user_id for m in (msg.mentions or []))
        )

        return InboundEvent(
            address=PlatformAddress(
                channel_type="discord",
                platform_id=str(msg.channel.id),
                thread_id=None,
            ),
            message=InboundMessage(
                id=str(msg.id),
                content=content,
                timestamp=msg.created_at.astimezone(timezone.utc) if msg.created_at else datetime.now(timezone.utc),
                kind="chat",
                sender_id=str(msg.author.id) if msg.author else None,
                is_mention=is_mention,
                is_group=not is_dm,
            ),
        )

    # ------------------------------------------------------------------
    # Outbound: send_text
    # ------------------------------------------------------------------

    async def send_text(self, *, address: PlatformAddress, text: str) -> str:
        """POST one text chunk to the channel; returns the message id.

        Maps Discord's documented rate-limit response (``code 20028 / 30007``
        / ``retry_after``) to :class:`ConnectionError` so the bridge retries.
        """
        bot_token = self.require_secret("bot_token")

        payload: dict[str, Any] = {"content": text}
        result = await asyncio.to_thread(
            self._post_message,
            bot_token,
            address.platform_id,
            payload,
        )
        # A rate-limited Discord response carries ``retry_after`` and either
        # ``code: 0`` with ``global: bool`` or one of the bucket codes.
        if "retry_after" in result:
            raise ConnectionError(
                f"discord rate-limit: retry_after={result.get('retry_after')}"
            )
        if result.get("code") or result.get("message"):
            if result.get("code"):
                raise RuntimeError(
                    f"discord send failed: {result.get('message', result)} "
                    f"(code={result.get('code')})"
                )
        msg_id = result.get("id")
        if not msg_id:
            raise RuntimeError(f"discord send returned no id: {result}")
        return str(msg_id)

    @staticmethod
    def _post_message(bot_token: str, channel_id: str, payload: dict) -> dict:
        """Synchronous ``POST /channels/{id}/messages`` (run via ``asyncio.to_thread``).

        Returns Discord's JSON body for both 2xx and 4xx — Discord encodes
        errors as ``{"code": ..., "message": ...}``. Re-raises 429/5xx
        ``HTTPError`` as :class:`ConnectionError` so the bridge's
        send-with-retry wrapper kicks in on transient platform errors.
        """
        from urllib.error import HTTPError as _HTTPError

        from channels.bridge import raise_if_retryable

        url = f"{DISCORD_API_BASE}/channels/{channel_id}/messages"
        body = json.dumps(payload).encode("utf-8")
        req = URLRequest(
            url,
            data=body,
            headers={
                "Authorization": f"Bot {bot_token}",
                "Content-Type": "application/json; charset=utf-8",
                "User-Agent": "AntonCoWork (https://github.com/mindsdb/cowork, 1.0)",
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
# Adapter factory — runs on import so the registry sees Discord immediately
# ---------------------------------------------------------------------------


async def _discord_adapter_factory() -> ChannelAdapter | None:
    """Build a DiscordBridge from the first usable Discord connection.

    Resolution order:
      1. ``ANTON_DISCORD_ACCOUNT`` env var — explicit account name.
      2. Any ``discord-*`` entry in the local DataVault.
      3. ``DS_DISCORD_DEFAULT__*`` env vars — pure env-var config.
    """
    try:
        from anton.core.datasources.data_vault import LocalDataVault
        vault = LocalDataVault()
        vault_conns = [
            c for c in vault.list_connections() if c.get("engine") == "discord"
        ]
    except Exception:
        vault, vault_conns = None, []

    explicit = os.environ.get("ANTON_DISCORD_ACCOUNT", "").strip()
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
                stored = vault.load("discord", name) or {}
                fields.update({k: str(v) for k, v in stored.items() if v})
            except Exception:
                pass
        fields.update(load_channel_secrets("discord", name))
        if fields.get("bot_token"):
            logger.info(
                "DiscordBridge factory selected account=%s",
                name,
            )
            return DiscordBridge(account=name, channel_secrets=fields)

    logger.debug("No usable Discord credentials found; adapter not started.")
    return None


if _DISPATCH_AVAILABLE:
    register_channel_adapter("discord", _discord_adapter_factory)
else:
    logger.warning(
        "anton.core.dispatch not available; Discord adapter not registered. "
        "Install anton from a dispatch-enabled build to enable Discord dispatch."
    )


# ---------------------------------------------------------------------------
# Interactions webhook — slash commands + button presses
# ---------------------------------------------------------------------------


@router.post("/discord/interactions")
async def discord_interactions(request: Request):
    """Discord interactions entry point.

    Order of operations:
        1. Read raw body — Ed25519 verification signs ``timestamp + body``.
        2. Verify signature — reject 401 on mismatch.
        3. PING (type=1) → respond with PONG (type=1). No dispatch.
        4. APPLICATION_COMMAND → forward via parse_inbound, ack with a
           "thinking" deferred response so Discord doesn't time out the
           interaction (3 s budget).
    """
    if not _DISPATCH_AVAILABLE:
        raise HTTPException(503, detail="anton dispatch module not available")
    from anton.core.dispatch.registry import get_active_adapter

    adapter = get_active_adapter("discord")
    if adapter is None or not isinstance(adapter, DiscordBridge):
        raise HTTPException(503, detail="discord adapter not configured")

    body = await request.body()
    headers_lower = {k.lower(): v for k, v in request.headers.items()}

    try:
        adapter.verify_signature(body=body, headers=headers_lower)
    except SignatureMismatch as exc:
        logger.warning("discord interaction signature mismatch: %s", exc)
        raise HTTPException(401, detail="signature mismatch") from exc

    try:
        data = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(400, detail="invalid interaction payload") from exc

    # PING — Discord pings the URL when the operator sets it in the portal.
    if isinstance(data, dict) and data.get("type") == 1:
        return {"type": 1}

    events = await adapter.parse_inbound(body=body, headers=headers_lower)
    for event in events:
        try:
            if adapter._setup is not None:
                await adapter._setup.on_inbound(event)
        except Exception:
            logger.exception("discord interactions: on_inbound failed")

    # type=5 → DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE (ephemeral). Tells Discord
    # "I'll send the real response via follow-up" so the 3-second budget
    # doesn't trip while Anton is thinking.
    return {"type": 5, "data": {"flags": 64}}


# ---------------------------------------------------------------------------
# OAuth: install URL (Discord doesn't mint per-install bot tokens, so there's
# no token-exchange callback — the bot token is set once in config).
# ---------------------------------------------------------------------------


@router.post("/discord/oauth/install")
async def discord_oauth_install(redirect_uri: str = Query(...)):
    """Build the install URL that adds the bot to a guild.

    Discord's bot OAuth flow puts the bot in the chosen guild and redirects
    back to ``redirect_uri`` with a one-shot ``code``. We don't need the
    code (bot tokens aren't per-install), but we still mint a state nonce
    so the callback can confirm the install came from us.
    """
    client_id = _discord_env_value("DISCORD_CLIENT_ID")
    if not client_id:
        raise HTTPException(
            status_code=400,
            detail="DISCORD_CLIENT_ID must be set. Use the Configure panel in Dispatch.",
        )

    state = secrets_mod.token_urlsafe(24)
    issued = datetime.now(timezone.utc).isoformat()

    def _stash(state_dict):
        state_dict[DISCORD_OAUTH_STATE_KEY] = {
            "state": state,
            "redirect_uri": redirect_uri,
            "issued_at": issued,
        }
    update_state(_stash)

    params = {
        "client_id": client_id,
        "scope": " ".join(DISCORD_OAUTH_SCOPES),
        "permissions": str(DISCORD_BOT_PERMISSIONS),
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "state": state,
    }
    return {"install_url": f"{DISCORD_AUTHORIZE_URL}?{urlencode(params)}"}


@router.get("/discord/oauth/callback")
async def discord_oauth_callback(
    code: str = Query(""),
    state: str = Query(""),
    guild_id: str = Query(""),
):
    """Confirm the install and clear the state nonce.

    Discord redirects here with ``guild_id`` set when the bot was successfully
    added. We don't exchange ``code`` for tokens — the bot token is app-level
    and was configured once via ``/discord/config``.
    """
    saved = load_state().get(DISCORD_OAUTH_STATE_KEY) or {}
    if not saved or saved.get("state") != state:
        return Response(
            content=html_lib.escape("OAuth state mismatch — please retry."),
            status_code=400,
            media_type="text/html",
        )

    update_state(lambda s: s.pop(DISCORD_OAUTH_STATE_KEY, None))  # clear nonce

    if guild_id:
        # Stash the guild_id in DataVault so future routes can list which
        # guilds the bot is installed in without re-querying Discord.
        try:
            from anton.core.datasources.data_vault import LocalDataVault

            vault = LocalDataVault()
            existing = vault.load("discord", guild_id) or {}
            existing.setdefault("bot_token", os.environ.get("DS_DISCORD_DEFAULT__BOT_TOKEN", ""))
            existing.setdefault("public_key", os.environ.get("DS_DISCORD_DEFAULT__PUBLIC_KEY", ""))
            existing["guild_id"] = guild_id
            vault.save("discord", guild_id, existing)
        except Exception:
            logger.exception("could not store Discord guild in DataVault")

    body_html = (
        f"<h2>Discord bot installed</h2>"
        f"<p>Guild {html_lib.escape(guild_id or '<unknown>')} linked. "
        f"You can close this window.</p>"
    )
    return Response(content=body_html, media_type="text/html")


# ---------------------------------------------------------------------------
# Config panel — read/write Discord env vars from the UI
# ---------------------------------------------------------------------------

# CLIENT_ID / CLIENT_SECRET are app-level and used by the install-URL builder;
# BOT_TOKEN / PUBLIC_KEY are persisted under the DataVault env-var convention
# so the existing factory at _discord_adapter_factory finds them via
# load_channel_secrets().
DISCORD_CLIENT_ID_KEY = "DISCORD_CLIENT_ID"
DISCORD_CLIENT_SECRET_KEY = "DISCORD_CLIENT_SECRET"
DISCORD_BOT_TOKEN_KEY = "DS_DISCORD_DEFAULT__BOT_TOKEN"
DISCORD_PUBLIC_KEY_KEY = "DS_DISCORD_DEFAULT__PUBLIC_KEY"

DISCORD_ENV_KEYS = (
    DISCORD_CLIENT_ID_KEY,
    DISCORD_CLIENT_SECRET_KEY,
    DISCORD_BOT_TOKEN_KEY,
    DISCORD_PUBLIC_KEY_KEY,
)


class DiscordConfigPatch(BaseModel):
    client_id: str | None = None
    client_secret: str | None = None
    bot_token: str | None = None
    public_key: str | None = None


def _discord_env_value(key: str) -> str:
    """Resolve a Discord env var from process env first, then ~/.anton/.env."""
    val = os.environ.get(key, "").strip()
    if val:
        return val
    return _read_dotenv(GLOBAL_ENV_PATH).get(key, "").strip()


@router.get("/discord/config")
async def discord_get_config():
    """Return whether each Discord env var is set. Never returns the values themselves."""
    flags = {key: bool(_discord_env_value(key)) for key in DISCORD_ENV_KEYS}
    return {
        "client_id_set":     flags[DISCORD_CLIENT_ID_KEY],
        "client_secret_set": flags[DISCORD_CLIENT_SECRET_KEY],
        "bot_token_set":     flags[DISCORD_BOT_TOKEN_KEY],
        "public_key_set":    flags[DISCORD_PUBLIC_KEY_KEY],
        # Gateway runs as soon as bot_token is set.
        "gateway_ready":  flags[DISCORD_BOT_TOKEN_KEY],
        # Interactions endpoint additionally needs the app public key.
        "interactions_ready": flags[DISCORD_BOT_TOKEN_KEY] and flags[DISCORD_PUBLIC_KEY_KEY],
        # Install URL needs client_id.
        "install_ready":  flags[DISCORD_CLIENT_ID_KEY],
    }


@router.put("/discord/config")
async def discord_put_config(patch: DiscordConfigPatch):
    """Persist Discord credentials to ~/.anton/.env and mirror to os.environ.

    Empty strings clear the corresponding key. Restart the server (or trigger
    an adapter refresh elsewhere) for an already-instantiated bridge to pick
    up new secrets — this route does not hot-reload the live bridge.
    """
    fields: dict[str, str | None] = {
        DISCORD_CLIENT_ID_KEY:     patch.client_id,
        DISCORD_CLIENT_SECRET_KEY: patch.client_secret,
        DISCORD_BOT_TOKEN_KEY:     patch.bot_token,
        DISCORD_PUBLIC_KEY_KEY:    patch.public_key,
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

    return await discord_get_config()
