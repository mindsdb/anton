"""Where a turn's tokens actually went — MindsHub, a third party, or on-box.

ENG-1689. ``turn_completed`` could report *how much* a turn cost but not
*where it went*, and every available proxy for that is wrong:

- ``llm_provider`` names the **client shape**, not the destination. Our own
  gateway arrives under two different values — ``minds-cloud`` from the desktop
  DB overlay and ``openai-compatible`` from anything constructed through
  ``AntonSettings`` (the ``_map_minds_cloud_to_openai_compatible`` validator
  rewrites it, and cowork-server's ``setattr`` overlay bypasses the validator;
  ENG-1695). Measured 2026-08-20: counting the gateway as ``minds-cloud`` alone
  undercounts it by 18% — 382 turns, 146 people, 25.2M tokens over 14 days.
- ``planning_model`` needs a live ``/v1/models`` allowlist to interpret (73
  distinct models in two days) and can never be authoritative anyway, since
  nothing stops a user naming a local model ``sonnet``.

Only anton knows the base URL it resolved, so only anton can answer this.

**Scope: this is the PLANNING role's endpoint.** A turn has independent
``planning_provider``, ``coding_provider`` and ``router_provider`` settings, so
roles CAN reach different destinations — and this reports one label, matching
``llm_provider``, which is also planning-only. The exposure is small because
there is a single ``openai_base_url``: two roles both on ``openai-compatible``
share an endpoint, so they diverge only when the provider *types* differ.
Measured 14d: 12.9% of real turns ran a different coding model, but only 0.5%
of tokens, and most of those pairs sit on the same endpoint (``sonnet`` with
``haiku``, both ours). Read this as "where the planning role went", not as a
guarantee that every token in the turn went there.

**The values are a partition over the observed host, not over who pays.** That
distinction is what makes them mutually exclusive: "is this loopback" and "is
this our domain" are facts about a string we hold, whereas "whose money paid"
depends on a server-side gate verdict anton cannot see (that lives on
``usage_events``; see ENG-1611). A rule keyed on intent would put a loopback
proxy forwarding to our gateway in two buckets at once.

Consequences of reading the host and nothing else, both deliberate:

- A local proxy forwarding to our gateway reports ``local``, because that is
  genuinely all anton observes. ``local`` therefore means "the endpoint was on
  this machine or LAN", NOT "definitely not our gateway".
- A private/LAN address counts as ``local``, not only loopback — Ollama and
  LM Studio are routinely reached across the network, and a loopback-only rule
  would file those as ``third-party`` and understate on-box use, which is the
  population this exists to measure.

The raw base URL is **never** emitted: those carry internal corporate
hostnames. Only the three labels below leave the process.
"""

from __future__ import annotations

import ipaddress
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:  # pragma: no cover - typing only
    from anton.config.settings import AntonSettings

#: The endpoint was on this machine or its local network.
ENDPOINT_LOCAL = "local"
#: The endpoint was the MindsHub gateway (any of its host/path shapes).
ENDPOINT_MINDSHUB = "mindshub"
#: The endpoint was somebody else's — OpenRouter, Databricks, a vendor API.
ENDPOINT_THIRD_PARTY = "third-party"
#: Unrecognised provider. Absent beats wrong: a value here would be a guess.
ENDPOINT_UNKNOWN = ""

VALID_ENDPOINT_CLASSES = frozenset(
    {ENDPOINT_LOCAL, ENDPOINT_MINDSHUB, ENDPOINT_THIRD_PARTY}
)

# Registrable domains we serve the gateway from. `mdb.ai` is the legacy host
# and still live. Matched as an exact host or a dot-delimited subdomain —
# never as a substring, which `mindshub.ai.attacker.example` would satisfy.
_MINDSHUB_DOMAINS = ("mindshub.ai", "mdb.ai")

# Hostname suffixes that mean "this machine / this network" without being IPs.
# `.local` is mDNS (Bonjour), how an Ollama box on the LAN is usually named.
_LOCAL_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home.arpa")

# Providers whose destination is fixed by the provider itself, so the base URL
# is irrelevant (and, for `anthropic`, not even passed — see
# `LLMClient.from_settings`). `gemini` is absent from AntonSettings entirely
# and only ever arrives via cowork-server's overlay (ENG-1695).
_VENDOR_FIXED = {
    "anthropic": ENDPOINT_THIRD_PARTY,
    "gemini": ENDPOINT_THIRD_PARTY,
    "minds-cloud": ENDPOINT_MINDSHUB,
}

# Providers that DO read `openai_base_url`, so the URL decides. Both are in
# `from_settings`' dispatch dict with `base_url=settings.openai_base_url`;
# omitting plain `openai` here would misfile a user who points it at us.
_URL_DECIDED = frozenset({"openai", "openai-compatible"})


def _host_of(base_url: str) -> str:
    """Lowercased hostname from a base URL, tolerant of a missing scheme.

    ``urlsplit`` only populates ``hostname`` when it sees an authority, so a
    bare ``127.0.0.1:11434`` (which users do paste) parses as a *path* and
    yields nothing. Prefixing ``//`` when there is no ``//`` already recovers
    that case. Also strips any IPv6 brackets, which ``hostname`` already does.
    """
    raw = (base_url or "").strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw if "//" in raw else f"//{raw}")
        return (parts.hostname or "").strip().lower().rstrip(".")
    except Exception:  # pragma: no cover - defensive; urlsplit is lenient
        return ""


def _is_local_host(host: str) -> bool:
    """True for loopback, private, link-local, or a local-only name suffix."""
    if host == "localhost" or host.endswith(_LOCAL_SUFFIXES):
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(
        addr.is_loopback
        or addr.is_private
        or addr.is_link_local
        or addr.is_unspecified
    )


def _is_mindshub_host(host: str) -> bool:
    """Exact-or-subdomain match on our domains — deliberately not a substring.

    ``LLMClient``'s vision gate uses ``"mindshub.ai" in host``, which a
    hostile host like ``mindshub.ai.example.com`` satisfies. Unifying those
    call sites onto this helper is ENG-1695; this one is written correctly from
    the start so the new property is not spoofable.
    """
    return any(
        host == domain or host.endswith(f".{domain}") for domain in _MINDSHUB_DOMAINS
    )


def classify_base_url(base_url: str) -> str:
    """Classify a base URL by host alone.

    An empty or unparseable URL is ``third-party``, not unknown: the OpenAI
    SDK falls back to ``api.openai.com`` when handed no base URL, so "nothing
    configured" genuinely does reach a third party.
    """
    host = _host_of(base_url)
    if not host:
        return ENDPOINT_THIRD_PARTY
    if _is_local_host(host):
        return ENDPOINT_LOCAL
    if _is_mindshub_host(host):
        return ENDPOINT_MINDSHUB
    return ENDPOINT_THIRD_PARTY


def classify_endpoint(settings: AntonSettings) -> str:
    """``local`` | ``mindshub`` | ``third-party``, or ``""`` when unrecognised.

    Provider-aware on purpose. ``openai_base_url`` is derived from
    ``minds_url`` by ``AntonSettings.model_post_init`` — but *only* when a
    provider is ``openai-compatible``. So for a vendor-fixed provider the
    field can be set, stale, and pointed at us while the turn never went near
    us; consulting it there would invent gateway traffic. Reading
    ``planning_provider`` first is what prevents that.

    Never raises: this is called from the analytics path, where a failure must
    not disturb the turn that just ran.
    """
    try:
        provider = str(getattr(settings, "planning_provider", "") or "").strip().lower()
        fixed = _VENDOR_FIXED.get(provider)
        if fixed is not None:
            return fixed
        if provider in _URL_DECIDED:
            return classify_base_url(
                str(getattr(settings, "openai_base_url", "") or "")
            )
        return ENDPOINT_UNKNOWN
    except Exception:  # pragma: no cover - defensive
        return ENDPOINT_UNKNOWN
