"""Model/provider wiring for the eval harness + drift detection (ENG-381).

We eval against **minds-cloud** because it's the path real hub users get. The
catch (verified): minds-cloud only accepts its own ``latest:*`` aliases — it
rejects every pinned snapshot ID *and* the bare ``claude-sonnet-4-6`` alias — so
the model **cannot be pinned**. ``latest:sonnet`` is remapped by MindsHub at will
(today it resolves to ``claude-sonnet-4-6``).

Since we can't pin, we do the next best thing: every run records the *resolved*
snapshot (the ``model`` minds echoes back) + the reasoning effort. If that
resolved ID changes between a baseline run and a later one, the comparison is
invalid and we re-baseline — instead of drifting silently.

``provider``/``base_url``/``*_model`` are plain config so switching to
Anthropic-direct with a *real* pinned snapshot later is a one-line change (for a
durable, months-long baseline) — see README.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# minds-cloud default host. minds_chat_base_url logic: api.mindshub.ai -> /v1,
# legacy mdb.ai -> /api/v1.
_MINDS_HOST_DEFAULT = "https://api.mindshub.ai"


@dataclass
class ModelConfig:
    """Which model(s) the eval runs and judges against.

    Defaults reflect what hub users get today: ``latest:sonnet`` for the
    analytical/planning turn (the model ENG-380 targets), ``latest:haiku`` for
    scratchpad coding (mirrors anton's stock coding_model). Effort ``high``
    matches minds-cloud's advertised default for ``latest:sonnet``.
    """

    provider: str = "minds-cloud"
    base_url: str = f"{_MINDS_HOST_DEFAULT}/v1"
    planning_model: str = "latest:sonnet"
    coding_model: str = "latest:haiku"
    judge_model: str = "latest:sonnet"
    planning_effort: str | None = "high"
    coding_effort: str | None = None
    judge_effort: str | None = "high"


def minds_chat_base_url(host: str) -> str:
    base = host.rstrip("/")
    return f"{base}/api/v1" if "mdb.ai" in base else f"{base}/v1"


def load_minds_credentials() -> tuple[str, str]:
    """Read (base_url, api_key) from ``~/.anton/.env``.

    Returns the chat base URL (with the correct ``/v1`` vs ``/api/v1`` suffix)
    and the raw key. Raises if no key is found — the eval can't run blind.
    """
    host = _MINDS_HOST_DEFAULT
    key = ""
    env = Path.home() / ".anton" / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTON_MINDS_URL="):
                host = line.split("=", 1)[1].strip() or host
            elif line.startswith("ANTON_MINDS_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise RuntimeError(
            "No ANTON_MINDS_API_KEY in ~/.anton/.env — can't run the eval against "
            "minds-cloud. Configure MindsHub in the app, or set the key."
        )
    return minds_chat_base_url(host), key


@dataclass
class UsageMeter:
    """Accumulates the SUBJECT's LLM cost over one turn (C11 efficiency).

    Wired in at the provider boundary by ``build_llm_client`` so it sees every
    ``complete()`` / ``stream()`` call the turn makes — across planning, coding,
    and structured-output paths. The judge runs through a separate urllib path
    (``chat_completion``), so judge tokens are correctly excluded.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def record(self, usage) -> None:
        self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

    def as_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
        }


class _MeteredProvider:
    """Transparent proxy that counts calls + sums token usage, delegating
    everything else to the wrapped provider."""

    def __init__(self, inner, meter: UsageMeter) -> None:
        self._inner = inner
        self._meter = meter

    async def complete(self, *args, **kwargs):
        resp = await self._inner.complete(*args, **kwargs)
        self._meter.llm_calls += 1
        self._meter.record(getattr(resp, "usage", None))
        return resp

    async def stream(self, *args, **kwargs):
        from anton.core.llm.provider import StreamComplete

        self._meter.llm_calls += 1
        async for event in self._inner.stream(*args, **kwargs):
            if isinstance(event, StreamComplete):
                self._meter.record(getattr(event.response, "usage", None))
            yield event

    def __getattr__(self, name):  # pass-through (native_web_tools, attrs, …)
        return getattr(self._inner, name)


def build_llm_client(cfg: ModelConfig, api_key: str, base_url: str,
                     meter: UsageMeter | None = None):
    """Construct an anton LLMClient pinned to ``cfg`` — independent of the
    user's DB/.env provider settings (the whole point: we choose the model).

    Mirrors cowork-server's minds-cloud branch: an openai-compatible
    ``OpenAIProvider`` at the minds base URL, effort passed only when set. When
    ``meter`` is given, both providers are wrapped so the turn's token/call cost
    is captured for the C11 efficiency dimension.
    """
    from anton.core.llm.client import LLMClient
    from anton.core.llm.openai import OpenAIProvider

    def _provider(effort: str | None):
        kw = {"reasoning_effort": effort} if effort else {}
        p = OpenAIProvider(api_key=api_key, base_url=base_url, **kw)
        return _MeteredProvider(p, meter) if meter is not None else p

    return LLMClient(
        planning_provider=_provider(cfg.planning_effort),
        planning_model=cfg.planning_model,
        coding_provider=_provider(cfg.coding_effort),
        coding_model=cfg.coding_model,
    )


def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    *,
    system: str | None = None,
    user: str,
    max_tokens: int = 1024,
    effort: str | None = None,
) -> tuple[str, str]:
    """Minimal one-shot chat completion via the OpenAI-compatible endpoint.

    Used for (a) the resolution probe and (b) the LLM judge — both want a plain
    request/response without anton's full turn machinery. Returns
    ``(content, resolved_model)`` where ``resolved_model`` is the ``model`` the
    server echoes back (the snapshot ``latest:*`` resolved to).
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    body: dict = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if effort:
        body["reasoning_effort"] = effort
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Cloudflare (mindshub.ai sits behind it) 403s the default
            # "Python-urllib/x.y" UA, so set an explicit one.
            "User-Agent": "anton-evals/1.0 (+ENG-381)",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    resolved = data.get("model") or "?"
    return content, resolved


def resolve_model(base_url: str, api_key: str, model: str) -> str:
    """Cheap probe: what snapshot does ``model`` (e.g. ``latest:sonnet``) resolve
    to right now? Recorded into every run for drift detection."""
    try:
        _, resolved = chat_completion(
            base_url, api_key, model, user="ok", max_tokens=1
        )
        return resolved
    except urllib.error.HTTPError as exc:
        return f"<probe-http-{exc.code}>"
    except Exception as exc:  # noqa: BLE001 — probe must never crash the run
        return f"<probe-failed: {exc}>"
