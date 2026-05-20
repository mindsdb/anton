"""Passthrough LLM agent — dispatcher.

Routing is driven by ``self.config.api_kind``; the per-provider modules
(``anthropic``, ``openai``, ``gemini``) hold the translation, proxy, and
streaming logic. This module only:

1. Owns the ``PassthroughAgent`` class — the object the request handler
   instantiates and calls ``proxy()`` on.
2. Re-exports the names tests historically reached for via
   ``from minds.agents.passthrough_agent.agent import _translate_tools_for_anthropic``
   etc., so the file split is invisible to callers.

Tool translation per destination is gated by ``self.config.web_search_mode``;
provider-specific native search hooks in for OpenAI/Anthropic/Gemini, and
``"drop"`` silently strips generic ``web_search``/``fetch`` entries (used for
Fireworks today, which has no hosted search index).
"""

from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI
from starlette.responses import JSONResponse, StreamingResponse

# Provider modules are imported under aliases so the top-level Python
# packages (``anthropic``, ``openai``) referenced inside each one resolve
# to the SDK packages, not our submodules.
from minds.agents.passthrough_agent import anthropic as _anthropic_mod
from minds.agents.passthrough_agent import gemini as _gemini_mod
from minds.agents.passthrough_agent import openai as _openai_mod
from minds.agents.passthrough_agent.common import (
    UsageBox,
    _messages_to_dicts,
)
from minds.common.logger import setup_logging
from minds.common.passthrough_config import ApiKind, PassthroughModelConfig
from minds.schemas.chat import Message

# SDK client classes are imported at module top so tests can monkeypatch them
# at this module path (e.g. ``minds.agents.passthrough_agent.agent.AsyncAnthropic``);
# the instance-method ``_get_*_client`` factories below reference these names
# explicitly so the swap takes effect.

logger = setup_logging()


# ---------------------------------------------------------------------------
# PassthroughAgent
# ---------------------------------------------------------------------------


class PassthroughAgent:
    """Fully OpenAI-compatible proxy to the configured LLM provider.

    Forwards the complete request — including tools, tool_choice, temperature,
    max_tokens — to the upstream provider and returns the raw response.
    Supports both streaming and non-streaming for OpenAI, Anthropic and Gemini.
    """

    def __init__(
        self,
        config: PassthroughModelConfig,
        instrument: bool = True,
    ):
        self.config = config
        self._usage_box = UsageBox()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def proxy(
        self,
        messages: list[Message],
        stream: bool,
        request_id: str,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> StreamingResponse | JSONResponse:
        logger.debug(
            "proxy called",
            extra={
                "api_kind": self.config.api_kind,
                "provider_label": self.config.label,
                "model": self.config.model_name,
                "alias": self.config.alias,
                "reasoning_effort": self.config.reasoning_effort,
                "message_count": len(messages),
                "stream": stream,
                "request_id": request_id,
            },
        )
        msg_dicts = _messages_to_dicts(messages)
        kwargs: dict[str, Any] = {
            "messages": msg_dicts,
            "stream": stream,
            "request_id": request_id,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if self.config.api_kind == ApiKind.ANTHROPIC_MESSAGES:
            return await self._proxy_anthropic(**kwargs)
        if self.config.api_kind == ApiKind.GEMINI_NATIVE:
            return await self._proxy_gemini(**kwargs)
        # ApiKind.OPENAI_RESPONSES (default)
        return await self._proxy_openai(**kwargs)

    async def get_last_run_usage(self) -> tuple[int, int] | None:
        return self._usage_box.value

    # ------------------------------------------------------------------
    # Thin instance-method wrappers around the module-level proxies.
    # Kept on the class so tests and external callers that historically
    # exercised ``agent._proxy_openai(...)`` / ``agent._get_openai_client()``
    # continue to work after the file split.
    # ------------------------------------------------------------------

    async def _proxy_openai(self, **kwargs) -> StreamingResponse | JSONResponse:
        return await _openai_mod.proxy_openai(
            client=self._get_openai_client(),
            config=self.config,
            usage_box=self._usage_box,
            **kwargs,
        )

    async def _proxy_anthropic(self, **kwargs) -> StreamingResponse | JSONResponse:
        return await _anthropic_mod.proxy_anthropic(
            client=self._get_anthropic_client(),
            config=self.config,
            usage_box=self._usage_box,
            **kwargs,
        )

    async def _proxy_gemini(self, **kwargs) -> StreamingResponse | JSONResponse:
        return await _gemini_mod.proxy_gemini(
            client=self._get_gemini_client(),
            config=self.config,
            usage_box=self._usage_box,
            **kwargs,
        )

    # SDK-client factories live on the class so tests can monkeypatch them
    # (``monkeypatch.setattr(agent, "_get_anthropic_client", lambda: fake)``)
    # AND so tests can swap the SDK constructor at
    # ``minds.agents.passthrough_agent.agent.AsyncAnthropic`` and have the
    # change reach the construction site.

    def _get_openai_client(self) -> AsyncOpenAI:
        kwargs: dict[str, Any] = {"api_key": self.config.api_key}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        return AsyncOpenAI(**kwargs)

    def _get_anthropic_client(self) -> AsyncAnthropic:
        kwargs: dict[str, Any] = {"api_key": self.config.api_key}
        # base_url is set for Anthropic-compatible proxies (e.g., Fireworks);
        # leave unset to use the SDK default for direct Anthropic.
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        return AsyncAnthropic(**kwargs)

    def _get_gemini_client(self) -> genai.Client:
        return genai.Client(api_key=self.config.api_key)


# ---------------------------------------------------------------------------
# Backward-compat re-exports
# ---------------------------------------------------------------------------
#
# Tests and external callers used to do
# ``from minds.agents.passthrough_agent.agent import _translate_tools_for_anthropic``
# (etc.). Each provider module declares ``__all__`` listing the private
# helpers worth keeping reachable; star-import them here so those import
# sites work unchanged after the split.

from minds.agents.passthrough_agent.anthropic import *  # noqa: E402, F401, F403
from minds.agents.passthrough_agent.common import *  # noqa: E402, F401, F403
from minds.agents.passthrough_agent.gemini import *  # noqa: E402, F401, F403
from minds.agents.passthrough_agent.openai import *  # noqa: E402, F401, F403
