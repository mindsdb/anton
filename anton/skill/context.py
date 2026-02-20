"""LLM access for skills.

Skills that need LLM capabilities call ``get_llm()`` to get a pre-configured
client. Credentials and model selection are handled by Anton's runtime — skills
never need API keys.

Usage inside a skill::

    from anton.skill.context import get_llm

    llm = get_llm()
    response = await llm.complete(
        system="You are a helpful classifier.",
        messages=[{"role": "user", "content": text}],
    )
    answer = response.content
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anton.llm.provider import LLMProvider, LLMResponse


class SkillLLM:
    """LLM access pre-configured with Anton's credentials and model.

    Skills receive this via ``get_llm()`` — they never instantiate it directly.
    """

    def __init__(self, provider: LLMProvider, model: str) -> None:
        self._provider = provider
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Call the LLM. Same interface as LLMProvider.complete but model is pre-set."""
        return await self._provider.complete(
            model=self._model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
        )


_current_llm: ContextVar[SkillLLM | None] = ContextVar("_current_llm", default=None)


def get_llm() -> SkillLLM:
    """Get the LLM for the current skill execution.

    No credentials or model selection needed — Anton provides both.

    Raises:
        RuntimeError: If called outside of Anton's skill execution context.
    """
    llm = _current_llm.get()
    if llm is None:
        raise RuntimeError(
            "No LLM available. This function must be called from within a skill "
            "executed by Anton's runtime."
        )
    return llm


def set_skill_llm(provider: LLMProvider, model: str) -> None:
    """Set the LLM for skill execution. Called by the Agent before running skills."""
    _current_llm.set(SkillLLM(provider, model))
