"""Task completion verification for the Anton agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from .llm.provider import LLMProvider
from .llm.structured import generate_object

MAX_VERIFICATION_CONTINUATIONS = 3


class TaskVerification(BaseModel):
    """Result of a task completion check."""

    status: Literal["complete", "incomplete", "stuck"]
    reason: str
    remaining_work: list[str] = []
    blocker: str | None = None


VERIFICATION_SYSTEM_PROMPT = """You are a strict task-completion verifier.

You will receive:
1. The original task classification (summary, success criteria, expected artifacts).
2. Recent conversation history including tool calls and their results.

Determine whether the user's original request has been **fully** completed.

Rules:
- status="complete" — ALL success criteria are met and expected artifacts were produced.
- status="incomplete" — progress was made but some criteria or artifacts are missing. \
List specifically what remains in remaining_work.
- status="stuck" — the agent hit a blocker it cannot resolve on its own (missing permissions, \
unavailable data, API errors, etc). Identify the blocker.

Be strict:
- If the user asked for a dashboard and only got a text answer, that is INCOMPLETE.
- If a data query was expected but none ran, that is INCOMPLETE.
- If the user asked a question and got a correct answer, that is COMPLETE even without artifacts.
- If the agent produced an error and did not recover, that is STUCK.
"""


async def verify_task(
    *,
    llm_provider: LLMProvider,
    model: str,
    classification_context: str,
    history: list[dict],
) -> TaskVerification:
    """Run a structured verification check against conversation history."""
    summary_parts = [classification_context]

    # Take last ~10 messages to keep context reasonable
    recent = history[-10:]
    for msg in recent:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, str):
            summary_parts.append(f"[{role}]: {content[:1500]}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        summary_parts.append(f"[{role}]: {block['text'][:800]}")
                    elif block.get("type") == "tool_result":
                        summary_parts.append(f"[tool_result]: {str(block.get('content', ''))[:800]}")

    messages = [{"role": "user", "content": "\n\n".join(summary_parts)}]

    return await generate_object(
        TaskVerification,
        llm_provider=llm_provider,
        model=model,
        system=VERIFICATION_SYSTEM_PROMPT,
        messages=messages,
        max_tokens=512,
    )
