"""LLM-as-judge — uses Haiku to grade responses against a rubric."""

from __future__ import annotations

import json

from anton.llm.provider import LLMProvider

_JUDGE_SYSTEM = """\
You are an eval judge. You will be given:
1. A rubric describing what a correct response looks like
2. The actual response from an AI assistant
3. Any tool outputs produced during the response

Score the response according to the rubric. Return ONLY valid JSON:
{"score": <float 0.0 to 1.0>, "explanation": "<brief explanation>"}

A score >= 0.5 means PASS. Be strict but fair.\
"""


async def judge_with_llm(
    rubric: str,
    response_text: str,
    tool_outputs: str,
    *,
    provider: LLMProvider,
    model: str = "claude-haiku-4-5-20251001",
) -> tuple[bool, str]:
    """Send the response + rubric to an LLM judge.

    Args:
        rubric: The grading criteria (from the scorer's ``value`` field).
        response_text: Concatenated assistant response text.
        tool_outputs: Concatenated tool result text (e.g. scratchpad stdout).
        provider: An LLMProvider instance (typically the coding provider).
        model: Which model to use for judging.

    Returns:
        Tuple of (passed: bool, explanation: str).
    """
    user_message = (
        f"## Rubric\n{rubric}\n\n"
        f"## Assistant Response\n{response_text}\n\n"
        f"## Tool Outputs\n{tool_outputs or '(none)'}"
    )

    try:
        response = await provider.complete(
            model=model,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=512,
            temperature=0.0,
        )

        # Parse the JSON verdict
        raw = response.content.strip()
        # Handle markdown code fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        verdict = json.loads(raw)
        score = float(verdict.get("score", 0))
        explanation = verdict.get("explanation", "")
        return score >= 0.5, f"score={score:.2f}: {explanation}"

    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return False, f"Judge response parsing failed: {exc}"
    except Exception as exc:
        # Graceful fallback — don't crash the eval for judge failures
        return True, f"WARN: judge call failed ({exc}), defaulting to pass"
