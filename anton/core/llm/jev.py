"""Completion verdicts from TypeSafe's Jev, via MindsHub's ``/v1/decisions``.

Failures come back as an error class, never raised, so the caller can fall back.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from dataclasses import dataclass

STATUSES = ("COMPLETE", "WAITING", "INCOMPLETE", "STUCK")


@dataclass(frozen=True)
class JevVerdict:
    """One call. Content-free: labels, numbers and an error class only."""

    status: str = ""
    # Probability of the chosen status; `confidence` is a spread statistic, not this.
    p_status: float | None = None
    p_complete: float | None = None
    ms: int = 0
    model: str = ""
    error: str = ""
    # Billed usage the service reported; None when the reply carried none.
    input_tokens: int | None = None
    output_tokens: int | None = None


def build_questions(status_description: str, rubric: str, close_to_done_description: str) -> dict:
    """The verifier's rubric as one Choice over the four statuses, plus the extra yes/no field."""
    preamble = status_description[: status_description.index("- COMPLETE:")].strip()
    bullets = dict(re.findall(r"^- (COMPLETE|WAITING|INCOMPLETE|STUCK): (.*)$", status_description, re.M))
    if set(bullets) != set(STATUSES):
        raise ValueError("status description no longer has one bullet per status")
    return {
        "status": {"type": "choice", "instructions": f"{preamble} {rubric}", "criteria": bullets},
        "close_to_done": {"type": "noul", "instructions": close_to_done_description},
    }


def _usage(body: object) -> dict:
    """The reported token usage, or nothing when it is missing or malformed."""
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return {}
    tokens = {k: usage.get(k) for k in ("input_tokens", "output_tokens")}
    # bool is an int subclass; a JSON true is not a token count.
    if all(type(v) is int and v >= 0 for v in tokens.values()):
        return tokens
    return {}


def _probability(value: object) -> float:
    # bool is an int subclass; a JSON true must not pass as probability 1.
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError("probability is not a finite number")
    if not 0 <= value <= 1:
        raise ValueError("probability out of range")
    return float(value)


async def classify(
    *,
    base_url: str,
    api_key: str,
    model: str,
    state: dict,
    questions: dict,
    timeout_s: float,
    client=None,
) -> JevVerdict:
    """POST one decision request. Never raises; failures come back in ``error``."""
    import httpx2 as httpx

    started = time.monotonic()

    def elapsed() -> int:
        return round((time.monotonic() - started) * 1000)

    owned = client is None
    client = client or httpx.AsyncClient(timeout=timeout_s)
    try:
        # A hard wall-clock bound: httpx's own timeout is per phase, so a
        # trickling response could otherwise outlive it.
        async with asyncio.timeout(timeout_s):
            response = await client.post(
                f"{base_url.rstrip('/')}/decisions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "state": state, "questions": questions},
            )
    except TimeoutError:
        return JevVerdict(ms=elapsed(), error="timeout")
    except httpx.HTTPError:
        return JevVerdict(ms=elapsed(), error="transport")
    except Exception as exc:  # the shadow must never raise into the turn
        return JevVerdict(ms=elapsed(), error=type(exc).__name__)
    finally:
        if owned:
            await client.aclose()

    ms = elapsed()
    if response.status_code != 200:
        return JevVerdict(ms=ms, error=f"http_{response.status_code}")
    try:
        body = response.json()
    except Exception:
        return JevVerdict(ms=ms, error="malformed")
    # A reply that was billed keeps its usage even when its answer is unusable.
    usage = _usage(body)
    try:
        answer = body["answers"]["status"]
        status = answer["choice"]
        if status not in STATUSES:
            raise ValueError("unknown status")
        probabilities = answer["probabilities"]
        p_status = _probability(probabilities[status])
        p_complete = _probability(probabilities["COMPLETE"])
    except Exception:  # any shape problem is one error class
        return JevVerdict(ms=ms, error="malformed", **usage)
    return JevVerdict(
        status=status, p_status=p_status, p_complete=p_complete, ms=ms, model=str(body.get("model") or ""),
        **usage,
    )
