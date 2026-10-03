"""Completion verdicts from Jev, through MindsHub or an explicit TypeSafe connection.

Failures come back as an error class, never raised, so the caller can fall back.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

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
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_attempted: bool = False


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
    api_path: str = "decisions",
) -> JevVerdict:
    """POST one decision request. Never raises; failures come back in ``error``."""
    import httpx2 as httpx

    started = time.monotonic()

    def elapsed() -> int:
        return round((time.monotonic() - started) * 1000)

    receipt = {"call_id": uuid.uuid4().hex, "requested_model": model, "api_path": api_path}

    def finish(verdict: JevVerdict) -> JevVerdict:
        # Billing evidence must survive an uncertain verdict, fallback or cancelled turn.
        # Neither the credential nor the submitted conversation belongs in this record.
        logger.info("jev request_receipt=%s", json.dumps({**receipt, **asdict(verdict)}))
        return verdict

    owned = client is None
    client = client or httpx.AsyncClient(timeout=timeout_s)
    logger.info("jev request_start=%s", json.dumps(receipt))
    try:
        # A hard wall-clock bound: httpx's own timeout is per phase, so a
        # trickling response could otherwise outlive it.
        async with asyncio.timeout(timeout_s):
            response = await client.post(
                f"{base_url.rstrip('/')}/{api_path}",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "state": state, "questions": questions},
            )
    except asyncio.CancelledError:
        finish(JevVerdict(ms=elapsed(), error="cancelled", request_attempted=True))
        raise
    except TimeoutError:
        return finish(JevVerdict(ms=elapsed(), error="timeout", request_attempted=True))
    except httpx.HTTPError:
        return finish(JevVerdict(ms=elapsed(), error="transport", request_attempted=True))
    except Exception as exc:  # the shadow must never raise into the turn
        return finish(JevVerdict(ms=elapsed(), error=type(exc).__name__, request_attempted=True))
    finally:
        if owned:
            await client.aclose()

    ms = elapsed()
    if response.status_code != 200:
        return finish(JevVerdict(ms=ms, error=f"http_{response.status_code}", request_attempted=True))
    try:
        body = response.json()
        usage = body.get("usage") or {}
        tokens = [usage.get("input_tokens"), usage.get("output_tokens")] if isinstance(usage, dict) else []
        known_usage = len(tokens) == 2 and all(type(n) is int and n >= 0 for n in tokens)
        token_fields = {"input_tokens": tokens[0], "output_tokens": tokens[1]} if known_usage else {}
    except Exception:
        return finish(JevVerdict(ms=ms, error="malformed", request_attempted=True))
    try:
        answer = body["answers"]["status"]
        status = answer["choice"]
        if status not in STATUSES:
            raise ValueError("unknown status")
        probabilities = answer["probabilities"]
        if set(probabilities) != set(STATUSES):
            raise ValueError("incomplete status distribution")
        distribution = {label: _probability(probabilities[label]) for label in STATUSES}
        if not math.isclose(sum(distribution.values()), 1.0, abs_tol=0.001):
            raise ValueError("status probabilities do not sum to one")
        p_status = distribution[status]
        p_complete = distribution["COMPLETE"]
        if p_status < max(distribution.values()):
            raise ValueError("choice disagrees with probabilities")
    except Exception:  # any shape problem is one error class
        return finish(JevVerdict(ms=ms, error="malformed", request_attempted=True, **token_fields))
    return finish(JevVerdict(
        status=status, p_status=p_status, p_complete=p_complete, ms=ms, model=str(body.get("model") or ""),
        request_attempted=True, **token_fields,
    ))
