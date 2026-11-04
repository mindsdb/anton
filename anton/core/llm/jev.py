"""Completion verdicts from TypeSafe's Jev, via MindsHub's ``/v1/decisions``.

`classify` reads the completion verifier's one `status` choice; `decide` reads
every named answer of any question set (Chimera's sidekick checks).

Failures come back as an error class, never raised, so the caller can fall back.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class JevAnswer:
    """One named answer. ``p`` is the probability of yes for a `noul`
    question, of the chosen option for a `choice` one."""

    type: str
    p: float
    choice: str = ""
    probabilities: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class JevDecision:
    """One `decide` call: every answer by question name, or an error class.
    Content-free, like `JevVerdict`."""

    answers: Mapping[str, JevAnswer] = field(default_factory=dict)
    ms: int = 0
    model: str = ""
    error: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None

    def p(self, name: str) -> float | None:
        """The probability answered for ``name``, or None when there is none."""
        answer = self.answers.get(name)
        return answer.p if answer is not None else None


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


async def _post(
    *,
    base_url: str,
    api_key: str,
    model: str,
    state: dict,
    questions: dict,
    timeout_s: float,
    client,
    headers: dict[str, str] | None,
) -> tuple[object, int, str]:
    """POST one decision request: the parsed body, the milliseconds it took,
    and an error class ("" when a body parsed). Never raises."""
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
                headers={**(headers or {}), "Authorization": f"Bearer {api_key}"},
                json={"model": model, "state": state, "questions": questions},
            )
    except TimeoutError:
        return None, elapsed(), "timeout"
    except httpx.HTTPError:
        return None, elapsed(), "transport"
    except Exception as exc:  # the shadow must never raise into the turn
        return None, elapsed(), type(exc).__name__
    finally:
        if owned:
            await client.aclose()

    ms = elapsed()
    if response.status_code != 200:
        return None, ms, f"http_{response.status_code}"
    try:
        return response.json(), ms, ""
    except Exception:
        return None, ms, "malformed"


async def classify(
    *,
    base_url: str,
    api_key: str,
    model: str,
    state: dict,
    questions: dict,
    timeout_s: float,
    client=None,
    headers: dict[str, str] | None = None,
) -> JevVerdict:
    """POST one decision request. Never raises; failures come back in ``error``.

    ``headers`` are sent alongside the key, which they cannot override.
    """
    body, ms, error = await _post(
        base_url=base_url, api_key=api_key, model=model, state=state, questions=questions,
        timeout_s=timeout_s, client=client, headers=headers,
    )
    if error:
        return JevVerdict(ms=ms, error=error)
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


def _parse_answer(raw: object) -> JevAnswer:
    """One answer as the service returns it; raises on any shape problem."""
    if not isinstance(raw, dict):
        raise ValueError("answer is not an object")
    if "noul" in raw:
        return JevAnswer(type="noul", p=_probability(raw["noul"]))
    choice = raw["choice"]
    if not isinstance(choice, str):
        raise ValueError("choice is not a string")
    probabilities = {str(k): _probability(v) for k, v in raw["probabilities"].items()}
    return JevAnswer(type="choice", p=probabilities[choice], choice=choice, probabilities=probabilities)


async def decide(
    *,
    base_url: str,
    api_key: str,
    model: str,
    state: dict,
    questions: dict,
    timeout_s: float,
    client=None,
    headers: dict[str, str] | None = None,
) -> JevDecision:
    """POST one decision request and read every named answer, `noul` and
    `choice` alike. Never raises: a failure, or an answer missing for any
    question asked, comes back in ``error`` with no answers, so a caller that
    acts only on answers fails open.

    ``headers`` are sent alongside the key, which they cannot override.
    """
    body, ms, error = await _post(
        base_url=base_url, api_key=api_key, model=model, state=state, questions=questions,
        timeout_s=timeout_s, client=client, headers=headers,
    )
    if error:
        return JevDecision(ms=ms, error=error)
    usage = _usage(body)
    try:
        raw = body["answers"]
        answers = {name: _parse_answer(raw[name]) for name in questions}
    except Exception:  # any shape problem is one error class
        return JevDecision(ms=ms, error="malformed", **usage)
    return JevDecision(answers=answers, ms=ms, model=str(body.get("model") or ""), **usage)
