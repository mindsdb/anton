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
    headers: dict[str, str] | None = None,
) -> JevVerdict:
    """POST one decision request. Never raises; failures come back in ``error``.

    ``headers`` are sent alongside the key, which they cannot override.
    """
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
        return JevVerdict(ms=elapsed(), error="timeout")
    except httpx.HTTPError:
        return JevVerdict(ms=elapsed(), error="transport")
    except Exception as exc:  # the completion check must never raise into the turn
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


@dataclass(frozen=True)
class JevDecision:
    """Raw answers for any set of questions, keyed by question name.

    Each answer is Jev's own shape: ``{"choice": ..., "probabilities": {...}}``
    for a choice, ``{"noul": p}`` for a yes/no. Empty with ``error`` set on
    any failure, so the caller falls back the same way ``classify``'s do.
    """

    answers: dict
    ms: int = 0
    model: str = ""
    error: str = ""

    def choice(self, name: str) -> tuple[str, float]:
        """(choice, its probability), or ("", 0.0) when absent or malformed."""
        answer = self.answers.get(name)
        if not isinstance(answer, dict):
            return "", 0.0
        picked = answer.get("choice")
        try:
            p = _probability((answer.get("probabilities") or {})[picked])
        except Exception:
            return "", 0.0
        return (picked if isinstance(picked, str) else ""), p

    def yes(self, name: str) -> float:
        """Probability that a yes/no question is true; 0.0 when absent or malformed."""
        answer = self.answers.get(name)
        try:
            return _probability(answer["noul"])
        except Exception:
            return 0.0


# MindsHub's answer when Jev's routing policy is briefly not loaded: a 503
# with exactly this message. Nothing was decided, so the same request may be
# sent once more within the same deadline (yolocoder's edit router does).
POLICY_UNAVAILABLE = "The model policy is temporarily unavailable. Please retry shortly."


def policy_unavailable(response) -> bool:
    """Whether RESPONSE is MindsHub's 503 "policy temporarily unavailable" refusal."""
    if response.status_code != 503:
        return False
    try:
        text = response.text
    except Exception:
        return False
    if len(text) >= 8 << 10:
        return False
    if text.strip() == POLICY_UNAVAILABLE:
        return True
    try:
        import json

        body = json.loads(text)
    except ValueError:
        return False
    error = body.get("error") if isinstance(body, dict) else None
    return isinstance(error, dict) and error.get("message") == POLICY_UNAVAILABLE


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
    retry_policy_unavailable: bool = False,
) -> JevDecision:
    """POST one decision request with any questions. Never raises.

    ``retry_policy_unavailable``: send the request once more after MindsHub's
    503 "policy temporarily unavailable", inside the same ``timeout_s``.
    ``headers`` are sent alongside the key, which they cannot override.
    """
    import httpx2 as httpx

    started = time.monotonic()

    def elapsed() -> int:
        return round((time.monotonic() - started) * 1000)

    owned = client is None
    client = client or httpx.AsyncClient(timeout=timeout_s)
    try:
        async with asyncio.timeout(timeout_s):
            for attempt in range(2):
                response = await client.post(
                    f"{base_url.rstrip('/')}/decisions",
                    headers={**(headers or {}), "Authorization": f"Bearer {api_key}"},
                    json={"model": model, "state": state, "questions": questions},
                )
                if not (retry_policy_unavailable and attempt == 0 and policy_unavailable(response)):
                    break
    except TimeoutError:
        return JevDecision(answers={}, ms=elapsed(), error="timeout")
    except httpx.HTTPError:
        return JevDecision(answers={}, ms=elapsed(), error="transport")
    except Exception as exc:  # a decision must never raise into the turn
        return JevDecision(answers={}, ms=elapsed(), error=type(exc).__name__)
    finally:
        if owned:
            await client.aclose()

    ms = elapsed()
    if response.status_code != 200:
        return JevDecision(answers={}, ms=ms, error=f"http_{response.status_code}")
    try:
        body = response.json()
        answers = body["answers"]
        if not isinstance(answers, dict):
            raise ValueError("answers is not an object")
    except Exception:
        return JevDecision(answers={}, ms=ms, error="malformed")
    return JevDecision(answers=answers, ms=ms, model=str(body.get("model") or ""))


# ── which decisions model, if any (ENG-4227) ─────────────────────────

DEFAULT_DECISIONS_MODEL = "jev-1.13.0"
_DISABLED = ("", "off", "none", "false", "0")


def _host(url: str | None) -> str:
    from urllib.parse import urlsplit

    try:
        return (urlsplit((url or "").strip()).hostname or "").lower()
    except ValueError:
        return ""


MINDSHUB_API_HOST = "api.mindshub.ai"


def _provider(settings) -> str:
    return str(getattr(settings, "planning_provider", "") or "").strip().lower().replace("_", "-")


def _v1(url: str) -> str:
    base = str(url).strip().rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


def _active_mindshub(settings) -> tuple[str, str] | None:
    """``(base_url ending in /v1, api_key)`` when the configured LLM provider itself is MindsHub, else None.

    Only the active (planning) provider's own config counts: ``minds-cloud``
    (its ``minds_url`` / ``minds_api_key``), or ``openai-compatible`` whose
    base URL host is exactly api.mindshub.ai (its ``openai_base_url`` /
    ``openai_api_key``; AntonSettings reads ``minds-cloud`` this way). A
    MindsHub key or URL left over beside an Anthropic or OpenAI provider is
    never used.
    """
    provider = _provider(settings)
    if provider == "minds-cloud":
        url, key = getattr(settings, "minds_url", None), getattr(settings, "minds_api_key", None)
    elif provider == "openai-compatible":
        url, key = getattr(settings, "openai_base_url", None), getattr(settings, "openai_api_key", None)
    else:
        return None
    if not url or not key or _host(url) != MINDSHUB_API_HOST:
        return None
    return _v1(url), str(key)


def default_decisions_model(settings) -> str | None:
    """The decisions model when none is configured: Jev when the LLM provider is MindsHub, otherwise none."""
    return DEFAULT_DECISIONS_MODEL if _active_mindshub(settings) else None


def decisions_model(settings) -> str | None:
    """The configured decisions model (``ANTON_DECISIONS_MODEL``), the MindsHub default when unset, or None.

    An explicit empty value (or "off" / "none") disables it.
    """
    value = getattr(settings, "decisions_model", None)
    if value is None:
        return default_decisions_model(settings)
    value = str(value).strip()
    return None if value.lower() in _DISABLED else value


def decisions_config(settings) -> tuple[str, str, str] | None:
    """``(base_url, api_key, model)`` for MindsHub's ``/v1/decisions``, or None when there is no decisions model.

    Available only when the configured LLM provider itself is MindsHub (see
    :func:`_active_mindshub`); an explicit ``ANTON_DECISIONS_MODEL`` picks the
    model but never borrows a URL or key from anywhere else.
    """
    connection = _active_mindshub(settings)
    model = decisions_model(settings)
    if connection is None or not model:
        return None
    return connection[0], connection[1], model


def decisions_available(settings) -> bool:
    """Whether the quick-edit bar may route (and so edit directly). False: send every message to the agent."""
    return decisions_config(settings) is not None
