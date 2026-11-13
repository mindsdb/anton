"""TypeSafe's Jev via MindsHub's ``/v1/decisions``: completion verdicts and bulk labelling.

Single calls never raise; failures come back as an error class, so the caller
can fall back. ``decide_many`` raises ``JevUnavailable`` only when no item of
the batch can succeed.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import threading
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


# The provider's error type when a request exceeds Jev's context limits.
TOO_LONG = "max_tokens_exceeded"


def _provider_error_type(response) -> str:
    """``detail.error_type`` of a provider refusal, or "" when the body has none."""
    try:
        detail = response.json().get("detail")
        return str(detail.get("error_type") or "") if isinstance(detail, dict) else ""
    except Exception:
        return ""


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


async def decide(
    *,
    base_url: str,
    api_key: str,
    model: str,
    state: dict | str,
    questions: dict,
    timeout_s: float,
    client=None,
    headers: dict[str, str] | None = None,
) -> JevDecision:
    """POST one decision request with any questions. Never raises.

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
            response = await client.post(
                f"{base_url.rstrip('/')}/decisions",
                headers={**(headers or {}), "Authorization": f"Bearer {api_key}"},
                json={"model": model, "state": state, "questions": questions},
            )
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
        error = f"http_{response.status_code}"
        if response.status_code == 400 and _provider_error_type(response) == TOO_LONG:
            error = TOO_LONG
        return JevDecision(answers={}, ms=ms, error=error)
    try:
        body = response.json()
        answers = body["answers"]
        if not isinstance(answers, dict):
            raise ValueError("answers is not an object")
    except Exception:
        return JevDecision(answers={}, ms=ms, error="malformed")
    return JevDecision(answers=answers, ms=ms, model=str(body.get("model") or ""))


def detach_conversation(headers: dict[str, str]) -> dict[str, str]:
    """Move the Langfuse session id into the metadata header; returns a new dict.

    The console marks a conversation failed when any call in it fails, and a
    refused decision is one the turn recovers from. The conversation rides in
    the metadata instead, so the call can still be joined to it.
    """
    headers = dict(headers)
    conversation_id = headers.pop("Langfuse-Session-Id", None)
    if conversation_id and "Langfuse-Metadata" in headers:
        metadata = json.loads(headers["Langfuse-Metadata"])
        metadata["conversation_id"] = conversation_id
        headers["Langfuse-Metadata"] = json.dumps(metadata)
    return headers


# --- Bulk labelling for scratchpad code: classify() and classify_yes_no() ---
#
# Questions in one request are evaluated in isolation against a shared state,
# so each text rides in its own question and the state holds only the task.
# Texts placed in the state instead would all be context for every question,
# and neighbours measurably change the labels of borderline texts. Category
# descriptions stay in every question's criteria: moved once into the state
# they halve the tokens per text, but Jev then gives about 1.5 times as many
# texts p < 0.9, which the agent re-checks with the chat model.

CLASSIFY_MODEL = "jev-1.13.0"
# The MindsHub edge refuses an IP address for a minute once it sends about 300
# non-model requests in a minute, decisions included. Stay below with a margin.
CLASSIFY_PER_MINUTE = 270
CLASSIFY_CONCURRENCY = 4
CLASSIFY_TIMEOUT_S = 120.0
# Jev allows 64k tokens per request; estimate conservatively below it.
CLASSIFY_BATCH_TOKENS = 48_000
CLASSIFY_BATCH_ITEMS = 200
# The name of the question in each per-text decision handed back to callers.
ANSWER = "answer"
EMPTY_TEXT = "empty_text"
TEXT_TOO_LONG = "text_too_long"
BATCH_STATE = "Answer each question about the text that question contains. The questions are independent."

_MAX_ATTEMPTS = 4
_RATE_LIMIT_PAUSE_S = 60.0
# This many failed texts with no success means the request itself is wrong.
_FAILURES_BEFORE_ABORT = 20
# Refusals that would repeat for every remaining batch.
_FATAL = {
    "http_401": "the MindsHub key was rejected",
    "http_402": "the organization has no credit or free decision allowance left",
    "http_403": "this organization may not use Jev (zero data retention or an admin rule)",
}


class JevUnavailable(RuntimeError):
    """Jev cannot label this batch at all; the caller should use the LLM instead."""


class Pacer:
    """Spaces request starts ``60 / per_minute`` seconds apart.

    One instance serves a whole process, threads included, so concurrent calls
    share the budget.
    """

    def __init__(self, per_minute: int, clock=time.monotonic) -> None:
        self._interval = 60.0 / per_minute
        self._clock = clock
        self._next = 0.0
        self._lock = threading.Lock()

    def reserve(self) -> float:
        """Claim the next start slot; returns how long to wait for it."""
        with self._lock:
            now = self._clock()
            start = max(now, self._next)
            self._next = start + self._interval
            return start - now

    def pause(self, seconds: float) -> None:
        """Push every later start at least ``seconds`` from now."""
        with self._lock:
            self._next = max(self._next, self._clock() + seconds)


def _retryable(error: str) -> bool:
    return error in ("timeout", "transport") or error.startswith("http_5")


def item_question(question: dict, item) -> dict:
    """``question`` asked about one text, carried in the question itself."""
    text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, default=str)
    return {**question, "instructions": f"{question.get('instructions') or ''}\n\nText:\n{text}"}


def _estimate_tokens(question: dict) -> int:
    # About 3 characters a token: an overestimate for English, so batches stay under the limit.
    return len(json.dumps(question, ensure_ascii=False)) // 3 + 1


def plan_batches(questions: list[dict], *, max_tokens: int = CLASSIFY_BATCH_TOKENS,
                 max_items: int = CLASSIFY_BATCH_ITEMS) -> list[list[int]]:
    """Group question indexes, in order, into batches under both limits."""
    batches: list[list[int]] = []
    current: list[int] = []
    used = 0
    for i, question in enumerate(questions):
        cost = _estimate_tokens(question)
        if current and (used + cost > max_tokens or len(current) >= max_items):
            batches.append(current)
            current, used = [], 0
        current.append(i)
        used += cost
    if current:
        batches.append(current)
    return batches


async def decide_many(
    items: list,
    question: dict,
    *,
    base_url: str,
    api_key: str,
    pacer: Pacer,
    model: str = CLASSIFY_MODEL,
    headers: dict[str, str] | None = None,
    client=None,
    concurrency: int = CLASSIFY_CONCURRENCY,
    timeout_s: float = CLASSIFY_TIMEOUT_S,
    max_batch_tokens: int = CLASSIFY_BATCH_TOKENS,
    on_progress=None,
    sleep=asyncio.sleep,
) -> list[JevDecision]:
    """Ask ``question`` about every item; one decision per item, in order.

    Items go out in batches, each item in its own question. A batch over Jev's
    context limit is split in half until it fits; a single item that still
    does not fit fails alone. Per-item failures come back as decisions with
    ``error`` set; their answers hold the one answer under ``ANSWER``.
    Raises ``JevUnavailable`` when a refusal would repeat for every batch, or
    when the first ``_FAILURES_BEFORE_ABORT`` finished items all failed.
    """
    results: list[JevDecision | None] = [None] * len(items)
    counts = {"done": 0, "ok": 0, "failed": 0}
    questions: dict[int, dict] = {}
    for i, item in enumerate(items):
        if isinstance(item, str) and not item.strip():
            results[i] = JevDecision(answers={}, error=EMPTY_TEXT)
        else:
            questions[i] = item_question(question, item)
    indexes = list(questions)
    batches = [[indexes[k] for k in batch]
               for batch in plan_batches([questions[i] for i in indexes], max_tokens=max_batch_tokens)]
    queue: asyncio.Queue = asyncio.Queue()
    for batch in batches:
        queue.put_nowait(batch)

    def finish(i: int, decision: JevDecision) -> None:
        results[i] = decision
        counts["done"] += 1
        counts["failed" if decision.error else "ok"] += 1
        if counts["ok"] == 0 and counts["failed"] >= _FAILURES_BEFORE_ABORT:
            raise JevUnavailable(
                f"The first {counts['failed']} texts all failed (last error: {decision.error})"
            )

    async def send(batch: list[int]) -> JevDecision:
        decision = JevDecision(answers={})
        for attempt in range(_MAX_ATTEMPTS):
            wait = pacer.reserve()
            if wait > 0:
                await sleep(wait)
            decision = await decide(
                base_url=base_url, api_key=api_key, model=model, state=BATCH_STATE,
                questions={f"t{i}": questions[i] for i in batch},
                timeout_s=timeout_s, client=client, headers=headers,
            )
            error = decision.error
            if not error:
                break
            if error in _FATAL:
                raise JevUnavailable(f"Jev refused the request ({error}): {_FATAL[error]}")
            if error == "http_429":
                pacer.pause(_RATE_LIMIT_PAUSE_S)
            elif _retryable(error):
                await sleep(2**attempt)
            else:
                break
        return decision

    async def worker() -> None:
        while not queue.empty():
            batch = queue.get_nowait()
            decision = await send(batch)
            if decision.error == TOO_LONG and len(batch) > 1:
                half = len(batch) // 2
                queue.put_nowait(batch[:half])
                queue.put_nowait(batch[half:])
                continue
            for i in batch:
                if decision.error:
                    error = TEXT_TOO_LONG if decision.error == TOO_LONG else decision.error
                    finish(i, JevDecision(answers={}, ms=decision.ms, error=error))
                elif f"t{i}" in decision.answers:
                    finish(i, JevDecision(answers={ANSWER: decision.answers[f"t{i}"]},
                                          ms=decision.ms, model=decision.model))
                else:
                    finish(i, JevDecision(answers={}, ms=decision.ms, error="malformed"))
            if on_progress is not None:
                on_progress(counts["done"], len(questions))

    try:
        async with asyncio.TaskGroup() as group:
            for _ in range(max(1, min(concurrency, len(batches)))):
                group.create_task(worker())
    except* JevUnavailable as failures:
        raise failures.exceptions[0] from None
    return results  # type: ignore[return-value]


def choice_question(categories, question: str) -> dict:
    """A choice question over ``categories``: a list of names, or names mapped to descriptions."""
    if isinstance(categories, dict):
        criteria = {str(name): "" if text is None else text for name, text in categories.items()}
    elif isinstance(categories, str):
        raise TypeError("categories must be a list of names or a dict of name -> description")
    else:
        criteria = {}
        for name in categories:
            if str(name) in criteria:
                raise ValueError(f"duplicate category: {name!r}")
            criteria[str(name)] = ""
    if len(criteria) < 2 or not all(name.strip() for name in criteria):
        raise ValueError("categories need at least two non-empty names")
    return {"type": "choice", "instructions": question, "criteria": criteria}


def yes_no_question(question: str, *, yes: str | None = None, no: str | None = None) -> dict:
    """A yes/no question; ``yes`` and ``no`` optionally describe each outcome."""
    if not str(question or "").strip():
        raise ValueError("question must not be empty")
    built: dict = {"type": "noul", "instructions": question}
    if yes is not None or no is not None:
        built["criteria"] = {"true": yes, "false": no}
    return built


def choice_result(decision: JevDecision) -> dict:
    """{"label", "p", "probabilities"}, or label None with "error" set."""
    if decision.error:
        return {"label": None, "p": None, "probabilities": None, "error": decision.error}
    label, p = decision.choice(ANSWER)
    try:
        probabilities = {
            str(k): _probability(v) for k, v in decision.answers[ANSWER]["probabilities"].items()
        }
    except Exception:
        probabilities = None
    if not label or probabilities is None:
        return {"label": None, "p": None, "probabilities": None, "error": "malformed"}
    return {"label": label, "p": p, "probabilities": probabilities}


def yes_no_result(decision: JevDecision) -> dict:
    """{"yes", "p_yes"}, where yes means p_yes >= 0.5; yes None with "error" set on failure."""
    if decision.error:
        return {"yes": None, "p_yes": None, "error": decision.error}
    try:
        p_yes = _probability(decision.answers[ANSWER]["noul"])
    except Exception:
        return {"yes": None, "p_yes": None, "error": "malformed"}
    return {"yes": p_yes >= 0.5, "p_yes": p_yes}
