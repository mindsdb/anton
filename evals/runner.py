"""Eval runner — orchestrates case execution against real LLM APIs."""

from __future__ import annotations

import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from anton.chat import ChatSession
from anton.llm.client import LLMClient
from anton.memory.cortex import Cortex

from evals.judge import judge_with_llm
from evals.scoring import apply_scorer, apply_scorer_async, compute_overall_score
from evals.types import EvalCase, EvalResult, ScoreResult, ToolCallRecord, TurnRecord


class EvalRunner:
    """Orchestrates benchmark case execution via ChatSession.turn()."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        coding_provider: str = "anthropic",
        coding_api_key: str = "",
    ) -> None:
        self._llm = llm_client
        self._coding_provider = coding_provider
        self._coding_api_key = coding_api_key

    def _build_session(self, case: EvalCase) -> tuple[ChatSession, tempfile.TemporaryDirectory | None]:
        """Construct a fresh ChatSession for this case.

        Returns:
            (session, tmp_dir) — tmp_dir is set for memory cases and must be
            kept alive for the session's lifetime.
        """
        cortex = None
        tmp_dir = None

        if case.category == "memory":
            # Wire up a real Cortex with temporary directories so the
            # memorize/recall tools are available.
            tmp_dir = tempfile.TemporaryDirectory(prefix="anton_eval_")
            tmp_path = Path(tmp_dir.name)
            global_dir = tmp_path / "global" / "memory"
            project_dir = tmp_path / "project" / "memory"
            global_dir.mkdir(parents=True)
            project_dir.mkdir(parents=True)
            cortex = Cortex(
                global_dir=global_dir,
                project_dir=project_dir,
                mode="autopilot",
                llm_client=self._llm,
            )

        session = ChatSession(
            self._llm,
            cortex=cortex,
            coding_provider=self._coding_provider,
            coding_api_key=self._coding_api_key,
        )
        return session, tmp_dir

    async def run_case(self, case: EvalCase) -> EvalResult:
        """Run a single eval case and return the scored result."""
        turn_records: list[TurnRecord] = []
        tmp_dir = None

        try:
            try:
                session, tmp_dir = self._build_session(case)

                for turn_spec in case.turns:
                    history_before = len(session.history)
                    t0 = time.monotonic()

                    reply = await session.turn(turn_spec.user_input)

                    elapsed = time.monotonic() - t0

                    # Extract tool calls and token usage from history
                    tool_calls, tool_rounds, input_tokens, output_tokens = (
                        _extract_turn_data(session.history, history_before)
                    )

                    turn_records.append(
                        TurnRecord(
                            user_input=turn_spec.user_input,
                            response_text=reply,
                            tool_calls=tool_calls,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            latency_seconds=round(elapsed, 2),
                            tool_rounds=tool_rounds,
                        )
                    )

            except Exception as exc:
                return EvalResult(
                    case_id=case.id,
                    category=case.category,
                    description=case.description,
                    turns=turn_records,
                    overall_score=0.0,
                    total_tokens=sum(t.input_tokens + t.output_tokens for t in turn_records),
                    total_latency_seconds=sum(t.latency_seconds for t in turn_records),
                    error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                )

            # Apply scorers
            scores = await self._score(case, turn_records)
            overall = compute_overall_score(scores)

            return EvalResult(
                case_id=case.id,
                category=case.category,
                description=case.description,
                turns=turn_records,
                scores=scores,
                overall_score=round(overall, 4),
                total_tokens=sum(t.input_tokens + t.output_tokens for t in turn_records),
                total_latency_seconds=round(sum(t.latency_seconds for t in turn_records), 2),
            )
        finally:
            if tmp_dir is not None:
                tmp_dir.cleanup()

    async def _score(
        self, case: EvalCase, turns: list[TurnRecord]
    ) -> list[ScoreResult]:
        """Apply all scorers for a case."""
        scores: list[ScoreResult] = []

        # Build the judge function (closure over our LLM client)
        async def _judge_fn(rubric: str, response: str, outputs: str) -> tuple[bool, str]:
            return await judge_with_llm(
                rubric,
                response,
                outputs,
                provider=self._llm.coding_provider,
                model=self._llm.coding_model,
            )

        for scorer in case.scorers:
            result = await apply_scorer_async(
                scorer, turns, judge_fn=_judge_fn
            )
            scores.append(result)

        return scores

    async def run_suite(
        self,
        cases: list[EvalCase],
        *,
        budget_usd: float | None = None,
    ) -> list[EvalResult]:
        """Run multiple cases sequentially, with optional budget cap.

        Args:
            cases: The eval cases to run.
            budget_usd: If set, stop when estimated cumulative cost exceeds this.

        Returns:
            List of EvalResult, one per case (may be shorter than cases if budget hit).
        """
        results: list[EvalResult] = []
        cumulative_cost = 0.0

        for case in cases:
            if budget_usd is not None:
                cumulative_cost += case.estimated_cost_usd
                if cumulative_cost > budget_usd:
                    break

            result = await self.run_case(case)
            results.append(result)

        return results


# ---------------------------------------------------------------------------
# History inspection helpers
# ---------------------------------------------------------------------------


def _extract_turn_data(
    history: list[dict], start_idx: int
) -> tuple[list[ToolCallRecord], int, int, int]:
    """Extract tool calls, round count, and token estimates from session history.

    Walks the history from ``start_idx`` to the end, finding tool_use blocks
    in assistant messages and their matching tool_result blocks.

    Returns:
        (tool_calls, tool_rounds, estimated_input_tokens, estimated_output_tokens)
    """
    tool_calls: list[ToolCallRecord] = []
    tool_rounds = 0

    # Map tool_use ids to their names/inputs
    pending: dict[str, dict[str, Any]] = {}

    for msg in history[start_idx:]:
        role = msg.get("role")
        content = msg.get("content")

        if role == "assistant" and isinstance(content, list):
            # Count this as a tool round if it has tool_use blocks
            has_tool_use = False
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    has_tool_use = True
                    tc_id = block.get("id", "")
                    pending[tc_id] = {
                        "name": block.get("name", ""),
                        "input": block.get("input", {}),
                    }
            if has_tool_use:
                tool_rounds += 1

        elif role == "user" and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tc_id = block.get("tool_use_id", "")
                    info = pending.pop(tc_id, {"name": "unknown", "input": {}})
                    tool_calls.append(
                        ToolCallRecord(
                            name=info["name"],
                            input=info["input"],
                            result_text=block.get("content", ""),
                        )
                    )

    # Rough token estimates based on character count (4 chars ≈ 1 token)
    total_chars = sum(
        len(str(msg.get("content", ""))) for msg in history[start_idx:]
    )
    estimated_input = total_chars // 4
    estimated_output = len(str(history[-1].get("content", ""))) // 4 if history else 0

    return tool_calls, tool_rounds, estimated_input, estimated_output
