"""
SQL candidate selection agent.

Selects the best SQL candidate from multiple options using pairwise comparison.
Based on CHASE-SQL research showing pairwise comparison outperforms direct ranking.
"""

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from minds.agents.candidate_sql_agent.candidate_generator_agent.agent import SQLCandidate
from minds.agents.candidate_sql_agent.selection_agent.instructions_template import SELECTION_SYSTEM_PROMPT
from minds.agents.helpers import model_for
from minds.common.logger import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from minds.model.mind import Mind


class ComparisonResult(BaseModel):
    """Result of comparing two SQL candidates."""

    winner: str = Field(description="Which query is better: 'A' or 'B'")
    reasoning: str = Field(description="Brief explanation of why this query is better")


selection_agent = Agent(
    model=None,
    system_prompt=SELECTION_SYSTEM_PROMPT,
    output_type=ComparisonResult,
    retries=2,
)


class SelectionAgent:
    """
    Selects the best SQL candidate using pairwise comparison.

    Algorithm:
    1. Compare each pair of candidates
    2. Track wins per candidate
    3. Return candidate with most wins

    This approach is more robust than direct ranking because:
    - LLMs are better at binary comparisons than rankings
    - Reduces position bias
    - Allows for nuanced evaluation
    """

    def __init__(self, mind: "Mind"):
        self.mind = mind

    async def select(
        self,
        candidates: list[SQLCandidate],
        question: str,
        schema_context: str,
    ) -> SQLCandidate:
        """
        Select the best candidate using pairwise comparison.

        Args:
            candidates: List of SQL candidates to compare
            question: The original natural language question
            schema_context: The schema context string

        Returns:
            The best SQLCandidate
        """
        if not candidates:
            raise ValueError("No candidates to select from")

        if len(candidates) == 1:
            logger.info("Only one candidate, returning it directly")
            return candidates[0]

        # Prioritize executed candidates
        executed = [c for c in candidates if c.executed and not c.execution_error]

        # If only one executed successfully, return it
        if len(executed) == 1:
            logger.info("Only one candidate executed successfully, returning it")
            return executed[0]

        # If multiple executed, compare them
        if len(executed) > 1:
            logger.info(f"Comparing {len(executed)} successfully executed candidates")
            return await self._pairwise_select(executed, question, schema_context)

        # If none executed, compare all
        logger.info(f"No candidates executed, comparing all {len(candidates)}")
        return await self._pairwise_select(candidates, question, schema_context)

    async def _pairwise_select(
        self,
        candidates: list[SQLCandidate],
        question: str,
        schema_context: str,
    ) -> SQLCandidate:
        """Perform pairwise comparison tournament."""

        wins = {i: 0 for i in range(len(candidates))}

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                try:
                    winner_idx = await self._compare_pair(
                        candidates[i],
                        candidates[j],
                        i,
                        j,
                        question,
                        schema_context,
                    )
                    wins[winner_idx] += 1
                except Exception as e:
                    logger.warning(f"Comparison failed: {e}, giving point to first")
                    wins[i] += 1

        best_idx = max(wins, key=wins.get)
        logger.info(f"Selection complete. Winner: {candidates[best_idx].strategy} with {wins[best_idx]} wins")
        return candidates[best_idx]

    async def _compare_pair(
        self,
        candidate_a: SQLCandidate,
        candidate_b: SQLCandidate,
        idx_a: int,
        idx_b: int,
        question: str,
        schema_context: str,
    ) -> int:
        """Compare two candidates and return the winner's index."""
        # TODO: Move this to insturctions prompt
        prompt = f"""Which SQL query better answers the question?

Question: {question}

Schema:
{schema_context}

Query A ({candidate_a.strategy}):
```sql
{candidate_a.query}
```
{self._format_execution_info(candidate_a, "A")}

Query B ({candidate_b.strategy}):
```sql
{candidate_b.query}
```
{self._format_execution_info(candidate_b, "B")}

Which query is better: A or B?"""

        result = await selection_agent.run(
            prompt,
            model=model_for(self.mind),
        )

        winner = result.output.winner.upper().strip()
        if winner == "A":
            logger.debug(f"A wins: {result.output.reasoning[:100]}")
            return idx_a
        elif winner == "B":
            logger.debug(f"B wins: {result.output.reasoning[:100]}")
            return idx_b
        else:
            logger.warning(f"Unclear winner '{winner}', defaulting to A")
            return idx_a

    def _format_execution_info(self, candidate: SQLCandidate, label: str) -> str:
        """Format execution information for a candidate."""
        if candidate.executed and not candidate.execution_error:
            return f"Query {label} executed successfully."
        elif candidate.execution_error:
            return f"Query {label} failed with error: {candidate.execution_error[:200]}"
        else:
            return f"Query {label} has not been executed."


class QueryComplexityClassifier:
    """
    Classifies query complexity to determine if multi-path generation is needed.

    Simple queries can skip multi-path generation for efficiency.
    """

    def is_simple(self, question: str, num_tables: int) -> bool:
        """
        Detect if query is simple enough for fast path (single generation).

        Args:
            question: The natural language question
            num_tables: Number of tables identified by schema linker

        Returns:
            True if query is simple and can use fast path
        """
        question_lower = question.lower()

        complex_indicators = [
            "join",
            "group by",
            "having",
            "subquery",
            "union",
            "intersect",
            "except",
            "window",
            "partition",
            "rank",
            "dense_rank",
            "row_number",
            "lead",
            "lag",
            "compare",
            "difference",
            "growth",
            "trend",
            "percentage",
            "ratio",
        ]

        for indicator in complex_indicators:
            if indicator in question_lower:
                return False

        # Simple queries typically involve a single table; multiple tables generally imply joins/complexity.
        return num_tables <= 1
