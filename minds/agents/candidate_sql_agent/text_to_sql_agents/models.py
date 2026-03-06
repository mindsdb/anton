from enum import Enum

from pydantic import BaseModel, Field

from minds.agents.candidate_sql_agent.settings import CandidateSQLAgentSettings
from minds.common.settings.app_settings import get_app_settings

settings = get_app_settings()
agent_settings = CandidateSQLAgentSettings()

# =========================
# Text-to-SQL Agent Models
# =========================


class QueryAttempt(BaseModel):
    """
    Structured output for a query attempt.
    This includes the query and the result of the query.
    """

    query: str = Field(description="The query that was generated for the attempt.")
    error: str | None = Field(description="The error that was encountered if the query failed.", default=None)
    result: str | None = Field(description="The result of the query that was executed.", default=None)

    def to_string(self) -> str:
        """Convert the query attempt to a human-readable string."""
        attempt_str = f"Query: {self.query}"
        if self.error:
            attempt_str += f"\nError: {self.error}"
        if self.result:
            attempt_str += f"\nResult: {self.result}"
        return attempt_str


class AcquiredKnowledgeItem(BaseModel):
    """
    Structured output for an acquired knowledge item.
    This includes a single item of knowledge acquired by the agent by running an exploratory query.
    There could be multiple attempts to acquire the knowledge item.
    Failed attempts are also captured here.
    """

    step: str = Field(description="The step in the database agent process that the knowledge item was acquired in.")
    attempts: list[QueryAttempt] = Field(
        default_factory=list, description="The attempts to acquire the knowledge item."
    )

    def to_string(self) -> str:
        """Convert the acquired knowledge item to a human-readable string."""
        attempts_str = "\n".join([attempt.to_string() for attempt in self.attempts])
        return f"Step: {self.step}\nQuery Attempts: {attempts_str}"


class AcquiredKnowledge(BaseModel):
    """
    Structured output for acquired knowledge.
    This includes all of the knowledge acquired by the agent by running exploratory queries.
    """

    items: list[AcquiredKnowledgeItem] = Field(default_factory=list, description="The acquired knowledge items.")

    def add_item(self, item: AcquiredKnowledgeItem) -> None:
        """Add an acquired knowledge item to the acquired knowledge."""
        self.items.append(item)

    def to_string(self) -> str:
        """Convert the acquired knowledge to a human-readable string."""
        items_str = "\n\n".join([item.to_string() for item in self.items])
        return (
            f"Query results only display the first {agent_settings.max_display_rows_to_agent} rows. "
            f"Do not assume the results are complete."
            f"\n\n{items_str}"
        )


class QueryPlanStepType(Enum):
    """Enum for the type of query plan step."""

    EXPLORATORY = "exploratory"
    FINAL = "final"


class DataCatalogSubset(BaseModel):
    """
    Structured output for the subset of the data catalog that is relevant to a given step of the query plan.
    This includes the names of the chosen datasources and tables for a given step of the query plan.
    """

    datasources: list[str] | None = Field(default=None, description="The names of the chosen datasources.")
    tables: list[str] | None = Field(
        default=None,
        description=(
            "The fully-qualified names of the chosen tables. e.g. '<datasource>.<table>' or '<integration>.<table>'."
        ),
    )

    def to_string(self) -> str:
        """Convert the data catalog subset to a human-readable string."""
        datasources_str = ", ".join(self.datasources or [])
        tables_str = ", ".join(self.tables or [])
        return f"Datasources: {datasources_str}\nTables: {tables_str}"


class QueryPlanStep(BaseModel):
    """
    Structured output for a query plan step.
    This is a single step of the overall query plan, which includes a description of the step and its type.
    """

    description: str = Field(description="The description or reasoning for the step.")
    type: QueryPlanStepType = Field(description="The type of the step.")
    data_catalog_subset: DataCatalogSubset = Field(
        description="The subset of the data catalog that is relevant to the step."
    )
    final_action: str | None = Field(
        default="query",
        description=(
            "For FINAL steps only: 'summarize' to summarize from acquired knowledge, "
            "'query' to execute a SQL query. Defaults to 'query'. Ignored for EXPLORATORY steps."
        ),
    )

    def to_string(self) -> str:
        """Convert the query plan step to a human-readable string."""
        return (
            f"Description: {self.description}\n"
            f"Type: {self.type.value}\n"
            f"Data Catalog Subset: {self.data_catalog_subset.to_string()}"
        )


class QueryPlan(BaseModel):
    """
    Structured output for a query plan.
    This includes all of the steps of the query plan.
    """

    # No strict max length validation for steps
    steps: list[QueryPlanStep] = Field(
        description="The steps of the query plan including at least one exploratory step and one final step.",
    )

    def to_string(self) -> str:
        """Convert the query plan to a human-readable string."""
        return "\n\n".join([step.to_string() for step in self.steps])


class RefinedQueryPlan(BaseModel):
    """
    Structured output for a refined query plan.
    This includes the refined steps of the query plan.
    """

    query_plan: QueryPlan = Field(description="The refined query plan.")
    is_refined: bool = Field(
        default=False,
        description="Whether the query plan was refined. Defaults to False if not specified.",
    )
    description: str = Field(
        default="",
        description=("The description of any refinements made, if any. Empty string if no refinements."),
    )


class SQLQuery(BaseModel):
    """
    Structured output for a SQL query.
    This includes the SQL query generated for a given step of the query plan,
    and optionally a chart intent for visualizing the results.
    """

    query: str = Field(description="The SQL query to be executed without semicolons, ready to be executed.")
    chart_intent: dict | None = Field(
        default=None,
        description=(
            "Optional chart visualization intent. Include when query results are suitable for visualization "
            "(distributions, trends, comparisons) or when user explicitly requests a chart. "
            "For bar/line charts: {type, x, y, series?, aggregate?, limit?, max_series?, title?}. "
            "For pie charts: {type, label, value, aggregate?, limit?, title?}. "
            "For scatter charts: {type, x, y, series?, limit?, max_series?, title?}. "
            "Use column names from the SQL query. Set to null if no chart is appropriate."
        ),
    )


# class FinaSQLQuery(SQLQuery):
#     """
#     Structured output for the final SQL query for answering a user's question.`
#     This includes the SQL query generated for the final step of the query plan and some final thoughts for the user.
#     """

#     final_thoughts: str = Field(description="Some final thoughts for the user.")


class TextToSQLToolResult(BaseModel):
    """
    Structured output for the result of the text-to-SQL pipeline.
    This includes the final query, execution result, steps executed, and acquired knowledge.
    """

    final_query: str = Field(description="The final query that was executed.")
    execution_result: str = Field(description="The result of the final query.")
    steps_executed: int = Field(description="The number of steps executed.")
    acquired_knowledge: str = Field(description="The acquired knowledge.")
