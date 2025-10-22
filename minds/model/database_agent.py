from pydantic import BaseModel, Field


class QueryPlanResult(BaseModel):
    """Structured output for the planning step to narrow down relevant catalogs."""

    preferred_engine: str | None = Field(
        default=None,
        description=(
            "Preferred engine for answering the question "
            "(e.g., 'salesforce' for SOQL, 'sql'/'postgres'/'mysql' for SQL)."
        ),
    )
    selected_datasources: list[str] | None = Field(
        default=None,
        description=(
            "List of datasource or integration names to use "
            "(e.g., 'salesforce', 'postgres_db', 'snowflake_integration')."
        ),
    )
    selected_tables: list[str] | None = Field(
        default=None,
        description=(
            "List of fully-qualified tables to include, e.g., '<datasource>.<table>' or '<integration>.<table>'."
        ),
    )
    selected_fields: list[str] | None = Field(
        default=None,
        description="List of fields of interest, optionally qualified (e.g., 'Account.Name', 'opportunities.amount').",
    )
    rationale: str | None = Field(
        default=None,
        description="Brief reasoning for the selection to aid debugging and observability.",
    )
    error: str | None = Field(
        default=None,
        description="Error or inability to determine a plan.",
    )
    
    def to_string(self) -> str:
        """Convert the plan result to a human-readable string."""
        parts = []
        if self.preferred_engine:
            parts.append(f"Preferred Engine: {self.preferred_engine}")
        if self.selected_datasources:
            parts.append(f"Selected Datasources: {', '.join(self.selected_datasources)}")
        if self.selected_tables:
            parts.append(f"Selected Tables: {', '.join(self.selected_tables)}")
        if self.selected_fields:
            parts.append(f"Selected Fields: {', '.join(self.selected_fields)}")
        if self.rationale:
            parts.append(f"Rationale: {self.rationale}")
        if self.error:
            parts.append(f"Error: {self.error}")
        return "\n".join(parts)


class QueryGenerationResult(BaseModel):
    """Structured output for SQL generation via LLM."""

    query: str | None = Field(None, description="Final generated SQL query to be executed")
    error: str | None = Field(None, description="Human readable error message if query generation failed")


class QueryGenerationResultRetry(BaseModel):
    """Structured output for SQL generation via LLM."""

    query: str | None = Field(None, description="Final generated SQL query to be executed")
    corrected_issues: str | None = Field(
        None, description="What were the issues with the previous query that were corrected in this new query?"
    )
    error: str | None = Field(None, description="Human readable error message if query generation failed")
