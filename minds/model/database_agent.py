from typing import Optional, List

from pydantic import BaseModel, Field


class QueryPlanResult(BaseModel):
    """Structured output for the planning step to narrow down relevant catalogs."""

    preferred_engine: Optional[str] = Field(
        default=None,
        description="Preferred engine for answering the question (e.g., 'salesforce' for SOQL, 'sql'/'postgres'/'mysql' for SQL).",
    )
    selected_datasources: Optional[List[str]] = Field(
        default=None,
        description="List of datasource or integration names to use (e.g., 'salesforce', 'postgres_db', 'snowflake_integration').",
    )
    selected_tables: Optional[List[str]] = Field(
        default=None,
        description="List of fully-qualified tables to include, e.g., '<datasource>.<table>' or '<integration>.<table>'.",
    )
    selected_fields: Optional[List[str]] = Field(
        default=None,
        description="List of fields of interest, optionally qualified (e.g., 'Account.Name', 'opportunities.amount').",
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Brief reasoning for the selection to aid debugging and observability.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error or inability to determine a plan.",
    )


class QueryGenerationResult(BaseModel):
    """Structured output for SQL generation via LLM."""

    query: Optional[str] = Field(
        None, description="Final generated SQL query to be executed"
    )
    error: Optional[str] = Field(
        None, description="Human readable error message if query generation failed"
    )


class QueryGenerationResultRetry(BaseModel):
    """Structured output for SQL generation via LLM."""

    query: Optional[str] = Field(
        None, description="Final generated SQL query to be executed"
    )
    corrected_issues: Optional[str] = Field(
        None, description="What were the issues with the previous query that were corrected in this new query?"
    )
    error: Optional[str] = Field(
        None, description="Human readable error message if query generation failed"
    )
