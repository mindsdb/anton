from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """
    This is the base response model that all agents should return.
    """

    sql: str = Field(description="The final SQL query executed to answer the question.")
    answer: str = Field(description="Plain-English answer grounded in the query result.")
    notes: list[str] = Field(default_factory=list, description="Short bullet notes about assumptions/filters.")
