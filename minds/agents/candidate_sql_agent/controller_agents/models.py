from pydantic import BaseModel, Field


class RouterAgentResult(BaseModel):
    handoff: bool = Field(description="Whether to handoff to the text-to-SQL agent.")
    feedback: str | None = Field(
        default=None, description="Feedback to the user; this can be clarifying questions or a final answer."
    )


class AnswerFeedbackAgentResult(BaseModel):
    feedback: str = Field(description="Feedback to the user based on the execution result of the text-to-SQL pipeline.")
    next_steps: str = Field(description="Next steps for the user to take based on the feedback.")
