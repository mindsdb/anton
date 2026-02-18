from minds.model.base import BaseSQLModel
from minds.model.message_tracing import MessageTracing


class ChatCompletion(BaseSQLModel, MessageTracing, table=True):
    """Lightweight record to track per-request token usage for the Chat Completions API."""

    __tablename__ = "chat_completions"
