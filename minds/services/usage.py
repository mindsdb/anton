"""
Usage service layer for counting resource consumption.

This module contains the UsageService class that provides lightweight
aggregation queries for token and question usage. It is used by the
LimitsService to populate the ``usage`` fields in MindLimitsConfig.

Token and question counts are sourced from **two** tables because the
application exposes two distinct API surfaces:

- **Responses API** -- creates Message records (in conversations).
- **Chat Completions API** -- creates ChatCompletion records (stateless).

Both tables inherit ``input_tokens`` / ``output_tokens`` from the
MessageTracing mixin, so total token usage is the combined sum across
both tables.
"""

from datetime import datetime

from sqlmodel import Session, and_, func, select

from minds.common.logger import setup_logging
from minds.model.chat_completion import ChatCompletion
from minds.model.message import Message
from minds.requests.context import Context
from minds.schemas.chat import Role

logger = setup_logging()


class UsageServiceError(Exception):
    """Base exception for usage service errors."""

    pass


class UsageService:
    """
    Service class for counting token and question usage.

    All queries are scoped to a single organization and the user from the
    request ``Context``. Both ``count_tokens()`` and ``count_questions()``
    filter on ``self.user_id`` and ``self.organization_id`` set at
    initialization from the provided context.
    """

    def __init__(self, session: Session, context: Context):
        """
        Initialize the usage service.

        Args:
            session: Database session for running aggregation queries.
            context: Request context containing user_id and organization_id.
        """
        self.session = session
        self.context = context
        self.user_id = context.user_id
        self.organization_id = context.organization_id

        logger.debug(f"UsageService initialized for user {self.user_id} in organization {self.organization_id}")

    async def count_tokens(self, since: datetime | None = None) -> int:
        """
        Sum total tokens consumed across the Responses API and Chat Completions API.

        Tokens are tracked in two places:
        - ``messages`` table (Responses API) -- each message stores input/output tokens.
        - ``chat_completions`` table (Chat Completions API) -- each request stores input/output tokens.

        We use ``COALESCE(..., 0)`` so that an empty result set returns 0
        rather than NULL.

        Args:
            since: When provided, only count tokens from records created on or after
                   this datetime. Used for billing-cycle-scoped counts.

        Returns:
            int: Combined total of input + output tokens.
        """
        try:
            logger.debug(
                f"Counting tokens for organization {self.organization_id} (user_id={self.user_id}, since={since})"
            )

            # --- Tokens from the Responses API (messages table) ---
            msg_conditions = [
                Message.organization_id == self.organization_id,
                Message.user_id == self.user_id,
                Message.deleted_at.is_(None),
            ]
            # --- Tokens from the Chat Completions API (chat_completions table) ---
            cc_conditions = [
                ChatCompletion.organization_id == self.organization_id,
                ChatCompletion.user_id == self.user_id,
                ChatCompletion.deleted_at.is_(None),
            ]

            if since is not None:
                msg_conditions.append(Message.created_at >= since)
                cc_conditions.append(ChatCompletion.created_at >= since)

            msg_stmt = select(func.coalesce(func.sum(Message.input_tokens + Message.output_tokens), 0)).where(
                and_(*msg_conditions)
            )

            cc_stmt = select(
                func.coalesce(func.sum(ChatCompletion.input_tokens + ChatCompletion.output_tokens), 0)
            ).where(and_(*cc_conditions))

            msg_tokens = self.session.exec(msg_stmt).one()
            cc_tokens = self.session.exec(cc_stmt).one()
            total = msg_tokens + cc_tokens

            logger.debug(
                f"Token count for organization {self.organization_id} (user_id={self.user_id}): "
                f"messages={msg_tokens}, chat_completions={cc_tokens}, total={total}"
            )

            if total == 0:
                logger.debug(
                    f"Token count is 0 for organization {self.organization_id} "
                    f"(user_id={self.user_id}). This is expected for new users/orgs."
                )

            return total
        except Exception as e:
            logger.error(
                f"Error counting tokens for organization {self.organization_id} (user_id={self.user_id}): {str(e)}"
            )
            raise UsageServiceError(f"Failed to count tokens: {str(e)}") from None

    async def count_questions(self, since: datetime | None = None) -> int:
        """
        Count the total number of questions asked across both API surfaces.

        A "question" is defined as:
        - One row in ``messages`` with ``role = 'user'`` (Responses API).
          Each user message represents a single question sent through a conversation.
        - One row in ``chat_completions`` (Chat Completions API).
          Each record represents a single stateless request, i.e. one question.

        Args:
            since: When provided, only count questions from records created on or after
                   this datetime. Used for billing-cycle-scoped counts.

        Returns:
            int: Total number of questions.
        """
        try:
            logger.debug(
                f"Counting questions for organization {self.organization_id} (user_id={self.user_id}, since={since})"
            )

            # --- Questions from the Responses API (messages with role='user') ---
            msg_conditions = [
                Message.organization_id == self.organization_id,
                Message.user_id == self.user_id,
                Message.deleted_at.is_(None),
                Message.role == Role.user,
            ]
            # --- Questions from the Chat Completions API (one row = one question) ---
            cc_conditions = [
                ChatCompletion.organization_id == self.organization_id,
                ChatCompletion.user_id == self.user_id,
                ChatCompletion.deleted_at.is_(None),
            ]

            if since is not None:
                msg_conditions.append(Message.created_at >= since)
                cc_conditions.append(ChatCompletion.created_at >= since)

            msg_stmt = select(func.count(Message.id)).where(and_(*msg_conditions))
            cc_stmt = select(func.count(ChatCompletion.id)).where(and_(*cc_conditions))

            msg_count = self.session.exec(msg_stmt).one()
            cc_count = self.session.exec(cc_stmt).one()
            total = msg_count + cc_count

            logger.debug(
                f"Question count for organization {self.organization_id} (user_id={self.user_id}): "
                f"messages={msg_count}, chat_completions={cc_count}, total={total}"
            )

            return total
        except Exception as e:
            logger.error(
                f"Error counting questions for organization {self.organization_id} (user_id={self.user_id}): {str(e)}"
            )
            raise UsageServiceError(f"Failed to count questions: {str(e)}") from None
