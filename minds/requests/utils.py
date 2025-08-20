import traceback
import uuid

from langfuse.decorators import langfuse_context

from minds.common.logger import setup_logging
from minds.requests.context import create_langfuse_context, Context

# Set up logging
logger = setup_logging()


def setup_langfuse_observation(context: Context) -> str:
    """
    Set up Langfuse observation for the current request context.
    This function creates a Langfuse context from the provided request context,
    updates the current observation, and retrieves the trace ID.

    Args:
            context (Context): The context for the request, including user and metadata.
    Returns:
            str: The trace ID for the current observation, or a default request ID if an error occurs.
    """
    default_request_id = str(uuid.uuid4())

    try:
        current_langfuse_context = create_langfuse_context(context)

        # Create Langfuse context for tracing
        langfuse_context.update_current_observation(
            user_id=current_langfuse_context.user_id,
            metadata=current_langfuse_context.metadata,
            tags=current_langfuse_context.tags,
        )

        trace_id = langfuse_context.get_current_trace_id()

        if trace_id:
            logger.debug(f"Created langfuse context with trace ID: {trace_id}")
            return trace_id
        else:
            logger.error("Failed to retrieve trace ID from Langfuse context.")
            return default_request_id

    except Exception as e:
        logger.error(f"Error updating Langfuse observation: {e}")
        logger.error(traceback.format_exc())

        return default_request_id
