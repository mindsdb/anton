import traceback
import uuid

try:
    from langfuse import get_client
except ImportError:
    class NoOpClient:
        def get_current_trace_id(self): return "disabled"
        def get_current_observation_id(self): return "disabled"
        def update_current_trace(self, **kwargs): pass
    get_client = lambda: NoOpClient()
from minds.common.logger import setup_logging
from minds.requests.context import Context, create_langfuse_context

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

        langfuse_client = get_client()

        # Create Langfuse context for tracing
        langfuse_client.update_current_trace(
            user_id=current_langfuse_context.user_id,
            metadata=current_langfuse_context.metadata,
            tags=current_langfuse_context.tags,
        )

        trace_id = langfuse_client.get_current_trace_id()
        logger.debug(f"Trace ID: {trace_id}")

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
