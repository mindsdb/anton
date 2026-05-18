import functools
import traceback

from minds.common.logger import get_logger
from minds.requests.context import Context, create_langfuse_context

logger = get_logger(__name__)


class _NoOpLangfuseClient:
    def get_current_trace_id(self):
        return "disabled"

    def get_current_observation_id(self):
        return "disabled"

    def update_current_trace(self, **kwargs):
        pass


def get_client():
    """Lazy accessor for the langfuse client. Defers the (heavy) langfuse import
    until first request, so workers don't pay the cost during startup."""
    try:
        from langfuse import get_client as _langfuse_get_client

        return _langfuse_get_client()
    except ImportError:
        return _NoOpLangfuseClient()


def lazy_observe(**observe_kwargs):
    """Lazy variant of langfuse's @observe decorator.

    Defers importing langfuse until the wrapped function is first called, so
    decorating a module-level function does not trigger the import at boot.
    """

    def decorator(fn):
        _wrapped = None

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            nonlocal _wrapped
            if _wrapped is None:
                try:
                    from langfuse import observe

                    _wrapped = observe(**observe_kwargs)(fn)
                except ImportError:
                    _wrapped = fn
            return await _wrapped(*args, **kwargs)

        return wrapper

    return decorator


def setup_langfuse_observation(context: Context):
    """
    Set up Langfuse observation for the current request context.
    This function creates a Langfuse context from the provided request context,
    updates the current observation, and retrieves the trace ID.

    Args:
            context (Context): The context for the request, including user and metadata.
    Returns:
            str: The trace ID for the current observation, or a default request ID if an error occurs.
    """
    logger.debug(f"Setting up Langfuse observation for context: {context}")

    try:
        current_langfuse_context = create_langfuse_context(context)

        tags_string = [str(item) for item in current_langfuse_context.tags]
        metadata_dict = current_langfuse_context.metadata.model_dump()

        langfuse_client = get_client()

        # Create Langfuse context for tracing
        langfuse_client.update_current_trace(
            user_id=str(context.user_id),
            metadata=metadata_dict,
            tags=tags_string,
        )

        trace_id = langfuse_client.get_current_trace_id()
        logger.debug(f"Trace ID: {trace_id}")

        if trace_id:
            logger.debug(f"Created langfuse context with trace ID: {trace_id}")
        else:
            logger.error("Failed to retrieve trace ID from Langfuse context.")

    except Exception as e:
        logger.error(f"Error updating Langfuse observation: {e}")
        logger.error(traceback.format_exc())


def get_langfuse_trace_id():
    """
    Get the trace ID for the current Langfuse observation.
    Returns:
            str: The trace ID for the current observation, or a default request ID if an error occurs.
    """
    try:
        langfuse_client = get_client()
        return langfuse_client.get_current_trace_id()
    except Exception as e:
        logger.error(f"Error getting Langfuse trace ID: {e}")
        logger.error(traceback.format_exc())
        return None
