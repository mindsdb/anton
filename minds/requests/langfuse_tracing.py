import contextlib
import functools
import traceback
from typing import Any

from minds.common.logger import get_logger
from minds.requests.context import Context, create_langfuse_context

class _NoOpObservation:
    """No-op observation returned by NoOpClient.start_observation."""

    def end(self, **kwargs):
        return self

    def update(self, **kwargs):
        return self

class NoOpClient:
    def get_current_trace_id(self):
        return "disabled"

logger = get_logger(__name__)


class _NoOpLangfuseClient:
    def get_current_trace_id(self):
        return "disabled"

    def update_current_generation(self, **kwargs):
        pass

    def start_observation(self, **kwargs):
        return _NoOpObservation()

    def get_current_observation_id(self):
        return "disabled"

    def get_client():
        return NoOpClient()

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


def capture_langfuse_generation_context() -> dict | None:
    """
    Capture the current Langfuse trace_id + observation_id so a generation can
    be enriched (or a child generation attached) AFTER the surrounding @observe
    scope has closed (e.g. inside a streaming background task that outlives the
    decorated request handler).

    Must be called while the @observe-decorated function is still on the call
    stack — typically right after setup_langfuse_observation(...).

    Returns:
        A dict {"trace_id": str, "parent_span_id": str} compatible with
        langfuse.types.TraceContext, or None if Langfuse is disabled / no
        active span / on error.
    """
    try:
        langfuse_client = get_client()
        trace_id = langfuse_client.get_current_trace_id()
        observation_id = langfuse_client.get_current_observation_id()

        if not trace_id or trace_id == "disabled":
            return None

        ctx: dict = {"trace_id": trace_id}
        if observation_id and observation_id != "disabled":
            ctx["parent_span_id"] = observation_id
        return ctx
    except Exception as e:
        logger.error(f"Error capturing Langfuse generation context: {e}")
        logger.error(traceback.format_exc())
        return None


def update_generation_usage(
    usage: tuple[int, int] | None,
    *,
    model: str | None = None,
    trace_context: dict | None = None,
    name: str = "llm-usage",
    input: Any = None,
    output: Any = None,
) -> None:
    """
    Record LLM token usage on a Langfuse generation.

    Two modes:
    - In-scope (``trace_context is None``): updates the *current* @observe
      generation in place via ``client.update_current_generation``. Use this
      from any code path that runs synchronously inside the @observe-decorated
      handler (all non-streaming paths).
    - Detached (``trace_context`` provided): creates a child generation
      observation attached to the trace via
      ``client.start_observation(trace_context=..., as_type="generation")`` and
      ends it. Use this when usage is captured AFTER the @observe scope has
      closed (every streaming path: the body iterator is consumed by the ASGI
      server only after the decorated handler returns).

    Token counts are emitted under Langfuse's canonical generic keys
    (``input`` / ``output`` / ``total``) so cost rolls up via Langfuse's model
    registry against the supplied ``model``.

    Args:
        usage: (input_tokens, output_tokens). If None, this is a no-op.
        model: Model identifier as stored on the DB row (e.g. "gpt-4o-mini").
        trace_context: Optional dict with "trace_id" and optional
            "parent_span_id". If provided, a child generation is attached to
            that trace; otherwise the current generation is updated in place.
        name: Name for the child generation in detached mode (ignored in
            in-scope mode).
        input: Optional input payload to attach (Langfuse displays it).
        output: Optional output payload to attach (Langfuse displays it).
    """
    if usage is None:
        logger.debug("update_generation_usage: usage is None, skipping")
        return

    input_tokens, output_tokens = usage
    in_t = int(input_tokens or 0)
    out_t = int(output_tokens or 0)
    usage_details = {
        "input": in_t,
        "output": out_t,
        "total": in_t + out_t,
    }

    try:
        langfuse_client = get_client()
        if trace_context is None:
            langfuse_client.update_current_generation(
                model=model,
                usage_details=usage_details,
                input=input,
                output=output,
            )
            logger.debug(f"Updated current Langfuse generation with usage_details={usage_details} model={model}")
        else:
            obs = langfuse_client.start_observation(
                trace_context=trace_context,
                name=name,
                as_type="generation",
                model=model,
                usage_details=usage_details,
                input=input,
                output=output,
            )
            # Defensive: ending may fail on stub clients; this is non-fatal.
            with contextlib.suppress(Exception):
                obs.end()
            logger.debug(
                "Attached child Langfuse generation "
                f"trace_id={trace_context.get('trace_id')} "
                f"parent_span_id={trace_context.get('parent_span_id')} "
                f"usage_details={usage_details} model={model}"
            )
    except Exception as e:
        logger.error(f"Error updating Langfuse generation usage: {e}")
        logger.error(traceback.format_exc())
